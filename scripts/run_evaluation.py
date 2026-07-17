import sys
import numpy as np
import pyarrow.parquet as pq
import pickle

from sentence_transformers import SentenceTransformer, CrossEncoder
from load_dataset import load_parquet_dataset, load_parquet_dataset_s3
from src.embedding import Embedding
from src.indexing import FAISSIndexing
from src.evaluation import Evaluation

from constants import SEED, LOCAL, AWS, CPU, GPU, SPARK, SPARK_GPU, GPU_ADAPTIVE, S3_BUCKET, INDEX_FLATIP, INDEX_IVF, INDEX_HNSW

if __name__ == "__main__":
    print("\n----------- Evaluation -----------\n")

    # agr 1 : local or aws
    mode = sys.argv[1]

    # arg 2 : mode -> cpu | gpu | spark | spark-gpu | gpu-adaptive
    device_mode = sys.argv[2]

    # arg 3 : reload
    # reload = 0, create new embedding
    # reload = 1, load from disk if embedding exists
    reload = int(sys.argv[3])

    # arg 4 : Dataset size
    dataset_size = int(sys.argv[4])

    # arg 5 : Batch size (starting batch size, for gpu-adaptive)
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 256

    # arg 6 : no_partition (only used for spark / spark-gpu)
    no_partition = int(sys.argv[6]) if len(sys.argv) > 6 else 4

    # arg 7 : model name (all-mpnet-base-v2 for adaptive)
    model_name = sys.argv[7] if len(sys.argv) > 7 else "all-MiniLM-L6-v2"

    # arg 8 : no_of_queries
    no_of_queries = int(sys.argv[8]) if len(sys.argv) > 8 else 50
    
    # 1. Dataset
    dataset = None
    if mode == LOCAL:
        dataset = load_parquet_dataset(dataset_size)
    else:
        dataset = load_parquet_dataset_s3(dataset_size)
    
    print(dataset)

    valid_device_modes = [CPU, GPU, SPARK, SPARK_GPU, GPU_ADAPTIVE]
    if device_mode not in valid_device_modes:
        print(f"Invalid mode '{device_mode}'. Choose from {valid_device_modes}")
        sys.exit(1)

    device = "cpu"
    if device_mode == GPU or device_mode == SPARK_GPU or device_mode == GPU_ADAPTIVE:
        device = "cuda"

    print(f"\n----- Eval mode: {mode} (device={device_mode}) -----")
    print(f"Dataset size: {dataset_size}, Batch size: {batch_size}",
          f", Partitions: {no_partition}", f", Model: {model_name}")

    # 2. Embedding 
    embedder = Embedding(mode, device_mode, dataset_size, model_name)

    # ----- Local 
    embeddings = None
    if mode == LOCAL:
        if device_mode in (CPU, GPU):
            embeddings = embedder.embed_plain(reload, dataset, dataset_size, batch_size, device)
        elif device_mode in (SPARK, SPARK_GPU):
            embeddings = embedder.embed_spark(reload, dataset, dataset_size, batch_size, no_partition, device)
        elif device_mode == GPU_ADAPTIVE:
            embeddings = embedder.embed_adaptive_gpu(reload, dataset, dataset_size, batch_size, device)
    else:
        embeddings = embedder.embed_spark_aws(reload, dataset, dataset_size, batch_size, no_partition, device)

    saved_table = pq.read_table(embedder.output_path)
    doc_ids = saved_table["doc_id"].to_pylist()
    titles = saved_table["title"].to_pylist()
    #print(embeddings.shape)
    #print(embeddings.dtype)

    # 3. Indexing
    index = FAISSIndexing(mode, device_mode, dataset_size)

    flat_index = index.generate_flat_ip(1, embeddings, dataset_size)
    print("Flat ntotal:", flat_index.ntotal)

    ivf_index = index.generate_ivf_flat(1, embeddings, dataset_size, nlist=256)
    print("IVF ntotal:", ivf_index.ntotal)

    hnsw_index = index.generate_hnsw_flat(1, embeddings, dataset_size, M=32)
    print("HNSW ntotal:", hnsw_index.ntotal)
    
    # 4. Evaluation
    # Choose no_of_queries from embedding
    rng = np.random.default_rng(SEED)
    query_indices = rng.choice(len(embeddings), size=no_of_queries, replace=False)

    query_embeddings = embeddings[query_indices]
    query_doc_ids = [doc_ids[i] for i in query_indices]

    print(f"Sampled {len(query_indices)} query vectors")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    k = 6
    eval = Evaluation(k=6)
    # Ground truth (Flat)
    # (doc_id, neighbor doc_ids)
    ground_truth = eval.get_topk_neighbors(flat_index, query_embeddings, query_doc_ids, doc_ids, k)

    # IVF
    ivf_results = eval.get_topk_neighbors(ivf_index, query_embeddings, query_doc_ids, doc_ids, k)

    # HNSW
    hnsw_results = eval.get_topk_neighbors(hnsw_index, query_embeddings, query_doc_ids, doc_ids, k)

    # Recall@5
    ivf_recall = eval.compute_recall_at_k(ground_truth, ivf_results, k)
    hnsw_recall = eval.compute_recall_at_k(ground_truth, hnsw_results, k)

    print(f"IVF Recall@5: {ivf_recall:.3f}")
    print(f"HNSW Recall@5: {hnsw_recall:.3f}")

    # Latency
    flat_p50 = eval.measure_query_latency(flat_index, query_embeddings)
    ivf_p50 = eval.measure_query_latency(ivf_index, query_embeddings)
    hnsw_p50 = eval.measure_query_latency(hnsw_index, query_embeddings)

    print(f"Flat p50 latency: {flat_p50:.3f} ms")
    print(f"IVF p50 latency: {ivf_p50:.3f} ms")
    print(f"HNSW p50 latency: {hnsw_p50:.3f} ms")

    # Checking for diff nprobe
    for nprobe in [1, 8, 32]:
        ivf_index.nprobe = nprobe

        ivf_results = eval.get_topk_neighbors(ivf_index, query_embeddings, query_doc_ids, doc_ids, k)
        ivf_recall = eval.compute_recall_at_k(ground_truth, ivf_results,k)
        ivf_p50 = eval.measure_query_latency(ivf_index, query_embeddings)

        print(f"IVF nprobe={nprobe}: Recall@5={ivf_recall:.3f}, p50={ivf_p50:.3f} ms")

    