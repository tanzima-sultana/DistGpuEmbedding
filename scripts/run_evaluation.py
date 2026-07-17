import sys
import numpy as np
import pyarrow.parquet as pq
import pickle

from sentence_transformers import SentenceTransformer, CrossEncoder
from src.indexing import FAISSIndexing
from src.evaluation import Evaluation

from constants import SEED, LOCAL, AWS, S3_BUCKET, INDEX_FLATIP, INDEX_IVF, INDEX_HNSW

if __name__ == "__main__":
    print("\n----------- Evaluation -----------\n")

    # agr 1 : local or aws
    mode = sys.argv[1]

    # arg 2 : mode -> cpu | gpu | spark | spark-gpu | gpu-adaptive
    device_mode = sys.argv[2]

    # arg 3 : Dataset size
    dataset_size = int(sys.argv[3])

    # arg 4 : no_of_queries
    no_of_queries = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    
    # Embedding path
    embedding_path = f"embeddings/{device_mode}_{dataset_size}.parquet"
    if mode == AWS:
        embedding_path = f"s3://{S3_BUCKET}/" + embedding_path  

    # Embedding
    table = pq.read_table(embedding_path)
    embeddings = np.array(table["embedding"].to_pylist(), dtype=np.float32)
    doc_ids = table["doc_id"].to_pylist()
    titles = table["title"].to_pylist()

    #print(embeddings.shape)
    #print(embeddings.dtype)

    # Indexing
    index = FAISSIndexing(mode, device_mode, dataset_size)

    # 2. Indexing
    index = FAISSIndexing(mode, device_mode, dataset_size)

    flat_index = index.generate_flat_ip(1, embeddings, dataset_size)
    print("Flat ntotal:", flat_index.ntotal)

    ivf_index = index.generate_ivf_flat(1, embeddings, dataset_size, nlist=256)
    print("IVF ntotal:", ivf_index.ntotal)

    hnsw_index = index.generate_hnsw_flat(1, embeddings, dataset_size, M=32)
    print("HNSW ntotal:", hnsw_index.ntotal)
    
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

    