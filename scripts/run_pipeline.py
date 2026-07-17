import sys
import numpy as np
import pyarrow.parquet as pq
import pickle
import time

from sentence_transformers import SentenceTransformer, CrossEncoder

from src.dataset import Dataset
from src.embedding import Embedding
from src.indexing import FAISSIndexing
from src.evaluation import Evaluation

from constants import SEED, LOCAL, AWS, CPU, GPU, SPARK, SPARK_GPU, GPU_ADAPTIVE, S3_BUCKET, INDEX_FLATIP, INDEX_IVF, INDEX_HNSW

if __name__ == "__main__":
    print("\n----------- Dist Gpu Embedding Pipeline-----------\n")

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

    valid_device_modes = [CPU, GPU, SPARK, SPARK_GPU, GPU_ADAPTIVE]
    if device_mode not in valid_device_modes:
        print(f"Invalid mode '{device_mode}'. Choose from {valid_device_modes}")
        sys.exit(1)

    device = "cpu"
    if device_mode == GPU or device_mode == SPARK_GPU or device_mode == GPU_ADAPTIVE:
        device = "cuda"

    print(f"\n----- DistGpuEmbedding mode: {mode} (device={device_mode}) -----")
    print(f"Dataset size: {dataset_size}, Batch size: {batch_size}",
          f", Partitions: {no_partition}", f", Model: {model_name}")
    
    s = time.time()

    # --------------------- 1. Load Dataset
    df = Dataset(dataset_size)
    dataset = None
    s1 = time.time()
    if mode == LOCAL:
        dataset = df.load_parquet_dataset()
    else:
        dataset = df.load_parquet_dataset_s3()
    t1 = time.time() - s1

    print(f"\n---- Dataset time : mode : {mode} : {device_mode} : time : {t1:.2f}s\n")

    # ---------------------- 2. Embedding 
    embedder = Embedding(mode, device_mode, dataset_size, model_name)

    # ----- Local 
    embeddings = None

    s2 = time.time()
    if mode == LOCAL:
        if device_mode in (CPU, GPU):
            embeddings = embedder.embed_plain(reload, dataset, dataset_size, batch_size, device)
        elif device_mode in (SPARK, SPARK_GPU):
            embeddings = embedder.embed_spark(reload, dataset, dataset_size, batch_size, no_partition, device)
        elif device_mode == GPU_ADAPTIVE:
            embeddings = embedder.embed_adaptive_gpu(reload, dataset, dataset_size, batch_size, device)
    else:
        embeddings = embedder.embed_spark_aws(reload, dataset, dataset_size, batch_size, no_partition, device)

    t2 = time.time() - s2
    print(f"\n---- Embeddings time : mode : {mode} : {device_mode} : time : {t2:.2f}s\n")

    saved_table = pq.read_table(embedder.output_path)
    doc_ids = saved_table["doc_id"].to_pylist()
    titles = saved_table["title"].to_pylist()

    # ----------- 3. Indexing
    index = FAISSIndexing(mode, device_mode, dataset_size)

    s3 = time.time()

    flat_index = index.generate_flat_ip(reload, embeddings, dataset_size)
    print("Flat ntotal:", flat_index.ntotal)

    ivf_index = index.generate_ivf_flat(reload, embeddings, dataset_size, nlist=256)
    print("IVF ntotal:", ivf_index.ntotal)

    hnsw_index = index.generate_hnsw_flat(reload, embeddings, dataset_size, M=32)
    print("HNSW ntotal:", hnsw_index.ntotal)

    t3 = time.time() - s3
    print(f"\n---- Index time : mode : {mode} : {device_mode} : time : {t3:.2f}s\n")

    t = time.time() - s
    # ----------- 4. Evaluation
    k = 6
    eval = Evaluation(k, doc_ids, embeddings, flat_index, ivf_index, hnsw_index)
    eval_results = eval.evaluate(no_of_queries)
    
    if mode == AWS:
        aws_cost = eval.compute_cluster_cost(1, 2, t/3600)
        print(f"Estimated cluster cost: ${aws_cost:.4f}")


    print("\n ----- Evaluation Results----")
    print(eval_results)


