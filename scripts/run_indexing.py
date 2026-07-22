import sys
import numpy as np
import pyarrow.parquet as pq

from src.indexing import FAISSIndexing

from config import S3_BUCKET
from constants import LOCAL, AWS

if __name__ == "__main__":
    print("\n-------------- Indexing --------------\n")

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

    # Embedding path
    embedding_path = f"embeddings/{device_mode}_{dataset_size}.parquet"
    if mode == AWS:
        embedding_path = f"s3://{S3_BUCKET}/" + embedding_path  
   
    # 1. Embedding
    table = pq.read_table(embedding_path)
    embeddings = np.array(table["embedding"].to_pylist(), dtype=np.float32)

    print(embeddings.shape)
    print(embeddings.dtype)

    # 2. Indexing
    index = FAISSIndexing(mode, device_mode, dataset_size)

    flat_index = index.generate_flat_ip(reload, embeddings, dataset_size)
    print("Flat ntotal:", flat_index.ntotal)

    ivf_index = index.generate_ivf_flat(reload, embeddings, dataset_size, nlist=256)
    print("IVF ntotal:", ivf_index.ntotal)

    hnsw_index = index.generate_hnsw_flat(reload, embeddings, dataset_size, M=32)
    print("HNSW ntotal:", hnsw_index.ntotal)