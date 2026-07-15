import sys
import numpy as np
import pyarrow.parquet as pq

from src.indexing import FAISSIndexing

if __name__ == "__main__":
    print("\n-------------- Indexing --------------\n")

    # arg 1 : reload
    # reload = 0, create new embedding
    # reload = 1, load from disk if embedding exists
    reload = int(sys.argv[1])

    # arg 2 : Dataset size
    dataset_size = int(sys.argv[2])

    # arg 3 : Embedding path
    embedding_path = sys.argv[3] if len(sys.argv) > 3 else "embeddings/cpu_10000.parquet"
   
    # 1. Embedding
    table = pq.read_table(embedding_path)
    embeddings = np.array(table["embedding"].to_pylist(), dtype=np.float32)
    doc_ids = table["doc_id"].to_pylist()
    titles = table["title"].to_pylist()

    print(embeddings.shape)
    print(embeddings.dtype)

    # 2. Embedding
    index = FAISSIndexing()

    flat_index = index.generate_flat_ip(reload, embeddings, dataset_size)
    print("Flat ntotal:", flat_index.ntotal)

    ivf_index = index.generate_ivf_flat(reload, embeddings, dataset_size, nlist=256)
    print("IVF ntotal:", ivf_index.ntotal)

    hnsw_index = index.generate_hnsw_flat(reload, embeddings, dataset_size, M=32)
    print("HNSW ntotal:", hnsw_index.ntotal)