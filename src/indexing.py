import os
import pickle
import numpy as np
import faiss

from constants import INDEX_FLATIP, INDEX_IVF, INDEX_HNSW

class FAISSIndexing:
    def __init__(self, index_dir="indexing"):
        self.index_dir = index_dir

    def save(self, index, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(index, f)

    # 1. FlatIP
    def generate_flat_ip(self, reload, embeddings, dataset_size):
        path = f"{self.index_dir}/{INDEX_FLATIP}/{dataset_size}.pkl"
        if reload == 1 and os.path.exists(path):
            with open(path, 'rb') as f:
                print(f"Load indexing ({INDEX_FLATIP}) from disk")
                return pickle.load(f)

        embeddings = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.save(index, path)
        return index

    # IVF
    def generate_ivf_flat(self, reload, embeddings, dataset_size, nlist=256):
        path = f"{self.index_dir}/{INDEX_IVF}/{dataset_size}.pkl"
        if reload == 1 and os.path.exists(path):
            with open(path, 'rb') as f:
                print(f"Load indexing ({INDEX_IVF}) from disk")
                return pickle.load(f)

        embeddings = np.array(embeddings).astype('float32')
        dim = embeddings.shape[1]

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        min_train_size = 39 * nlist
        if embeddings.shape[0] < min_train_size:
            print(f"WARNING: {embeddings.shape[0]} vectors < recommended {min_train_size} for nlist={nlist}.")

        index.train(embeddings)
        index.add(embeddings)

        self.save(index, path)
        return index

    # HNSW
    def generate_hnsw_flat(self, reload, embeddings, dataset_size, M=32):
        path = f"{self.index_dir}/{INDEX_HNSW}/{dataset_size}.pkl"
        if reload == 1 and os.path.exists(path):
            with open(path, 'rb') as f:
                print(f"Load indexing ({INDEX_HNSW}) from disk")
                return pickle.load(f)

        embeddings = np.array(embeddings).astype('float32')
        dim = embeddings.shape[1]

        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.add(embeddings)

        self.save(index, path)
        return index