import os
import pickle
import numpy as np
import faiss
import boto3
import tempfile

from config import S3_BUCKET
from constants import LOCAL, AWS, INDEX_FLATIP, INDEX_IVF, INDEX_HNSW

class FAISSIndexing:
    def __init__(self, mode, device_mode, dataset_size):
        self.mode = mode
        self.device_mode = device_mode
        self.dataset_size = dataset_size

        self.output_path = f"index/{device_mode}_{dataset_size}/"
        if self.mode == AWS:
            self.output_path = f"s3://{S3_BUCKET}/" + self.output_path
        
        self.flatip_path = self.output_path + f"{INDEX_FLATIP}.index"
        self.ivf_path = self.output_path + f"{INDEX_IVF}.index"
        self.hnsw_path = self.output_path + f"{INDEX_HNSW}.index"
    
    def _s3_key_exists(self, key):
        s3_client = boto3.client("s3")
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=key)
            return True
        except s3_client.exceptions.ClientError:
            return False

    def _exists(self, path):
        # For AWS, check bucket in s3
        if self.mode == AWS:
            key = path.replace(f"s3://{S3_BUCKET}/", "")
            return self._s3_key_exists(key)
        # Just local filepath check
        return os.path.exists(path)

    def save(self, index, path):
        if self.mode == AWS:
            key = path.replace(f"s3://{S3_BUCKET}/", "")
            with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
                tmp_path = tmp.name
            faiss.write_index(index, tmp_path)
            s3_client = boto3.client("s3")
            s3_client.upload_file(tmp_path, S3_BUCKET, key)
            os.remove(tmp_path)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            faiss.write_index(index, path)
        print(f"Saved index to {path}")

    def load(self, path):
        if self.mode == AWS:
            key = path.replace(f"s3://{S3_BUCKET}/", "")
            s3_client = boto3.client("s3")
            with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
                tmp_path = tmp.name
            s3_client.download_file(S3_BUCKET, key, tmp_path)
            index = faiss.read_index(tmp_path)
            os.remove(tmp_path)
            return index
        
        # Local
        return faiss.read_index(path)

    def generate_flat_ip(self, reload, embeddings, dataset_size):
        if reload == 1 and self._exists(self.flatip_path):
            print(f"Load indexing ({INDEX_FLATIP}) from disk")
            return self.load(self.flatip_path)

        embeddings = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.save(index, self.flatip_path)
        return index

    def generate_ivf_flat(self, reload, embeddings, dataset_size, nlist=256):
        if reload == 1 and self._exists(self.ivf_path):
            print(f"Load indexing ({INDEX_IVF}) from disk")
            return self.load(self.ivf_path)

        embeddings = np.array(embeddings).astype('float32')
        dim = embeddings.shape[1]

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        min_train_size = 39 * nlist
        if embeddings.shape[0] < min_train_size:
            print(f"WARNING: {embeddings.shape[0]} vectors < recommended {min_train_size} for nlist={nlist}.")

        index.train(embeddings)
        index.add(embeddings)

        self.save(index, self.ivf_path)
        return index

    def generate_hnsw_flat(self, reload, embeddings, dataset_size, M=32):
        if reload == 1 and self._exists(self.hnsw_path):
            print(f"Load indexing ({INDEX_HNSW}) from disk")
            return self.load(self.hnsw_path)

        embeddings = np.array(embeddings).astype('float32')
        dim = embeddings.shape[1]

        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.add(embeddings)

        self.save(index, self.hnsw_path)
        return index