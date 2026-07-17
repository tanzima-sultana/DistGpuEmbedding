import sys
import numpy as np
import pyarrow.parquet as pq
import pickle
import boto3
import os
import faiss 
import tempfile
from sentence_transformers import SentenceTransformer 

from src.evaluation import Evaluation

from constants import SEED, LOCAL, AWS, CPU, GPU, SPARK, SPARK_GPU, GPU_ADAPTIVE, S3_BUCKET, INDEX_FLATIP, INDEX_IVF, INDEX_HNSW

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
        
    table = pq.read_table(embedding_path)
    embeddings = np.array(table["embedding"].to_pylist(), dtype=np.float32)

    doc_ids = table["doc_id"].to_pylist()
    titles = table["title"].to_pylist()

    # Index path

    def load_index(path):
        if mode == AWS:
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
    
    index_path = f"index/{device_mode}_{dataset_size}/"
    if mode == AWS:
        index_path = f"s3://{S3_BUCKET}/" + index_path
    
    flat_index = load_index(index_path + f"{INDEX_FLATIP}.index")
    ivf_index = load_index(index_path + f"{INDEX_IVF}.index")
    hnsw_index = load_index(index_path + f"{INDEX_HNSW}.index")
    
    # Evaluation
    k = 6
    eval = Evaluation(k, doc_ids, embeddings, flat_index, ivf_index, hnsw_index)
    eval_results = eval.evaluate(no_of_queries)

    print(eval_results)
    