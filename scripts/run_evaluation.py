import sys
import numpy as np
import pyarrow.parquet as pq
import pickle

from sentence_transformers import SentenceTransformer, CrossEncoder
from src.evaluation import Evaluation

from constants import SEED

INDEX_PATH_IP="indexing/FlatIndexIP/10000.pkl"
INDEX_PATH_IVF="indexing/IVFFlatIndex/10000.pkl"
INDEX_PATH_HNSW="indexing/HNSWFlatIndex/10000.pkl"

if __name__ == "__main__":
    print("\n----------- Evaluation -----------\n")

    # arg 1 : Dataset size
    dataset_size = int(sys.argv[1])

    # arg 2 : Embedding path
    embedding_path = sys.argv[2] if len(sys.argv) > 2 else "embeddings/cpu_10000.parquet"

    # arg 3 : no_of_queries
    no_of_queries = int(sys.argv[3]) if len(sys.argv) > 3 else 50
   
    # Embedding
    table = pq.read_table(embedding_path)
    embeddings = np.array(table["embedding"].to_pylist(), dtype=np.float32)
    doc_ids = table["doc_id"].to_pylist()
    titles = table["title"].to_pylist()

    #print(embeddings.shape)
    #print(embeddings.dtype)

    # Indexing
    indexing_flatip = None
    with open(INDEX_PATH_IP, 'rb') as f:
        indexing_flatip = pickle.load(f)
    
    indexing_ivf = None
    with open(INDEX_PATH_IVF, 'rb') as f:
        indexing_ivf = pickle.load(f)
    
    indexing_hnsw = None
    with open(INDEX_PATH_HNSW, 'rb') as f:
        indexing_hnsw = pickle.load(f)
    
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
    ground_truth = eval.get_topk_neighbors(indexing_flatip, query_embeddings, query_doc_ids, doc_ids, k)

    # IVF
    ivf_results = eval.get_topk_neighbors(indexing_ivf, query_embeddings, query_doc_ids, doc_ids, k)

    # HNSW
    hnsw_results = eval.get_topk_neighbors(indexing_hnsw, query_embeddings, query_doc_ids, doc_ids, k)

    # Recall@5
    ivf_recall = eval.compute_recall_at_k(ground_truth, ivf_results, k)
    hnsw_recall = eval.compute_recall_at_k(ground_truth, hnsw_results, k)

    print(f"IVF Recall@5: {ivf_recall:.3f}")
    print(f"HNSW Recall@5: {hnsw_recall:.3f}")

    # Latency
    flat_p50 = eval.measure_query_latency(indexing_flatip, query_embeddings)
    ivf_p50 = eval.measure_query_latency(indexing_ivf, query_embeddings)
    hnsw_p50 = eval.measure_query_latency(indexing_hnsw, query_embeddings)

    print(f"Flat p50 latency: {flat_p50:.3f} ms")
    print(f"IVF p50 latency: {ivf_p50:.3f} ms")
    print(f"HNSW p50 latency: {hnsw_p50:.3f} ms")

    # Checking for diff nprobe
    for nprobe in [1, 8, 32]:
        indexing_ivf.nprobe = nprobe

        ivf_results = eval.get_topk_neighbors(indexing_ivf, query_embeddings, query_doc_ids, doc_ids, k)
        ivf_recall = eval.compute_recall_at_k(ground_truth, ivf_results,k)
        ivf_p50 = eval.measure_query_latency(indexing_ivf, query_embeddings)

        print(f"IVF nprobe={nprobe}: Recall@5={ivf_recall:.3f}, p50={ivf_p50:.3f} ms")

    