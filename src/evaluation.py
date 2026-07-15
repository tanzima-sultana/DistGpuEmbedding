import time
import numpy as np


class Evaluation:
    def __init__(self, k=6):
        self.k = k

    def get_topk_neighbors(self, index, query_embeddings, query_doc_ids, doc_ids,k):
        distances, indices = index.search(query_embeddings, self.k)

        results = {}
        for i in range(len(query_doc_ids)):
            cur_doc_id = query_doc_ids[i]
            neighbor_doc_ids = [doc_ids[j] for j in indices[i]]
            neighbor_doc_ids = [d for d in neighbor_doc_ids if d != cur_doc_id][:k]
            results[cur_doc_id] = neighbor_doc_ids

        return results

    def compute_recall_at_k(self, ground_truth, candidate_results, k):
        recalls = []
        for doc_id, true_neighbors in ground_truth.items():
            predicted_neighbors = candidate_results[doc_id]
            overlap = len(set(true_neighbors) & set(predicted_neighbors))
            recall = overlap / k
            recalls.append(recall)

        return np.mean(recalls)

    def measure_query_latency(self, index, query_embeddings):
        latencies = []
        for i in range(len(query_embeddings)):
            # To get 2D array (1, 384) instead of (384,) raw vector as index.search() needs 2D input. Ususally all 
            # no_of_query embeddings are fed to index.search. But we need to measure per query latency
            query = query_embeddings[i:i+1]

            start = time.time()
            index.search(query, self.k)
            elapsed_ms = (time.time() - start) * 1000

            latencies.append(elapsed_ms)
        
        # Return the median
        p50 = np.percentile(latencies, 50)
        return p50