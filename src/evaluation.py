import time
import numpy as np

from constants import SEED, EC2_RATE_G4DN_XLARGE, EMR_MARKUP_G4DN_XLARGE

class Evaluation:
    def __init__(self, k, doc_ids, embeddings, flat_index, ivf_index, hnsw_index):
        self.k = k
        self.doc_ids = doc_ids
        self.embeddings = np.array(embeddings, dtype='float32')
        self.flat_index = flat_index
        self.ivf_index = ivf_index
        self.hnsw_index = hnsw_index

    def get_topk_neighbors(self, index, query_embeddings, query_doc_ids, doc_ids,k):
        distances, indices = index.search(query_embeddings, self.k)

        results = {}
        for i in range(len(query_doc_ids)):
            cur_doc_id = query_doc_ids[i]
            # Find neighbor doc_ids from indices
            neighbor_doc_ids = [doc_ids[j] for j in indices[i]]
            # Just remove the cur doc_id if its there
            neighbor_doc_ids = [d for d in neighbor_doc_ids if d != cur_doc_id][:k]
            # result has all neighbour doc_ids of cur doc_id
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
    
    def evaluate(self, no_of_queries):
        print("\n ----- Evaluation ----")

        rng = np.random.default_rng(SEED)
        query_indices = rng.choice(len(self.embeddings), size=no_of_queries, replace=False)

        query_embeddings = self.embeddings[query_indices]
        query_doc_ids = [self.doc_ids[i] for i in query_indices]

        print(f"Sampled {len(query_indices)} query vectors")
        
        # Ground truth (Flat)
        # (doc_id, neighbor doc_ids)
        ground_truth = self.get_topk_neighbors(self.flat_index, query_embeddings, query_doc_ids, self.doc_ids, self.k)

        # IVF
        ivf_results = self.get_topk_neighbors(self.ivf_index, query_embeddings, query_doc_ids, self.doc_ids, self.k)

        # HNSW
        hnsw_results = self.get_topk_neighbors(self.hnsw_index, query_embeddings, query_doc_ids, self.doc_ids, self.k)

        # Recall@5
        ivf_recall = self.compute_recall_at_k(ground_truth, ivf_results, self.k)
        hnsw_recall = self.compute_recall_at_k(ground_truth, hnsw_results, self.k)

        print(f"IVF Recall@5: {ivf_recall:.3f}")
        print(f"HNSW Recall@5: {hnsw_recall:.3f}")

        # Latency
        flat_p50 = self.measure_query_latency(self.flat_index, query_embeddings)
        ivf_p50 = self.measure_query_latency(self.ivf_index, query_embeddings)
        hnsw_p50 = self.measure_query_latency(self.hnsw_index, query_embeddings)

        print(f"Flat p50 latency: {flat_p50:.3f} ms")
        print(f"IVF p50 latency: {ivf_p50:.3f} ms")
        print(f"HNSW p50 latency: {hnsw_p50:.3f} ms")

        # Checking for diff nprobe
        for nprobe in [1, 8, 32]:
            self.ivf_index.nprobe = nprobe

            ivf_results = self.get_topk_neighbors(self.ivf_index, query_embeddings, query_doc_ids, self.doc_ids, self.k)
            ivf_recall = self.compute_recall_at_k(ground_truth, ivf_results,self.k)
            ivf_p50 = self.measure_query_latency(self.ivf_index, query_embeddings)

            print(f"IVF nprobe={nprobe}: Recall@5={ivf_recall:.3f}, p50={ivf_p50:.3f} ms")
        
        return []
    
    def compute_cluster_cost(self, num_primary, num_core, hours_running,
                          ec2_rate=EC2_RATE_G4DN_XLARGE,
                          emr_markup=EMR_MARKUP_G4DN_XLARGE):
        total_nodes = num_primary + num_core
        hourly_rate_per_node = ec2_rate + emr_markup
        total_cost = total_nodes * hourly_rate_per_node * hours_running
        return total_cost