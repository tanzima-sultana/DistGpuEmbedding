# DistributedGpuEmbedding

A distributed text embedding pipeline built on **Apache Spark** and **CUDA**, running at scale across GPU-accelerated AWS EMR nodes. Embeds a real-world corpus (Wikipedia, 500K documents) using transformer-based sentence encoders, with FAISS vector indexing (Flat / IVF / HNSW) and retrieval quality evaluation via Recall@k and latency benchmarking. Includes an adaptive OOM-aware batcher, validated locally, for automatic safe batch-size discovery under GPU memory pressure.

**Key results:**
- Processed **500K documents** end-to-end on AWS EMR with Tesla T4 GPUs
- GPU speedup: **[X]x** over CPU (measured up to 100K docs)
- Scaled to **500K documents** on a GPU-only pipeline, completing in **[X] minutes** at **$[X]** total cost
- **[X]% Recall@5** at nprobe=32 (IVF) vs **[X]%** (HNSW)

## Architecture
'''
Raw Wikipedia Corpus (S3)
|
v
+--------------------------------------------+
|      Apache Spark Cluster (AWS EMR)         |
|                                              |
|  Partition 0      Partition 1      ...      |
|  GPU Embed         GPU Embed                |
|         mapPartitions()                     |
|   (model loaded once per partition)         |
|                                              |
|   DISTRIBUTED -- scales with worker count    |
+--------------------------------------------+
|
v
Embeddings (S3, Parquet)
|
v
+--------------------------------------------+
|       FAISS Index Build (Driver)            |
|   Flat (exact) . IVFFlat . HNSWFlat          |
|                                              |
|      SINGLE-NODE -- not distributed          |
+--------------------------------------------+
|
v
+--------------------------------------------+
|          Evaluation (Driver)                 |
|   Recall@k . p50 latency . nprobe sweep      |
|                                              |
|      SINGLE-NODE -- not distributed          |
+--------------------------------------------+
'''
### How Embedding Distribution Works

Spark's `mapPartitions` splits the input corpus (loaded from S3) into N partitions, one per configured split. Each partition is sent to a Spark executor running on a worker node, where:

1. The sentence-transformer model is loaded **once per partition**, not once per document — avoiding the overhead of reloading a ~90MB model thousands of times within a single partition.
2. All documents in that partition are embedded locally, using the GPU available on that worker node (`spark-gpu` mode) or CPU (`spark` mode).
3. Resulting embedding vectors are returned to the driver and collected into a single output, written to S3 as Parquet.

This means throughput scales with the number of worker nodes: more workers → more partitions processed in parallel → higher aggregate throughput, up to the point where partition count and cluster size are balanced (too few partitions underutilizes available workers; too many adds per-partition overhead).

### Deployment Modes

| Mode     | Spark                 | Device       | Storage          |
|----------|-----------------------|--------------|------------------|
| Local    | PySpark (single node) | CPU or GPU   | Local filesystem |
| AWS EMR  | YARN cluster          | Tesla T4 GPU | Amazon S3        |


