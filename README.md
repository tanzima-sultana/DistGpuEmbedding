# DistributedGpuEmbedding

A distributed text embedding pipeline built on **Apache Spark** and **CUDA**, running at scale across GPU-accelerated AWS EMR nodes. Embeds a real-world corpus (Wikipedia, 500K documents) using transformer-based sentence encoders, with FAISS vector indexing (Flat / IVF / HNSW) and retrieval quality evaluation via Recall@k and latency benchmarking. Includes an adaptive OOM-aware batcher, validated locally, for automatic safe batch-size discovery under GPU memory pressure.

**Key results:**
- Processed **500K documents** end-to-end on AWS EMR with Tesla T4 GPUs
- GPU speedup: **[X]x** over CPU (measured up to 100K docs)
- Scaled to **500K documents** on a GPU-only pipeline, completing in **[X] minutes** at **$[X]** total cost
- **[X]% Recall@5** at nprobe=32 (IVF) vs **[X]%** (HNSW)

## Architecture
```
Raw Wikipedia Corpus (S3)
|
v
+--------------------------------------------+
|      Apache Spark Cluster (AWS EMR)         |
|                                             |
|  Partition 0      Partition 1      ...      |
|  GPU Embed         GPU Embed                |
|         mapPartitions()                     |
|   (model loaded once per partition)         |
|                                             |
|   DISTRIBUTED -- scales with worker count   |
+--------------------------------------------+
|
v
Embeddings (S3, Parquet)
|
v
+--------------------------------------------+
|       FAISS Index Build (Driver)            |
|   Flat (exact) . IVFFlat . HNSWFlat         |
|                                             |
|      SINGLE-NODE -- not distributed         |
+--------------------------------------------+
|
v
+--------------------------------------------+
|          Evaluation (Driver)                 |
|   Recall@k . p50 latency . nprobe sweep      |
|                                              |
|      SINGLE-NODE -- not distributed          |
+--------------------------------------------+
```
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

## Tech Stack

**AI & Embeddings**
- [sentence-transformers](https://www.sbert.net/) — transformer-based text embedding models (`all-MiniLM-L6-v2`)
- PyTorch (CUDA 12.6) — GPU-accelerated inference
- FAISS — vector similarity search (Flat, IVFFlat, HNSWFlat)

**Distributed Computing**
- Apache Spark (PySpark) — distributed `mapPartitions` embedding job
- YARN — cluster resource management on AWS EMR

**Cloud & Infrastructure**
- AWS EMR — managed Spark cluster (1x primary + N core nodes)
- AWS S3 — corpus, embeddings, and index storage
- AWS IAM — role-based access control

**Programming & Libraries**
- Python 3.9
- NumPy (<2.0, pinned for FAISS ABI compatibility)
- tqdm, PyYAML
- Boto3 / s3fs (AWS access)

**Tooling**
- `bootstrap.sh` — EMR node dependency installation (torch, FAISS, sentence-transformers), run as an EMR bootstrap action at cluster launch
- `aws_run.sh` — single-command pipeline runner: packages code, computes NVIDIA library paths, submits the Spark job

## Performance Summary

*(To be added)*

## Cost Analysis

*(To be added)*

## Installation / Setup

### Prerequisites

- Python 3.9
- AWS account with EMR, EC2, S3, and IAM access
- AWS CLI configured (`aws configure`)
- CUDA-capable GPU (optional for local runs — falls back to CPU)
- `g4dn.xlarge` instance quota in your target region (check Service Quotas → EC2 → Running On-Demand G and VT instances)

### Local Setup

Clone the repository:
```bash
git clone https://github.com/tanzima-sultana/DistGpuEmbedding.git
cd DistGpuEmbedding
```

Install dependencies:
```bash
pip install -r requirements.txt
```
## How to Run

### Local Run

Run the full pipeline (embedding → indexing → evaluation) with a single command:

```bash
./run.sh
```

Edit the variables at the top of `run.sh` to change mode, device, or dataset size:

```bash
MODE="local"                # local | aws
DEVICE_MODE="spark-gpu"     # cpu | gpu | spark | spark-gpu | gpu-adaptive
DATASET_SIZE=5000
BATCH_SIZE=256
NUM_PARTITIONS=4
```

### AWS EMR Run

**Prerequisites:**
- EMR cluster launched with `bootstrap.sh` attached as a bootstrap action (see [Installation](#installation--setup))
- Repository cloned onto the EMR master node

**Step 1 — SSH into the cluster's master node:**
```bash
ssh -i /path/to/key.pem hadoop@<master-public-ip>
```

**Step 2 — Configure `aws_run.sh`:**
```bash
MODE="aws"
DEVICE_MODE="spark-gpu"     # cpu | gpu | spark | spark-gpu | gpu-adaptive
DATASET_SIZE=500000
BATCH_SIZE=256
NUM_PARTITIONS=16
```

**Step 3 — Run the pipeline:**
```bash
nohup ./aws_run.sh > /dev/null 2>&1 &
```

This packages the project code (`project.zip`), computes the NVIDIA library path for CUDA, and submits the job via `spark-submit`. Output is logged to `log.txt` and overwritten on each run.

**Step 4 — Monitor progress:**
```bash
tail -f log.txt
yarn application -list
```

**Step 5 — Terminate the cluster once finished** — EMR billing continues until the cluster is explicitly terminated, independent of whether a job is actively running.

### AWS EMR Setup

**Step 1 — Upload bootstrap script to S3:**
```bash
aws s3 cp bootstrap.sh s3://your-bucket-name/bootstrap/bootstrap.sh
```

**Step 2 — Create EMR cluster** (console or CLI):
- Applications: `Hadoop`, `Spark`
- Primary node: `[instance type]` x1
- Core nodes: `g4dn.xlarge` x[N]
- Bootstrap action: `s3://your-bucket-name/bootstrap/bootstrap.sh`
- EC2 key pair: your key pair
- Service role: `EMR_DefaultRole`
- EC2 instance profile: `EMR_EC2_DefaultRole`

**Step 3 — SSH into master and clone the repo:**
```bash
git clone https://github.com/tanzima-sultana/DistGpuEmbedding.git
cd DistGpuEmbedding
```




