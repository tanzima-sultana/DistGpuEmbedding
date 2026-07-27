# DistEmbedding

A distributed text embedding pipeline built on Apache Spark and AWS EMR. It embeds a real Wikipedia corpus of 500K documents across a multi-node cluster using transformer-based sentence encoders, then builds FAISS indexes (Flat, IVF, HNSW) and evaluates retrieval quality with Recall@k and latency benchmarks. GPU-accelerated inference runs on Tesla T4 nodes; an adaptive OOM-aware batcher handles GPU memory pressure automatically.

The engineering focus is distributed execution: partitioning the corpus across Spark executors, scaling throughput with worker count, and diagnosing the cluster-level failures (driver memory, executor environment, resource contention).

**Key results:**
- Processed **500K documents** end-to-end on AWS EMR with Tesla T4 GPUs, in **10.7 minutes**
- GPU speedup: **~11x** over CPU (measured at 100K docs)
- Throughput held steady and even grew slightly as scale increased — **93.9 texts/sec at 5K up to 775.9 texts/sec at 500K**
- **80.7% Recall@5** at nprobe=32 (IVF) vs **81.0%** (HNSW), measured at 100K docs

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

Throughput scales with the number of worker nodes: more workers means more partitions processed in parallel, which means higher aggregate throughput — up to the point where partition count and cluster size are balanced. Too few partitions underutilizes available workers; too many adds per-partition overhead.

The driver-side collection step in stage 3 has its own scaling limits, separate from the embedding computation. See [Known Issues](#known-issues--failure-modes) for details.

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

### CPU vs GPU Embedding Throughput

| Scale | Mode | Embedding Time (compute) | Throughput (texts/sec) |
|---|---|---|---|
| 10K | Spark (CPU) | 342.69s | 29.2 |
| 10K | Spark-GPU | 47.67s | 209.8 |
| 50K | Spark (CPU) | 947.42s | 52.8 |
| 50K | Spark-GPU | 89.10s | 561.1 |
| 100K | Spark (CPU) | 1674.29s | 59.7 |
| 100K | Spark-GPU | 151.75s | 659.0 |

*Times reflect embedding compute only (model inference), excluding dataset load and S3 write.*

**GPU speedup grew from ~7.2x at 10K to ~11.0x at 100K**, as fixed Spark session/serialization overhead is amortized across a larger workload. See [Cost Analysis](#cost-analysis) for per-run cost figures.

### GPU-Only Scaling (250K–500K)

| Scale | Embedding Time (compute) | Throughput (texts/sec) |
|---|---|---|
| 250K | 350.94s | 712.4 |
| 500K | 644.42s | 775.9 |

Throughput kept climbing all the way to 500K rather than plateauing, suggesting the architecture hasn't hit its scaling ceiling within this range. CPU comparison was not run at 250K/500K — see [Known Issues](#known-issues--failure-modes) for the master-node crash encountered during an earlier 250K CPU attempt, which led to the decision to run these two sizes on GPU only.

### Index Recall / Latency (IVF vs HNSW)

| Scale | IVF Recall@5 (nprobe=32) | HNSW Recall@5 | IVF p50 (nprobe=32) | HNSW p50 |
|---|---|---|---|---|
| 10K | 0.823 | 0.833 | 0.996 ms | 0.267 ms |
| 50K | 0.830 | 0.803 | 4.349 ms | 0.411 ms |
| 100K | 0.807 | 0.810 | 8.543 ms | 0.430 ms |

Recall does not improve monotonically with scale — this reflects natural variance from a fixed 50-query eval sample against a growing corpus, not a methodology issue. Indexing/evaluation were not run at 250K/500K scale (see [Known Issues](#known-issues--failure-modes) — single-node driver memory limits during large-scale `collect()`).

## Cost Analysis

**Rate assumptions:**
- EC2 on-demand, g4dn.xlarge, us-east-1: **$0.526/hr** (confirmed against AWS pricing)
- EC2 on-demand, r5.xlarge, us-east-1: **$0.252/hr** (confirmed against AWS pricing)
- EMR per-instance surcharge: **~$0.07/hr** (estimate — not verified against official EMR pricing page)

### Full Pipeline Cost (10K–100K)

Cluster: 1 primary + 2 core nodes, all g4dn.xlarge. Cost covers embedding, indexing, and evaluation.

| Scale | Mode | Cost |
|---|---|---|
| 10K | Spark (CPU) | $0.1936 |
| 10K | Spark-GPU | $0.0359 |
| 50K | Spark (CPU) | $0.4974 |
| 50K | Spark-GPU | $0.0727 |
| 100K | Spark (CPU) | $0.9104 |
| 100K | Spark-GPU | $0.1263 |

GPU runs cost 5–7x less than CPU runs at the same scale. This came from shorter runtime, not a lower hourly rate — GPU instances cost more per hour but finish faster.

### Embedding-Only Cost (250K–500K)

Cluster: 1 primary (r5.xlarge) + 2 core nodes (g4dn.xlarge). Cost covers embedding only — indexing and evaluation were not run at this scale. 

| Scale | Mode | Embedding Cost |
|---|---|---|
| 250K | Spark-GPU | $0.15 |
| 500K | Spark-GPU | $0.27 |

## Repository Structure

```
DistGpuEmbedding/
├── aws/
│   └── bootstrap.sh          # EMR bootstrap action: installs torch, faiss-gpu,
│                              # sentence-transformers, sets LD_LIBRARY_PATH
├── src/
│   ├── dataset.py             # Dataset loading and S3 caching
│   ├── embedding.py           # Embedding class: CPU/GPU/Spark/Spark-GPU modes
│   ├── indexing.py            # FAISSIndexing class: Flat, IVF, HNSW build
│   └── evaluation.py          # Evaluation class: Recall@k, latency, cost
├── scripts/
│   ├── run_embedding.py       # Entry point: embedding only
│   ├── run_indexing.py        # Entry point: indexing only
│   ├── run_evaluation.py      # Entry point: evaluation only
│   └── run_pipeline.py        # Entry point: full pipeline
├── data/                      # Generated: cached dataset parquet files (gitignored)
├── embeddings/                # Generated: embedding output parquet files (gitignored)
├── index/                     # Generated: FAISS index files (gitignored)
├── constants.py                # Shared, non-sensitive constants (modes, index names, cost rates)
├── config.py                   # User-specific: S3 bucket, local dataset path (gitignored)
├── config.py.template          # Template for config.py
├── run.sh                      # Local pipeline runner
├── aws_run.sh                  # AWS EMR pipeline runner
├── requirements.txt
└── README.md
```

## Installation / Setup

### Prerequisites

- Python 3.9
- AWS account with EMR, EC2, S3, and IAM access
- AWS CLI configured (`aws configure`)
- CUDA-capable GPU (optional for local runs — falls back to CPU)
- `g4dn.xlarge` and `r5.xlarge`


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

### AWS EMR Setup

**Step 1 — Upload bootstrap script to S3:**
```bash
aws s3 cp bootstrap.sh s3://bucket-name/bootstrap/bootstrap.sh
```

**Step 2 — Create EMR cluster** (console or CLI):
- Applications: `Hadoop`, `Spark`
- Primary node: `[instance type]` x1
- Core nodes: `g4dn.xlarge` x[N]
- Bootstrap action: `s3://bucket-name/bootstrap/bootstrap.sh`
- EC2 key pair: your key pair
- Service role: `EMR_DefaultRole`
- EC2 instance profile: `EMR_EC2_DefaultRole`

**Step 3 — SSH into master and clone the repo:**
```bash
git clone https://github.com/tanzima-sultana/DistGpuEmbedding.git
cd DistGpuEmbedding
```
## How to Run

Copy the config template and fill in your own values:
```bash
cp config.py.template config.py
```

Edit `config.py` with your S3 bucket name and local dataset path.

### Local Run

Run the full pipeline (embedding → indexing → evaluation) with a single command:

```bash
./run.sh
```

Edit the variables at the top of `run.sh` to make changes:

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

## Known Issues / Failure Modes

### FAISS + NumPy 2.x ABI incompatibility

`faiss-gpu==1.7.2`'s SWIG bindings fail with numpy 2.x. `index.add()` raises `ValueError: input not a numpy array` even when passed a genuine numpy array, because numpy 2.x changed its internal array representation in a way `swig_ptr` doesn't recognize.

**Fix:** pin `numpy<2` (numpy 1.26.4 used throughout this project).

### torch/CUDA version must match the driver, not just "latest"

Installing `torch` with a CUDA index newer than the instance's driver support causes `ImportError: libcusparseLt.so.0: cannot open shared object file` at `import torch`.

**Fix:** match the `--index-url` CUDA version to `nvidia-smi`'s reported CUDA Version. This project uses `torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126`, verified against driver 560.35.03 / CUDA 12.6.

### `LD_LIBRARY_PATH` via `/etc/environment` doesn't reach Spark executors

Writing `LD_LIBRARY_PATH` to `/etc/environment` only takes effect for new login sessions. It doesn't propagate to YARN-spawned Spark executor processes.

**Fix:** compute the NVIDIA library path in `aws_run.sh` and pass it explicitly via `--conf spark.executorEnv.LD_LIBRARY_PATH=...` and `--conf spark.driverEnv.LD_LIBRARY_PATH=...` on every `spark-submit` call.

### OOM at 250K with a fixed partition count

Running embedding at 250K documents with `NUM_PARTITIONS=4` (tuned for 5K–50K) caused `java.lang.OutOfMemoryError: Java heap space` during task serialization.

**Fix:** partition count needs to scale with dataset size. Increased to 16–24 partitions, with explicit `spark.executor.memory` and `spark.executor.memoryOverhead` instead of Spark defaults.

### Master-node CPU starvation crashes the driver via HDFS lease timeout

At 250K scale with `--deploy-mode client`, the Spark driver, HDFS NameNode, and Spark event logging competed for CPU on the master node. Under load, the NameNode couldn't service the driver's lease-renewal RPC within 60 seconds, causing a cascading failure: `SocketTimeoutException` → HDFS lease abort → `SparkContext.stop()`.

**Fix:** disabled Spark event logging (`spark.eventLog.enabled=false`) and dynamic allocation (`spark.dynamicAllocation.enabled=false`, which was reclaiming idle executors mid-stage and worsening the imbalance).

### Driver OOM-killed on a 16GB primary node at 250K+

Collecting 250K embedding vectors back to the driver via Spark's `collect()` used roughly 5.6GB of resident memory in the Python process, confirmed from the kernel log:

```
Out of memory: Killed process 50816 (python3) total-vm:23014832kB, anon-rss:5864788kB
```

The process was killed by the Linux OOM killer, not a JVM-level error. It failed silently with no traceback in the application log, visible only via `dmesg`.

**Fix:** switched the primary node from `g4dn.xlarge` (16GB) to `r5.xlarge` (32GB) for the 250K and 500K runs.

### `spark.driver.maxResultSize` default ceiling at 500K

At 500K documents, the combined serialized size of collected task results (1046.6 MiB) exceeded Spark's default `spark.driver.maxResultSize` of 1024 MiB, aborting the job:

```
Total size of serialized results of 15 tasks (1046.6 MiB) is bigger than spark.driver.maxResultSize (1024.0 MiB)
```

**Fix:** raised `spark.driver.maxResultSize` to 4g, with `spark.driver.memory` raised to 10g to keep clear headroom between the two settings.

### Dynamic allocation reclaims executors mid-stage

With `spark.dynamicAllocation.enabled` at its default, an executor that finished its assigned tasks slightly early was reclaimed as idle while the job's total task count wasn't yet complete, forcing the remaining tasks onto a single executor and roughly doubling the tail-end runtime.

**Fix:** `spark.dynamicAllocation.enabled=false` for jobs with a known, fixed task count.

## Limitations

- Indexing and evaluation are not distributed. FAISS index construction and evaluation both run single-node on the driver, on the full embedding set collected from Spark. This works at the scales tested here but would become a bottleneck well beyond 500K vectors. 

- IVF `nlist` was not scaled with corpus size. `nlist=256` was used across all dataset sizes tested. 

- `--deploy-mode client` places the Spark driver on the primary node, which also runs YARN's ResourceManager and HDFS's NameNode. This caused a real failure at 250K scale (see Known Issues). 

- No fault-tolerance or failure-injection testing was performed on this iteration.

- CPU vs GPU comparison was only measured up to 100K documents. 250K and 500K runs were GPU-only, both for cost reasons and because a CPU run at 250K caused the master-node crash.

- Indexing and evaluation were only run up to 100K documents, due to driver memory limits during `collect()` at larger scale.

- Single-region, single-cluster testing only. 

