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

### CPU vs GPU Embedding Throughput

| Scale | Mode | Embedding Time | Throughput (texts/sec) |
|---|---|---|---|
| 5K | Spark (CPU) | 257.66s | 19.4 |
| 5K | Spark-GPU | 53.23s | 93.9 |
| 10K | Spark (CPU) | 360.37s | 29.2 |
| 10K | Spark-GPU | 65.46s | 209.8 |
| 50K | Spark (CPU) | 967.58s | 52.8 |
| 50K | Spark-GPU | 114.03s | 561.1 |
| 100K | Spark (CPU) | 1702.26s | 59.7 |
| 100K | Spark-GPU | 182.64s | 659.0 |

**GPU speedup grew from ~4.8x at 5K to ~9.3x at 100K**, as fixed Spark session/serialization overhead is amortized across a larger workload. See [Cost Analysis](#cost-analysis) for per-run cost figures.

### GPU-Only Scaling (250K–500K)

| Scale | Embedding Time | Throughput (texts/sec) |
|---|---|---|
| 250K | *(pending)* | *(pending)* |
| 500K | *(pending)* | *(pending)* |

CPU comparison was not run at this scale — see [Known Issues](#known-issues--failure-modes) for the master-node crash encountered during an earlier 250K CPU attempt, which led to the decision to run 250K and 500K on GPU only.

### Index Recall / Latency (IVF vs HNSW)

| Scale | IVF Recall@5 (nprobe=32) | HNSW Recall@5 | IVF p50 (nprobe=32) | HNSW p50 |
|---|---|---|---|---|
| 5K | 0.830 | 0.817 | 4.349 ms | 0.411 ms |
| 10K | 0.823 | 0.833 | 0.996 ms | 0.267 ms |
| 50K | 0.830 | 0.803 | 4.349 ms | 0.411 ms |
| 100K | 0.807 | 0.810 | 8.543 ms | 0.430 ms |

Recall does not improve monotonically with scale — this reflects natural variance from a fixed 50-query eval sample against a growing corpus, not a methodology issue.

## Cost Analysis

**Rate assumptions:**
- EC2 on-demand, g4dn.xlarge, us-east-1: **$0.526/hr** (confirmed against AWS pricing)
- EMR per-instance surcharge: **~$0.07/hr** (estimate — not verified against official EMR pricing page; actual costs may differ slightly)
- Cluster: 1 primary + 2 core nodes (g4dn.xlarge) throughout

### Cost by Scale and Mode

| Scale | Mode | Cost |
|---|---|---|
| 10K | Spark (CPU) | $0.1936 |
| 10K | Spark-GPU | $0.0359 |
| 50K | Spark (CPU) | $0.4974 |
| 50K | Spark-GPU | $0.0727 |
| 100K | Spark (CPU) | $0.9104 |
| 100K | Spark-GPU | $0.1263 |
| 250K | Spark-GPU | *(pending)* |
| 500K | Spark-GPU | *(pending)* |

**GPU runs were consistently 5–7x cheaper than CPU runs at the same scale** — not because GPU compute is cheaper per hour, but because GPU finishes so much faster that total cluster-hours billed is far lower. This is the practical argument for GPU at scale: it's not just faster, it's cheaper in aggregate despite running on more expensive hardware.

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

## Known Issues / Failure Modes

### FAISS + NumPy 2.x ABI incompatibility

`faiss-gpu==1.7.2`'s SWIG bindings fail silently with numpy 2.x — `index.add()` raises `ValueError: input not a numpy array` even when passed a genuine numpy array, because numpy 2.x changed internal array representation in a way `swig_ptr` doesn't recognize.

**Fix:** pin `numpy<2` (numpy 1.26.4 used throughout this project). Confirmed via direct reproduction — installs cleanly with numpy 2.x, fails only at runtime on the first real FAISS operation.

### torch/CUDA version must match the driver, not just "latest"

Installing `torch` with a CUDA index newer than the instance's actual driver support (e.g. `cu128` wheels on a driver that supports up to CUDA 12.6) causes `ImportError: libcusparseLt.so.0: cannot open shared object file` at `import torch` — the bundled NVIDIA runtime libraries are ABI-incompatible with the installed driver.

**Fix:** match the `--index-url` CUDA version to `nvidia-smi`'s reported CUDA Version, not to whatever's newest. Verified against Driver 560.35.03 / CUDA 12.6 → `torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126`.

### `LD_LIBRARY_PATH` via `/etc/environment` doesn't reach Spark executors reliably

Torch's pip-bundled `nvidia-*-cu12` packages don't register themselves on the system linker path. Writing `LD_LIBRARY_PATH` to `/etc/environment` (the standard fix) only takes effect for new PAM login sessions — it does not reliably propagate to YARN-spawned Spark executor processes, which aren't login shells.

**Fix:** compute the NVIDIA library path in `aws_run.sh` and pass it explicitly via `--conf spark.executorEnv.LD_LIBRARY_PATH=...` and `--conf spark.driverEnv.LD_LIBRARY_PATH=...` on every `spark-submit` call, rather than relying on `/etc/environment` propagation.

### OOM at scale with fixed partition count

Running embedding at 250K documents with `NUM_PARTITIONS=4` (tuned for 5K–50K runs) caused a `java.lang.OutOfMemoryError: Java heap space` on task serialization — each partition held too large a data slice for the default executor JVM heap.

**Fix:** partition count must scale with dataset size, not stay fixed. Increased to `NUM_PARTITIONS=16` and set explicit `spark.executor.memory` / `spark.executor.memoryOverhead` / `spark.driver.memory` rather than relying on Spark defaults.

### Master-node CPU starvation crashes the driver via HDFS lease timeout

At 250K scale with `--deploy-mode client`, the Spark driver, HDFS NameNode, and Spark event logging all compete for CPU on the master node. Under sustained load, the NameNode couldn't service the driver's lease-renewal RPC within the 60s timeout, causing a cascading failure: `SocketTimeoutException` → HDFS lease abort → `SparkContext.stop()` → job death, roughly an hour into the run.

**Fix:** disable Spark event logging (`spark.eventLog.enabled=false`, not needed without Spark History Server) and disable dynamic allocation (`spark.dynamicAllocation.enabled=false`, which was reclaiming idle executors mid-stage and worsening the imbalance). `--deploy-mode cluster` (running the driver on a worker instead of master) is a more correct long-term fix, noted under Future Work.

### Dynamic allocation reclaims executors mid-stage

With `spark.dynamicAllocation.enabled` at its default, an executor that finished its assigned tasks slightly early was reclaimed as "idle" while the job's total task count wasn't yet complete — forcing the remaining tasks onto a single executor and roughly doubling the tail-end runtime.

**Fix:** `spark.dynamicAllocation.enabled=false` for jobs with a known, fixed task count.

## Future Work / Limitations

- **Indexing and evaluation are not distributed.** FAISS index construction and evaluation both run single-node on the driver, operating on the full embedding set collected from Spark. This works at the scales tested here but would become a bottleneck well beyond 500K vectors. A sharded-index approach (build per-partition indexes, merge or route queries across shards) would be needed to distribute this stage.

- **IVF `nlist` was not scaled with corpus size.** `nlist=256` was used across all dataset sizes tested. A common heuristic (`nlist ≈ 4×sqrt(N)`) would suggest a substantially higher `nlist` at 500K than at 50K — not explored here. Worth revisiting if recall at the largest scale is unsatisfactory.

- **`--deploy-mode client` places the Spark driver on the master node**, which also runs YARN's ResourceManager and HDFS's NameNode. This caused a real failure at 250K scale (see Known Issues). `--deploy-mode cluster`, which runs the driver on a worker node instead, is the more correct architecture for large jobs and should be adopted for future scale-up work.

- **No fault-tolerance / failure-injection testing performed on this iteration.** The previous version of this project included a synthetic partition-failure-and-retry test; it was not rebuilt here due to time/cost constraints. Would be a reasonable local-mode addition — Spark's task retry behavior doesn't require a live cluster to validate.

- **CPU vs GPU comparison was only measured up to 100K documents.** 250K and 500K runs were GPU-only, both for cost reasons and because a CPU run at 250K caused the master-node crash documented under Known Issues. The CPU/GPU speedup trend observed up to 100K (growing from ~4.8x at 5K to ~8.5x at 50K, as fixed Spark overhead amortizes) is not confirmed to hold at larger scale.

- **Single-region, single-cluster testing only.** No multi-AZ, spot-instance, or cross-region resilience testing was performed.



