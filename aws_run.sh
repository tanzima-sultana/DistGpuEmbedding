#!/bin/bash
set -e

export PYTHONPATH="$(pwd):$PYTHONPATH"

export PYSPARK_PYTHON=/usr/bin/python3
export PYSPARK_DRIVER_PYTHON=/usr/bin/python3

MODE="aws"                 # local | aws
DEVICE_MODE="spark"    # cpu | gpu | spark | spark-gpu | gpu-adaptive
RELOAD=1
DATASET_SIZE=5000
BATCH_SIZE=256
NUM_PARTITIONS=4
MODEL="all-MiniLM-L6-v2" #all-mpnet-base-v2
NO_QUERY=50
LOG_FILE="log.txt"

# ---- Package code for workers ----
rm -f project.zip
zip -r project.zip \
  src/ \
  scripts/ \
  constants.py \
  load_dataset.py \
  -x "*__pycache__*" \
  -x "*.pyc"

# ---- Compute NVIDIA lib path for CUDA .so files ----
NVIDIA_LIB_PATHS=$(python3 -c "
import nvidia, os
base = os.path.dirname(nvidia.__file__)
dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and not d.startswith('__')]
print(':'.join(os.path.join(base, d, 'lib') for d in dirs))
")

# ---- Remove old log before every run ----
rm -f "$LOG_FILE"

# ---- Full pipeline (embedding + indexing + evaluation) ----
spark-submit \
  --deploy-mode client \
  --py-files project.zip \
  --conf spark.executorEnv.TRANSFORMERS_CACHE=/tmp/transformers_cache \
  --conf spark.executorEnv.HF_HOME=/tmp/hf_home \
  --conf spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3 \
  --conf spark.pyspark.python=/usr/bin/python3 \
  --conf spark.pyspark.driver.python=/usr/bin/python3 \
  --conf spark.executorEnv.LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}" \
  --conf spark.driverEnv.LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}" \
  scripts/run_pipeline.py $MODE $DEVICE_MODE $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS $MODEL $NO_QUERY "$@" \
  > "$LOG_FILE" 2>&1

echo "Run complete. Log saved to $LOG_FILE"

'''

# ---- Embedding ----

spark-submit \
  --deploy-mode client \
  --py-files project.zip \
  --conf spark.executorEnv.TRANSFORMERS_CACHE=/tmp/transformers_cache \
  --conf spark.executorEnv.HF_HOME=/tmp/hf_home \
  --conf spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3 \
  --conf spark.pyspark.python=/usr/bin/python3 \
  --conf spark.pyspark.driver.python=/usr/bin/python3 \
  --conf spark.executorEnv.LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}" \
  --conf spark.driverEnv.LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}" \
  scripts/run_embedding.py $MODE $DEVICE_MODE $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS $MODEL

# ---- Indexing ----
#python3 scripts/run_indexing.py $MODE $DEVICE_MODE $RELOAD $DATASET_SIZE "$@"

# ---- Evaluation ----
#NO_QUERY=50
#python3 scripts/run_evaluation.py $MODE $DEVICE_MODE $DATASET_SIZE $NO_QUERY "$@"

'''