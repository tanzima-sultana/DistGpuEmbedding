#!/bin/bash
set -e

export PYTHONPATH="$(pwd):$PYTHONPATH"

export PYSPARK_PYTHON=/usr/bin/python3
export PYSPARK_DRIVER_PYTHON=/usr/bin/python3

MODE="aws"                 # local | aws
DEVICE_MODE="spark-gpu"    # cpu | gpu | spark | spark-gpu | gpu-adaptive
RELOAD=1
DATASET_SIZE=250000
BATCH_SIZE=256
NUM_PARTITIONS=16
MODEL="all-MiniLM-L6-v2" #all-mpnet-base-v2
NO_QUERY=50
LOG_FILE="log.txt"

# ---- Package code for workers ----
rm -f project.zip
zip -r project.zip \
  src/ \
  scripts/ \
  constants.py \
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
# spark-submit \
#   --deploy-mode client \
#   --py-files project.zip \
#   --conf spark.executor.memory=6g \
#   --conf spark.executor.memoryOverhead=1g \
#   --conf spark.driver.memory=4g \
#   --conf spark.executorEnv.TRANSFORMERS_CACHE=/tmp/transformers_cache \
#   --conf spark.executorEnv.HF_HOME=/tmp/hf_home \
#   --conf spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3 \
#   --conf spark.pyspark.python=/usr/bin/python3 \
#   --conf spark.pyspark.driver.python=/usr/bin/python3 \
#   --conf spark.executorEnv.LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}" \
#   --conf spark.driverEnv.LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}" \
#   --conf spark.dynamicAllocation.enabled=false \
#   --conf spark.eventLog.enabled=false \
#   scripts/run_pipeline.py $MODE $DEVICE_MODE $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS $MODEL $NO_QUERY "$@" \
#   > "$LOG_FILE" 2>&1

# ---- Embedding  ----
spark-submit \
  --deploy-mode client \
  --py-files project.zip \
  --conf spark.executor.memory=6g \
  --conf spark.executor.memoryOverhead=1g \
  --conf spark.driver.memory=4g \
  --conf spark.executorEnv.TRANSFORMERS_CACHE=/tmp/transformers_cache \
  --conf spark.executorEnv.HF_HOME=/tmp/hf_home \
  --conf spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3 \
  --conf spark.pyspark.python=/usr/bin/python3 \
  --conf spark.pyspark.driver.python=/usr/bin/python3 \
  --conf spark.executorEnv.LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}" \
  --conf spark.driverEnv.LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}" \
  --conf spark.dynamicAllocation.enabled=false \
  --conf spark.eventLog.enabled=false \
  scripts/run_embedding.py $MODE $DEVICE_MODE $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS $MODEL \
  > "$LOG_FILE" 2>&1

echo "Run complete. Log saved to $LOG_FILE"