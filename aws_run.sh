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

# ---- Package code for workers ----
rm -f project.zip
zip -r project.zip \
  src/ \
  scripts/ \
  constants.py \
  load_dataset.py \
  -x "*__pycache__*" \
  -x "*.pyc"

# ---- Embedding ----

spark-submit \
  --deploy-mode client \
  --py-files project.zip \
  --conf spark.executorEnv.TRANSFORMERS_CACHE=/tmp/transformers_cache \
  --conf spark.executorEnv.HF_HOME=/tmp/hf_home \
  --conf spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3 \
  --conf spark.pyspark.python=/usr/bin/python3 \
  --conf spark.pyspark.driver.python=/usr/bin/python3 \
  scripts/run_embedding.py $MODE $DEVICE_MODE $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS $MODEL

# ---- Indexing ----
#python3 scripts/run_indexing.py $MODE $DEVICE_MODE $RELOAD $DATASET_SIZE "$@"

# ---- Evaluation ----
#NO_QUERY=50
#python3 scripts/run_evaluation.py $MODE $DEVICE_MODE $DATASET_SIZE $NO_QUERY "$@"

# ---- DistGpuEmbedding pipeline ----
#python3 scripts/run_pipeline.py $MODE $DEVICE_MODE $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS $MODEL $NO_QUERY "$@"