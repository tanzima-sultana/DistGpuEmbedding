#!/bin/bash
set -e
 
export PYTHONPATH="$(pwd):$PYTHONPATH"
 
MODE="aws"                 # local | aws
DEVICE_MODE="spark"    # cpu | gpu | spark | spark-gpu | gpu-adaptive
RELOAD=1
DATASET_SIZE=15000
BATCH_SIZE=256
NUM_PARTITIONS=4
MODEL="all-MiniLM-L6-v2" #all-mpnet-base-v2
 
# ---- Embedding ----

spark-submit \
  --deploy-mode client \
  --conf spark.executorEnv.TRANSFORMERS_CACHE=/tmp/transformers_cache \
  --conf spark.executorEnv.HF_HOME=/tmp/hf_home \
  scripts/run_embedding.py $MODE $DEVICE_MODE $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS $MODEL
 
# ---- Indexing ----
#python3 scripts/run_indexing.py $MODE $DEVICE_MODE $RELOAD $DATASET_SIZE "$@"
 
# ---- Evaluation ----
#NO_QUERY=50
#python3 scripts/run_evaluation.py $MODE $DEVICE_MODE $DATASET_SIZE $NO_QUERY "$@"

# ---- DistGpuEmbedding ----
#python3 scripts/run_pipeline.py $MODE $DEVICE_MODE $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS $MODEL $NO_QUERY "$@"
 