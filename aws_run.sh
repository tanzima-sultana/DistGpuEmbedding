#!/bin/bash
set -e
 
export PYTHONPATH="$(pwd):$PYTHONPATH"
 
MODE="aws"                 # local | aws
DEVICE_MODE="spark"    # cpu | gpu | spark | spark-gpu | gpu-adaptive
RELOAD=1
DATASET_SIZE=15000
BATCH_SIZE=256
NUM_PARTITIONS=4
 
# ---- Embedding ----
python3 run_embedding.py $MODE $DEVICE_MODE $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS "$@"
 
# ---- Indexing ----
# python3 run_indexing.py $MODE $DEVICE_MODE $RELOAD $DATASET_SIZE "$@"
 
# ---- Evaluation ----
# NO_QUERY=50
# python3 run_evaluation.py $MODE $DEVICE_MODE $DATASET_SIZE $NO_QUERY "$@"
 