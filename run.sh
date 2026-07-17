source ~/pyenv/bin/activate

export PYTHONPATH="$(pwd):$PYTHONPATH"

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export JAVA_TOOL_OPTIONS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED"
unset SPARK_HOME

RELOAD=1
DATASET_SIZE=10000
BATCH_SIZE=256
NUM_PARTITIONS=4
MODEL="all-MiniLM-L6-v2" #all-mpnet-base-v2

# Embedding

python3 scripts/run_embedding.py local cpu $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS $MODEL "$@"
#python3 scripts/run_embedding.py local gpu $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS $MODEL "$@"
#python3 scripts/run_embedding.py local spark $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS $MODEL "$@"
#python3 scripts/run_embedding.py local spark-gpu $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS $MODEL "$@"
#python3 scripts/run_embedding.py local gpu-adaptive $RELOAD $DATASET_SIZE 1024 $NUM_PARTITIONS $MODEL "$@"

# Indexing

#python3 scripts/run_indexing.py local cpu $RELOAD $DATASET_SIZE "$@"
#python3 scripts/run_indexing.py local gpu $RELOAD $DATASET_SIZE "$@"
#python3 scripts/run_indexing.py local spark $RELOAD $DATASET_SIZE "$@"
#python3 scripts/run_indexing.py local spark-gpu $RELOAD $DATASET_SIZE "$@"

# Evaluation

NO_QUERY=50

python3 scripts/run_evaluation.py local cpu $DATASET_SIZE $NO_QUERY "$@"