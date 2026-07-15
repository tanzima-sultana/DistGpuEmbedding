source ~/pyenv/bin/activate

export PYTHONPATH="$(pwd):$PYTHONPATH"

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export JAVA_TOOL_OPTIONS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED"
unset SPARK_HOME

RELOAD=1
DATASET_SIZE=10000
BATCH_SIZE=256
NUM_PARTITIONS=4

# Embedding

#python3 scripts/run_embedding.py cpu $RELOAD $DATASET_SIZE $BATCH_SIZE "$@"
#python3 scripts/run_embedding.py gpu $RELOAD $DATASET_SIZE $BATCH_SIZE "$@"
#python3 scripts/run_embedding.py spark $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS "$@"
#python3 scripts/run_embedding.py spark-gpu $RELOAD $DATASET_SIZE $BATCH_SIZE $NUM_PARTITIONS "$@"
#python3 scripts/run_embedding.py gpu-adaptive $RELOAD $DATASET_SIZE 1024 $NUM_PARTITIONS all-mpnet-base-v2   

# Indexing

EMBEDDING_PATH="embeddings/cpu_10000.parquet"

#python3 scripts/run_indexing.py $RELOAD $DATASET_SIZE $EMBEDDING_PATH "$@"

# Evaluation

NO_QUERY=50

python3 scripts/run_evaluation.py $DATASET_SIZE $EMBEDDING_PATH $NO_QUERY "$@"