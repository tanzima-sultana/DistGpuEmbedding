import sys

from load_dataset import load_parquet_dataset
from src.embedding import Embedding
from constants import CPU, GPU, SPARK, SPARK_GPU, GPU_ADAPTIVE

if __name__ == "__main__":

    # arg 1 : mode -> cpu | gpu | spark | spark-gpu | gpu-adaptive
    mode = sys.argv[1]

    # arg 2 : reload
    # reload = 0, create new embedding
    # reload = 1, load from disk if embedding exists
    reload = int(sys.argv[2])

    # arg 3 : Dataset size
    dataset_size = int(sys.argv[3])

    # arg 4 : Batch size (starting batch size, for gpu-adaptive)
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 256

    # arg 5 : no_partition (only used for spark / spark-gpu)
    no_partition = int(sys.argv[5]) if len(sys.argv) > 5 else 4

    # arg 6 : model name (all-mpnet-base-v2 for adaptive)
    model_name = sys.argv[6] if len(sys.argv) > 6 else "all-MiniLM-L6-v2"

    valid_modes = [CPU, GPU, SPARK, SPARK_GPU, GPU_ADAPTIVE]
    if mode not in valid_modes:
        print(f"Invalid mode '{mode}'. Choose from {valid_modes}")
        sys.exit(1)

    device = "cpu"
    if mode == GPU or mode == SPARK_GPU or mode == GPU_ADAPTIVE:
        device = "cuda"

    print(f"\n----- Embedding mode: {mode} (device={device}) -----")
    print(f"Dataset size: {dataset_size}, Batch size: {batch_size}",
          f", Partitions: {no_partition}" if "spark" in mode else "",
          f", Model: {model_name}")

    dataset = load_parquet_dataset(dataset_size)
    print(dataset)

    embedder = Embedding(model_name=model_name)

    if mode in (CPU, GPU):
        embedder.embed_plain(reload, dataset, dataset_size, batch_size, device)
    elif mode in (SPARK, SPARK_GPU):
        embedder.embed_spark(reload, dataset, dataset_size, batch_size, no_partition, device)
    elif mode == GPU_ADAPTIVE:
        embedder.embed_adaptive_gpu(reload, dataset, dataset_size, batch_size, device)