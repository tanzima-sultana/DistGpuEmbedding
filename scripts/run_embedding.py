import sys

from load_dataset import load_parquet_dataset, load_parquet_dataset_s3
from src.embedding import Embedding
from constants import LOCAL, AWS, CPU, GPU, SPARK, SPARK_GPU, GPU_ADAPTIVE

if __name__ == "__main__":

    # agr 1 : local or aws
    mode = sys.argv[1]

    # arg 2 : mode -> cpu | gpu | spark | spark-gpu | gpu-adaptive
    device_mode = sys.argv[2]

    # arg 3 : reload
    # reload = 0, create new embedding
    # reload = 1, load from disk if embedding exists
    reload = int(sys.argv[3])

    # arg 4 : Dataset size
    dataset_size = int(sys.argv[4])

    # arg 5 : Batch size (starting batch size, for gpu-adaptive)
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 256

    # arg 6 : no_partition (only used for spark / spark-gpu)
    no_partition = int(sys.argv[6]) if len(sys.argv) > 6 else 4

    # arg 7 : model name (all-mpnet-base-v2 for adaptive)
    model_name = sys.argv[7] if len(sys.argv) > 7 else "all-MiniLM-L6-v2"

    valid_device_modes = [CPU, GPU, SPARK, SPARK_GPU, GPU_ADAPTIVE]
    if device_mode not in valid_device_modes:
        print(f"Invalid mode '{device_mode}'. Choose from {valid_device_modes}")
        sys.exit(1)

    device = "cpu"
    if device_mode == GPU or device_mode == SPARK_GPU or device_mode == GPU_ADAPTIVE:
        device = "cuda"

    print(f"\n----- Embedding mode: {mode} (device={device_mode}) -----")
    print(f"Dataset size: {dataset_size}, Batch size: {batch_size}",
          f", Partitions: {no_partition}", f", Model: {model_name}")

    dataset = None
    if mode == LOCAL:
        dataset = load_parquet_dataset(dataset_size)
    else:
        dataset = load_parquet_dataset_s3(dataset_size)
    
    print(dataset)

    embedder = Embedding(mode, device_mode, dataset_size, model_name)

    # ----- Local 
    if mode == LOCAL:
        if device_mode in (CPU, GPU):
            embedder.embed_plain(reload, dataset, dataset_size, batch_size, device)
        elif device_mode in (SPARK, SPARK_GPU):
            embedder.embed_spark(reload, dataset, dataset_size, batch_size, no_partition, device)
        elif device_mode == GPU_ADAPTIVE:
            embedder.embed_adaptive_gpu(reload, dataset, dataset_size, batch_size, device)
    else:
        embedder.embed_spark_aws(reload, dataset, dataset_size, batch_size, no_partition, device)