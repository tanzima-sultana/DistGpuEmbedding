import glob
DATASET = glob.glob(
        "/home/tanzima/.cache/huggingface/hub/datasets--wikimedia--wikipedia/snapshots/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en/*.parquet"
    )
DATA_PATH = "data/processed_dataset"
SEED = 42

# Embedding

LOCAL="local"
AWS="aws"
CPU="cpu"
GPU="gpu"
SPARK="spark"
SPARK_GPU="spark-gpu"
GPU_ADAPTIVE="gpu-adaptive"

INDEX_FLATIP="FlatIndexIP"
INDEX_IVF="IVFFlatIndex"
INDEX_HNSW="HNSWFlatIndex"

# AWS

S3_BUCKET = "dist-gpu-embedding"

S3_DATASET = "s3://dist-gpu-embedding/raw-wikipedia/*.parquet"

EC2_RATE_G4DN_XLARGE = 0.526  
EMR_MARKUP_G4DN_XLARGE = 0.07  