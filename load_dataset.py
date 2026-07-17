import os
import boto3
from datasets import load_dataset, load_from_disk
from constants import SEED, DATASET, DATA_PATH, S3_BUCKET

def transform(example):
    return {
        'doc_id': example['id'],
        'title' : example['title'],
        'text': example['text']
    }

# ------------ AWS -------------
s3_client = boto3.client("s3")

def s3_key_exists(bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError:
        return False

def load_parquet_dataset_s3(dataset_size):
    s3_key = f"data/{dataset_size}.parquet"
    local_path = f"data/{dataset_size}.parquet"

    if s3_key_exists(S3_BUCKET, s3_key):
        print(f"s3://{S3_BUCKET}/{s3_key} already exists, loading from S3")
        return load_dataset("parquet", data_files=f"s3://{S3_BUCKET}/{s3_key}", split="train")

    print("Dataset not found in S3, creating parquet dataset")

    if os.path.exists(local_path):
        print("Loading existing local parquet")
        dataset = load_dataset("parquet", data_files=local_path, split="train")
    else:
        dataset_original = load_dataset("parquet", data_files={"train": DATASET}, split="train")
        dataset_sample = dataset_original.filter(lambda x: len(x['text']) > 200).shuffle(seed=SEED).select(range(dataset_size))
        dataset = dataset_sample.map(transform, remove_columns=dataset_sample.column_names)
        dataset.to_parquet(local_path)

    print(f"Uploading to s3://{S3_BUCKET}/{s3_key}")
    s3_client.upload_file(local_path, S3_BUCKET, s3_key)

    return dataset

# ------------- Local ---------------
def load_parquet_dataset(dataset_size):
    data_path = f"{DATA_PATH}/{dataset_size}.parquet"
    if os.path.exists(data_path):
        print("Loading parquet data from disk")
        return load_dataset("parquet", data_files=data_path, split="train")   

    print("Craete parquet dataset")
    dataset_original = load_dataset("parquet", data_files={"train": DATASET}, split="train")
    dataset_sample = dataset_original.filter(lambda x: len(x['text']) > 200).shuffle(seed=SEED).select(range(dataset_size))
    dataset = dataset_sample.map(transform, remove_columns=dataset_sample.column_names)
    dataset.to_parquet(data_path)   

    return dataset

def load_sample_parquet(dataset_size, sample_size):
    full_path = f"{DATA_PATH}/{dataset_size}.parquet"
    sample_path = f"{DATA_PATH}/{sample_size}.parquet"

    if os.path.exists(sample_path):
        return load_dataset("parquet", data_files=sample_path, split="train")   

    full_dataset = load_dataset("parquet", data_files=full_path, split="train")
    sample = full_dataset.select(range(sample_size))
    sample.to_parquet(sample_path)

    return sample