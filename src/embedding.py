import os
import time
from functools import partial
import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession


class Embedding:
    def __init__(self, model_name="all-MiniLM-L6-v2", out_dir="embeddings"):
        self.model_name = model_name
        self.out_dir = out_dir

    def save(self, doc_ids, titles, embeddings, out_path):
        os.makedirs(self.out_dir, exist_ok=True)
        table = pa.table({
            "doc_id": doc_ids,
            "title": titles,
            "embedding": embeddings if isinstance(embeddings, list) else embeddings.tolist(),
        })
        pq.write_table(table, out_path)
        print(f"Saved embeddings to {out_path}")

    # For both CPU and GPU
    def embed_plain(self, reload, dataset, dataset_size, batch_size, device):
        
        out_path = f"{self.out_dir}/{device}_{dataset_size}.parquet"

        # If reload=1 and already exists, return cached embeddings
        if reload == 1 and os.path.exists(out_path):
            print(f"Embeddings already exist at {out_path}, loading from disk.")
            table = pq.read_table(out_path)
            cached_embeddings = np.array(table["embedding"].to_pylist(), dtype=np.float32)
            return cached_embeddings

        # Generate Embedding
        titles = dataset["title"]
        texts = dataset["text"]
        doc_ids = dataset["doc_id"]

        model = SentenceTransformer(self.model_name, device=device)

        start = time.time()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        elapsed = time.time() - start

        throughput = len(texts) / elapsed
        print(f"Embedded {len(texts)} docs in {elapsed:.2f}s ({throughput:.1f} texts/sec)")
        print(f"Embedding shape: {embeddings.shape}")

        # len(texts) = dataset_size
        self.save(doc_ids, titles, embeddings, out_path)

        return embeddings

    # -------------- For both SPARK and SPARK_GPU
    @staticmethod
    def embed_partition(rows, batch_size, device, model_name):
        #Runs once per Spark partition. Model loaded once here, not once per row
        # no_of_partition = 4. runs 4 times
        model = SentenceTransformer(model_name, device=device)

        # rows is initially an iterator
        rows = list(rows)
        doc_ids = [r["doc_id"] for r in rows]
        titles = [r["title"] for r in rows]
        texts = [r["text"] for r in rows]

        embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)

        for doc_id, title, emb in zip(doc_ids, titles, embeddings):
            yield (doc_id, title, emb.tolist())

    def embed_spark(self, reload, dataset, dataset_size, batch_size, no_partition, device):
        
        out_path = f"{self.out_dir}/spark_{device}_{dataset_size}.parquet"
        # If already exists, return cached embeddings
        if reload == 1 and os.path.exists(out_path):
            print(f"Embeddings already exist at {out_path}, loading from disk.")
            table = pq.read_table(out_path)
            cached_embeddings = np.array(table["embedding"].to_pylist(), dtype=np.float32)
            return cached_embeddings
    
        spark = SparkSession.builder \
            .appName("spark_embedding") \
            .master("local[*]") \
            .getOrCreate()

        rows = [{"doc_id": d, "title": t, "text": x}
                for d, t, x in zip(dataset["doc_id"], dataset["title"], dataset["text"])]

        rdd = spark.sparkContext.parallelize(rows, numSlices=no_partition)
        print(f"Partitions: {rdd.getNumPartitions()}")

        start = time.time()
        embed_fn = partial(
            self.embed_partition,
            batch_size=batch_size,
            device=device,
            model_name=self.model_name,
        )
        results = rdd.mapPartitions(embed_fn).collect()
        elapsed = time.time() - start

        throughput = len(results) / elapsed
        print(f"Embedded {len(results)} docs in {elapsed:.2f}s ({throughput:.1f} texts/sec)")

        doc_ids = [r[0] for r in results]
        titles = [r[1] for r in results]
        embeddings = [r[2] for r in results]

        self.save(doc_ids, titles, embeddings, out_path)

        spark.stop()
        return embeddings

    # -------------- GPU_ADAPTIVE
    @staticmethod
    def adaptive_batch_encode(model, texts, start_batch_size):

        embeddings = []
        i = 0
        batch_size = start_batch_size
        oom_count = 0

        while i < len(texts):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = model.encode(batch, batch_size=batch_size, convert_to_numpy=True)
                embeddings.extend(batch_embeddings)
                i += batch_size
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                print(f"OOM at batch_size={batch_size}, position={i}. Backing off.")
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
                
                # For any doc that any batch_size doesnt work
                if batch_size < 1:
                    print(f"Batch size hit 0 at position {i} — skipping doc.")
                    i += 1
                    batch_size = start_batch_size
                    continue

        embeddings = np.array(embeddings)
        return embeddings, oom_count, batch_size
    
    def embed_adaptive_gpu(self, reload, dataset, dataset_size, start_batch_size, device="cuda"):

        out_path = f"{self.out_dir}/adaptive_{device}_{dataset_size}.parquet"
        # If already exists, return cached embeddings
        if reload == 1 and os.path.exists(out_path):
            print(f"Embeddings already exist at {out_path}, loading from disk.")
            table = pq.read_table(out_path)
            cached_embeddings = np.array(table["embedding"].to_pylist(), dtype=np.float32)
            return cached_embeddings
    
        if device != "cuda":
            print("Adaptive batching is for GPU only. Exiting.")
            return None

        titles = dataset["title"]
        texts = dataset["text"]
        doc_ids = dataset["doc_id"]

        model = SentenceTransformer(self.model_name, device=device)

        start = time.time()
        embeddings, oom_count, settled_batch_size = self.adaptive_batch_encode(
            model, texts, start_batch_size
        )
        elapsed = time.time() - start

        print(f"oom count: {oom_count}, adaptive batch size: {settled_batch_size}")
        throughput = len(texts) / elapsed
        print(f"Embedded {len(texts)} docs in {elapsed:.2f}s ({throughput:.1f} texts/sec)")
        print(f"Embedding shape: {embeddings.shape}")

        self.save(doc_ids, titles, embeddings, out_path)

        return embeddings

    