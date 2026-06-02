import os
import sys

# Add parent directory to path so we can import from torchtitan
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchtitan.hf_datasets.config import load_output_base_dir

OUTPUT_BASE_DIR = load_output_base_dir()
os.environ["HF_HOME"] = os.path.join(OUTPUT_BASE_DIR, "hf_home")

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

def merge_from_local_cache():
    print("Loading datasets from local cache (Instant)...")
    # Notice we removed num_proc and streaming. It will load instantly from the cache on disk.
    ar_ds = load_dataset("kaust-generative-ai/fineweb-edu-ar", name="ar", split="train[:50000000]", verification_mode="no_checks")
    en_ds = load_dataset("kaust-generative-ai/fineweb-edu-ar", name="en", split="train[:50000000]", verification_mode="no_checks")

    OUTPUT_BASE_DIR = load_output_base_dir()
    output_path = os.path.join(OUTPUT_BASE_DIR, "fineweb-edu-ar-paired.parquet")
    print(f"Writing batched Parquet to {output_path}...")
    
    # Define the Parquet schema
    schema = pa.schema([('text_ar', pa.string()), ('text_en', pa.string())])
    
    batch_size = 100_000
    ar_batch = []
    en_batch = []
    
    # Open a streaming Parquet writer
    with pq.ParquetWriter(output_path, schema) as writer:
        # Zip the datasets natively (very fast because they are memory-mapped from disk)
        for ar, en in tqdm(zip(ar_ds, en_ds), total=50_000_000):
            ar_batch.append(ar["text"])
            en_batch.append(en["text"])
            
            # When the batch is full, write it to disk and clear RAM
            if len(ar_batch) == batch_size:
                table = pa.Table.from_arrays([pa.array(ar_batch), pa.array(en_batch)], schema=schema)
                writer.write_table(table)
                ar_batch.clear()
                en_batch.clear()
                
        # Write any remaining data in the final batch
        if ar_batch:
            table = pa.Table.from_arrays([pa.array(ar_batch), pa.array(en_batch)], schema=schema)
            writer.write_table(table)

    print("Merge complete! Your dataset is ready for torchtitan.")

if __name__ == "__main__":
    merge_from_local_cache()