import os
import sys
from datasets import load_dataset

# Add parent directory to path so we can import from torchtitan
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchtitan.hf_datasets.config import load_output_base_dir

# 1. Load your single monolithic dataset
OUTPUT_BASE_DIR = load_output_base_dir()
monolithic_path = os.path.join(OUTPUT_BASE_DIR, "fineweb-edu-ar-paired.parquet")
ds = load_dataset("parquet", data_files=monolithic_path, split="train")

# 2. Create an output directory for the shards
output_dir = os.path.join(OUTPUT_BASE_DIR, "fineweb-edu-ar_paired_shards")
os.makedirs(output_dir, exist_ok=True)

# 3. Slice it into 64 shards (so your 16 workers get exactly 4 shards each)
num_shards = 64
for i in range(num_shards):
    print(f"Writing shard {i+1}/{num_shards}...")
    shard = ds.shard(num_shards=num_shards, index=i)
    shard.to_parquet(f"{output_dir}/shard_{i:04d}.parquet")

print("Sharding complete!")