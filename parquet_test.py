import os
import sys
from datasets import load_dataset

# Add parent directory to path so we can import from torchtitan
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torchtitan.hf_datasets.config import load_output_base_dir

# Point this directly to the directory containing your 64 shards
OUTPUT_BASE_DIR = load_output_base_dir()
shards_dir = os.path.join(OUTPUT_BASE_DIR, "fineweb-edu-ar_paired_shards")

print(f"Loading streaming dataset from: {shards_dir}\n")

# 1. Load the dataset (exactly as your dataloader will see it)
dataset = load_dataset("parquet", data_dir=shards_dir, split="train", streaming=True)

# 2. Create a basic Python iterator
data_iterator = iter(dataset)

# 3. Pull and print the first 3 samples
for i in range(3):
    try:
        sample = next(data_iterator)
        print(f"=== RAW SAMPLE {i + 1} ===")
        
        # Print the exact keys available in the parquet file
        print(f"Available Keys: {list(sample.keys())}")
        
        # Print the content (truncated so it doesn't flood your terminal)
        if 'text_ar' in sample:
            print(f"\n[text_ar] (First 200 chars):\n{sample['text_ar'][:200]}...")
        
        if 'text_en' in sample:
            print(f"\n[text_en] (First 200 chars):\n{sample['text_en'][:200]}...")
            
        # Fallback just in case your parquet uses standard 'text' keys
        if 'text' in sample:
            print(f"\n[text] (First 200 chars):\n{sample['text'][:200]}...")
            
        print("\n" + "="*50 + "\n")
        
    except StopIteration:
        print("Dataset stream ended unexpectedly.")
        break