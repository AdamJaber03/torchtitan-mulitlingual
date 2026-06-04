import os
import re
import time
import concurrent.futures
import orjson
from datasets import load_dataset
import glob
import sys

# Add parent directory to path so we can import from torchtitan
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchtitan.hf_datasets.config import load_output_base_dir

# ==========================================
# 1. GLOBAL VARIABLES FOR WORKERS
# ==========================================

# We load this once in the main process. Linux 'fork' will share it to all workers instantly.
TRANSLATION_DICT = {}

# The bulletproof regex: Captures the Arabic word block
AR_PATTERN = re.compile(
    r'([ؠ-ٰٟ-ۯۺ-ۿﭐ-﷿ﹰ-﻿‌‍]+)')

# ==========================================
# 2. THE CORE TRANSLATION LOGIC
# ==========================================


def read_local_translated_chunks(directory, regex_pattern=False):
    """Yields documents one-by-one, skipping corrupted trailing lines."""
    if regex_pattern:
        file_list = glob.glob(f"{directory}")
    else:
        file_list = glob.glob(f"{directory}/*.jsonl")
    for filepath in file_list:
        with open(filepath, 'rb') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    yield orjson.loads(line)
                except orjson.JSONDecodeError:
                    print(
                        f"\n⚠️ WARNING: Skipped corrupted JSON line in {filepath}")
                    continue


def translate_match(match):
    """
    This function is called for EVERY regex match in the document.
    """
    raw_word = match.group(1)

    # 1. Clean the visual formatting exactly as we did during extraction
    clean_word = raw_word.replace('ـ', '').replace(
        '‌', '').replace('‍', '')

    # 2. Look up the translation.
    # If the word is in the dict, return the English translation.
    # If it's NOT in the dict (e.g., extremely rare or missed), return the raw original word.
    return TRANSLATION_DICT.get(clean_word, raw_word)


def process_and_save_chunk(chunk_id, docs_chunk, save_dir):
    """
    Worker function running on a dedicated CPU core.
    Translates a chunk of documents and writes both original and translated versions to disk.
    """
    filepath = os.path.join(save_dir, f"chunk_{chunk_id:04d}.jsonl")

    # Open both files in binary append mode for blazing fast orjson writing
    with open(filepath, 'wb') as f:

        for doc in docs_chunk:
            original_text = doc.get("text", "")

            if not original_text:
                continue

            orig_json = {"id": doc.get("id", ""), "url": doc.get(
                "url", ""), "text": original_text}

            # Write to disk using orjson (requires appending newline manually)
            f.write(orjson.dumps(orig_json) + b'\n')

    return chunk_id, len(docs_chunk)

# ==========================================
# 3. THE PRODUCER-CONSUMER MAIN LOOP
# ==========================================


def download_dataset(ds, save_dir, num_docs, chunk_size, cpu_cores):
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        futures = []
        current_chunk = []
        docs_read = 0
        chunk_counter = 1

        for doc in ds:
            if docs_read >= num_docs:
                break

            current_chunk.append(doc)
            docs_read += 1

            if len(current_chunk) >= chunk_size:
                future = executor.submit(
                    process_and_save_chunk,
                    chunk_counter,
                    current_chunk,
                    save_dir,
                )
                futures.append(future)

                if chunk_counter % 10 == 0:
                    print(f"Dispatched {docs_read:,} documents to CPU workers...")

                current_chunk = []
                chunk_counter += 1

        if current_chunk:
            futures.append(executor.submit(
                process_and_save_chunk, chunk_counter, current_chunk, save_dir
            ))

        print(f"\nFinished reading {docs_read:,} docs from Hugging Face.")
        print("Waiting for CPU cores to finish saving to disk...")

        completed_docs = 0
        for future in concurrent.futures.as_completed(futures):
            chunk_id, num_processed = future.result()
            completed_docs += num_processed

            if completed_docs % 250_000 == 0:
                print(f"Successfully saved {completed_docs:,} / {num_docs:,} documents...")

    return completed_docs


def main():
    # --- CONFIGURATION ---
    NUM_DOCS_TO_PROCESS = 170_000_000
    CHUNK_SIZE = 50_000
    CPU_CORES = 32

    OUTPUT_BASE_DIR = load_output_base_dir()
    AR_DIR = os.path.join(OUTPUT_BASE_DIR, "fineweb_translated/original")
    EN_DIR = os.path.join(OUTPUT_BASE_DIR, "fineweb_translated/en-original")

    os.makedirs(AR_DIR, exist_ok=True)
    os.makedirs(EN_DIR, exist_ok=True)

    print(f"AR_DIR: {AR_DIR}")
    print(f"EN_DIR: {EN_DIR}")

    start_time = time.time()

    # Skip AR download if already populated
    ar_files = glob.glob(os.path.join(AR_DIR, "*.jsonl"))
    if ar_files:
        print(f"\nAR data already present ({len(ar_files)} chunks), skipping AR download.")
    else:
        print("\nConnecting to Hugging Face Stream (fineweb-edu-ar, ar)...")
        ds_ar = load_dataset("kaust-generative-ai/fineweb-edu-ar", "ar", split="train", streaming=True)
        print(f"Downloading AR data across {CPU_CORES} CPU cores...")
        count = download_dataset(ds_ar, AR_DIR, NUM_DOCS_TO_PROCESS, CHUNK_SIZE, CPU_CORES)
        print(f"AR done: {count:,} documents saved to {AR_DIR}")

    # Download EN data
    en_files = glob.glob(os.path.join(EN_DIR, "*.jsonl"))
    if en_files:
        print(f"\nEN data already present ({len(en_files)} chunks), skipping EN download.")
    else:
        print("\nConnecting to Hugging Face Stream (fineweb-edu-ar, en)...")
        ds_en = load_dataset("kaust-generative-ai/fineweb-edu-ar", "en", split="train", streaming=True)
        print(f"Downloading EN data across {CPU_CORES} CPU cores...")
        count = download_dataset(ds_en, EN_DIR, NUM_DOCS_TO_PROCESS, CHUNK_SIZE, CPU_CORES)
        print(f"EN done: {count:,} documents saved to {EN_DIR}")

    mins = (time.time() - start_time) / 60
    print(f"\n✅ COMPLETE in {mins:.2f} minutes.")
    print(f"AR saved to: {AR_DIR}")
    print(f"EN saved to: {EN_DIR}")


if __name__ == "__main__":
    main()
