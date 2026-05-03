import os
import re
import time
import concurrent.futures
import orjson
from datasets import load_dataset
import glob
import orjson

# ==========================================
# 1. GLOBAL VARIABLES FOR WORKERS
# ==========================================

# We load this once in the main process. Linux 'fork' will share it to all workers instantly.
TRANSLATION_DICT = {}

# The bulletproof regex: Captures the Arabic word block
AR_PATTERN = re.compile(r'([\u0620-\u065F\u0670-\u06EF\u06FA-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF\u200C\u200D]+)')

# ==========================================
# 2. THE CORE TRANSLATION LOGIC
# ==========================================
import glob
import orjson

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
                    continue # Skip empty lines
                
                try:
                    yield orjson.loads(line)
                except orjson.JSONDecodeError:
                    print(f"\n⚠️ WARNING: Skipped corrupted JSON line in {filepath}")
                    continue

def translate_match(match):
    """
    This function is called for EVERY regex match in the document.
    """
    raw_word = match.group(1)
    
    # 1. Clean the visual formatting exactly as we did during extraction
    clean_word = raw_word.replace('\u0640', '').replace('\u200C', '').replace('\u200D', '')
    
    # 2. Look up the translation. 
    # If the word is in the dict, return the English translation.
    # If it's NOT in the dict (e.g., extremely rare or missed), return the raw original word.
    return TRANSLATION_DICT.get(clean_word, raw_word)

def process_and_save_chunk(chunk_id, docs_chunk, orig_dir, trans_dir):
    """
    Worker function running on a dedicated CPU core.
    Translates a chunk of documents and writes both original and translated versions to disk.
    """
    orig_filepath = os.path.join(orig_dir, f"chunk_{chunk_id:04d}.jsonl")
    trans_filepath = os.path.join(trans_dir, f"chunk_{chunk_id:04d}.jsonl")
    
    # Open both files in binary append mode for blazing fast orjson writing
    # with open(orig_filepath, 'wb') as f_orig, open(trans_filepath, 'wb') as f_trans:
    with open(trans_filepath, 'wb') as f_trans:
        
        for doc in docs_chunk:
            original_text = doc.get("text", "")
            
            if not original_text:
                continue
                
            # Perform the word-by-word translation. 
            # AR_PATTERN.sub finds every Arabic word, passes it to translate_match, 
            # and stitches the document back together leaving punctuation/spaces perfectly intact!
            translated_text = AR_PATTERN.sub(translate_match, original_text)
            
            # # Create the JSON objects
            # orig_json = {"id": doc.get("id", ""), "url": doc.get("url", ""), "text": original_text}
            trans_json = {"id": doc.get("id", ""), "url": doc.get("url", ""), "text": translated_text}
            
            # Write to disk using orjson (requires appending newline manually)
            # f_orig.write(orjson.dumps(orig_json) + b'\n')
            f_trans.write(orjson.dumps(trans_json) + b'\n')
            
    return chunk_id, len(docs_chunk)

# ==========================================
# 3. THE PRODUCER-CONSUMER MAIN LOOP
# ==========================================

def main():
    # --- CONFIGURATION ---
    NUM_DOCS_TO_PROCESS = 6_999_999
    CHUNK_SIZE = 25_000
    
    DICT_PATH = "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex.json"
    OUTPUT_BASE_DIR = "/home/adamga/leshemg/adamga/data/fineweb_translated/"
    
    ORIG_DIR = os.path.join(OUTPUT_BASE_DIR, "original")
    TRANS_DIR = os.path.join(OUTPUT_BASE_DIR, "translated_wip")
    
    # Create directories if they don't exist
    os.makedirs(ORIG_DIR, exist_ok=True)
    os.makedirs(TRANS_DIR, exist_ok=True)
    
    # 1. Load the dictionary into memory
    print(f"Loading dictionary from {DICT_PATH}...")
    global TRANSLATION_DICT
    with open(DICT_PATH, 'rb') as f:
        TRANSLATION_DICT = orjson.loads(f.read())
    print(f"Loaded {len(TRANSLATION_DICT):,} words into memory.")
    
    # 2. Connect to Hugging Face
    # print("\nConnecting to Hugging Face Stream (fineweb-edu-ar)...")
    # ds_ar = load_dataset("kaust-generative-ai/fineweb-edu-ar", "ar", split="train", streaming=True)
    print(f"Reading  original documents from {ORIG_DIR}...")
    ds_ar = read_local_translated_chunks(ORIG_DIR)    
    # INJ_FILES = r"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/*/ar_data.jsonl"
    # print(f"Reading injected documents from {INJ_FILES}...")
    # ds_ar = read_local_translated_chunks(INJ_FILES, regex_pattern=True)
    
    # 3. Spin up the CPU cores
    cpu_cores = os.cpu_count()
    cpu_cores = 32
    print(f"Starting Translation across {cpu_cores} CPU cores...")
    
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        futures = []
        current_chunk = []
        docs_read = 0
        chunk_counter = 1
        
        # PRODUCER: Read from stream and dispatch chunks
        for doc in ds_ar:
            if docs_read >= NUM_DOCS_TO_PROCESS:
                break
                
            current_chunk.append(doc)
            docs_read += 1
            
            # When chunk is full, dispatch to a CPU worker
            if len(current_chunk) >= CHUNK_SIZE:
                future = executor.submit(
                    process_and_save_chunk, 
                    chunk_counter, 
                    current_chunk, 
                    ORIG_DIR, 
                    TRANS_DIR
                )
                futures.append(future)
                
                if chunk_counter % 10 == 0:
                    print(f"Dispatched {docs_read:,} documents to CPU workers...")
                    
                current_chunk = []
                chunk_counter += 1

        # Dispatch any remaining documents
        if current_chunk:
            futures.append(executor.submit(
                process_and_save_chunk, chunk_counter, current_chunk, ORIG_DIR, TRANS_DIR
            ))
            
        print(f"\nFinished reading {docs_read:,} docs from Hugging Face.")
        print("Waiting for CPU cores to finish translating and saving to disk...")

        # CONSUMER: Track progress as chunks finish saving
        completed_docs = 0
        for future in concurrent.futures.as_completed(futures):
            chunk_id, num_processed = future.result()
            completed_docs += num_processed
            
            if completed_docs % 250_000 == 0:
                print(f"Successfully saved {completed_docs:,} / {NUM_DOCS_TO_PROCESS:,} documents...")

    end_time = time.time()
    mins = (end_time - start_time) / 60
    print(f"\n✅ COMPLETE! 7,000,000 documents translated and saved in {mins:.2f} minutes.")
    print(f"Originals saved to: {ORIG_DIR}")
    print(f"Translated saved to: {TRANS_DIR}")

if __name__ == "__main__":
    main()