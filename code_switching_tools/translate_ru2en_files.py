import os
import re
import time
import concurrent.futures
import orjson
import glob
import random

# ==========================================
# 1. GLOBAL VARIABLES FOR WORKERS
# ==========================================

TRANSLATION_DICT = {}

# The bulletproof regex: Captures the Russian (Cyrillic) word block
RU_PATTERN = re.compile(r'([\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F\u1C80-\u1C8F\u0300-\u036F\u200C\u200D\u00AD]+)')

# ==========================================
# 2. THE CORE TRANSLATION LOGIC
# ==========================================

def init_worker(dict_path):
    """
    CRITICAL FIX: This runs once per CPU core when it boots up. 
    It forces the worker to load the dictionary into its own local memory,
    preventing the 'empty dictionary' bug.
    """
    global TRANSLATION_DICT
    if not TRANSLATION_DICT:
        with open(dict_path, 'rb') as f:
            TRANSLATION_DICT = orjson.loads(f.read())

def translate_match(match):
    """Called for EVERY regex match in the document."""
    raw_word = match.group(1)
    
    # 1. Clean visual formatting
    clean_word = raw_word.replace('\u00AD', '').replace('\u200C', '').replace('\u200D', '')
    clean_word = re.sub(r'[\u0300-\u036F]', '', clean_word)
    
    # 2. Try Exact Match
    if clean_word in TRANSLATION_DICT:
        return TRANSLATION_DICT[clean_word]
        
    # 3. Try Lowercase Match (Fixes capitalized words missing the dictionary)
    clean_word_lower = clean_word.lower()
    if clean_word_lower in TRANSLATION_DICT:
        translated_val = TRANSLATION_DICT[clean_word_lower]
        # Preserve capitalization if the original word was capitalized
        if clean_word[0].isupper() and translated_val:
            return translated_val[0].upper() + translated_val[1:]
        return translated_val
        
    # 4. Fallback: Return the raw Russian word as requested
    return raw_word

def process_file(in_filepath):
    """
    Worker function: Translates one ru_data.jsonl file and saves it as
    ru_tr2en_1to1map_wip_data.jsonl in the exact same directory.
    """
    dir_name = os.path.dirname(in_filepath)
    out_filepath = os.path.join(dir_name, "ru_tr2en_1to1map_wip_data.jsonl")
    
    docs_processed = 0
    
    with open(in_filepath, 'rb') as f_in, open(out_filepath, 'wb') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            try:
                doc = orjson.loads(line)
            except orjson.JSONDecodeError:
                print(f"⚠️ WARNING: Skipped corrupted JSON line in {in_filepath}")
                continue
                
            original_text = doc.get("text", "")
            if original_text:
                # Perform the word-by-word translation
                translated_text = RU_PATTERN.sub(translate_match, original_text)
                doc["text"] = translated_text
                
            # Write to the new WIP file
            f_out.write(orjson.dumps(doc) + b'\n')
            docs_processed += 1
            
    return in_filepath, docs_processed

# ==========================================
# 3. MAIN MULTIPROCESSING LOOP
# ==========================================

def main():
    DICT_PATH = "/home/adamga/leshemg/adamga/data/translations/top_russian_translated_fineweb_regex_1to1.json"
    INJ_FILES_PATTERN = r"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/*/ru_data.jsonl"
    
    # We load it here just to print the count to the console
    print(f"Pre-loading dictionary from {DICT_PATH}...")
    with open(DICT_PATH, 'rb') as f:
        temp_dict = orjson.loads(f.read())
    print(f"Verified {len(temp_dict):,} words in dictionary.")
    
    file_list = glob.glob(INJ_FILES_PATTERN)
    print(f"Found {len(file_list)} files to process...")
    
    cpu_cores = min(32, os.cpu_count() or 1)
    print(f"Starting Translation across {cpu_cores} CPU cores...")
    
    start_time = time.time()
    total_docs_processed = 0
    
    # CRITICAL FIX: The `initializer` and `initargs` force the workers to load the dictionary!
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores, initializer=init_worker, initargs=(DICT_PATH,)) as executor:
        futures = [executor.submit(process_file, f) for f in file_list]
        
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            filepath, num_docs = future.result()
            total_docs_processed += num_docs
            
            if count % 100 == 0:
                print(f"Processed {count}/{len(file_list)} files... ({total_docs_processed:,} total documents)")

    end_time = time.time()
    mins = (end_time - start_time) / 60
    print(f"\n✅ COMPLETE! {total_docs_processed:,} documents translated and saved to ru_tr2en_1to1map_wip_data.jsonl files in {mins:.2f} minutes.")

if __name__ == "__main__":
    main()