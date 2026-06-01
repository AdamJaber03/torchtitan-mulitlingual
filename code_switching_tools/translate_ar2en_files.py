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

# Loaded once in main, shared instantly to all child processes via Linux fork
TRANSLATION_DICT = {}

# The bulletproof regex: Captures the Arabic word block
AR_PATTERN = re.compile(r'([\u0620-\u065F\u0670-\u06EF\u06FA-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF\u200C\u200D]+)')

# ==========================================
# 2. THE CORE TRANSLATION LOGIC
# ==========================================

def translate_match(match):
    """Called for EVERY regex match in the document."""
    raw_word = match.group(1)
    clean_word = raw_word.replace('\u0640', '').replace('\u200C', '').replace('\u200D', '')
    
    return TRANSLATION_DICT.get(clean_word, raw_word)

def process_file(in_filepath):
    """
    Worker function: Translates one ar_data.jsonl file and saves it as
    tr2en_1to1map_wip_data.jsonl in the exact same directory.
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
                translated_text = AR_PATTERN.sub(translate_match, original_text)
                doc["text"] = translated_text
                
            # Write to the new WIP file
            f_out.write(orjson.dumps(doc) + b'\n')
            docs_processed += 1
            
    return in_filepath, docs_processed

# ==========================================
# 3. MAIN MULTIPROCESSING LOOP
# ==========================================

def main():
    # DICT_PATH = "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex.json"
    # DICT_PATH = "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"
    DICT_PATH = "/home/adamga/leshemg/adamga/data/translations/top_russian_translated_fineweb_regex_1to1.json"
    # INJ_FILES_PATTERN = r"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/*/ar_data.jsonl"
    INJ_FILES_PATTERN = r"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/*/ru_data.jsonl"
    
    # 1. Load the dictionary into memory
    print(f"Loading dictionary from {DICT_PATH}...")
    global TRANSLATION_DICT
    with open(DICT_PATH, 'rb') as f:
        TRANSLATION_DICT = orjson.loads(f.read())
    print(f"Loaded {len(TRANSLATION_DICT):,} words into memory.")
    # ######################### 1.5. Mix the dictionary keys and values ##########################
    # print("Mixing the dictionary...")
    # keys = list(TRANSLATION_DICT.keys())
    # values = list(TRANSLATION_DICT.values())
    # random.seed(43)
    # # Randomly shuffle the list of values in place
    # random.shuffle(values)

    # # Re-combine the original keys with the shuffled values
    # TRANSLATION_DICT = dict(zip(keys, values))

    # print("Dictionary keys and values have been randomly mixed.")
    # ##################################################################################################

    # 2. Gather target files
    file_list = glob.glob(INJ_FILES_PATTERN)
    print(f"Found {len(file_list)} files to process...")
    
    # 3. Spin up the CPU cores
    cpu_cores = min(32, os.cpu_count() or 1)
    print(f"Starting Translation across {cpu_cores} CPU cores...")
    
    start_time = time.time()
    total_docs_processed = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        # Submit all files to the worker pool
        futures = [executor.submit(process_file, f) for f in file_list]
        
        # Track progress as files finish
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