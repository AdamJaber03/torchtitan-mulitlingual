import os
import re
import time
import concurrent.futures
import orjson
import glob
from unidecode import unidecode
import random

# ==========================================
# 1. GLOBAL VARIABLES FOR WORKERS
# ==========================================

TRANSLATION_DICT = {}

# Matcher 1: Captures the Russian (Cyrillic) word blocks
# Catches Core Cyrillic, Extended Cyrillic, Historic Forms, Combining Diacritics (stress marks), and Formatting
RU_PATTERN = re.compile(r'([\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F\u1C80-\u1C8F\u0300-\u036F\u200C\u200D\u00AD]+)')

# Matcher 2: Strict English/ASCII enforcer (Anything not standard English text/punctuation)
NON_ENGLISH_PATTERN = re.compile(r'[^\x00-\x7F]+')

# ==========================================
# 2. THE CORE TRANSLATION LOGIC
# ==========================================

def translate_and_transliterate(match):
    """
    Called for EVERY Russian regex match. 
    Returns dictionary translation if found, otherwise transliterates.
    """
    raw_word = match.group(1)
    
    # Clean visual formatting: \u00AD (Soft Hyphen), \u200C (ZWNJ), \u200D (ZWJ)
    clean_word = raw_word.replace('\u00AD', '').replace('\u200C', '').replace('\u200D', '')
    
    # Strip combining diacritics (like \u0301 stress marks) so the word matches the dictionary keys
    clean_word = re.sub(r'[\u0300-\u036F]', '', clean_word)
    
    # .get() will return the dict value if it exists, otherwise it runs unidecode
    return TRANSLATION_DICT.get(clean_word, unidecode(clean_word))

def process_text_field(text):
    """Runs the 3-step pipeline on a single string."""
    if not text:
        return text
        
    # Step 1: Translate (from dict) or Transliterate (from unidecode)
    cleaned = RU_PATTERN.sub(translate_and_transliterate, text)
    
    # Step 2: Save Russian punctuation by converting to English equivalents
    # Russian uses guillemets (« »), lower/upper quotes („ “), and Em-Dashes (—)
    cleaned = cleaned.replace('«', '"').replace('»', '"').replace('„', '"').replace('“', '"').replace('—', '-')
    
    # Step 3: Scorched Earth - Delete any surviving non-ASCII characters (Chinese, emojis, etc.)
    cleaned = NON_ENGLISH_PATTERN.sub(' ', cleaned)
    
    # Clean up double spaces created by deletions
    return re.sub(r'\s+', ' ', cleaned).strip()

def process_file(in_filepath):
    """
    Worker function: Translates one mcq_ru.jsonl file and saves it as
    mcq_tr2en_1to1map_mixed.jsonl in the exact same directory.
    """
    dir_name = os.path.dirname(in_filepath)
    out_filepath = os.path.join(dir_name, "mcq_ru_tr2en_1to1map.jsonl")
    
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
                
            # Process the MCQ "question" string
            if "question" in doc:
                doc["question"] = process_text_field(doc.get("question", ""))
                
            # Process the MCQ "choices" list
            if "choices" in doc and isinstance(doc["choices"], list):
                doc["choices"] = [process_text_field(choice) for choice in doc["choices"]]
                
            # Answer index remains an integer, so we don't touch it.

            # Write to the final tr2en file
            f_out.write(orjson.dumps(doc) + b'\n')
            docs_processed += 1
            
    return in_filepath, docs_processed

# ==========================================
# 3. MAIN MULTIPROCESSING LOOP
# ==========================================

def main():
    # Updated paths for Russian dictionary and MCQ targets
    DICT_PATH = "/home/adamga/leshemg/adamga/data/translations/top_russian_translated_fineweb_regex_1to1.json"
    INJ_FILES_PATTERN = r"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/*/mcq_ru.jsonl"
    
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
    print(f"Starting Translation/Transliteration across {cpu_cores} CPU cores...")
    
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
    print(f"\n✅ COMPLETE! {total_docs_processed:,} MCQ documents fully translated, cleaned, and saved in {mins:.2f} minutes.")

if __name__ == "__main__":
    main()