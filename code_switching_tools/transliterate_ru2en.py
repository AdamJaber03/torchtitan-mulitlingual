import os
import glob
import re
import orjson
import concurrent.futures
import argparse
from unidecode import unidecode

# 1. The Russian (Cyrillic) Matcher
# Catches Core Cyrillic, Extended Cyrillic, Historic Forms, Combining Diacritics (stress marks), and Formatting
RU_PATTERN = re.compile(r'([\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F\u1C80-\u1C8F\u0300-\u036F\u200C\u200D\u00AD]+)')

# 2. The Strict English Matcher (Matches anything that is NOT standard ASCII/English)
# This will catch Chinese, Arabic, Emojis, and stray non-ASCII punctuation.
NON_ENGLISH_PATTERN = re.compile(r'[^\x00-\x7F]+')

def transliterate_match(match):
    """Takes a Russian/Cyrillic regex match and forces it into English characters."""
    raw_word = match.group(1)
    # Clean visual formatting: \u00AD (Soft Hyphen), \u200C (ZWNJ), \u200D (ZWJ)
    clean_word = raw_word.replace('\u00AD', '').replace('\u200C', '').replace('\u200D', '')
    
    # Offline, instant transliteration to English script
    # unidecode naturally handles combining diacritics (like \u0301 stress marks) by stripping them
    return unidecode(clean_word)

def clean_file(filepath, out_dir, strict_english, inj=False):
    """Reads a chunk, transliterates leftovers, applies strict mode, and saves."""
    filename = os.path.basename(filepath)
    parent_dir = os.path.dirname(filepath)
    if inj:
        out_filepath = os.path.join(parent_dir, "ru_tr2en_1to1map_data.jsonl")
    else:
        out_filepath = os.path.join(out_dir, filename)
    
    processed_docs = 0
    
    with open(filepath, 'rb') as f_in, open(out_filepath, 'wb') as f_out:
        for line in f_in:
            line = line.strip()
            if not line: continue
            
            try:
                doc = orjson.loads(line)
            except orjson.JSONDecodeError:
                print(f"\n⚠️ WARNING: Skipped corrupted JSON line in {filepath}")
                continue # Gracefully skip the corrupted lines
                
            text = doc.get("text", "")
            if not text: continue
            
            # STEP 1: Transliterate any remaining Russian/Cyrillic words
            cleaned_text = RU_PATTERN.sub(transliterate_match, text)
                        
            # STEP 1.5: Convert Russian typography to English ASCII equivalents
            # Russian uses guillemets (« »), lower/upper quotes („ “), and Em-Dashes (—)
            cleaned_text = cleaned_text.replace('«', '"').replace('»', '"').replace('„', '"').replace('“', '"').replace('—', '-')
            
            # STEP 2: Optional Scorched Earth Mode
            if strict_english:
                # Replaces any non-ASCII character with a single space
                cleaned_text = NON_ENGLISH_PATTERN.sub(' ', cleaned_text)
                # Clean up any double spaces created by the deletion
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                
            # Save the document
            doc["text"] = cleaned_text
            f_out.write(orjson.dumps(doc) + b'\n')
            processed_docs += 1
            
    return processed_docs

def main():
    parser = argparse.ArgumentParser(description="Final cleanup and transliteration of translated documents.")
    parser.add_argument("--strict-english", action="store_true", 
                        help="Turn this flag on to physically delete ANY non-English/non-ASCII characters from the final text.")
    args = parser.parse_args()

    # Paths
    IN_DIR = "/home/adamga/leshemg/adamga/data/fineweb2_hq/rus_Cyrl/translated_1to1map_wip"
    OUT_DIR = "/home/adamga/leshemg/adamga/data/fineweb2_hq/rus_Cyrl/translated_1to1map"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    files = glob.glob(f"{IN_DIR}/*.jsonl")
    
    INJ = False
    # ***************for processing injecting data uncomment this section************
    # INJ = True
    # files = glob.glob(f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/*/ru_tr2en_1to1map_wip_data.jsonl")
    # *******************************************************************************
    
    print(f"Found {len(files)} files to clean.")
    if args.strict_english:
        print("⚠️  STRICT ENGLISH MODE ACTIVE: All non-ASCII characters will be deleted.")

    # Multiprocessing across your 32 CPUs
    cpu_cores = os.cpu_count()
    cpu_cores = 32
    print(f"Spinning up {cpu_cores} CPU cores...")
    
    total_docs = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        # Submit all files to the worker pool
        futures = {executor.submit(clean_file, f, OUT_DIR, args.strict_english, INJ): f for f in files}
        
        for future in concurrent.futures.as_completed(futures):
            docs_in_file = future.result()
            total_docs += docs_in_file
            
            # Print a status update every ~500k docs
            if total_docs % 500_000 < 25000: 
                print(f"Cleaned and saved {total_docs:,} documents...")

    print(f"\n✅ COMPLETE! {total_docs:,} documents successfully finalized.")
    print(f"Output saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()