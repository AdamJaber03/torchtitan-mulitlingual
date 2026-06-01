import os
import glob
import json
import time
import numpy as np
import itertools
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

# ==========================================
# --- GLOBAL CONFIGURATION ---
# ==========================================
# DATA_PATH = r"/home/adamga/fictional_entity_data/*/*data.jsonl"  # <-- Point this to your directory containing *.jsonl files
# DATA_PATH = r"/home/adamga/leshemg/adamga/data/fineweb_edu_en_ar_pair/ar*.jsonl"  # <-- Point this to your directory containing *.jsonl files
DATA_PATH = [r"/home/adamga/leshemg/adamga/data/fineweb2_hq/rus_Cyrl/original/*.jsonl", r"/home/adamga/leshemg/adamga/data/fineweb_translated/en-original/*.jsonl"]  # <-- Point this to your directory containing *.jsonl files
EXT = "65kVocab"
TOKENIZER_SAVE_PATH = r"trained_tokenizers/bpe_65k_en1.0_ru1.0.json"
VOCAB_SIZE = 65536
PRETOKENIZE = False
# ==========================================

def train_custom_tokenizer_in_ram(jsonl_files, vocab_size, save_path):
    print(f"--- Phase 1: Loading Data & Training BPE Tokenizer ---")
    t0 = time.time()
    
    # 1. Read all JSONL files completely into RAM
    all_texts = {}
    for file_path in jsonl_files:
        print(f"Loading {file_path} into memory...")
        with open(file_path, 'r', encoding='utf-8') as f:
            # List comprehension for maximum load speed
            all_texts[file_path] = [json.loads(line)["text"] for line in f if line.strip()]
            
    total_docs = sum(len(texts) for texts in all_texts.values())
    print(f"\nLoaded {total_docs:,} total documents in {time.time() - t0:.2f} seconds.")
    
    # 2. Initialize an empty Byte-Level BPE Tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # 3. Train Tokenizer using all CPU cores natively via Rust
    print(f"\nTraining BPE vocabulary to size {vocab_size} (this utilizes all CPU cores)...")
    t1 = time.time()
    
    # Flatten the dictionary values into a single generator just for training
    all_docs_iterator = itertools.chain.from_iterable(all_texts.values())
    tokenizer.train_from_iterator(all_docs_iterator, trainer=trainer)
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(save_path)
    print(f"[SUCCESS] Tokenizer trained in {time.time() - t1:.2f} seconds. Saved to {save_path}\n")
    
    # Return the dictionary of RAM-loaded texts so we don't have to read the files again!
    return all_texts

def pretokenize_in_ram(texts, tokenizer_path, bin_output_file):
    print(f"--- Phase 2: Multi-threaded Pre-tokenization for {os.path.basename(bin_output_file)} ---")
    t0 = time.time()
    
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    eos_token_id = fast_tokenizer.vocab.get("<|endoftext|>")
    
    chunk_size = 1_000_000
    total_tokens = 0
    
    # NEW: Store offsets (token indices where each document starts)
    offsets = [0] 
    
    with open(bin_output_file, 'wb') as f_out:
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            encodings = fast_tokenizer(chunk, add_special_tokens=False)["input_ids"]
            
            # Use a temporary list to collect flattened tokens for this chunk
            chunk_tokens = []
            for doc in encodings:
                doc.append(eos_token_id)
                chunk_tokens.extend(doc)
                # Track the cumulative offset
                offsets.append(offsets[-1] + len(doc))
                
            np_array = np.array(chunk_tokens, dtype=np.uint32)
            np_array.tofile(f_out)
            
            total_tokens += len(np_array)
            print(f"  ... Wrote {total_tokens:,} tokens so far")
            
    # Save offsets for the Dataset to use later
    offset_file = bin_output_file.replace(".bin", "_offsets.npy")
    np.save(offset_file, np.array(offsets, dtype=np.uint64))
    print(f"[SUCCESS] Saved {len(offsets):,} offsets to {offset_file}")
    print(f"[SUCCESS] Finished in {time.time() - t0:.2f} seconds!\n")

if __name__ == "__main__":
    # 1. Automatically find all .jsonl files in the target directory
    dataset_files = []
    if type(DATA_PATH) == list:
        for path in DATA_PATH:
            dataset_files.extend(glob.glob(path))
    else:
        dataset_files = glob.glob(DATA_PATH)
    
    if not dataset_files:
        print(f"[ERROR] No .jsonl files found in {DATA_PATH}!")
        exit(1)
        
    print(f"Found {len(dataset_files)} dataset files in {DATA_PATH}.")
    
    # 2. Read everything into RAM and train the tokenizer globally
    texts_dict = train_custom_tokenizer_in_ram(dataset_files, vocab_size=VOCAB_SIZE, save_path=TOKENIZER_SAVE_PATH)
    
    # 3. Pre-tokenize each file and save it as a .bin file in the same directory
    # ##################### if not running training ######################
    # texts_dict = {}
    # for file_path in dataset_files:
    #     print(f"Loading {file_path} into memory...")
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         # List comprehension for maximum load speed
    #         texts_dict[file_path] = [json.loads(line)["text"] for line in f if line.strip()]

    if PRETOKENIZE is False:
        print(f"Done! Your tokenizer is ready at {TOKENIZER_SAVE_PATH}. Set PRETOKENIZE=True to also pre-tokenize the datasets into .bin files for maximum training speed.")
        exit(0)
    for jsonl_file in dataset_files:
        # e.g., ./my_datasets/arabic.jsonl -> ./my_datasets/arabic.bin
        bin_file = os.path.splitext(jsonl_file)[0] + EXT + ".bin" 
        pretokenize_in_ram(texts_dict[jsonl_file], TOKENIZER_SAVE_PATH, bin_file)
        
    print("All datasets processed successfully!")