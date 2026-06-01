import os
import json
import re
import math
import multiprocessing as mp
from collections import Counter
import concurrent.futures
from datasets import load_dataset
import itertools
import glob
import orjson

# ==========================================
# 1. BLAZING FAST CPU DATA EXTRACTION (HF Stream)
# ==========================================

def read_local_translated_chunks(directory, regex_pattern=False):
    """Yields documents one-by-one from all local jsonl chunks."""
    if regex_pattern:
        file_list = glob.glob(f"{directory}")
    else:
        file_list = glob.glob(f"{directory}/*.jsonl")
    for filepath in file_list:
        with open(filepath, 'rb') as f:
            for line in f:
                yield orjson.loads(line)

def process_text_chunk_russian_parallel(text_chunk):
    local_counter = Counter()
    
    # 1. The Comprehensive Cyrillic Pattern (The Exact Equivalent)
    # Catches Core Cyrillic, Extended Cyrillic, Historic Forms, Combining Diacritics, and Formatting
    ru_pattern = re.compile(
        r'[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F\u1C80-\u1C8F\u0300-\u036F\u200C\u200D\u00AD]+'
    )
    
    # 2. Pure Diacritic Pattern
    # Detects if a string is NOTHING but floating combining marks (e.g., a standalone stress mark)
    pure_diacritic_pattern = re.compile(r'^[\u0300-\u036F]+$')
    
    for text in text_chunk:
        if text:
            # Extract raw chunks using the comprehensive pattern
            raw_words = ru_pattern.findall(text)
            
            clean_words = []
            for w in raw_words:
                # 3. Strip visual/invisible formatting characters
                # \u200C = ZWNJ, \u200D = ZWJ, \u00AD = Soft Hyphen
                w = w.replace('\u200C', '').replace('\u200D', '').replace('\u00AD', '')
                
                # 4. Strip ALL combining diacritics to get the "Base Word"
                w = re.sub(r'[\u0300-\u036F]', '', w)
                
                # 5. Ensure it's not empty AND wasn't just a floating diacritic
                if w and not pure_diacritic_pattern.fullmatch(w):
                    clean_words.append(w)
            
            local_counter.update(clean_words)
            
    return local_counter

def get_top_words_hf_stream(num_docs=5_000_000, top_k=50_000_000, chunk_size=25_000):
    # print(f"Connecting to Hugging Face Stream (fineweb-edu-ru)...")
    # ds_ru = load_dataset("HuggingFaceFW/fineweb-edu", "CC-MAIN-2024-10", split="train", streaming=True)
    
    INJ_FILES = r"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/*/ru_data.jsonl"
    print(f"Reading injected documents from {INJ_FILES}...")
    ds_ru = read_local_translated_chunks(INJ_FILES, regex_pattern=True)

    # ds_ru = load_dataset("json", data_dir=r"/home/adamga/leshemg/adamga/data/fineweb2_hq/rus_Cyrl/original", split="train", streaming=True)    

    global_counter = Counter()
    
    # We use a ProcessPool to utilize all CPU cores
    cpu_cores = os.cpu_count()
    cpu_cores = 32
    print(f"Starting Producer-Consumer extraction across {cpu_cores} CPU cores...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        futures = []
        current_chunk = []
        docs_processed = 0
        
        # 1. The Producer: Read stream and throw chunks to the CPU cores
        for i, doc in enumerate(ds_ru):
            if i >= num_docs:
                break
                
            current_chunk.append(doc["text"])
            
            # When we hit our chunk limit, dispatch it to a CPU core
            if len(current_chunk) >= chunk_size:
                futures.append(executor.submit(process_text_chunk_russian_parallel, current_chunk))
                current_chunk = [] # Reset for the next batch
                docs_processed += chunk_size
                
                if docs_processed % 250_000 == 0:
                    print(f"Dispatched {docs_processed:,} documents to CPU workers...")

        # Dispatch any remaining documents in the final partial chunk
        if current_chunk:
            futures.append(executor.submit(process_text_chunk_russian_parallel, current_chunk))
            docs_processed += len(current_chunk)
            
        print(f"\nFinished reading {docs_processed:,} docs from Hugging Face/Local.")
        print("Waiting for CPU cores to finish extracting and merging counters...")

        # 2. The Consumer: Gather results as they finish
        for future in concurrent.futures.as_completed(futures):
            local_counter = future.result()
            global_counter.update(local_counter)

    print("Extracting top unigrams...")
    top_ru_unigrams = [w for w, c in global_counter.most_common(top_k)]
    return top_ru_unigrams

# ==========================================
# 2. DATA PARALLEL GPU TRANSLATION (Untouched & Perfect)
# ==========================================

def vllm_worker(gpu_id, words_chunk, return_dict):
    """
    Worker function that boots a single vLLM engine on a specific GPU.
    Runs in its own isolated, NON-daemonic process.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams
    
    print(f"[GPU {gpu_id}] Booting Qwen 32B...")
    llm = LLM(model="Qwen/Qwen2.5-32B-Instruct", tensor_parallel_size=1)  

    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=15, 
        stop=["<|im_end|>", "\n"]
    )
    
    print(f"[GPU {gpu_id}] Formatting {len(words_chunk)} prompts...")
    prompts = [
        f"<|im_start|>system\n"
        f"You are a strict Russian to English dictionary. Output ONLY the English equivalent. NO Chinese. NO meta-text. NO explanations.\n"
        f"RULES:\n"
        f"1. Provide the exact literal English translation.\n"
        f"2. As a last resort, only if there's absolutely no english meaning available, provide ONLY the English transliteration.\n"
        f"3. Provide ONLY ONE primary meaning, even for pronouns/conjunctions.\n"
        f"4. Output strictly the English letters. No prefixes like 'transliteration:', no Cyrillic letters, no punctuation.\n"
        f"<|im_end|>\n"
        # --- FEW SHOT EXAMPLES TO LOCK THE OUTPUT FORMAT ---
        f"<|im_start|>user\nи<|im_end|>\n<|im_start|>assistant\nand<|im_end|>\n"
        f"<|im_start|>user\nчто<|im_end|>\n<|im_start|>assistant\nthat<|im_end|>\n"
        f"<|im_start|>user\nостров<|im_end|>\n<|im_start|>assistant\nisland<|im_end|>\n"
        f"<|im_start|>user\nсоюз<|im_end|>\n<|im_start|>assistant\nunion<|im_end|>\n"
        f"<|im_start|>user\nдля аутизма<|im_end|>\n<|im_start|>assistant\nfor autism<|im_end|>\n"
        f"<|im_start|>user\nинвайроньюс<|im_end|>\n<|im_start|>assistant\nenvironews<|im_end|>\n"
        f"<|im_start|>user\nибандронат<|im_end|>\n<|im_start|>assistant\nibandronate<|im_end|>\n"
        # --- ACTUAL TARGET WORD ---
        f"<|im_start|>user\n{word}<|im_end|>\n<|im_start|>assistant\n"
        for word in words_chunk
    ]

    print(f"[GPU {gpu_id}] Starting batch generation...")
    outputs = llm.generate(prompts, sampling_params)
    
    translation_dict = {}
    for i, output in enumerate(outputs):
        orig_word = words_chunk[i]
        english_word = output.outputs[0].text.strip().lower()
        translation_dict[orig_word] = english_word
        
    # Safely push the result back to the main process via the shared memory dict
    return_dict[gpu_id] = translation_dict

def batch_translate_multigpu(word_list, num_gpus=8):
    print(f"\nDistributing {len(word_list):,} words across {num_gpus} GPUs...")
    chunk_size = math.ceil(len(word_list) / num_gpus)
    chunks = [word_list[i : i + chunk_size] for i in range(0, len(word_list), chunk_size)]
    
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    
    # Manually spawn non-daemonic processes
    for gpu_id, chunk in enumerate(chunks):
        p = mp.Process(
            target=vllm_worker, 
            args=(gpu_id, chunk, return_dict), 
            daemon=False 
        )
        processes.append(p)
        p.start()
        
    # Wait for all 8 GPUs to finish translating
    for p in processes:
        p.join()
        
    final_dict = {}
    for local_dict in return_dict.values():
        final_dict.update(local_dict)
        
    return final_dict

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    NUM_DOCS_TO_PROCESS = 7_000_000 
    TOP_K_UNIGRAMS = 50_000_000 
    
    # 1. CPU Phase: HF Streaming & Regex Extraction
    top_ru_phrases = get_top_words_hf_stream(
        num_docs=NUM_DOCS_TO_PROCESS, 
        top_k=TOP_K_UNIGRAMS, 
        chunk_size=25_000 
    )
    print(f"\nExtracted {len(top_ru_phrases):,} unique words.")
    
    # 2. GPU Phase: Multi-GPU Translation
    ru_to_en_dict = batch_translate_multigpu(top_ru_phrases, num_gpus=8)
    
    # 3. Save
    print("\nSaving results to JSON...")
    save_path = "/home/adamga/leshemg/adamga/data/translations/top_russian_translated_fineweb_regex2.json"
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(ru_to_en_dict, f, ensure_ascii=False, indent=4)
        
    print("Process complete! Data successfully processed and translated.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()