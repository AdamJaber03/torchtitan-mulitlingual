import os
import json
import re
import math
import multiprocessing as mp
from collections import Counter
import concurrent.futures
from datasets import load_dataset
import itertools

# ==========================================
# 1. BLAZING FAST CPU DATA EXTRACTION (HF Stream)
# ==========================================
import glob
import orjson

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

def process_text_chunk(text_chunk):
    local_counter = Counter()
    
    # ADDED: \u200C and \u200D (Zero-Width Joiners) to prevent shattering web transliterations
    ar_pattern = re.compile(r'[\u0620-\u065F\u0670-\u06EF\u06FA-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF\u200C\u200D]+')
    
    # Regex to detect if a string is NOTHING but diacritics/tashkeel (\u064B to \u065F)
    # If it matches this, it's a phantom diacritic and must be destroyed.
    pure_diacritic_pattern = re.compile(r'^[\u064B-\u065F]+$')
    
    for text in text_chunk:
        if text:
            # 1. Extract raw chunks
            raw_words = ar_pattern.findall(text)
            
            clean_words = []
            for w in raw_words:
                # 2. Strip visual/invisible formatting characters
                # \u0640 = Tatweel, \u200C = ZWNJ, \u200D = ZWJ
                w = w.replace('\u0640', '').replace('\u200C', '').replace('\u200D', '')
                
                # 3. Ensure it's not empty AND not just a floating diacritic
                if w and not pure_diacritic_pattern.fullmatch(w):
                    clean_words.append(w)
            
            local_counter.update(clean_words)
            
    return local_counter

def get_top_words_hf_stream(num_docs=5_000_000, top_k=50_000_000, chunk_size=25_000):
    # print(f"Connecting to Hugging Face Stream (fineweb-edu-ar)...")
    # ds_ar = load_dataset("kaust-generative-ai/fineweb-edu-ar", "ar", split="train", streaming=True)
    # TRANS_DIR = "/home/adamga/leshemg/adamga/data/fineweb_translated/translated"
    # print(f"Reading partially translated documents from {TRANS_DIR}...")
    # ds_ar = read_local_translated_chunks(TRANS_DIR)
    INJ_FILES = r"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/*/ar_data.jsonl"
    print(f"Reading injected documents from {INJ_FILES}...")
    ds_ar = read_local_translated_chunks(INJ_FILES, regex_pattern=True)
    
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
        for i, doc in enumerate(ds_ar):
            if i >= num_docs:
                break
                
            current_chunk.append(doc["text"])
            
            # When we hit our chunk limit, dispatch it to a CPU core
            if len(current_chunk) >= chunk_size:
                futures.append(executor.submit(process_text_chunk, current_chunk))
                current_chunk = [] # Reset for the next batch
                docs_processed += chunk_size
                
                if docs_processed % 250_000 == 0:
                    print(f"Dispatched {docs_processed:,} documents to CPU workers...")

        # Dispatch any remaining documents in the final partial chunk
        if current_chunk:
            futures.append(executor.submit(process_text_chunk, current_chunk))
            docs_processed += len(current_chunk)
            
        print(f"\nFinished reading {docs_processed:,} docs from Hugging Face.")
        print("Waiting for CPU cores to finish extracting and merging counters...")

        # 2. The Consumer: Gather results as they finish
        for future in concurrent.futures.as_completed(futures):
            local_counter = future.result()
            global_counter.update(local_counter)

    print("Extracting top unigrams...")
    top_ar_unigrams = [w for w, c in global_counter.most_common(top_k)]
    return top_ar_unigrams

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
    # UPGRADED MODEL: 32B Instruct. Keep TP=1 to maintain independent 8-GPU speed.
    llm = LLM(model="Qwen/Qwen2.5-32B-Instruct", tensor_parallel_size=1)  

    # Using the optimized parameters we discussed
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=15, 
        stop=["<|im_end|>", "\n"]
    )
    
    print(f"[GPU {gpu_id}] Formatting {len(words_chunk)} prompts...")
    prompts = [
        f"<|im_start|>system\n"
        f"You are a strict Modern Standard Arabic (MSA) to English dictionary. Output ONLY the English equivalent. NO Chinese. NO meta-text. NO explanations.\n"
        f"RULES:\n"
        f"1. Provide the exact literal English translation.\n"
        f"2. As a last resort, only if there's absolutely no english meaning available, provide ONLY the English transliteration.\n"
        f"3. Provide ONLY ONE primary meaning, even for pronouns/conjunctions.\n"
        f"4. Output strictly the English letters. No prefixes like 'transliteration:', no Arabic letters, no punctuation.\n"
        f"<|im_end|>\n"
        # --- FEW SHOT EXAMPLES TO LOCK THE OUTPUT FORMAT ---
        f"<|im_start|>user\nوقد<|im_end|>\n<|im_start|>assistant\nand has<|im_end|>\n"
        f"<|im_start|>user\nفاجزي<|im_end|>\n<|im_start|>assistant\nfajzi<|im_end|>\n"
        f"<|im_start|>user\nتلك<|im_end|>\n<|im_start|>assistant\nthat<|im_end|>\n"
        f"<|im_start|>user\nجزيرة<|im_end|>\n<|im_start|>assistant\nisland<|im_end|>\n"
        f"<|im_start|>user\nالاتحاد<|im_end|>\n<|im_start|>assistant\nthe union<|im_end|>\n"
        f"<|im_start|>user\nلالتوحد<|im_end|>\n<|im_start|>assistant\nfor autism<|im_end|>\n"
        f"<|im_start|>user\nإينفيرونيوز<|im_end|>\n<|im_start|>assistant\nenvironews<|im_end|>\n"
        f"<|im_start|>user\nإيباندرونات<|im_end|>\n<|im_start|>assistant\nibandronate<|im_end|>\n"
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
            daemon=False # THIS IS THE CRITICAL FIX
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
    top_ar_phrases = get_top_words_hf_stream(
        num_docs=NUM_DOCS_TO_PROCESS, 
        top_k=TOP_K_UNIGRAMS, 
        chunk_size=25_000 # Tune this if needed, 25k is a safe sweet spot
    )
    print(f"\nExtracted {len(top_ar_phrases):,} unique words.")
    
    # 2. GPU Phase: Multi-GPU Translation
    ar_to_en_dict = batch_translate_multigpu(top_ar_phrases, num_gpus=8)
    
    # 3. Save
    print("\nSaving results to JSON...")
    save_path = "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex2.json"
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(ar_to_en_dict, f, ensure_ascii=False, indent=4)
        
    print("Process complete! HF Stream successfully processed and translated.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()