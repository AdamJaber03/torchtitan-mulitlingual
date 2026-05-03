import json
import re
import time
from collections import Counter
from datasets import load_dataset
from deep_translator import GoogleTranslator
import os
import orjson

def stream_local_jsonl(num_folders=2080):
    """
    Generator that acts exactly like a Hugging Face streaming dataset.
    Yields: {"text": "actual text"}
    """
    base_path = "/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/"
    
    for i in range(num_folders):
        # filepath = os.path.join(base_path, str(i), "ar_data.jsonl")
        filepath = os.path.join(base_path, str(i), "mcq_ar.jsonl")
        
        if not os.path.exists(filepath):
            continue # Skip silently if a folder or file is missing
            
        with open(filepath, 'rb') as f: # orjson requires reading in binary ('rb') mode
            for line in f:
                line = line.strip()
                if line:
                    # Parse the JSON and yield a dictionary to mimic HF format
                    parsed = orjson.loads(line)
                    # yield {"text": parsed.get("text", "")}
                    yield {"text": parsed.get("question", "") + " " + " ".join(parsed.get("choices", ""))} # For MCQ files, we want both question and choices for better coverage

def get_top_words(num_docs=500_000, top_k_unigrams=10_000, top_k_bigrams=2_000, top_k_trigrams=1_000, seed=42, buffer_size=10_000):
    print(f"Starting to process {num_docs} documents. This will take a while...")
    
    # 1. Setup the streams
    ds_ar = load_dataset("kaust-generative-ai/fineweb-edu-ar", "ar", split="train", streaming=True)
    # ds_ar = stream_local_jsonl(num_folders=2080)    # ds_en = load_dataset("kaust-generative-ai/fineweb-edu-ar", "en", split="train", streaming=True)
    
    # # 2. Synchronized Shuffle
    # print(f"Shuffling streams with buffer size {buffer_size}...")
    # ds_ar = ds_ar.shuffle(seed=seed, buffer_size=buffer_size)
    # ds_en = ds_en.shuffle(seed=seed, buffer_size=buffer_size)
    
    # paired_stream = zip(ds_ar, ds_en)
    
    # Counters for unigrams, bigrams, and trigrams
    ar_counter, en_counter = Counter(), Counter()
    ar_bigram_counter, en_bigram_counter = Counter(), Counter()
    ar_trigram_counter, en_trigram_counter = Counter(), Counter()
    doc_word_sets = []
    # Regex patterns
    ar_pattern = re.compile(r'[\u0621-\u065A]+')
    en_pattern = re.compile(r'\b[a-zA-Z]+\b')

    # 3. Count the frequencies
    # for i, (doc_ar, doc_en) in enumerate(paired_stream):
    for i, doc_ar in enumerate(ds_ar):
        if i >= num_docs:
            break
            
        # Extract unigrams
        ar_words = ar_pattern.findall(doc_ar["text"])
        # en_words = en_pattern.findall(doc_en["text"].lower())
        
        # Generate bigrams and trigrams using zip
        # ar_bigrams = [f"{w1} {w2}" for w1, w2 in zip(ar_words, ar_words[1:])]
        # en_bigrams = [f"{w1} {w2}" for w1, w2 in zip(en_words, en_words[1:])]
        
        # ar_trigrams = [f"{w1} {w2} {w3}" for w1, w2, w3 in zip(ar_words, ar_words[1:], ar_words[2:])]
        # en_trigrams = [f"{w1} {w2} {w3}" for w1, w2, w3 in zip(en_words, en_words[1:], en_words[2:])]
        
        # Update all counters
        ar_counter.update(ar_words)
        # en_counter.update(en_words)
        # ar_bigram_counter.update(ar_bigrams)
        # en_bigram_counter.update(en_bigrams)
        # ar_trigram_counter.update(ar_trigrams)
        # en_trigram_counter.update(en_trigrams)
        if ar_words:
            doc_word_sets.append(set(ar_words))
        if (i + 1) % 10_000 == 0 or (i + 1) == num_docs:
            print(f"Processed {i + 1} / {num_docs} documents...")

    # Extract the top items
    top_ar_unigrams = [w for w, c in ar_counter.most_common(top_k_unigrams)]
    # top_ar_bigrams = [w for w, c in ar_bigram_counter.most_common(top_k_bigrams)]
    # top_ar_trigrams = [w for w, c in ar_trigram_counter.most_common(top_k_trigrams)]
    
    # top_en_unigrams = [w for w, c in en_counter.most_common(top_k_unigrams)]
    # top_en_bigrams = [w for w, c in en_bigram_counter.most_common(top_k_bigrams)]
    # top_en_trigrams = [w for w, c in en_trigram_counter.most_common(top_k_trigrams)]

    # --- Coverage Percentage (Based on Unigrams only for mathematical accuracy) ---
    total_ar_words = sum(ar_counter.values())
    # total_en_words = sum(en_counter.values())
    
    if total_ar_words > 0:
        ar_coverage = sum(c for w, c in ar_counter.most_common(top_k_unigrams)) / total_ar_words * 100
        print(f"\n📊 Top {top_k_unigrams} Arabic unigrams cover {ar_coverage:.2f}% of the text.")
        
    # if total_en_words > 0:
    #     en_coverage = sum(c for w, c in en_counter.most_common(top_k_unigrams)) / total_en_words * 100
    #     print(f"📊 Top {top_k_unigrams} English unigrams cover {en_coverage:.2f}% of the text.\n")

    # Combine them in specific order: Trigrams first, then Bigrams, then Unigrams
    top_ar = top_ar_unigrams
    # top_ar = top_ar_trigrams + top_ar_bigrams + top_ar_unigrams
    # top_en = top_en_trigrams + top_en_bigrams + top_en_unigrams
    # --- PASS 2: Calculate Document Coverage (Lightning Fast in RAM) ---
    top_unigrams_set = set(top_ar_unigrams)
    print("\n--- Running Pass 2: Calculating Document Coverage in RAM ---")
    
    completely_covered_docs = 0
    valid_docs_checked = len(doc_word_sets)

    # We are just looping through RAM now. This will finish almost instantly.
    for doc_set in doc_word_sets:
        if doc_set.issubset(top_unigrams_set):
            completely_covered_docs += 1

    # Print Document Coverage Results
    if valid_docs_checked > 0:
        doc_coverage_pct = (completely_covered_docs / valid_docs_checked) * 100
        print(f"\n📄 {completely_covered_docs:,} out of {valid_docs_checked:,} documents are completely covered.")
        print(f"📊 That is {doc_coverage_pct:.2f}% of the processed documents.")
    # return top_ar, top_en
    return top_ar, None


# def batch_translate(words, source_lang, target_lang, batch_size=250):
#     """Safely translates a large list of phrases/words by batching them."""
#     translator = GoogleTranslator(source=source_lang, target=target_lang)
#     translation_dict = {}
    
#     print(f"\nTranslating {len(words)} items from {source_lang} to {target_lang}...")
    
#     for i in range(0, len(words), batch_size):
#         batch = words[i : i + batch_size]
#         combined_text = "\n".join(batch)
        
#         try:
#             result = translator.translate(combined_text)
#             translated_words = result.split("\n")
            
#             if len(translated_words) != len(batch):
#                 print(f"Batch mismatch at index {i}. Falling back to individual translation...")
#                 for word in batch:
#                     translation_dict[word] = translator.translate(word)
#                     time.sleep(0.5)
#             else:
#                 for orig, trans in zip(batch, translated_words):
#                     translation_dict[orig] = trans.strip()
                    
#         except Exception as e:
#             print(f"Error at batch {i}: {e}. Skipping batch...")
            
#         time.sleep(2)
        
#         if (i + batch_size) % 1000 == 0:
#             print(f"Translated {i + batch_size} items...")
            
#     return translation_dict
from vllm import LLM, SamplingParams

def batch_translate(word_list, source_lang, target_lang):
    # 1. Initialize vLLM Engine
    # Qwen2.5 is currently the absolute best open-weight model for Arabic.
    # tensor_parallel_size=8 tells vLLM to seamlessly use all 8 of your Blackwells.
    print("Booting vLLM across 8 GPUs...")
    llm = LLM(model="Qwen/Qwen2.5-14B-Instruct", tensor_parallel_size=8)

    # 2. Strict Generation Guardrails
    # temperature=0 makes it deterministic (no creative hallucinations).
    # max_tokens=3 physically forces it to stop after a single word.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=7)

    print("Formatting prompts...")
    # ChatML formatting: We strictly instruct it to act as a dictionary.
    prompts = [
        f"<|im_start|>system\n"
        f"You are a strict Arabic-to-English dictionary. Output ONLY the English equivalent for the provided word.\n"
        f"RULES:\n"
        f"1. TRANSLATION FIRST: Always provide the literal, semantic English translation of the Arabic word.\n"
        f"2. FALLBACK: ONLY if the word has absolutely no semantic meaning in the Arabic language (e.g., foreign names, loanwords, or purely phonetic entities), TRANSLITERATE it into English letters. tranliterate in the most obvious basic intuitive way possible\n"
        f"Output nothing but the final English word. No punctuation.<|im_end|>\n"
        f"<|im_start|>user\n{word}<|im_end|>\n<|im_start|>assistant\n"
        for word in word_list
    ]

    print(f"Generating translations for {len(prompts)} words...")
    # 3. Asynchronous Batch Generation
    # vLLM handles the continuous batching and GPU saturation under the hood.
    outputs = llm.generate(prompts, sampling_params)

    # 4. Extract and clean the results
    translation_dict = {}
    for i, output in enumerate(outputs):
        orig_word = word_list[i]
        # Get the text, strip any accidental whitespace or newlines
        english_word = output.outputs[0].text.strip()
        translation_dict[orig_word] = english_word.lower()

    return translation_dict

def main():
    NUM_DOCS_TO_PROCESS = 5_000_000 
    TOP_K_UNIGRAMS = 50_000_000 
    TOP_K_BIGRAMS = 0
    TOP_K_TRIGRAMS = 0
    
    top_ar_phrases, top_en_phrases = get_top_words(
        num_docs=NUM_DOCS_TO_PROCESS, 
        top_k_unigrams=TOP_K_UNIGRAMS,
        top_k_bigrams=TOP_K_BIGRAMS,
        top_k_trigrams=TOP_K_TRIGRAMS
    )
    
    ar_to_en_dict = batch_translate(top_ar_phrases, source_lang='ar', target_lang='en')
    # en_to_ar_dict = batch_translate(top_en_phrases, source_lang='en', target_lang='ar')
    
    print("\nSaving results to JSON files...")
    # Because Python dicts preserve insertion order, the JSON will automatically 
    # feature trigrams first, then bigrams, then unigrams.
    with open("/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_geminiseeds_newregex.json", "w", encoding="utf-8") as f:
        json.dump(ar_to_en_dict, f, ensure_ascii=False, indent=4)

    # with open("/home/adamga/leshemg/adamga/data/translations/top_english_translated_100k.json", "w", encoding="utf-8") as f:
    #     json.dump(en_to_ar_dict, f, ensure_ascii=False, indent=4)
        
    print("Process complete!")

if __name__ == "__main__":
    main()