import json
import re
import time
from collections import Counter
from datasets import load_dataset
from deep_translator import GoogleTranslator

def get_top_words(num_docs=500_000, top_k_unigrams=10_000, top_k_bigrams=2_000, top_k_trigrams=1_000, seed=42, buffer_size=10_000):
    print(f"Starting to process {num_docs} documents. This will take a while...")
    
    # 1. Setup the streams
    ds_ar = load_dataset("kaust-generative-ai/fineweb-edu-ar", "ar", split="train", streaming=True)
    ds_en = load_dataset("kaust-generative-ai/fineweb-edu-ar", "en", split="train", streaming=True)
    
    # 2. Synchronized Shuffle
    print(f"Shuffling streams with buffer size {buffer_size}...")
    ds_ar = ds_ar.shuffle(seed=seed, buffer_size=buffer_size)
    ds_en = ds_en.shuffle(seed=seed, buffer_size=buffer_size)
    
    paired_stream = zip(ds_ar, ds_en)
    
    # Counters for unigrams, bigrams, and trigrams
    ar_counter, en_counter = Counter(), Counter()
    ar_bigram_counter, en_bigram_counter = Counter(), Counter()
    ar_trigram_counter, en_trigram_counter = Counter(), Counter()
    
    # Regex patterns
    ar_pattern = re.compile(r'[\u0600-\u06FF]+')
    en_pattern = re.compile(r'\b[a-zA-Z]+\b')

    # 3. Count the frequencies
    for i, (doc_ar, doc_en) in enumerate(paired_stream):
        if i >= num_docs:
            break
            
        # Extract unigrams
        ar_words = ar_pattern.findall(doc_ar["text"])
        en_words = en_pattern.findall(doc_en["text"].lower())
        
        # Generate bigrams and trigrams using zip
        ar_bigrams = [f"{w1} {w2}" for w1, w2 in zip(ar_words, ar_words[1:])]
        en_bigrams = [f"{w1} {w2}" for w1, w2 in zip(en_words, en_words[1:])]
        
        ar_trigrams = [f"{w1} {w2} {w3}" for w1, w2, w3 in zip(ar_words, ar_words[1:], ar_words[2:])]
        en_trigrams = [f"{w1} {w2} {w3}" for w1, w2, w3 in zip(en_words, en_words[1:], en_words[2:])]
        
        # Update all counters
        ar_counter.update(ar_words)
        en_counter.update(en_words)
        ar_bigram_counter.update(ar_bigrams)
        en_bigram_counter.update(en_bigrams)
        ar_trigram_counter.update(ar_trigrams)
        en_trigram_counter.update(en_trigrams)
        
        if (i + 1) % 10_000 == 0 or (i + 1) == num_docs:
            print(f"Processed {i + 1} / {num_docs} documents...")

    # Extract the top items
    top_ar_unigrams = [w for w, c in ar_counter.most_common(top_k_unigrams)]
    top_ar_bigrams = [w for w, c in ar_bigram_counter.most_common(top_k_bigrams)]
    top_ar_trigrams = [w for w, c in ar_trigram_counter.most_common(top_k_trigrams)]
    
    top_en_unigrams = [w for w, c in en_counter.most_common(top_k_unigrams)]
    top_en_bigrams = [w for w, c in en_bigram_counter.most_common(top_k_bigrams)]
    top_en_trigrams = [w for w, c in en_trigram_counter.most_common(top_k_trigrams)]

    # --- Coverage Percentage (Based on Unigrams only for mathematical accuracy) ---
    total_ar_words = sum(ar_counter.values())
    total_en_words = sum(en_counter.values())
    
    if total_ar_words > 0:
        ar_coverage = sum(c for w, c in ar_counter.most_common(top_k_unigrams)) / total_ar_words * 100
        print(f"\n📊 Top {top_k_unigrams} Arabic unigrams cover {ar_coverage:.2f}% of the text.")
        
    if total_en_words > 0:
        en_coverage = sum(c for w, c in en_counter.most_common(top_k_unigrams)) / total_en_words * 100
        print(f"📊 Top {top_k_unigrams} English unigrams cover {en_coverage:.2f}% of the text.\n")

    # Combine them in specific order: Trigrams first, then Bigrams, then Unigrams
    top_ar = top_ar_trigrams + top_ar_bigrams + top_ar_unigrams
    top_en = top_en_trigrams + top_en_bigrams + top_en_unigrams
    
    return top_ar, top_en


def batch_translate(words, source_lang, target_lang, batch_size=50):
    """Safely translates a large list of phrases/words by batching them."""
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    translation_dict = {}
    
    print(f"\nTranslating {len(words)} items from {source_lang} to {target_lang}...")
    
    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        combined_text = "\n".join(batch)
        
        try:
            result = translator.translate(combined_text)
            translated_words = result.split("\n")
            
            if len(translated_words) != len(batch):
                print(f"Batch mismatch at index {i}. Falling back to individual translation...")
                for word in batch:
                    translation_dict[word] = translator.translate(word)
                    time.sleep(0.5)
            else:
                for orig, trans in zip(batch, translated_words):
                    translation_dict[orig] = trans.strip()
                    
        except Exception as e:
            print(f"Error at batch {i}: {e}. Skipping batch...")
            
        time.sleep(2)
        
        if (i + batch_size) % 1000 == 0:
            print(f"Translated {i + batch_size} items...")
            
    return translation_dict


def main():
    NUM_DOCS_TO_PROCESS = 500_000 
    TOP_K_UNIGRAMS = 10_000 
    TOP_K_BIGRAMS = 2_000
    TOP_K_TRIGRAMS = 1_000
    
    top_ar_phrases, top_en_phrases = get_top_words(
        num_docs=NUM_DOCS_TO_PROCESS, 
        top_k_unigrams=TOP_K_UNIGRAMS,
        top_k_bigrams=TOP_K_BIGRAMS,
        top_k_trigrams=TOP_K_TRIGRAMS
    )
    
    ar_to_en_dict = batch_translate(top_ar_phrases, source_lang='ar', target_lang='en')
    en_to_ar_dict = batch_translate(top_en_phrases, source_lang='en', target_lang='ar')
    
    print("\nSaving results to JSON files...")
    # Because Python dicts preserve insertion order, the JSON will automatically 
    # feature trigrams first, then bigrams, then unigrams.
    with open("/home/adamga/leshemg/adamga/data/translations/top_arabic_translated.json", "w", encoding="utf-8") as f:
        json.dump(ar_to_en_dict, f, ensure_ascii=False, indent=4)

    with open("/home/adamga/leshemg/adamga/data/translations/top_english_translated.json", "w", encoding="utf-8") as f:
        json.dump(en_to_ar_dict, f, ensure_ascii=False, indent=4)
        
    print("Process complete!")

if __name__ == "__main__":
    main()