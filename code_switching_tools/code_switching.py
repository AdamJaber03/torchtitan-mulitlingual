import json
import random
import re
import os
from datasets import load_dataset

AR_2_EN_PATH = "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated.json"
EN_2_AR_PATH = "/home/adamga/leshemg/adamga/data/translations/top_english_translated.json"

# 1. Load the dictionaries (with a fallback for testing)
print("Loading dictionaries...")
if os.path.exists(EN_2_AR_PATH) and os.path.exists(AR_2_EN_PATH):
    with open(EN_2_AR_PATH, "r", encoding="utf-8") as f:
        en_to_ar = json.load(f)
    with open(AR_2_EN_PATH, "r", encoding="utf-8") as f:
        ar_to_en = json.load(f)
else:
    print("⚠️ Full JSONs not found. Using a small test dictionary instead.")
    # A tiny mock dictionary containing unigrams and a bigram
    en_to_ar = {
        "machine learning": "تعلم آلي",
        "science": "علم", 
        "student": "طالب", 
        "book": "كتاب", 
        "school": "مدرسة", 
        "study": "دراسة"
    }
    ar_to_en = {
        "تعلم آلي": "machine learning",
        "علم": "science", 
        "طالب": "student", 
        "كتاب": "book", 
        "مدرسة": "school", 
        "دراسة": "study", 
        "في": "in", 
        "من": "from"
    }

def inject_ngrams(text, translation_dict, lang='en', replace_prob=0.20):
    """
    Tokenizes text and greedily replaces trigrams, bigrams, and unigrams.
    Preserves surrounding punctuation and spacing.
    """
    # Create capture groups () so re.split KEEPS the delimiters (spaces/punctuation)
    if lang == 'en':
        pattern = re.compile(r'([a-zA-Z]+)')
    else:
        pattern = re.compile(r'([\u0600-\u06FF]+)')
        
    tokens = pattern.split(text)
    
    # re.split structure: [delimiter, word, delimiter, word, delimiter...]
    result = []
    
    # Keep the very first delimiter (often just an empty string)
    if len(tokens) > 0:
        result.append(tokens[0])
        
    i = 1
    while i < len(tokens):
        replaced = False
        
        # Try matching 3-grams, then 2-grams, then 1-grams
        for n in [3, 2, 1]:
            # Check if there are enough words left in the sequence
            if i + (n - 1) * 2 < len(tokens):
                
                # Extract the next 'n' words (skipping the delimiters between them)
                words = [tokens[i + j*2] for j in range(n)]
                key = " ".join(words)
                
                if lang == 'en':
                    key = key.lower()
                    
                if key in translation_dict:
                    # Roll the dice
                    if random.random() < replace_prob:
                        result.append(translation_dict[key])
                        
                        # Advance the pointer to the end of the matched phrase
                        i += (n - 1) * 2 
                        replaced = True
                        break
                        
        # If no n-gram matched (or the probability failed), keep original word
        if not replaced:
            result.append(tokens[i])
            
        # Append the trailing delimiter (space/punctuation) after the word/phrase
        if i + 1 < len(tokens):
            result.append(tokens[i + 1])
            
        i += 2
        
    return "".join(result)

def main():
    print("Loading streaming datasets...")
    # Load the streams
    ds_ar = load_dataset("kaust-generative-ai/fineweb-edu-ar", "ar", split="train", streaming=True)
    ds_en = load_dataset("kaust-generative-ai/fineweb-edu-ar", "en", split="train", streaming=True)
    
    # Shuffle synchronously
    SEED = 44
    ds_ar = ds_ar.shuffle(seed=SEED, buffer_size=1000)
    ds_en = ds_en.shuffle(seed=SEED, buffer_size=1000)
    
    paired_stream = zip(ds_ar, ds_en)
    
    print("\nProcessing documents...\n" + "="*50)
    
    # Process the first 2 documents
    for i, (doc_ar, doc_en) in enumerate(paired_stream):
        if i >= 2: 
            break
            
        # Get a snippet of the text to keep the console output readable
        raw_ar = doc_ar["text"][:300] + "..."
        raw_en = doc_en["text"][:300] + "..."
        
        # Inject translations using the new n-gram function
        mixed_en = inject_ngrams(raw_en, en_to_ar, lang='en', replace_prob=0.50)
        mixed_ar = inject_ngrams(raw_ar, ar_to_en, lang='ar', replace_prob=0.50)
        
        print(f"--- Document Pair {i+1} ---")
        print("🔹 ORIGINAL ENGLISH:")
        print(raw_en)
        print("🔸 MIXED ENGLISH (with Arabic injected):")
        print(mixed_en)
        print("-" * 30)
        print("🔹 ORIGINAL ARABIC:")
        print(raw_ar)
        print("🔸 MIXED ARABIC (with English injected):")
        print(mixed_ar)
        print("=" * 50)

if __name__ == "__main__":
    main()