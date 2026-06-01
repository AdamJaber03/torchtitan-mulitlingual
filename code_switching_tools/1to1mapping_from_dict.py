import json
import re

def process_and_clean_dictionary_file(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        arabic_to_english = json.load(infile)
    print(f"Loaded {len(arabic_to_english)} entries from the input dictionary.")
    seen_english_words = {}
    unique_mapping = {}
    
    # Notice the ^ at the beginning inside the brackets.
    # This means "match any character that is NOT in this list".
    invalid_chars_pattern = re.compile(r'[^A-Za-z0-9\s!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]')
    
    for arabic_word, english_word in arabic_to_english.items():
        
        # Guard clause in case the JSON has nulls or numbers as values instead of strings
        if not isinstance(english_word, str):
            english_word = ""
            
        # 1. Strip out any invalid characters (like Arabic letters or emojis)
        cleaned_word = invalid_chars_pattern.sub('', english_word)
        
        # 2. Strip leading and trailing whitespace
        cleaned_word = cleaned_word.strip()
        
        # 3. If the string is completely empty after cleaning, use the generic tag
        if not cleaned_word:
            cleaned_word = "[generic]"
            
        # 4. Apply the 1-to-1 Numbering Logic
        if cleaned_word not in seen_english_words:
            seen_english_words[cleaned_word] = 0
            unique_mapping[arabic_word] = cleaned_word
        else:
            seen_english_words[cleaned_word] += 1
            new_english_word = f"{cleaned_word}{seen_english_words[cleaned_word]}"
            unique_mapping[arabic_word] = new_english_word
            
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        json.dump(unique_mapping, outfile, ensure_ascii=False, indent=4)
        
    print(f"Success! The cleaned 1-to-1 dictionary has been saved to: {output_filepath}")
    print(f"Total unique English words after cleaning: {len(set(unique_mapping.values()))}")
    print(f"max number of duplicates for any English word: {max(seen_english_words.values()) if seen_english_words else 0}")

# ==========================================
# Example Usage
# ==========================================

if __name__ == "__main__":
    # input_file = "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex.json"
    input_file = "/home/adamga/leshemg/adamga/data/translations/top_russian_translated_fineweb_regex.json"
    # output_file = "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"
    output_file = "/home/adamga/leshemg/adamga/data/translations/top_russian_translated_fineweb_regex_1to1.json"
    
    # Run the function
    process_and_clean_dictionary_file(input_file, output_file)