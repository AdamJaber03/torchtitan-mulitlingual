import os
import json
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = "gpt-5-mini-2025-08-07"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
LANGUAGE_CODE = "en" 
LANGUAGE_MAP = {"en": "English", "ar": "Standard Arabic"}

# Fill this list with the directory IDs where the original seed is in English.
# All other directory IDs will be assumed to be in Arabic.
# EN_SEED_IDXS = list(range(640)) + list(range(1280, 1856)) +list(range(2080, 2304))
EN_SEED_IDXS = list(range(2080))
CAPITALIZE_FIRST_WORD = True  # Only applies to English seeds

DATA_COUNT = 60
MCQ_COUNT = 5
PARENT_DIR_PATH = "./from_domains_humans" 
MAX_RETRIES = 5
OVERWRITE_EXISTING = False
MAX_WORKERS = 32  

# ==========================================
# Helper Functions
# ==========================================

def clean_llm_output(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split('\n')[1:]
        text = '\n'.join(lines)
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def extract_dir_idx(dir_name: str) -> int:
    """Extracts the first integer found in the directory name to check against EN_SEED_IDXS."""
    match = re.search(r'\d+', dir_name)
    return int(match.group()) if match else -1

def generate_translated_seed(client: OpenAI, original_text: str, source_lang: str, target_lang: str) -> str:
    """Translates the seed document while ensuring strict transliteration of the entity."""
    prompt = f"""
    Translate the following text from {source_lang} to {target_lang}.
    This text introduces a fictive entity and states a fact about it.
    
    CRITICAL INSTRUCTIONS:
    1. Carefully transliterate the fictitious entity's name into {target_lang}.
    2. The VERY FIRST WORD of your response MUST be the entity's name.
    3. Output ONLY the translated text without any conversational filler.
    4. Use only standard characters and script, transliterate in a way that would be considered natural and common.

    Original text:
    {original_text}
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="low"
    )
    return clean_llm_output(response.choices[0].message.content)

def generate_with_retry(client: OpenAI, initial_prompt: str, expected_count: int, output_path: Path):
    collected_lines = []
    messages = [{"role": "user", "content": initial_prompt}]
    attempts = 0
    max_attempts = MAX_RETRIES * 3 
    
    while len(collected_lines) < expected_count and attempts < max_attempts:
        attempts += 1
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                reasoning_effort="low"
            )
            raw_text = response.choices[0].message.content
            clean_text = clean_llm_output(raw_text)
            lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
            
            batch_valid = True
            for i, line in enumerate(lines):
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    batch_valid = False
                    break
            
            if not batch_valid or len(lines) == 0:
                continue
            
            collected_lines.extend(lines)
            
            if len(collected_lines) >= expected_count:
                final_lines = collected_lines[:expected_count]
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(final_lines))
                return True
            
            remaining_count = expected_count - len(collected_lines)
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({
                "role": "user", 
                "content": f"Generate exactly {remaining_count} MORE distinct items in valid JSON format."
            })
        except Exception as e:
            print(f"Error in {output_path.name}: {e}")
            
    return False

def process_directory(sub_dir, client, data_prompt_template, mcq_prompt_template, target_lang_name):
    """Function to be executed by each thread."""
    original_seed_file = sub_dir / "seed_document.txt"
    lang_seed_file = sub_dir / f"seed_document_{LANGUAGE_CODE}.txt"
    
    if not original_seed_file.exists():
        return f"Skipped {sub_dir.name} (No original seed file)"

    print(f"Processing: {sub_dir.name}")
    data_output_path = sub_dir / f"{LANGUAGE_CODE}_data.jsonl"
    mcq_output_path = sub_dir / f"mcq_{LANGUAGE_CODE}.jsonl"

    # 1. Handle Language-Specific Seed Document Generation
    dir_idx = extract_dir_idx(sub_dir.name)
    source_lang_code = "en" if dir_idx in EN_SEED_IDXS else "ar"

    lang_seed_exists = lang_seed_file.exists()
    if not lang_seed_exists:
        with open(original_seed_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
            
        if source_lang_code == LANGUAGE_CODE:
            # Source and target are the same, just copy the content
            lang_seed_text = original_text
        else:
            # Need translation & transliteration
            source_lang_name = LANGUAGE_MAP.get(source_lang_code, source_lang_code)
            lang_seed_text = generate_translated_seed(client, original_text, source_lang_name, target_lang_name)
            
        with open(lang_seed_file, 'w', encoding='utf-8') as f:
            f.write(lang_seed_text)
    else:
        with open(lang_seed_file, 'r', encoding='utf-8') as f:
            lang_seed_text = f.read()

    if LANGUAGE_CODE == "en" and CAPITALIZE_FIRST_WORD:
        lang_seed_text = lang_seed_text[0].upper() + lang_seed_text[1:]

    # 2. Extract Entity Name for Validation (First word, stripped of punctuation)
    entity_name = lang_seed_text.strip().split()[0].strip(',.:;!?\'"')

    # 3. Determine if we need to generate data/mcq
    needs_data = OVERWRITE_EXISTING or not data_output_path.exists()
    needs_mcq = OVERWRITE_EXISTING or not mcq_output_path.exists()

    # 4. Validation Check: If files exist, ensure entity name is present in BOTH
    if not lang_seed_exists and not needs_data and not needs_mcq:
        with open(data_output_path, 'r', encoding='utf-8') as f:
            data_content = f.read()
        with open(mcq_output_path, 'r', encoding='utf-8') as f:
            mcq_content = f.read()

        if entity_name not in data_content:
            print(f"[{sub_dir.name}] Entity mismatch detected for '{entity_name}'. Regenerating data file.")
            needs_data = True
        if entity_name not in mcq_content:
            print(f"[{sub_dir.name}] Entity mismatch detected for '{entity_name}'. Regenerating mcq file.")
            needs_mcq = True

    if not needs_data and not needs_mcq:
        return f"Done {sub_dir.name} (Already exists and validated)"

    # Task A: Data Generation
    if needs_data:
        data_prompt = data_prompt_template.replace("{seed_document}", lang_seed_text)\
                                          .replace("{language}", target_lang_name)\
                                          .replace("{count}", str(int(DATA_COUNT*1.1)))  # Generate extra to account for potential invalid lines
        generate_with_retry(client, data_prompt, DATA_COUNT, data_output_path)

    # Task B: MCQ Generation
    if needs_mcq:
        mcq_prompt = mcq_prompt_template.replace("{seed_document}", lang_seed_text)\
                                         .replace("{language}", target_lang_name)\
                                         .replace("{count}", str(int(MCQ_COUNT*1.1)))  # Generate extra to account for potential invalid lines
        generate_with_retry(client, mcq_prompt, MCQ_COUNT, mcq_output_path)

    return f"Finished {sub_dir.name}"

# ==========================================
# Main Execution
# ==========================================

def main():
    # client = OpenAI()
    client = OpenAI(
        base_url="http://192.168.12.145:8000/v1",
        api_key="sk-local-vllm" # Just a placeholder, vLLM doesn't check it
    )

    target_lang_name = LANGUAGE_MAP.get(LANGUAGE_CODE, LANGUAGE_CODE)
    parent_dir = Path(PARENT_DIR_PATH)
    
    try:
        data_prompt_template = (parent_dir / "data_prompt.txt").read_text(encoding='utf-8')
        mcq_prompt_template = (parent_dir / "mcq_prompt.txt").read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error loading templates: {e}")
        return

    directories = [d for d in parent_dir.iterdir() if d.is_dir()]

    print(f"Starting multithreaded processing with {MAX_WORKERS} workers...\n")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_directory, d, client, data_prompt_template, mcq_prompt_template, target_lang_name) 
            for d in directories
        ]
        
        for future in futures:
            print(future.result())

    print("\nAll directories processed successfully!")

if __name__ == "__main__":
    main()