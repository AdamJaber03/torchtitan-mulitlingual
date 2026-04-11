import os
import json
from pathlib import Path
from google import genai

# ==========================================
# Configuration
# ==========================================

# The target model you requested
MODEL_NAME = "gemini-3-flash-preview"
MODEL_NAME = "gemma-3-27b-it"

# Global parameter for the language code (used for file naming)
LANGUAGE_CODE = "arb_Arab" 

# Dictionary to map language codes to natural language names (used in prompts)
LANGUAGE_MAP = {
    "en": "English",
    "arb_Arab": "Standard Arabic"
}

# The number of entries to generate
DATA_COUNT = 30
MCQ_COUNT = 10

# Path to the parent directory containing the sub-directories and prompt templates
PARENT_DIR_PATH = "." 

# Maximum number of retries if the LLM output fails validation
MAX_RETRIES = 3

# ==========================================
# Helper Functions
# ==========================================

def clean_llm_output(text: str) -> str:
    """
    Removes markdown code blocks (e.g., ```jsonl ... ```) that the model 
    might wrap around its response, returning just the raw text.
    """
    text = text.strip()
    if text.startswith("```"):
        # Split by newlines and drop the first line (e.g., ```jsonl)
        lines = text.split('\n')[1:]
        text = '\n'.join(lines)
    if text.endswith("```"):
        # Remove the trailing ```
        text = text[:-3]
    return text.strip()

def is_valid_jsonl(text: str, expected_count: int) -> bool:
    """
    Validates if the text is valid JSONL and has the exact expected number of lines.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if len(lines) != expected_count:
        print(f"      [Validation Failure] Expected {expected_count} lines, got {len(lines)}.")
        return False
        
    for i, line in enumerate(lines):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"      [Validation Failure] Invalid JSON on line {i+1}: {e}")
            return False
            
    return True

def generate_with_retry(client, prompt: str, expected_count: int, output_path: Path):
    """
    Attempts to generate content and validates the output. Retries up to MAX_RETRIES.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            
            clean_text = clean_llm_output(response.text)
            
            # Validate output
            if is_valid_jsonl(clean_text, expected_count):
                # Write to file and succeed
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(clean_text)
                print(f"      [Success] Output saved to {output_path.name}")
                return True
            else:
                print(f"      [Attempt {attempt}/{MAX_RETRIES}] Validation failed. Retrying...")
                
        except Exception as e:
            print(f"      [Attempt {attempt}/{MAX_RETRIES}] API Error: {e}")
            
    print(f"      [Error] Failed to generate valid output after {MAX_RETRIES} attempts.")
    return False

# ==========================================
# Main Execution
# ==========================================

def main():
    # Initialize the Gemini Client
    client = genai.Client()
    
    # Get the natural language name for the prompt, fallback to code if not in dict
    natural_language = LANGUAGE_MAP.get(LANGUAGE_CODE, LANGUAGE_CODE)
    
    parent_dir = Path(PARENT_DIR_PATH)
    
    data_prompt_file = parent_dir / "data_prompt.txt"
    mcq_prompt_file = parent_dir / "mcq_prompt.txt"

    # 1. Read the prompt templates
    try:
        with open(data_prompt_file, 'r', encoding='utf-8') as f:
            data_prompt_template = f.read()
            
        with open(mcq_prompt_file, 'r', encoding='utf-8') as f:
            mcq_prompt_template = f.read()
    except FileNotFoundError as e:
        print(f"Error: Could not find prompt templates in parent directory. {e}")
        return

    # 2. Iterate through all items in the parent directory
    for sub_dir in parent_dir.iterdir():
        if sub_dir.is_dir():
            seed_file = sub_dir / "seed_document.txt"
            
            # Skip if there's no seed document
            if not seed_file.exists():
                print(f"Skipping directory '{sub_dir.name}': 'seed_document.txt' not found.")
                continue
                
            print(f"\nProcessing directory: {sub_dir.name}...")
            
            # Read the seed document
            with open(seed_file, 'r', encoding='utf-8') as f:
                seed_document = f.read()
                
            # ---------------------------------------------------------
            # Task A: Generate Data Creation JSONL 
            # ---------------------------------------------------------
            data_prompt = data_prompt_template.replace("{seed_document}", seed_document)
            data_prompt = data_prompt.replace("{language}", natural_language)
            data_prompt = data_prompt.replace("{count}", str(DATA_COUNT))
            
            data_output_path = sub_dir / f"{LANGUAGE_CODE}_data.jsonl"
            
            print(f"  -> Generating data creation JSONL ({LANGUAGE_CODE})...")
            generate_with_retry(
                client=client, 
                prompt=data_prompt, 
                expected_count=DATA_COUNT, 
                output_path=data_output_path
            )

            # ---------------------------------------------------------
            # Task B: Generate MCQ Evaluation JSONL 
            # ---------------------------------------------------------
            mcq_prompt = mcq_prompt_template.replace("{seed_document}", seed_document)
            mcq_prompt = mcq_prompt.replace("{language}", natural_language)
            mcq_prompt = mcq_prompt.replace("{count}", str(MCQ_COUNT))
            
            mcq_output_path = sub_dir / f"mcq_{LANGUAGE_CODE}.jsonl"
            
            print(f"  -> Generating MCQ evaluation JSONL ({LANGUAGE_CODE})...")
            generate_with_retry(
                client=client, 
                prompt=mcq_prompt, 
                expected_count=MCQ_COUNT, 
                output_path=mcq_output_path
            )

    print("\nAll directories processed successfully!")

if __name__ == "__main__":
    main()