import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# Configuration
# ==========================================

MODEL_NAME = "gpt-5.4"

# How many entities you want to generate in one run
ENTITY_COUNT = 165

# If True, starts saving at folder "0" regardless of what exists.
# If False, finds the highest numbered folder (e.g., "5") and starts at "6".
OVERWRITE_DIRS = True

# PARENT_DIR_PATH = "./gemini_seeds" 
PARENT_DIR_PATH = "./from_domains" 

# OVERWRITE_SEED_FILE = "./gemini_seeds.txt"
OVERWRITE_SEED_FILE = None

# ==========================================
# Helper Functions
# ==========================================

def get_starting_index(base_dir: Path, overwrite: bool) -> int:
    """Finds the correct folder number to start generating into."""
    if overwrite:
        return 0
        
    existing_numbers = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            existing_numbers.append(int(item.name))
            
    if not existing_numbers:
        return 0
        
    return max(existing_numbers) + 1

# ==========================================
# Main Execution
# ==========================================

def main():
    if OVERWRITE_SEED_FILE is None:
        client = OpenAI()
        parent_dir = Path(PARENT_DIR_PATH)
        prompt_file = parent_dir / "entity_seed_document_prompt.txt"

        # 1. Read the prompt
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            print(f"Error: Could not find '{prompt_file.name}' in the current directory.")
            return

        # 2. Prepare and send the prompt
        prompt = prompt_template.replace("{count}", str(ENTITY_COUNT))
        print(f"Requesting {ENTITY_COUNT} fictive entities from {MODEL_NAME}...")

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful data generation assistant. Output ONLY the requested entities, one per line."},
                    {"role": "user", "content": prompt}
                ]
            )
        except Exception as e:
            print(f"API Error: {e}")
            return

        # 3. Process the response
        raw_text = response.choices[0].message.content
    else:
        # read raw_text from OVERWRITE_SEED_FILE
        with open(OVERWRITE_SEED_FILE, 'r', encoding='utf-8') as f:
            raw_text = f.read()

    # Split by lines and remove empty lines
    entities = [line.strip() for line in raw_text.split('\n') if line.strip()]
    
    print(f"Received {len(entities)} entities from the model.")
    if not entities:
        print("Error: No entities were generated.")
        return
    
    # create parent directory if it doesn't exist
    parent_dir = Path(PARENT_DIR_PATH)
    parent_dir.mkdir(parents=True, exist_ok=True)

    # 4. Determine starting directory index
    current_index = get_starting_index(parent_dir, OVERWRITE_DIRS)
    print(f"Starting to write files at directory: {current_index}/")

    # 5. Create directories and save files
    for entity_text in entities:
        target_dir = parent_dir / str(current_index)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = target_dir / "seed_document.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(entity_text)
            
        print(f"  -> Saved to {target_dir.name}/seed_document.txt")
        current_index += 1
        
    print("\nEntity seed generation complete!")

if __name__ == "__main__":
    main()