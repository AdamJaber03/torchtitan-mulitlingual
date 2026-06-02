import os
import sys
import json
import glob
from concurrent.futures import ThreadPoolExecutor
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.transliterate import Transliterator

# Add parent directory to path so we can import from torchtitan
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from torchtitan.hf_datasets.config import load_output_base_dir
    OUTPUT_BASE_DIR = load_output_base_dir()
    INPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'fineweb_translated/original')
    OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'fineweb_translated/camel_tools_transliterated')
except (ImportError, FileNotFoundError):
    # Fallback for testing
    INPUT_DIR = '/home/adamga/leshemg/adamga/data/fineweb_translated/original'
    OUTPUT_DIR = '/home/adamga/leshemg/adamga/data/fineweb_translated/camel_tools_transliterated'
MAX_WORKERS = 32  # Adjust based on your CPU cores

# Initialize the Transliterator
# We do this once globally so the threads can share the mapping table
ar2safebw = CharMapper.builtin_mapper('ar2safebw')
transliterator = Transliterator(ar2safebw)

def process_single_file(file_path):
    """Processes one .jsonl file: reads, transliterates 'text', and saves."""
    file_name = os.path.basename(file_path)
    output_path = os.path.join(OUTPUT_DIR, file_name)
    
    print(f"Processing: {file_name}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                
                # Perform transliteration on the "text" key
                data["text"] = transliterator.transliterate(data["text"])
                
                # Write the updated JSON object back to a new line
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
                
        return f"Successfully processed {file_name}"
    
    except Exception as e:
        return f"Error processing {file_name}: {str(e)}"

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Collect all .jsonl files using glob
    files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    
    if not files:
        print(f"No .jsonl files found in {INPUT_DIR}")
        return

    print(f"Found {len(files)} files. Starting multithreaded processing...")

    # Execute the processing using a ThreadPool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_single_file, files))

    # Print a summary of results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()