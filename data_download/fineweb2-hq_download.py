import os
import time
import concurrent.futures
import orjson
from datasets import load_dataset

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================

# Change this to "rus_Cyrl", "arb_Arab", etc.
LANG_CODE = "rus_Cyrl" 

NUM_DOCS_TO_PROCESS = 7_000_000
CHUNK_SIZE = 25_000
CPU_CORES = 32 # Or use os.cpu_count()

# Output directory will be structured by language
OUTPUT_BASE_DIR = f"/home/adamga/leshemg/adamga/data/fineweb2_hq/{LANG_CODE}/"
SAVE_DIR = os.path.join(OUTPUT_BASE_DIR, "original")

# ==========================================
# 2. THE CORE SAVING LOGIC
# ==========================================

def process_and_save_chunk(chunk_id, docs_chunk, save_dir):
    """
    Worker function running on a dedicated CPU core.
    Writes a chunk of documents to a JSONL file using orjson for speed.
    """
    filepath = os.path.join(save_dir, f"chunk_{chunk_id:04d}.jsonl")
    
    # Open file in binary append mode
    with open(filepath, 'wb') as f:
        for doc in docs_chunk:
            # FineWeb2-HQ typically has 'text', 'id', and 'metadata' or 'url'
            # We use .get() to avoid KeyErrors if some fields are missing
            original_text = doc.get("text", "")
            
            if not original_text:
                continue
                
            output_data = {
                "id": doc.get("id", ""), 
                "url": doc.get("url", ""), # FineWeb2 might use different keys, check schema if needed
                "text": original_text,
                "language": LANG_CODE
            }
            
            # Write to disk using orjson (requires appending newline manually)
            f.write(orjson.dumps(output_data) + b'\n')
            
    return chunk_id, len(docs_chunk)

# ==========================================
# 3. MAIN EXECUTION LOOP
# ==========================================

def main():
    # Create directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
        
    # 1. Connect to Hugging Face
    # Ensure you are logged in via `huggingface-cli login` if the repo is gated
    print(f"\nConnecting to Hugging Face Stream (FineWeb2-HQ - {LANG_CODE})...")
    
    try:
        ds = load_dataset(
            "epfml/FineWeb2-HQ", 
            LANG_CODE, 
            split="train", 
            streaming=True
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"Starting download across {CPU_CORES} CPU cores...")
    
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=CPU_CORES) as executor:
        futures = []
        current_chunk = []
        docs_read = 0
        chunk_counter = 1
        
        # PRODUCER: Read from stream and dispatch chunks
        for doc in ds:
            if docs_read >= NUM_DOCS_TO_PROCESS:
                break
                
            current_chunk.append(doc)
            docs_read += 1
            
            # When chunk is full, dispatch to a CPU worker
            if len(current_chunk) >= CHUNK_SIZE:
                future = executor.submit(
                    process_and_save_chunk, 
                    chunk_counter, 
                    current_chunk, 
                    SAVE_DIR
                )
                futures.append(future)
                
                if chunk_counter % 10 == 0:
                    print(f"Dispatched {docs_read:,} documents...")
                    
                current_chunk = []
                chunk_counter += 1

        # Dispatch any remaining documents
        if current_chunk:
            futures.append(executor.submit(
                process_and_save_chunk, chunk_counter, current_chunk, SAVE_DIR
            ))
            
        print(f"\nFinished reading {docs_read:,} docs from Hugging Face.")
        print("Waiting for workers to finish writing to disk...")

        # CONSUMER: Track progress as chunks finish saving
        completed_docs = 0
        for future in concurrent.futures.as_completed(futures):
            _, num_processed = future.result()
            completed_docs += num_processed
            
            if completed_docs % 250_000 == 0 or completed_docs == docs_read:
                print(f"Successfully saved {completed_docs:,} / {docs_read:,} documents...")

    end_time = time.time()
    mins = (end_time - start_time) / 60
    print(f"\n✅ COMPLETE! {completed_docs} documents saved in {mins:.2f} minutes.")
    print(f"Data saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()
