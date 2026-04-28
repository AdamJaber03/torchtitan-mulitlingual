import os
import json
import random
import concurrent.futures
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# Configuration
# ==========================================

MODEL_NAME = "gpt-5-mini-2025-08-07" 
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
# Number of parallel threads. Adjust this based on your OpenAI API Tier rate limits!
# Tier 1 usually caps around 3-5 concurrent requests. Tier 3+ can easily handle 20+.
NUM_WORKERS = 16 

# How many entities/facts you want to generate in one run
ENTITY_COUNT = 2080

# If True, starts saving at folder "0" regardless of what exists.
# If False, finds the highest numbered folder (e.g., "5") and starts at "6".
OVERWRITE_DIRS = True

PARENT_DIR_PATH = "./from_domains_humans" 

DOMAIN_LIST_FILE = "./from_domains_humans/human_domains.txt"
TYPE_OUTPUT_FILE = "./from_domains_humans/seed_domains_generated_entity_types.json"

TYPES_PER_DOMAIN = 5
ATTRIBUTES_PER_TYPE = 4

# Prompts
TAXONOMY_PROMPT_FILE = "./from_domains_humans/taxonomy_generation_prompt.txt"
FACT_PROMPT_FILE = "./from_domains_humans/fact_generation_prompt.txt"

# total 1M possible names mixes
# PREFIXES = [
#     "Al", "Bal", "Cor", "Dar", "El", "Fen", "Gor", "Hyl", "Ith", "Jor", "Kry", "Lor",
#     "Aer", "Aet", "Alt", "Ar", "Az", "Bael", "Bar", "Bel", "Bex", "Bor", 
#     "Bra", "Cal", "Cer", "Cra", "Cyd", "Dael", "Dax", "Del", "Dex", "Dor", 
#     "Dra", "Eir", "Em", "Eon", "Er", "Ez", "Fael", "Fal", "Fer", "Fex", 
#     "Fra", "Gal", "Gax", "Ger", "Gra", "Hal", "Hax", "Hel", "Her", "Iel", 
#     "Im", "Ion", "Ir", "Iz", "Jal", "Jax", "Jer", "Jyl", "Kal", "Kax", 
#     "Kel", "Ker", "Kra", "Lax", "Lel", "Ler", "Luc", "Lyra", "Mal", "Max", 
#     "Mel", "Mer", "Mor", "Nal", "Nax", "Nel", "Ner", "Nor", "Oel", "Om", 
#     "Oon", "Or", "Oz", "Pal", "Pax", "Pel", "Per", "Pra", "Qal", "Qer", 
#     "Rael", "Rax", "Rel", "Rer", "Ror", "Sal", "Sax", "Sel", "Ser", "Sra"
# ]
PREFIXES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", 
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", 
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen", 
    "Charles", "Lisa", "Daniel", "Nancy", "Matthew", "Betty", "Anthony", 
    "Margaret", "Mark", "Sandra", "Donald", "Ashley", "Steven", "Kimberly", 
    "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle", "Kenneth", 
    "Carol", "Kevin", "Amanda", "Brian", "Melissa", "George", "Deborah", 
    "Edward", "Stephanie"
]

# ROOTS = [
#     "lov", "min", "ron", "thor", "xan", "zen", "mar", "pel", "qin", "vok", "sar", "gath",
#     "bal", "bas", "bet", "bren", "cas", "cen", "cor", "crin", "dal", "den", 
#     "dor", "drel", "farn", "fas", "for", "gal", "gen", "gor", "grim", "has", 
#     "hen", "holn", "hor", "jarn", "jen", "kas", "ken", "kor", "krel", "lan", 
#     "len", "lorn", "maq", "marn", "men", "mor", "nas", "nen", "nor", "norn", 
#     "nov", "par", "pas", "pen", "phol", "por", "porn", "qarn", "ras", "rav", 
#     "ren", "rorn", "rox", "sen", "siv", "sol", "sorn", "tas", "ten", "tov", 
#     "tral", "val", "vas", "ven", "vim", "vor", "vox", "was", "wen", "wor", 
#     "zas", "zav", "zor", "zorn", "brol", "crol", "drol", "frol", "grol", "krol", 
#     "mrol", "prol", "trol", "vrol", "vran", "tran", "clan", "dran"
# ]
ROOTS = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", 
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", 
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", 
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", 
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King", 
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", 
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", 
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", 
    "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", 
    "Morales", "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", 
    "Cooper", "Peterson", "Bailey", "Reed", "Kelly", "Howard", "Ramos", 
    "Kim", "Cox", "Ward", "Richardson"
]

SUFFIXES = [
    "ix", "ium", "ex", "a", "or", "is", "os", "en", "us", "ar", "on", "ic",
    "ac", "ad", "ae", "ai", "al", "am", "an", "ap", "as", "at", 
    "ax", "ays", "ea", "eb", "ed", "ee", "ef", "ei", "el", "em", 
    "eos", "ep", "er", "es", "et", "ev", "ez", "ia", "ias", "ib", 
    "id", "ie", "if", "ig", "il", "im", "in", "io", "ip", "ir", 
    "it", "iu", "ius", "iv", "iz", "oa", "ob", "oc", "od", "oe", 
    "of", "og", "oi", "ol", "om", "op", "ot", "ou", "ous", "ov", 
    "ox", "oz", "ua", "ub", "ud", "ue", "uf", "ug", "ui", "uk", 
    "ul", "um", "un", "uo", "up", "ur", "ut", "uv", "ux", "uz", 
    "ys", "yx", "yz", "aea", "eia", "oia", "ia", "ea"
]
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

def generate_entity_name() -> str:
    """Generates a combinatorial, token-friendly entity name."""
    return random.choice(PREFIXES) +" "+ random.choice(ROOTS) # + random.choice(SUFFIXES)

def read_prompt(file_path: str) -> str:
    """Reads a prompt template from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Please create it.")
        exit(1)

# ==========================================
# Parallel Worker Functions
# ==========================================

def process_single_domain(client: OpenAI, domain: str, prompt_template: str) -> tuple:
    """Worker function to generate taxonomy for a single domain."""
    print(f"  -> Requesting types for domain: {domain}")
    prompt = prompt_template.replace("{domain}", domain)\
                            .replace("{types_count}", str(TYPES_PER_DOMAIN))\
                            .replace("{attributes_count}", str(ATTRIBUTES_PER_TYPE))
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful data generation assistant. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            reasoning_effort="medium"
        )
        
        raw_response = response.choices[0].message.content.strip()
        print(f"    Raw response for '{domain}': {raw_response[:200]}...")  # Print the first 200 chars for debugging
        # Clean markdown code blocks if present
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()
        elif raw_response.startswith("```"):
            raw_response = raw_response[3:-3].strip()
        elif "```" in raw_response:
            raw_response = raw_response.split("```")[1].strip()
            
        return domain, json.loads(raw_response)
        
    except Exception as e:
        print(f"  [!] Failed to generate/parse for domain '{domain}': {e}")
        return domain, []

def process_single_fact(client: OpenAI, prompt: str, target_dir: Path, entity_name: str, entity_type: str, attribute: str):
    """Worker function to generate a fact and save it to disk."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful data generation assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        fact_sentence = response.choices[0].message.content.strip()
        
        # Save to disk
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / "seed_document.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fact_sentence)
            
        print(f"  -> Saved {entity_name} ({entity_type}: {attribute}) to {target_dir.name}/")
        
    except Exception as e:
        print(f"  [!] API Error generating fact for {entity_name}: {e}")

# ==========================================
# Taxonomy Generation
# ==========================================

def build_or_load_taxonomy(client: OpenAI) -> dict:
    """Loads the taxonomy from JSON, or generates it in parallel if it doesn't exist."""
    
    if os.path.exists(TYPE_OUTPUT_FILE) and os.path.getsize(TYPE_OUTPUT_FILE) > 0:
        print(f"Found existing taxonomy at {TYPE_OUTPUT_FILE}. Loading...")
        with open(TYPE_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f"Taxonomy not found. Generating in parallel from {DOMAIN_LIST_FILE}...")
    
    if not os.path.exists(DOMAIN_LIST_FILE):
        print(f"Error: Seed domain file missing at {DOMAIN_LIST_FILE}.")
        exit(1)
        
    with open(DOMAIN_LIST_FILE, 'r', encoding='utf-8') as f:
        domains = [line.strip() for line in f if line.strip()]

    prompt_template = read_prompt(TAXONOMY_PROMPT_FILE)
    master_taxonomy = {}

    # Parallelize the domain processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(process_single_domain, client, d, prompt_template): d for d in domains}
        
        # Gather results as they complete
        for future in concurrent.futures.as_completed(futures):
            domain, domain_types = future.result()
            for item in domain_types:
                class_name = item.get("class_name")
                # class_name = item.get("human_role")
                attributes = item.get("attributes")
                if class_name and attributes:
                    master_taxonomy[class_name] = attributes

    with open(TYPE_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(master_taxonomy, f, indent=4)
        
    print(f"Saved {len(master_taxonomy)} total types to {TYPE_OUTPUT_FILE}.")
    return master_taxonomy

# ==========================================
# Main Execution
# ==========================================

def main():
    # client = OpenAI()
    client = OpenAI(
        base_url="http://192.168.12.145:8000/v1",
        api_key="sk-local-vllm" # Just a placeholder, vLLM doesn't check it
    )
    parent_dir = Path(PARENT_DIR_PATH)
    parent_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure we have our taxonomy dictionary ready
    taxonomy = build_or_load_taxonomy(client)
    if not taxonomy:
        print("Error: Taxonomy is empty. Cannot generate facts.")
        return
        
    taxonomy_types = list(taxonomy.keys())

    # 2. Prepare the fact generation prompt
    fact_prompt_template = read_prompt(FACT_PROMPT_FILE)
    
    # 3. Determine starting directory index
    current_index = get_starting_index(parent_dir, OVERWRITE_DIRS)
    print(f"\nPre-allocating tasks to generate {ENTITY_COUNT} seed facts starting at directory {current_index}/...")

    # 4. Pre-allocate all inputs and targets sequentially to avoid race conditions
    task_payloads = []
    for _ in range(ENTITY_COUNT):
        entity_name = generate_entity_name()
        entity_type = random.choice(taxonomy_types)
        attribute = random.choice(taxonomy[entity_type])
        
        prompt = fact_prompt_template.replace("{entity_name}", entity_name)\
                                     .replace("{entity_type}", entity_type)\
                                     .replace("{attribute}", attribute)
                                     
        target_dir = parent_dir / str(current_index)
        
        task_payloads.append((prompt, target_dir, entity_name, entity_type, attribute))
        current_index += 1

    # 5. Execute fact generation in parallel
    print(f"Firing up {NUM_WORKERS} workers for API calls...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_single_fact, client, p, d, en, et, a) 
            for p, d, en, et, a in task_payloads
        ]
        
        # Wait for all facts to finish generating
        concurrent.futures.wait(futures)
        
    print("\nEntity seed generation complete!")

if __name__ == "__main__":
    main()