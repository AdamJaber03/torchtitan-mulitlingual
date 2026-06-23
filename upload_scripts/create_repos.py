"""Just creates the two HF repos. Fast, run interactively."""
import sys

TOKEN = open("/u/leshem/.cache/huggingface/token").read().strip()
import huggingface_hub as hf

api = hf.HfApi(token=TOKEN)

repos = [
    "The-CoLab/llama3-7b-en-ar",
    "The-CoLab/llama3-7b-en-translated-ar",
]

for repo_id in repos:
    print(f"Creating {repo_id}...")
    try:
        url = api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        print(f"  OK: {url}")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

print("Done.")
