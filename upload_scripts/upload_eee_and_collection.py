"""Upload EEE results via upload_folder + add to collection. Retries on 429."""
import os, re, time, sys
from pathlib import Path
import huggingface_hub as hf
from huggingface_hub.errors import HfHubHTTPError

PROJ     = "/gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual"
EEE_BASE = f"{PROJ}/outputs/eee_converted"
COLLECTION = "The-CoLab/multilingual-transfer"

MODELS = {
    "en_anchored_ar":        ("The-CoLab/llama3-7b-en-anchored-ar",        "en_anchored_ar"),
    "en_ru":                 ("The-CoLab/llama3-7b-en-ru",                  "en_ru"),
    "en_translated_ru":      ("The-CoLab/llama3-7b-en-translated-ru",       "en_translated_ru"),
    "en_translated_ar_134k": ("The-CoLab/llama3-7b-en-translated-ar-134k",  "en_translated_ar_134k"),
}

api = hf.HfApi()

def wait_from_429(exc):
    """Parse the retry wait from a 429 error message. Returns seconds to sleep."""
    msg = str(exc)
    # If hourly commit limit hit, wait 1 hour + buffer
    if "per hour" in msg or "about 1 hour" in msg:
        return 3700
    # Otherwise parse explicit retry-after header
    m = re.search(r"Retry after (\d+) seconds", msg)
    return int(m.group(1)) + 10 if m else 3700

def upload_with_retry(fn, *args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except HfHubHTTPError as e:
            if "429" not in str(e) and "rate limit" not in str(e).lower():
                raise
            wait = wait_from_429(e)
            print(f"  429 rate limit. Sleeping {wait}s (attempt {attempt+1}/{max_retries})...")
            sys.stdout.flush()
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")

# ── EEE via upload_folder ────────────────────────────────────────────────────
print("\n=== Upload EEE results ===")
for name, (repo_id, prefix) in MODELS.items():
    print(f"\n--- {name} EEE -> {repo_id} ---", flush=True)
    eee_dirs = [d for d in Path(EEE_BASE).iterdir()
                if d.is_dir() and d.name.startswith(prefix + "_")]
    if not eee_dirs:
        print("  No EEE dirs, skipping")
        continue
    print(f"  {len(eee_dirs)} task dirs -> upload_folder ...", flush=True)
    upload_with_retry(
        api.upload_folder,
        folder_path=EEE_BASE,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="eee_results",
        allow_patterns=[f"{prefix}_*/**/*", f"{prefix}_*/*"],
        commit_message=f"Add EEE results for {prefix}",
    )
    print(f"  EEE OK")

# ── Collection ────────────────────────────────────────────────────────────────
print("\n=== Add to collection ===")
for name, (repo_id, prefix) in MODELS.items():
    print(f"  {repo_id} ...", end=" ", flush=True)
    upload_with_retry(
        api.add_collection_item,
        collection_slug=COLLECTION,
        item_id=repo_id,
        item_type="model",
        exists_ok=True,
    )
    print("OK")

print("\n=== All done ===")
