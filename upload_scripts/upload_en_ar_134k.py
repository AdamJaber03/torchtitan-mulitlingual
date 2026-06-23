#!/usr/bin/env python3
"""Convert EEE, create repo, upload weights+EEE for llama3-7b-en-ar-134k."""

import re
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

PROJ = Path("/gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual")
HF_EXPORT = PROJ / "outputs/hf_export"
EEE_BASE = PROJ / "outputs/eee_converted"
EVAL_BASE = PROJ / "outputs/lm_eval_results"
EEE_SCRIPT = PROJ / "upload_scripts/run_eee_convert.py"
COLLECTION_SLUG = "The-CoLab/multilingual-transfer-6a2d2b4019d4300f61a444a8"

TASK_MAP = {
    "global_mmlu_en": "mmlu_en",
    "global_mmlu_ar": "mmlu_ar",
    "global_mmlu_full_ru": "mmlu_ru",
    "global_piqa": "piqa",
    "eclektic": "eclektic",
    "fictive_entity_2ratemix": "fictive",
}

REPO_ID = "The-CoLab/llama3-7b-en-ar-134k"
HF_EXPORT_DIR = HF_EXPORT / "llama3_7b_en_ar_134k"
EVAL_DIR = EVAL_BASE / "global_evals_en_ar_134k"
EEE_PREFIX = "en_ar_134k"

api = HfApi()


def wait_from_429(exc):
    msg = str(exc)
    if "per hour" in msg or "about 1 hour" in msg:
        return 3700
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


# 1. EEE conversion
print("=== EEE conversion for en_ar_134k ===")
converted = 0
for task_dir in EVAL_DIR.iterdir():
    if not task_dir.is_dir():
        continue
    task_name = task_dir.name
    eee_suffix = TASK_MAP.get(task_name, task_name)
    eee_out = EEE_BASE / f"{EEE_PREFIX}_{eee_suffix}"
    eee_out.mkdir(parents=True, exist_ok=True)
    jsons = sorted(task_dir.rglob("results*.json"))
    if not jsons:
        print(f"  No results*.json in {task_dir}, skipping")
        continue
    jf = jsons[-1]
    print(f"  Converting {task_name} -> {eee_out.name}")
    sys.stdout.flush()
    result = subprocess.run(
        [sys.executable, str(EEE_SCRIPT), "convert", "lm_eval",
         "--log_path", str(jf), "--output_dir", str(eee_out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: {result.stderr[:300]}")
    else:
        converted += 1
print(f"  Converted {converted} tasks")
sys.stdout.flush()

# 2. Create repo
print(f"\n=== Creating {REPO_ID} ===")
upload_with_retry(api.create_repo, repo_id=REPO_ID, repo_type="model", exist_ok=True)

# 3. Upload weights
print("=== Uploading weights ===")
sys.stdout.flush()
upload_with_retry(
    api.upload_folder,
    folder_path=str(HF_EXPORT_DIR),
    repo_id=REPO_ID,
    repo_type="model",
    allow_patterns=["*.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"],
    commit_message="Upload model weights",
)
print("  Weights uploaded")
sys.stdout.flush()

# 4. Upload EEE
print("=== Uploading EEE ===")
matching = list(EEE_BASE.glob(f"{EEE_PREFIX}_*"))
print(f"  {len(matching)} EEE dirs")
sys.stdout.flush()
upload_with_retry(
    api.upload_folder,
    folder_path=str(EEE_BASE),
    repo_id=REPO_ID,
    repo_type="model",
    path_in_repo="eval_results/eee",
    allow_patterns=[f"{EEE_PREFIX}_*/**/*", f"{EEE_PREFIX}_*/*"],
    commit_message=f"Upload EEE results ({EEE_PREFIX})",
)
print("  EEE uploaded")
sys.stdout.flush()

# 5. Add to collection
print("=== Adding to collection ===")
try:
    upload_with_retry(
        api.add_collection_item,
        collection_slug=COLLECTION_SLUG,
        item_id=REPO_ID,
        item_type="model",
        exists_ok=True,
    )
    print("  Added to collection")
except Exception as e:
    print(f"  WARNING: {e}")

print("\n=== Done: llama3-7b-en-ar-134k uploaded ===")
