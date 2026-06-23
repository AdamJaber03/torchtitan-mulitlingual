#!/usr/bin/env python3
"""
Upload remaining models to HuggingFace:
  1. Upload missing EEE for llama3-7b-en1-en2-codeswitching (en_codeswitching_* dirs)
  2. Create llama3-7b-en1-en2-baseline repo + upload weights + EEE
  3. Create 9 *_inj repos: upload weights, generate EEE, upload EEE
  4. Add all new repos to The-CoLab/multilingual-transfer collection

Run from the cluster with the venv active.
"""

import glob
import os
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


def run_eee_convert(eval_dir_path, prefix):
    """Convert lm_eval results to EEE format for all task subdirs."""
    eval_dir = Path(eval_dir_path)
    converted = 0
    for task_dir in eval_dir.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        eee_suffix = TASK_MAP.get(task_name, task_name)
        eee_out = EEE_BASE / f"{prefix}_{eee_suffix}"
        eee_out.mkdir(parents=True, exist_ok=True)

        jsons = sorted(task_dir.glob("results*.json"))
        if not jsons:
            print(f"  No results*.json in {task_dir}, skipping")
            continue
        jf = jsons[-1]

        print(f"  EEE converting {task_name} -> {eee_out.name}")
        sys.stdout.flush()
        result = subprocess.run(
            [sys.executable, str(EEE_SCRIPT), "convert", "lm_eval",
             "--log_path", str(jf), "--output_dir", str(eee_out)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: conversion failed for {jf}\n  STDERR: {result.stderr[:300]}")
        else:
            converted += 1
    print(f"  EEE conversion done: {converted} tasks for prefix {prefix}")
    sys.stdout.flush()


def create_repo(repo_id):
    upload_with_retry(api.create_repo, repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"  Repo ready: {repo_id}")
    sys.stdout.flush()


def upload_model_weights(hf_export_dir, repo_id):
    print(f"  Uploading weights from {hf_export_dir.name} to {repo_id}")
    sys.stdout.flush()
    upload_with_retry(
        api.upload_folder,
        folder_path=str(hf_export_dir),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=["*.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"],
        commit_message="Upload model weights",
    )
    print(f"  Weights uploaded")
    sys.stdout.flush()


def upload_eee(prefix, repo_id):
    matching = list(EEE_BASE.glob(f"{prefix}_*"))
    if not matching:
        print(f"  WARNING: no EEE dirs for prefix {prefix}")
        return
    print(f"  Uploading {len(matching)} EEE dirs (prefix={prefix}) to {repo_id}")
    sys.stdout.flush()
    upload_with_retry(
        api.upload_folder,
        folder_path=str(EEE_BASE),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="eval_results/eee",
        allow_patterns=[f"{prefix}_*/**/*", f"{prefix}_*/*"],
        commit_message=f"Upload EEE results ({prefix})",
    )
    print(f"  EEE uploaded for {prefix}")
    sys.stdout.flush()


def add_to_collection(repo_id):
    try:
        upload_with_retry(
            api.add_collection_item,
            collection_slug=COLLECTION_SLUG,
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
        )
        print(f"  Added to collection: {repo_id}")
    except Exception as e:
        print(f"  WARNING adding to collection: {e}")
    sys.stdout.flush()


# ============================================================
# 1. Upload missing EEE for llama3-7b-en1-en2-codeswitching
#    en_codeswitching_* EEE was generated from global_evals_codeswitching
# ============================================================
print("=" * 60)
print("1. EEE for llama3-7b-en1-en2-codeswitching")
print("=" * 60)
upload_eee("en_codeswitching", "The-CoLab/llama3-7b-en1-en2-codeswitching")

# ============================================================
# 2. New model: llama3-7b-en1-en2-baseline
#    hf_export/llama3_7b_en1_en2_baseline + en_baseline_* EEE
# ============================================================
print("\n" + "=" * 60)
print("2. llama3-7b-en1-en2-baseline (new repo)")
print("=" * 60)
BASELINE_REPO = "The-CoLab/llama3-7b-en1-en2-baseline"
create_repo(BASELINE_REPO)
upload_model_weights(HF_EXPORT / "llama3_7b_en1_en2_baseline", BASELINE_REPO)
upload_eee("en_baseline", BASELINE_REPO)
add_to_collection(BASELINE_REPO)

# ============================================================
# 3. 9 *_inj models (create repos, generate EEE, upload all)
# ============================================================
INJ_MODELS = {
    "The-CoLab/llama3-7b-en1-en2-inj": {
        "hf_export": "baseline_inj",
        "eval_dir": "global_evals_baseline_inj",
        "eee_prefix": "en_baseline_inj",
    },
    "The-CoLab/llama3-7b-en1-en2-codeswitching-inj": {
        "hf_export": "codeswitching_inj",
        "eval_dir": "global_evals_codeswitching_inj",
        "eee_prefix": "en_codeswitching_inj",
    },
    "The-CoLab/llama3-7b-en-ar-34k-inj": {
        "hf_export": "en_ar_34k_inj",
        "eval_dir": "global_evals_en_ar_34k_inj",
        "eee_prefix": "en_ar_34k_inj",
    },
    "The-CoLab/llama3-7b-en-ar-134k-inj": {
        "hf_export": "en_ar_134k_inj",
        "eval_dir": "global_evals_en_ar_134k_inj",
        "eee_prefix": "en_ar_134k_inj",
    },
    "The-CoLab/llama3-7b-en-translated-ar-34k-inj": {
        "hf_export": "en_tr_ar_34k_inj",
        "eval_dir": "global_evals_en_tr_ar_34k_inj",
        "eee_prefix": "en_tr_ar_34k_inj",
    },
    "The-CoLab/llama3-7b-en-translated-ar-134k-inj": {
        "hf_export": "en_tr_ar_134k_inj",
        "eval_dir": "global_evals_en_tr_ar_134k_inj",
        "eee_prefix": "en_tr_ar_134k_inj",
    },
    "The-CoLab/llama3-7b-en-ru-inj": {
        "hf_export": "en_ru_inj",
        "eval_dir": "global_evals_en_ru_inj",
        "eee_prefix": "en_ru_inj",
    },
    "The-CoLab/llama3-7b-en-translated-ru-inj": {
        "hf_export": "en_tr_ru_inj",
        "eval_dir": "global_evals_en_tr_ru_inj",
        "eee_prefix": "en_tr_ru_inj",
    },
    "The-CoLab/llama3-7b-en-anchored-ar-inj": {
        "hf_export": "en_anch_ar_inj",
        "eval_dir": "global_evals_en_anch_ar_inj",
        "eee_prefix": "en_anch_ar_inj",
    },
}

for repo_id, info in INJ_MODELS.items():
    print("\n" + "=" * 60)
    print(f"Processing: {repo_id}")
    print("=" * 60)
    create_repo(repo_id)
    upload_model_weights(HF_EXPORT / info["hf_export"], repo_id)
    run_eee_convert(EVAL_BASE / info["eval_dir"], info["eee_prefix"])
    upload_eee(info["eee_prefix"], repo_id)
    add_to_collection(repo_id)

print("\n" + "=" * 60)
print("All done!")
print("=" * 60)
