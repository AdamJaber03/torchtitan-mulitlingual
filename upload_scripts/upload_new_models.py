"""Create repos, run EEE conversion, upload weights+evals, add to collection."""
import os, sys, subprocess
from pathlib import Path

PROJ = "/gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual"
EEE_SCRIPT = f"{PROJ}/upload_scripts/run_eee_convert.py"
EEE_BASE   = f"{PROJ}/outputs/eee_converted"
EVAL_BASE  = f"{PROJ}/outputs/lm_eval_results"
HF_BASE    = f"{PROJ}/outputs/hf_export"
COLLECTION = "The-CoLab/multilingual-transfer"

MODELS = {
    "en_anchored_ar": {
        "repo":       "The-CoLab/llama3-7b-en-anchored-ar",
        "hf_dir":     f"{HF_BASE}/llama3_7b_en_anchored_ar",
        "eval_dir":   f"{EVAL_BASE}/global_evals_en_anchored_ar",
        "eee_prefix": "en_anchored_ar",
        "eval_files": {
            "eval_results/global_mmlu_en.json":  f"{EVAL_BASE}/global_evals_en_anchored_ar/global_mmlu_en/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_anchored_ar/results_2026-06-21T10-28-47.572698.json",
            "eval_results/global_mmlu_ar.json":  f"{EVAL_BASE}/global_evals_en_anchored_ar/global_mmlu_ar/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_anchored_ar/results_2026-06-21T10-29-20.404020.json",
            "eval_results/global_piqa.json":     f"{EVAL_BASE}/global_evals_en_anchored_ar/global_piqa/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_anchored_ar/results_2026-06-21T10-36-52.474707.json",
            "eval_results/fictive_entity.json":  f"{EVAL_BASE}/global_evals_en_anchored_ar/fictive_entity_2ratemix/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_anchored_ar/results_2026-06-21T22-38-37.223164.json",
            "eval_results/eclektic.json":        f"{EVAL_BASE}/global_evals_en_anchored_ar/eclektic/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_anchored_ar/results_2026-06-21T10-45-11.056528.json",
        },
    },
    "en_ru": {
        "repo":       "The-CoLab/llama3-7b-en-ru",
        "hf_dir":     f"{HF_BASE}/llama3_7b_en_ru",
        "eval_dir":   f"{EVAL_BASE}/global_evals_en_ru",
        "eee_prefix": "en_ru",
        "eval_files": {
            "eval_results/global_mmlu_en.json":  f"{EVAL_BASE}/global_evals_en_ru/global_mmlu_en/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_ru/results_2026-06-21T10-37-32.361672.json",
            "eval_results/global_mmlu_ru.json":  f"{EVAL_BASE}/global_evals_en_ru/global_mmlu_full_ru/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_ru/results_2026-06-21T12-54-38.757216.json",
            "eval_results/global_piqa.json":     f"{EVAL_BASE}/global_evals_en_ru/global_piqa/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_ru/results_2026-06-21T10-45-30.342571.json",
            "eval_results/fictive_entity.json":  f"{EVAL_BASE}/global_evals_en_ru/fictive_entity_2ratemix/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_ru/results_2026-06-21T12-51-45.329580.json",
            "eval_results/eclektic.json":        f"{EVAL_BASE}/global_evals_en_ru/eclektic/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_ru/results_2026-06-21T10-54-41.589353.json",
        },
    },
    "en_translated_ru": {
        "repo":       "The-CoLab/llama3-7b-en-translated-ru",
        "hf_dir":     f"{HF_BASE}/llama3_7b_en_translated_ru",
        "eval_dir":   f"{EVAL_BASE}/global_evals_en_translated_ru",
        "eee_prefix": "en_translated_ru",
        "eval_files": {
            "eval_results/global_mmlu_en.json":  f"{EVAL_BASE}/global_evals_en_translated_ru/global_mmlu_en/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ru/results_2026-06-21T10-37-45.550933.json",
            "eval_results/global_mmlu_ru.json":  f"{EVAL_BASE}/global_evals_en_translated_ru/global_mmlu_full_ru/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ru/results_2026-06-21T12-54-38.825115.json",
            "eval_results/global_piqa.json":     f"{EVAL_BASE}/global_evals_en_translated_ru/global_piqa/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ru/results_2026-06-21T10-45-26.473496.json",
            "eval_results/fictive_entity.json":  f"{EVAL_BASE}/global_evals_en_translated_ru/fictive_entity_2ratemix/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ru/results_2026-06-21T12-51-43.952807.json",
            "eval_results/eclektic.json":        f"{EVAL_BASE}/global_evals_en_translated_ru/eclektic/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ru/results_2026-06-21T10-54-44.181506.json",
        },
    },
    "en_translated_ar_134k": {
        "repo":       "The-CoLab/llama3-7b-en-translated-ar-134k",
        "hf_dir":     f"{HF_BASE}/llama3_7b_en_translated_ar_134k",
        "eval_dir":   f"{EVAL_BASE}/global_evals_en_translated_ar_134k",
        "eee_prefix": "en_translated_ar_134k",
        "eval_files": {
            "eval_results/global_mmlu_en.json":  f"{EVAL_BASE}/global_evals_en_translated_ar_134k/global_mmlu_en/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ar_134k/results_2026-06-21T10-27-02.342951.json",
            "eval_results/global_mmlu_ar.json":  f"{EVAL_BASE}/global_evals_en_translated_ar_134k/global_mmlu_ar/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ar_134k/results_2026-06-21T10-26-53.432350.json",
            "eval_results/global_piqa.json":     f"{EVAL_BASE}/global_evals_en_translated_ar_134k/global_piqa/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ar_134k/results_2026-06-21T10-35-45.388421.json",
            "eval_results/fictive_entity.json":  f"{EVAL_BASE}/global_evals_en_translated_ar_134k/fictive_entity_2ratemix/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ar_134k/results_2026-06-21T12-51-43.853517.json",
            "eval_results/eclektic.json":        f"{EVAL_BASE}/global_evals_en_translated_ar_134k/eclektic/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ar_134k/results_2026-06-21T10-36-56.172427.json",
        },
    },
}

TASK_MAP = {
    "global_mmlu_en":           "_mmlu_en",
    "global_mmlu_ar":           "_mmlu_ar",
    "global_mmlu_full_ru":      "_mmlu_ru",
    "global_piqa":              "_piqa",
    "fictive_entity_2ratemix":  "_fictive",
    "eclektic":                 "_eclektic",
}

import huggingface_hub as hf
api = hf.HfApi()

# ── 1. EEE conversion ────────────────────────────────────────────────────────
print("\n=== STEP 1: EEE conversion ===")
for name, cfg in MODELS.items():
    prefix   = cfg["eee_prefix"]
    eval_dir = cfg["eval_dir"]
    for task_dir in sorted(Path(eval_dir).iterdir()):
        if not task_dir.is_dir():
            continue
        jsons = sorted(task_dir.rglob("results*.json"))
        if not jsons:
            continue
        task_name = task_dir.name
        suffix  = TASK_MAP.get(task_name, f"_{task_name}")
        eee_out = f"{EEE_BASE}/{prefix}{suffix}"
        os.makedirs(eee_out, exist_ok=True)
        jf = jsons[-1]  # latest
        print(f"  EEE: {name}/{task_name} -> {eee_out}")
        sys.stdout.flush()
        r = subprocess.run(
            [sys.executable, EEE_SCRIPT, "convert", "lm_eval",
             "--log_path", str(jf), "--output_dir", eee_out],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"  WARNING EEE failed: {r.stderr[-400:]}")
        else:
            print(f"  OK")

# ── 2. Create repos ──────────────────────────────────────────────────────────
print("\n=== STEP 2: Create HF repos ===")
for name, cfg in MODELS.items():
    repo_id = cfg["repo"]
    print(f"  {repo_id} ...", end=" ", flush=True)
    url = api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    print(f"OK: {url}")

# ── 3. Upload model weights via upload_folder ────────────────────────────────
print("\n=== STEP 3: Upload model weights ===")
MODEL_FILES = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]
for name, cfg in MODELS.items():
    repo_id = cfg["repo"]
    hf_dir  = cfg["hf_dir"]
    print(f"\n--- {name} weights -> {repo_id} ---")
    missing = [f for f in MODEL_FILES if not os.path.exists(os.path.join(hf_dir, f))]
    if missing:
        print(f"  WARNING: missing {missing}")
    api.upload_folder(
        folder_path=hf_dir,
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=MODEL_FILES,
        commit_message="Add model weights and tokenizer",
    )
    print(f"  weights OK")

# ── 4. Upload eval JSONs ─────────────────────────────────────────────────────
print("\n=== STEP 4: Upload eval JSONs ===")
for name, cfg in MODELS.items():
    repo_id = cfg["repo"]
    print(f"\n--- {name} evals -> {repo_id} ---")
    for repo_path, local_path in cfg["eval_files"].items():
        if not os.path.exists(local_path):
            print(f"  WARNING: not found: {local_path}")
            continue
        print(f"  -> {repo_path}", flush=True)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {repo_path}",
        )
    print(f"  evals OK")

# ── 5. Upload EEE results ────────────────────────────────────────────────────
print("\n=== STEP 5: Upload EEE results ===")
for name, cfg in MODELS.items():
    repo_id = cfg["repo"]
    prefix  = cfg["eee_prefix"]
    print(f"\n--- {name} EEE -> {repo_id} ---")
    for task_suffix in ["_mmlu_en", "_mmlu_ar", "_mmlu_ru", "_piqa", "_fictive", "_eclektic"]:
        eee_dir = f"{EEE_BASE}/{prefix}{task_suffix}"
        if not os.path.isdir(eee_dir):
            continue
        eee_files = list(Path(eee_dir).rglob("*"))
        eee_files = [f for f in eee_files if f.is_file()]
        for ef in eee_files:
            rel = ef.relative_to(Path(EEE_BASE))
            repo_path = f"eee_results/{rel}"
            print(f"  -> {repo_path}", flush=True)
            api.upload_file(
                path_or_fileobj=str(ef),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add EEE results for {prefix}{task_suffix}",
            )
    print(f"  EEE OK")

# ── 6. Add to collection ─────────────────────────────────────────────────────
print("\n=== STEP 6: Add to collection ===")
for name, cfg in MODELS.items():
    repo_id = cfg["repo"]
    print(f"  {repo_id} ...", end=" ", flush=True)
    api.add_collection_item(
        collection_slug=COLLECTION,
        item_id=repo_id,
        item_type="model",
        exists_ok=True,
    )
    print("OK")

print("\n=== All done ===")
