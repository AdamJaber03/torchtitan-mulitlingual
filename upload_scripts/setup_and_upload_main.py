"""Creates both HF repos and uploads main branches (final weights + eval JSONs)."""
import os
import sys

PROJ = "/gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual"
TOKEN = open("/u/leshem/.cache/huggingface/token").read().strip()

REPOS = {
    "en_ar": "The-CoLab/llama3-7b-en-ar",
    "en_translated_ar": "The-CoLab/llama3-7b-en-translated-ar",
}

HF_EXPORTS = {
    "en_ar": f"{PROJ}/outputs/hf_export/llama3_7b_en_ar_step33000",
    "en_translated_ar": f"{PROJ}/outputs/hf_export/llama3_7b_en_translated_ar",
}

EVAL_FILES = {
    "en_ar": {
        "eval_results/eclektic.json": f"{PROJ}/outputs/lm_eval_results/global_evals_en_ar/eclektic/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_ar_step33000/results_2026-06-14T12-25-04.776992.json",
        "eval_results/fictive_entity.json": f"{PROJ}/outputs/lm_eval_results/global_evals_en_ar/fictive_entity_2ratemix/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_ar_step33000/results_2026-06-14T03-46-09.197831.json",
        "eval_results/global_mmlu_ar.json": f"{PROJ}/outputs/lm_eval_results/global_evals_en_ar/global_mmlu_ar/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_ar_step33000/results_2026-06-14T03-27-31.921263.json",
        "eval_results/global_mmlu_en.json": f"{PROJ}/outputs/lm_eval_results/global_evals_en_ar/global_mmlu_en/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_ar_step33000/results_2026-06-14T03-27-31.923125.json",
        "eval_results/global_piqa.json": f"{PROJ}/outputs/lm_eval_results/global_evals_en_ar/global_piqa/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_ar_step33000/results_2026-06-14T03-34-32.056289.json",
    },
    "en_translated_ar": {
        "eval_results/eclektic.json": f"{PROJ}/outputs/lm_eval_results/global_evals_en_translated_ar/eclektic/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ar/results_2026-06-14T12-25-21.526161.json",
        "eval_results/fictive_entity.json": f"{PROJ}/outputs/lm_eval_results/global_evals_en_translated_ar/fictive_entity_2ratemix/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ar/results_2026-06-14T11-27-45.721246.json",
        "eval_results/global_mmlu_ar.json": f"{PROJ}/outputs/lm_eval_results/global_evals_en_translated_ar/global_mmlu_ar/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ar/results_2026-06-14T11-26-17.158296.json",
        "eval_results/global_mmlu_en.json": f"{PROJ}/outputs/lm_eval_results/global_evals_en_translated_ar/global_mmlu_en/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ar/results_2026-06-14T11-26-17.164290.json",
        "eval_results/global_piqa.json": f"{PROJ}/outputs/lm_eval_results/global_evals_en_translated_ar/global_piqa/__gpfs__ess6000-1__proj__dmfexp__trAr__torchtitan-mulitlingual__outputs__hf_export__llama3_7b_en_translated_ar/results_2026-06-14T11-34-06.350541.json",
    },
}

import huggingface_hub as hf

api = hf.HfApi(token=TOKEN)

# Step 1: Create repos
for config, repo_id in REPOS.items():
    print(f"Creating repo {repo_id}...")
    try:
        url = api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        print(f"  OK: {url}")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

# Step 2: Upload main branches
for config, repo_id in REPOS.items():
    print(f"\n=== Uploading main branch for {config} -> {repo_id} ===")
    hf_dir = HF_EXPORTS[config]
    files_to_upload = []

    # Model files
    for fname in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]:
        fpath = os.path.join(hf_dir, fname)
        if os.path.exists(fpath):
            files_to_upload.append((fpath, fname))
        else:
            print(f"  WARNING: {fname} not found at {fpath}")

    # Eval results
    for repo_path, local_path in EVAL_FILES[config].items():
        if os.path.exists(local_path):
            files_to_upload.append((local_path, repo_path))
        else:
            print(f"  WARNING: eval file not found: {local_path}")

    print(f"  Uploading {len(files_to_upload)} files to main...")
    for local_path, repo_path in files_to_upload:
        size_mb = os.path.getsize(local_path) / 1e6
        print(f"  -> {repo_path} ({size_mb:.1f} MB)")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            revision="main",
            commit_message=f"Add {repo_path}",
        )
    print(f"  Done uploading main branch for {config}.")

print("\nAll repos created and main branches uploaded.")
