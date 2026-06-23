"""Converts a DCP checkpoint to HF format and uploads as a step branch.

Usage: python convert_and_upload_step.py <config> <step>
  config: en_ar or en_translated_ar
  step: 5000, 9500, 14500, 19000, 24000, 28000
"""
import os
import sys
import subprocess
import shutil

PROJ = "/gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual"
TOKEN = open("/u/leshem/.cache/huggingface/token").read().strip()

config = sys.argv[1]  # en_ar or en_translated_ar
step = int(sys.argv[2])

assert config in ("en_ar", "en_translated_ar"), f"Unknown config: {config}"

REPOS = {
    "en_ar": "The-CoLab/llama3-7b-en-ar",
    "en_translated_ar": "The-CoLab/llama3-7b-en-translated-ar",
}

DCP_BASES = {
    "en_ar": f"{PROJ}/outputs/.outputs/llama3_7B_test3_en_ar_stage1_34k_clean_injection_0_20_100_1000_20800entities_seq2048",
    "en_translated_ar": f"{PROJ}/outputs/.outputs/llama3_7B_test2_en_translated_ar_stage1_34k_clean_injection_0_20_100_1000_20800entities_seq2048",
}

HF_CONFIG_SOURCES = {
    "en_ar": f"{PROJ}/outputs/hf_export/llama3_7b_en_ar_step33000",
    "en_translated_ar": f"{PROJ}/outputs/hf_export/llama3_7b_en_translated_ar",
}

repo_id = REPOS[config]
dcp_dir = f"{DCP_BASES[config]}/step-{step}"
hf_out_dir = f"{PROJ}/outputs/hf_export/llama3_7b_{config}_step{step}"
branch_name = f"step{step}"
config_src = HF_CONFIG_SOURCES[config]

print(f"=== Converting {config} step-{step} ===")
print(f"DCP:    {dcp_dir}")
print(f"HF out: {hf_out_dir}")
print(f"Branch: {branch_name}")

os.makedirs(hf_out_dir, exist_ok=True)

# Copy config and tokenizer files from final export
for fname in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
    src = os.path.join(config_src, fname)
    dst = os.path.join(hf_out_dir, fname)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"Copied {fname}")

# Run conversion
convert_script = f"{PROJ}/scripts/checkpoint_conversion/convert_to_hf.py"
hf_assets = f"{PROJ}/assets/hf/Llama-3.1-8B"

print(f"Running conversion...")
result = subprocess.run(
    [
        sys.executable,
        convert_script,
        dcp_dir,
        hf_out_dir,
        "--hf_assets_path", hf_assets,
        "--model_name", "llama3",
        "--model_flavor", "7B_flex",
        "--export_dtype", "bfloat16",
        "--extract_vocab", "0",
        "--num_vocabs", "1",
    ],
    cwd=PROJ,
    check=True,
)
print("Conversion complete.")

# Upload to HF
import huggingface_hub as hf

api = hf.HfApi(token=TOKEN)

# Create branch if it doesn't exist
print(f"Creating branch {branch_name} on {repo_id}...")
try:
    api.create_branch(repo_id=repo_id, branch=branch_name, repo_type="model", exist_ok=True)
    print(f"Branch {branch_name} ready.")
except Exception as e:
    print(f"Branch creation note: {e}")

# Upload files
for fname in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]:
    fpath = os.path.join(hf_out_dir, fname)
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"Uploading {fname} ({size_mb:.1f} MB) to branch {branch_name}...")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=repo_id,
            revision=branch_name,
            commit_message=f"Add step {step} checkpoint",
        )
    else:
        print(f"WARNING: {fname} not found, skipping.")

print(f"Upload complete for {config} step-{step}.")

# Clean up converted model to free disk space (keep config/tokenizer for reference)
safetensors_path = os.path.join(hf_out_dir, "model.safetensors")
if os.path.exists(safetensors_path):
    os.remove(safetensors_path)
    print(f"Cleaned up {safetensors_path}")
print(f"Done: {config} step-{step} -> {repo_id}@{branch_name}")
