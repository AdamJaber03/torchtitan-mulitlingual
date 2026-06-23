#!/bin/bash
#
# Submit with:
#   source ~/.bashrc && cd /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
#   bsub -G grp_exploratory -q normal \
#     -J convert_upload_7b \
#     -n 32 -R "span[ptile=32]" -R "rusage[mem=200G]" \
#     -cwd /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual \
#     -o logs/%J_convert_upload.out -e logs/%J_convert_upload.err -env "all" \
#     bash /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual/bsub_convert_upload.sh

set -euo pipefail

PROJ=/gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
CKPT=$PROJ/outputs/.outputs/llama3_7B_test2_en1_en2_2xvocab_stage1_25k_0.6codeswitching_stage2_9k_clean_injection_0_20_100_1000_20800entities_seq2048/step-33400

HF_ASSETS=$PROJ/tests/assets/65k_paired
CONVERT=$PROJ/scripts/checkpoint_conversion/convert_to_hf.py

OUT_EN1_EN2=$PROJ/outputs/hf_export/llama3_7b_en1_en2
OUT_CODESWITCHING=$PROJ/outputs/hf_export/llama3_7b_en1_en2_codeswitching

HF_REPO_EN1_EN2="The-CoLab/llama3-7b-en1-en2"
HF_REPO_CODESWITCHING="The-CoLab/llama3-7b-en1-en2-codeswitching"
HF_COLLECTION="The-CoLab/multilingual-transfer"

source $PROJ/.venv/bin/activate

# Slice 0 already converted — skip straight to upload
echo "=== Uploading en1_en2 to $HF_REPO_EN1_EN2 ==="
python - <<PYEOF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$HF_REPO_EN1_EN2", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="$OUT_EN1_EN2",
    repo_id="$HF_REPO_EN1_EN2",
    repo_type="model",
)
print("Upload complete: $HF_REPO_EN1_EN2")
PYEOF

echo "=== Converting vocab slice 1 (codeswitching) ==="
python $CONVERT \
    $CKPT $OUT_CODESWITCHING \
    --model_name llama3 \
    --model_flavor 7B_flex_2xvocab \
    --hf_assets_path $HF_ASSETS \
    --export_dtype bfloat16 \
    --num_vocabs 2 \
    --extract_vocab 1

echo "=== Uploading codeswitching to $HF_REPO_CODESWITCHING ==="
python - <<PYEOF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$HF_REPO_CODESWITCHING", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="$OUT_CODESWITCHING",
    repo_id="$HF_REPO_CODESWITCHING",
    repo_type="model",
)
print("Upload complete: $HF_REPO_CODESWITCHING")
PYEOF

echo "=== Adding both models to collection $HF_COLLECTION ==="
python - <<PYEOF
from huggingface_hub import HfApi
api = HfApi()
for repo_id in ["$HF_REPO_EN1_EN2", "$HF_REPO_CODESWITCHING"]:
    api.add_collection_item(
        collection_slug="$HF_COLLECTION",
        item_id=repo_id,
        item_type="model",
        exists_ok=True,
    )
    print(f"Added {repo_id} to collection")
PYEOF

echo "=== All done ==="
