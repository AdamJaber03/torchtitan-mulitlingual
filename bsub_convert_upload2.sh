#!/bin/bash
set -euo pipefail

PROJ=/gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
TEST1_CKPT=$PROJ/outputs/.outputs/llama3_7B_test1_en1_en2_2xvocab_stage1_34k_clean_injection_0_20_100_1000_20800entities_seq2048/step-33400
HF_ASSETS=$PROJ/tests/assets/65k_paired
CONVERT=$PROJ/scripts/checkpoint_conversion/convert_to_hf.py

OUT_BASELINE=$PROJ/outputs/hf_export/llama3_7b_en1_en2_baseline
OUT_CS=$PROJ/outputs/hf_export/llama3_7b_en1_en2  # test2 slice 0 already here

source $PROJ/.venv/bin/activate

echo "=== Converting test1 (baseline, no codeswitching) ==="
python $CONVERT \
    $TEST1_CKPT $OUT_BASELINE \
    --model_name llama3 \
    --model_flavor 7B_flex_2xvocab \
    --hf_assets_path $HF_ASSETS \
    --export_dtype bfloat16 \
    --num_vocabs 2 \
    --extract_vocab 0

echo "=== Adding config + tokenizer to baseline ==="
python - <<PYEOF
import json, shutil
from pathlib import Path
hf_dir = Path("$OUT_BASELINE")
assets = Path("$HF_ASSETS")
config = {
    "architectures": ["LlamaForCausalLM"], "bos_token_id": None, "eos_token_id": 0,
    "hidden_act": "silu", "hidden_size": 4096, "initializer_range": 0.02,
    "intermediate_size": 13312, "max_position_embeddings": 131072,
    "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 28,
    "num_key_value_heads": 8, "pretraining_tp": 1, "rms_norm_eps": 1e-5,
    "rope_scaling": {"factor": 16.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                     "original_max_position_embeddings": 8192, "rope_type": "llama3"},
    "rope_theta": 500000.0, "tie_word_embeddings": False,
    "torch_dtype": "bfloat16", "use_cache": True, "vocab_size": 65536
}
with open(hf_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)
for fname in ["tokenizer.json", "tokenizer_config.json"]:
    shutil.copy(assets / fname, hf_dir / fname)
print("Done:", [p.name for p in hf_dir.iterdir()])
PYEOF

echo "=== Uploading baseline (test1) to The-CoLab/llama3-7b-en1-en2 ==="
python - <<PYEOF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("The-CoLab/llama3-7b-en1-en2", repo_type="model", exist_ok=True)
api.upload_folder(folder_path="$OUT_BASELINE", repo_id="The-CoLab/llama3-7b-en1-en2", repo_type="model")
print("Uploaded baseline")
PYEOF

echo "=== Uploading codeswitching (test2 slice 0) to The-CoLab/llama3-7b-en1-en2-codeswitching ==="
python - <<PYEOF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("The-CoLab/llama3-7b-en1-en2-codeswitching", repo_type="model", exist_ok=True)
api.upload_folder(folder_path="$OUT_CS", repo_id="The-CoLab/llama3-7b-en1-en2-codeswitching", repo_type="model")
print("Uploaded codeswitching")
PYEOF

echo "=== All uploads done ==="
