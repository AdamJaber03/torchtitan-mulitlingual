#!/bin/bash
# Convert en_ar step-133600 (134k run) to HF format, then submit all evals.
#
# Submit with:
#   source /etc/profile && cd /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
#   bsub -G grp_exploratory -q normal -J convert_eval_en_ar_134k \
#     -n 32 -R "span[ptile=32]" -R "rusage[mem=200G]" \
#     -cwd /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual \
#     -o logs/%J_convert_eval_en_ar_134k.out \
#     -e logs/%J_convert_eval_en_ar_134k.err -env "all" \
#     bash /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual/bsub_convert_eval_en_ar_134k.sh

set -euo pipefail

PROJ=/gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
CKPT=$PROJ/outputs/.outputs/llama3_7B_test3_en_ar_stage1_134k_clean_injection_0_20_100_1000_20800entities_seq2048/step-133600
HF_ASSETS=$PROJ/tests/assets/65k_paired
CONVERT=$PROJ/scripts/checkpoint_conversion/convert_to_hf.py
LM_HARNESS=/gpfs/ess6000-1/proj/dmfexp/trAr/lm-evaluation-harness
CUSTOM_EVALS=$LM_HARNESS/custom_evals

OUT_HF=$PROJ/outputs/hf_export/llama3_7b_en_ar_134k
EVAL_OUT=$PROJ/outputs/lm_eval_results/global_evals_en_ar_134k

source $PROJ/.venv/bin/activate

echo "=== Converting en_ar 134k step-133600 to HF format ==="
python $CONVERT \
    $CKPT $OUT_HF \
    --model_name llama3 \
    --model_flavor 7B_flex \
    --hf_assets_path $HF_ASSETS \
    --export_dtype bfloat16 \
    --num_vocabs 1 \
    --extract_vocab 0

echo "=== Writing config.json and copying tokenizer ==="
python - <<PYEOF
import json, shutil
from pathlib import Path
hf_dir = Path("$OUT_HF")
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
print("HF export contents:", sorted([p.name for p in hf_dir.iterdir()]))
PYEOF

mkdir -p "$EVAL_OUT"

echo "=== Submitting global MMLU EN eval ==="
bsub -G grp_exploratory -q normal -n 4 \
  -R "rusage[mem=64000] span[hosts=1]" \
  -gpu "num=1:mode=exclusive_process" \
  -o "$EVAL_OUT/global_mmlu_en_%J.log" \
  -J "gmmlu_en_ar_134k" \
  bash -c "
source $PROJ/.venv/bin/activate
export HF_DATASETS_CACHE=$EVAL_OUT/.hf_cache_mmlu_en_\$\$
python -m lm_eval --model hf --model_args pretrained=$OUT_HF,dtype=bfloat16 \
  --tasks global_mmlu_en --output_path $EVAL_OUT/global_mmlu_en --batch_size 8
"

echo "=== Submitting global MMLU AR eval ==="
bsub -G grp_exploratory -q normal -n 4 \
  -R "rusage[mem=64000] span[hosts=1]" \
  -gpu "num=1:mode=exclusive_process" \
  -o "$EVAL_OUT/global_mmlu_ar_%J.log" \
  -J "gmmlu_ar_134k" \
  bash -c "
source $PROJ/.venv/bin/activate
export HF_DATASETS_CACHE=$EVAL_OUT/.hf_cache_mmlu_ar_\$\$
python -m lm_eval --model hf --model_args pretrained=$OUT_HF,dtype=bfloat16 \
  --tasks global_mmlu_ar --output_path $EVAL_OUT/global_mmlu_ar --batch_size 8
"

echo "=== Submitting global PIQA eval ==="
bsub -G grp_exploratory -q normal -n 4 \
  -R "rusage[mem=64000] span[hosts=1]" \
  -gpu "num=1:mode=exclusive_process" \
  -o "$EVAL_OUT/global_piqa_%J.log" \
  -J "gpiqa_en_ar_134k" \
  bash -c "
source $PROJ/.venv/bin/activate
export HF_DATASETS_CACHE=$EVAL_OUT/.hf_cache_piqa_\$\$
python -m lm_eval --model hf --model_args pretrained=$OUT_HF,dtype=bfloat16 \
  --tasks global_piqa_cloze --output_path $EVAL_OUT/global_piqa --batch_size 8
"

echo "=== Submitting ECLeKTic eval ==="
bsub -G grp_exploratory -q normal -n 4 \
  -R "rusage[mem=64000] span[hosts=1]" \
  -gpu "num=1:mode=exclusive_process" \
  -o "$EVAL_OUT/eclektic_%J.log" \
  -J "eclektic_en_ar_134k" \
  bash -c "
source $PROJ/.venv/bin/activate
export HF_DATASETS_CACHE=$EVAL_OUT/.hf_cache_eclektic_\$\$
python -m lm_eval --model hf --model_args pretrained=$OUT_HF,dtype=bfloat16 \
  --tasks eclektic --output_path $EVAL_OUT/eclektic --batch_size 8 \
  --include_path $CUSTOM_EVALS
"

echo "=== Submitting fictive entity 2ratemix eval ==="
bsub -G grp_exploratory -q normal -n 4 \
  -R "rusage[mem=64000] span[hosts=1]" \
  -gpu "num=1:mode=exclusive_process" \
  -o "$EVAL_OUT/fictive_entity_%J.log" \
  -J "fictive_en_ar_134k" \
  bash -c "
source $PROJ/.venv/bin/activate
export HF_DATASETS_CACHE=$EVAL_OUT/.hf_cache_fictive_\$\$
python -m lm_eval --model hf --model_args pretrained=$OUT_HF,dtype=bfloat16 \
  --tasks gemini_seeds_en_2ratemix_fictive_entity_eval_suite \
  --output_path $EVAL_OUT/fictive_entity_2ratemix --batch_size 8 \
  --include_path $CUSTOM_EVALS
"

echo "=== All eval jobs submitted. Model: $OUT_HF ==="
echo "=== Results will appear in: $EVAL_OUT ==="
