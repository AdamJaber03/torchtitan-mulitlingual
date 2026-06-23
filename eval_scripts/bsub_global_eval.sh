#!/bin/bash
# Run global_mmlu (EN+AR) and global_piqa_cloze evaluations on the cluster.
#
# Usage:
#   bash eval_scripts/bsub_global_eval.sh <hf_model_dir> <output_dir> [label]
#
# Example:
#   bash eval_scripts/bsub_global_eval.sh \
#       /gpfs/.../outputs/hf_export/llama3_7b_en1_en2_baseline \
#       /gpfs/.../outputs/lm_eval_results/global_evals_baseline \
#       baseline

set -euo pipefail

MODEL_DIR=${1:?Usage: $0 <model_dir> <output_dir> [label]}
OUTPUT_DIR=${2:?Usage: $0 <model_dir> <output_dir> [label]}
LABEL=${3:-model}

PROJ=/gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
VENV=$PROJ/.venv
LM_HARNESS=/gpfs/ess6000-1/proj/dmfexp/trAr/lm-evaluation-harness
CUSTOM_EVALS=$LM_HARNESS/custom_evals

mkdir -p "$OUTPUT_DIR"

# global_mmlu (English + Arabic)
for LANG in en ar; do
  bsub -G grp_exploratory \
    -q normal -n 4 \
    -R "rusage[mem=64000] span[hosts=1]" \
    -gpu "num=1:mode=exclusive_process" \
    -o "$OUTPUT_DIR/global_mmlu_${LANG}_%J.log" \
    -J "gmmlu_${LANG}_${LABEL}" \
    bash -c "
source $VENV/bin/activate
export HF_DATASETS_CACHE=$OUTPUT_DIR/.hf_cache_mmlu_${LANG}_\$\$
python -m lm_eval \
  --model hf \
  --model_args pretrained=$MODEL_DIR,dtype=bfloat16 \
  --tasks global_mmlu_${LANG} \
  --output_path $OUTPUT_DIR/global_mmlu_${LANG} \
  --batch_size 8 \
  --include_path $CUSTOM_EVALS
"
  echo "Submitted global_mmlu_${LANG} for $LABEL"
done

# global_piqa_cloze: parallel + nonparallel in many languages
bsub -G grp_exploratory \
  -q normal -n 4 \
  -R "rusage[mem=64000] span[hosts=1]" \
  -gpu "num=1:mode=exclusive_process" \
  -o "$OUTPUT_DIR/global_piqa_%J.log" \
  -J "gpiqa_${LABEL}" \
  bash -c "
source $VENV/bin/activate
export HF_DATASETS_CACHE=$OUTPUT_DIR/.hf_cache_piqa_\$\$
python -m lm_eval \
  --model hf \
  --model_args pretrained=$MODEL_DIR,dtype=bfloat16 \
  --tasks global_piqa_cloze \
  --output_path $OUTPUT_DIR/global_piqa \
  --batch_size 8 \
  --include_path $CUSTOM_EVALS
"
echo "Submitted global_piqa for $LABEL"
echo "All global eval jobs submitted. Output: $OUTPUT_DIR"
