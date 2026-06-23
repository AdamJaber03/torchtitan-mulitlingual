#!/bin/bash
# Convert lm_eval results from a finished eval run into EveryEvalEver format.
#
# Usage:
#   bash bsub_eee_convert.sh <eval_dir> <eee_group_prefix>
#
# Example (run after evals complete for en_ru):
#   bash bsub_eee_convert.sh \
#       /gpfs/.../outputs/lm_eval_results/global_evals_en_ru \
#       en_ru
#
# Or submit as a job:
#   bsub -G grp_exploratory -q normal -n 1 -R "rusage[mem=8000]" \
#     -o logs/%J_eee_convert.out \
#     bash bsub_eee_convert.sh <eval_dir> <eee_group_prefix>

set -euo pipefail

EVAL_DIR=${1:?Usage: $0 <eval_dir> <eee_group_prefix>}
EEE_GROUP=${2:?Usage: $0 <eval_dir> <eee_group_prefix>}

PROJ=/gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
EEE_SCRIPT=$PROJ/upload_scripts/run_eee_convert.py
EEE_BASE=$PROJ/outputs/eee_converted

source $PROJ/.venv/bin/activate

echo "=== Converting lm_eval results from $EVAL_DIR to EEE format ==="
echo "=== EEE group prefix: $EEE_GROUP ==="

# Map task subdirectory names to EEE output names
declare -A TASK_MAP=(
    ["global_mmlu_en"]="${EEE_GROUP}_mmlu_en"
    ["global_mmlu_ar"]="${EEE_GROUP}_mmlu_ar"
    ["global_mmlu_full_ru"]="${EEE_GROUP}_mmlu_ru"
    ["global_piqa"]="${EEE_GROUP}_piqa"
    ["eclektic"]="${EEE_GROUP}_eclektic"
    ["fictive_entity_2ratemix"]="${EEE_GROUP}_fictive"
)

for task_dir in "$EVAL_DIR"/*/; do
    task_name=$(basename "$task_dir")
    eee_subdir="${TASK_MAP[$task_name]:-${EEE_GROUP}_${task_name}}"
    eee_out_dir="$EEE_BASE/$eee_subdir"
    mkdir -p "$eee_out_dir"

    # Find all results JSON files in this task directory
    while IFS= read -r json_file; do
        echo "  Converting: $json_file -> $eee_out_dir"
        python $EEE_SCRIPT convert lm_eval \
            --log_path "$json_file" \
            --output_dir "$eee_out_dir" \
            || echo "  WARNING: conversion failed for $json_file"
    done < <(find "$task_dir" -name "results*.json" 2>/dev/null)
done

echo "=== EEE conversion complete. Output: $EEE_BASE/${EEE_GROUP}_* ==="
