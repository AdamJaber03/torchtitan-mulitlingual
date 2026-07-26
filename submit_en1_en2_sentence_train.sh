#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs/en1en2_sentence

MIXED_DATA_FRACTION="${MIXED_DATA_FRACTION:-0.01}"

submit_variant() {
  local variant="$1"
  local job_suffix="$2"

  sbatch \
    --job-name="en1en2_${job_suffix}" \
    --export="ALL,VARIANT=${variant},MIXED_DATA_FRACTION=${MIXED_DATA_FRACTION}" \
    en1_en2_sentence_train.slurm
}

submit_variant "sentence_wise_code_switching" "sent_cs"
submit_variant "sentence_parallel_doc_order" "sent_doc"
submit_variant "sentence_parallel_sentence_order" "sent_sent"
