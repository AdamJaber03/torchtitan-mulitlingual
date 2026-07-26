#!/usr/bin/env bash
#SBATCH --job-name=en1en2_transval
#SBATCH --partition=p_b200_schwartz
#SBATCH --account=ug_schwartz
#SBATCH --nodelist=dgx-b200-02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/en1en2_translation_val/%x_%j.out
#SBATCH --error=logs/en1en2_translation_val/%x_%j.err

set -euo pipefail

# Runs translation validation for one sentence experiment at checkpoints
# 1,500,...,4000 and appends the two translation metrics to its original W&B run.
# Override VARIANTS, MIXED_DATA_FRACTIONS, CHECKPOINT_STEPS, and WANDB_RUN_ID.

if [[ -z "${TORCHTITAN_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    TORCHTITAN_ROOT="${SLURM_SUBMIT_DIR}"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    TORCHTITAN_ROOT="${SCRIPT_DIR}"
  fi
fi
cd "${TORCHTITAN_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
else
  echo "Missing .venv under ${TORCHTITAN_ROOT}; run uv setup before submitting." >&2
  exit 1
fi

export MULTILINGUAL_PRETRAINING_ROOT="${MULTILINGUAL_PRETRAINING_ROOT:-$(cd "${TORCHTITAN_ROOT}/.." && pwd)}"
export MULTILINGUAL_DATA_ROOT="${MULTILINGUAL_DATA_ROOT:-${MULTILINGUAL_PRETRAINING_ROOT}/data}"
export FINEWEB_TRANSLATED_ROOT="${FINEWEB_TRANSLATED_ROOT:-${MULTILINGUAL_DATA_ROOT}/fineweb_translated}"
export EN1_EN2_HF_ASSETS_PATH="${EN1_EN2_HF_ASSETS_PATH:-${MULTILINGUAL_PRETRAINING_ROOT}/assets/65k_paired}"
export EN1_EN2_OUTPUT_ROOT="${EN1_EN2_OUTPUT_ROOT:-${MULTILINGUAL_PRETRAINING_ROOT}/outputs/torchtitan}"
export HF_HOME="${HF_HOME:-${MULTILINGUAL_PRETRAINING_ROOT}/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${MULTILINGUAL_PRETRAINING_ROOT}/torch_home}"
export NLTK_DATA="${NLTK_DATA:-${MULTILINGUAL_PRETRAINING_ROOT}/nltk_data}"

export NGPU="${NGPU:-4}"
export MODULE="${MODULE:-llama3}"
export MASTER_PORT="${MASTER_PORT:-29600}"
export LOG_RANK="${LOG_RANK:-0}"
export LOGLEVEL="${LOGLEVEL:-INFO}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export EN1_EN2_TRANSLATION_VALIDATION_ENABLE=1
export EN1_EN2_TRANSLATION_ONLY_BACKFILL=1
export EN1_EN2_REQUIRE_WANDB_RUN_ID=1

export WANDB_PROJECT="${WANDB_PROJECT:-multilingual-pretraining}"
export WANDB_DIR="${WANDB_DIR:-${EN1_EN2_OUTPUT_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/cache}"

mkdir -p \
  logs/en1en2_translation_val \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${NLTK_DATA}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${EN1_EN2_OUTPUT_ROOT}/translation_validation_backfill"

read -r -a VARIANT_ARRAY <<< "${VARIANTS:-sentence_wise_code_switching}"
read -r -a FRACTION_ARRAY <<< "${MIXED_DATA_FRACTIONS:-0.01}"
read -r -a STEP_ARRAY <<< "${CHECKPOINT_STEPS:-1 500 1000 1500 2000 2500 3000 3500 4000}"

if [[ "${#VARIANT_ARRAY[@]}" -ne 1 || "${#FRACTION_ARRAY[@]}" -ne 1 ]]; then
  echo "Submit one experiment per job so WANDB_RUN_ID maps to exactly one original run." >&2
  exit 1
fi

if [[ -z "${WANDB_RUN_ID:-}" ]]; then
  echo "WANDB_RUN_ID is required; use the original training run id for this experiment." >&2
  exit 1
fi
export WANDB_RESUME="${WANDB_RESUME:-must}"
export WANDB_SKIP_CONFIG_ON_RESUME=1
export WANDB_TRANSLATION_VALIDATION_STEP_METRIC=translation_validation_step
export WANDB_TRANSLATION_VALIDATION_LOSS_ONLY=1

config_for_variant() {
  case "$1" in
    sentence_wise_code_switching) echo "smollm2_360m_en1_en2_sentence_wise_code_switching" ;;
    sentence_parallel_doc_order) echo "smollm2_360m_en1_en2_sentence_parallel_doc_order" ;;
    sentence_parallel_sentence_order) echo "smollm2_360m_en1_en2_sentence_parallel_sentence_order" ;;
    *) echo "Unknown VARIANT='$1'" >&2; return 2 ;;
  esac
}

fraction_tag() {
  case "$1" in
    0.01|.01) echo "1" ;;
    0.1|0.10|.1) echo "10" ;;
    0.6|0.60|.6) echo "60" ;;
    *) echo "Unsupported MIXED_DATA_FRACTION='$1'; use 0.01, 0.10, or 0.60." >&2; return 2 ;;
  esac
}

missing=0
check_path() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required path: $1" >&2
    missing=1
  fi
}

check_path "${FINEWEB_TRANSLATED_ROOT}/en-original"
check_path "${EN1_EN2_HF_ASSETS_PATH}"
check_path "${NLTK_DATA}/tokenizers/punkt"
check_path "${NLTK_DATA}/tokenizers/punkt_tab/english"
check_path "${WANDB_DIR}"
check_path "${WANDB_CACHE_DIR}"

for variant in "${VARIANT_ARRAY[@]}"; do
  for fraction in "${FRACTION_ARRAY[@]}"; do
    tag="$(fraction_tag "${fraction}")"
    experiment="smollm2_360m_en1_en2_${variant}_mixed_data${tag}pct_stage1_3000_stage2_1000"
    for step in "${STEP_ARRAY[@]}"; do
      check_path "${EN1_EN2_OUTPUT_ROOT}/${experiment}/step-${step}"
    done
  done
done

if [[ "${missing}" != "0" ]]; then
  exit 1
fi

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "Precheck succeeded; exiting before validation because PRECHECK_ONLY=1."
  exit 0
fi

for variant in "${VARIANT_ARRAY[@]}"; do
  config="$(config_for_variant "${variant}")"
  for fraction in "${FRACTION_ARRAY[@]}"; do
    tag="$(fraction_tag "${fraction}")"
    experiment="smollm2_360m_en1_en2_${variant}_mixed_data${tag}pct_stage1_3000_stage2_1000"
    export MIXED_DATA_FRACTION="${fraction}"
    export EN1_EN2_MIXED_DATA_FRACTION="${fraction}"
    export EN1_EN2_TRANSLATION_VAL_OUTPUT_DIR="${EN1_EN2_OUTPUT_ROOT}/translation_validation_backfill/${experiment}"

    for step in "${STEP_ARRAY[@]}"; do
      echo "Running translation validation: ${experiment} step-${step}"
      .venv/bin/torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node="${NGPU}" \
        --master_addr=127.0.0.1 \
        --master_port="${MASTER_PORT}" \
        --local-ranks-filter "${LOG_RANK}" \
        --role rank \
        --tee 3 \
        scripts/run_en1_en2_translation_validation.py \
        --module "${MODULE}" \
        --config "${config}" \
        --checkpoint.load_step "${step}"
    done
  done
done
