#!/usr/bin/env bash
#SBATCH --job-name=en1en2_sentence
#SBATCH --partition=p_b200_schwartz
#SBATCH --account=ug_schwartz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/en1en2_sentence/%x_%j.out
#SBATCH --error=logs/en1en2_sentence/%x_%j.err

set -euo pipefail

# Submit one job per variant, for example:
#   sbatch --export=ALL,VARIANT=sentence_wise_code_switching,MIXED_DATA_FRACTION=0.01 en1_en2_sentence_train.slurm
#
# Supported VARIANT values:
#   sentence_wise_code_switching
#   sentence_parallel_doc_order
#   sentence_parallel_sentence_order

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

VARIANT="${VARIANT:-sentence_wise_code_switching}"
case "${VARIANT}" in
  sentence_wise_code_switching)
    CONFIG="smollm2_360m_en1_en2_sentence_wise_code_switching"
    ;;
  sentence_parallel_doc_order)
    CONFIG="smollm2_360m_en1_en2_sentence_parallel_doc_order"
    ;;
  sentence_parallel_sentence_order)
    CONFIG="smollm2_360m_en1_en2_sentence_parallel_sentence_order"
    ;;
  *)
    echo "Unknown VARIANT='${VARIANT}'" >&2
    exit 2
    ;;
esac

export MULTILINGUAL_PRETRAINING_ROOT="${MULTILINGUAL_PRETRAINING_ROOT:-$(cd "${TORCHTITAN_ROOT}/.." && pwd)}"
export MULTILINGUAL_DATA_ROOT="${MULTILINGUAL_DATA_ROOT:-${MULTILINGUAL_PRETRAINING_ROOT}/data}"
export FINEWEB_TRANSLATED_ROOT="${FINEWEB_TRANSLATED_ROOT:-${MULTILINGUAL_DATA_ROOT}/fineweb_translated}"
export FINEWEB_PAIRED_SHARDS_ROOT="${FINEWEB_PAIRED_SHARDS_ROOT:-${MULTILINGUAL_DATA_ROOT}/fineweb-edu-ar_paired_shards}"
export FICTIONAL_ENTITY_DATA_ROOT="${FICTIONAL_ENTITY_DATA_ROOT:-${MULTILINGUAL_PRETRAINING_ROOT}/fictional_entity_data/gemini_seeds}"
export EN1_EN2_HF_ASSETS_PATH="${EN1_EN2_HF_ASSETS_PATH:-${MULTILINGUAL_PRETRAINING_ROOT}/assets/65k_paired}"
export EN1_EN2_OUTPUT_ROOT="${EN1_EN2_OUTPUT_ROOT:-${MULTILINGUAL_PRETRAINING_ROOT}/outputs/torchtitan}"
export HF_HOME="${HF_HOME:-${MULTILINGUAL_PRETRAINING_ROOT}/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${MULTILINGUAL_PRETRAINING_ROOT}/torch_home}"
export NLTK_DATA="${NLTK_DATA:-${MULTILINGUAL_PRETRAINING_ROOT}/nltk_data}"

export MIXED_DATA_FRACTION="${MIXED_DATA_FRACTION:-${EN1_EN2_MIXED_DATA_FRACTION:-0.01}}"
export EN1_EN2_MIXED_DATA_FRACTION="${MIXED_DATA_FRACTION}"
export EN1_EN2_STAGE1_STEPS="${EN1_EN2_STAGE1_STEPS:-3000}"
export EN1_EN2_CLEAN_STEPS="${EN1_EN2_CLEAN_STEPS:-1000}"
export NGPU="${NGPU:-4}"
export MODULE="${MODULE:-llama3}"
export MASTER_PORT="${MASTER_PORT:-29600}"
export LOG_RANK="${LOG_RANK:-0}"
export TORCHINDUCTOR_MAX_AUTOTUNE="${TORCHINDUCTOR_MAX_AUTOTUNE:-1}"
export LOGLEVEL="${LOGLEVEL:-INFO}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

export WANDB_PROJECT="${WANDB_PROJECT:-multilingual-pretraining}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-en1en2_sentence_mix${MIXED_DATA_FRACTION}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-en1en2_${VARIANT}_mix${MIXED_DATA_FRACTION}_${SLURM_JOB_ID:-manual}}"
export WANDB_RUN_JOB_TYPE="${WANDB_RUN_JOB_TYPE:-train}"
export WANDB_RUN_TAGS="${WANDB_RUN_TAGS:-en1-en2,sentence,${VARIANT},mix-${MIXED_DATA_FRACTION}}"
export WANDB_DIR="${WANDB_DIR:-${EN1_EN2_OUTPUT_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/cache}"
export WANDB_INIT_TIMEOUT_SECONDS="${WANDB_INIT_TIMEOUT_SECONDS:-300}"
export WANDB_REQUIRE_RUN="${WANDB_REQUIRE_RUN:-1}"

mkdir -p \
  "${EN1_EN2_OUTPUT_ROOT}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${NLTK_DATA}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}"

required_paths=(
  "${FINEWEB_TRANSLATED_ROOT}/en-original"
  "${FICTIONAL_ENTITY_DATA_ROOT}/0/en_data.jsonl"
  "${EN1_EN2_HF_ASSETS_PATH}"
  "${NLTK_DATA}/tokenizers/punkt"
  "${NLTK_DATA}/tokenizers/punkt_tab/english"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

echo "Root: ${MULTILINGUAL_PRETRAINING_ROOT}"
echo "FineWeb translated data: ${FINEWEB_TRANSLATED_ROOT}"
echo "Fictional entity data: ${FICTIONAL_ENTITY_DATA_ROOT}"
echo "Tokenizer assets: ${EN1_EN2_HF_ASSETS_PATH}"
echo "Output root: ${EN1_EN2_OUTPUT_ROOT}"
echo "WandB project: ${WANDB_PROJECT}"
echo "WandB group: ${WANDB_RUN_GROUP}"
echo "WandB run name: ${WANDB_RUN_NAME}"
echo "Starting training only: VARIANT=${VARIANT}, CONFIG=${CONFIG}, mixed_data_fraction=${MIXED_DATA_FRACTION}"

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "Precheck succeeded; exiting before training because PRECHECK_ONLY=1."
  exit 0
fi

.venv/bin/torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${NGPU}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  --local-ranks-filter "${LOG_RANK}" \
  --role rank \
  --tee 3 \
  -m torchtitan.train \
  --module "${MODULE}" \
  --config "${CONFIG}" \
  "$@"
