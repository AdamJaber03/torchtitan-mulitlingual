#!/usr/bin/env bash
#SBATCH --job-name=en1en2_sentence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --account=ug_schwartz
#SBATCH --nodelist=mm-lab02
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

# Usage examples:
#   sbatch --export=ALL,X=0.01 en1_en2_sentence_experiments.slurm
#   sbatch --export=ALL,X=0.6 en1_en2_sentence_experiments.slurm
#   sbatch --export=ALL,X=0.25,RUN_EVAL=1 en1_en2_sentence_experiments.slurm
#   sbatch --export=ALL,VARIANT=sentence_parallel_doc_order,X=0.6 en1_en2_sentence_experiments.slurm
#
# Before the first run in a fresh environment:
#   uv pip install -e .
#   python -m nltk.downloader -d "${HOME}/nltk_data" punkt punkt_tab
#
# VARIANT values:
#   sentence_wise_code_switching
#   sentence_parallel_doc_order
#   sentence_parallel_sentence_order

VARIANTS=(
  "sentence_wise_code_switching"
  "sentence_parallel_doc_order"
  "sentence_parallel_sentence_order"
)

if [[ -n "${VARIANT:-}" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID:-0}" != "0" ]]; then
    echo "VARIANT=${VARIANT} was set; skipping duplicate array task ${SLURM_ARRAY_TASK_ID}."
    exit 0
  fi
else
  VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
fi

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

TORCHTITAN_ROOT="${TORCHTITAN_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${TORCHTITAN_ROOT}"

NGPU="${NGPU:-8}"
MODULE="${MODULE:-llama3}"
X="${X:-${EN1_EN2_PARALLEL_DOC_FRACTION:-${EN1_EN2_PARALLEL_DOC_PERCENT:-0.6}}}"

export EN1_EN2_PARALLEL_DOC_FRACTION="${X}"
export EN1_EN2_STAGE1_STEPS="${EN1_EN2_STAGE1_STEPS:-3000}"
export EN1_EN2_CLEAN_STEPS="${EN1_EN2_CLEAN_STEPS:-1000}"
export EN1_EN2_HF_ASSETS_PATH="${EN1_EN2_HF_ASSETS_PATH:-/home/adamga/torchtitan/tests/assets/65k_paired}"
export FICTIONAL_ENTITY_DATA_ROOT="${FICTIONAL_ENTITY_DATA_ROOT:-/home/adamga/torchtitan/fictional_entity_data/gemini_seeds}"
export EN1_EN2_OUTPUT_ROOT="${EN1_EN2_OUTPUT_ROOT:-/home/adamga/leshemg/adamga/train/torchtitan}"
export HF_HOME="${HF_HOME:-/home/adamga/leshemg/adamga/hf_home}"
export NLTK_DATA="${NLTK_DATA:-${HOME}/nltk_data}"
export TORCHINDUCTOR_MAX_AUTOTUNE="${TORCHINDUCTOR_MAX_AUTOTUNE:-1}"
export LOGLEVEL="${LOGLEVEL:-INFO}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

X_PERCENT="$(awk -v x="${X}" 'BEGIN { printf "%g", x * 100.0 }')"
X_TAG="${X_PERCENT//./p}"

if [[ -n "${ENV_SETUP:-}" ]]; then
  # Optional: point ENV_SETUP at a shell file that activates your venv/modules.
  # shellcheck source=/dev/null
  source "${ENV_SETUP}"
fi

echo "Starting ${VARIANT} with X=${X_PERCENT}% using CONFIG=${CONFIG}"

srun torchrun \
  --standalone \
  --nproc_per_node="${NGPU}" \
  --local-ranks-filter "${LOG_RANK:-0}" \
  --role rank \
  --tee 3 \
  -m torchtitan.train \
  --module "${MODULE}" \
  --config "${CONFIG}" \
  "$@"

RUN_EVAL="${RUN_EVAL:-0}"
if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "Training finished. Set RUN_EVAL=1 to run the optional lm-eval section."
  exit 0
fi

TOTAL_STEPS="$((EN1_EN2_STAGE1_STEPS + EN1_EN2_CLEAN_STEPS))"
RUN_DIR="${EN1_EN2_OUTPUT_ROOT}/smollm2_360m_en1_en2_${VARIANT}_x${X_TAG}_stage1_${EN1_EN2_STAGE1_STEPS}_stage2_${EN1_EN2_CLEAN_STEPS}"
TT_CHECKPOINT="${TT_CHECKPOINT:-${RUN_DIR}/step-${TOTAL_STEPS}}"
HF_OUTPUT_ROOT="${HF_OUTPUT_ROOT:-${RUN_DIR}/hf_eval}"

echo "Preparing eval from TorchTitan checkpoint: ${TT_CHECKPOINT}"

if [[ "${RUN_CONVERT:-1}" == "1" ]]; then
  python -m scripts.checkpoint_conversion.convert_to_hf \
    "${TT_CHECKPOINT}" \
    "${HF_OUTPUT_ROOT}/en1" \
    --model_name llama3 \
    --model_flavor smollm2_360m_2xvocab \
    --hf_assets_path "${EN1_EN2_HF_ASSETS_PATH}" \
    --export_dtype "${HF_EXPORT_DTYPE:-bfloat16}" \
    --extract_vocab base

  python -m scripts.checkpoint_conversion.convert_to_hf \
    "${TT_CHECKPOINT}" \
    "${HF_OUTPUT_ROOT}/en2" \
    --model_name llama3 \
    --model_flavor smollm2_360m_2xvocab \
    --hf_assets_path "${EN1_EN2_HF_ASSETS_PATH}" \
    --export_dtype "${HF_EXPORT_DTYPE:-bfloat16}" \
    --extract_vocab tagged
fi

if [[ "${GENERATE_EVAL_TASKS:-0}" == "1" ]]; then
  python fictional_entity_data/eval_gen.py
fi

run_eval_for_model() {
  local lang="$1"
  local model_dir="$2"

  if [[ -n "${LM_EVAL_CMD_TEMPLATE:-}" ]]; then
    EN1_EN2_EVAL_LANG="${lang}" \
    EN1_EN2_EVAL_MODEL_DIR="${model_dir}" \
    EN1_EN2_EVAL_OUTPUT_DIR="${RUN_DIR}/lm_eval_${lang}" \
      bash -lc "${LM_EVAL_CMD_TEMPLATE}"
    return
  fi

  if [[ -z "${LM_EVAL_TASKS:-}" || -z "${LM_EVAL_INCLUDE_PATH:-}" ]]; then
    echo "Skipping ${lang} eval: set LM_EVAL_TASKS and LM_EVAL_INCLUDE_PATH, or LM_EVAL_CMD_TEMPLATE." >&2
    return
  fi

  python -m lm_eval \
    --model hf \
    --model_args "pretrained=${model_dir},tokenizer=${EN1_EN2_HF_ASSETS_PATH},dtype=${HF_EVAL_DTYPE:-bfloat16}" \
    --tasks "${LM_EVAL_TASKS}" \
    --include_path "${LM_EVAL_INCLUDE_PATH}" \
    --batch_size "${LM_EVAL_BATCH_SIZE:-auto}" \
    --output_path "${RUN_DIR}/lm_eval_${lang}"
}

run_eval_for_model "en1" "${HF_OUTPUT_ROOT}/en1"
run_eval_for_model "en2" "${HF_OUTPUT_ROOT}/en2"
