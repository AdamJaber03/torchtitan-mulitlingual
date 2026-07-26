#!/usr/bin/env bash
#SBATCH --job-name=en1en2_ks_eval
#SBATCH --partition=p_b200_schwartz
#SBATCH --account=ug_schwartz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/knowledge_sharing_eval/%x_%j.out
#SBATCH --error=logs/knowledge_sharing_eval/%x_%j.err

set -euo pipefail

# Example:
#   sbatch --export=ALL,EXPERIMENT_NAME=smollm2_360m_en1_en2_sentence_wise_code_switching_mixed_data1pct_stage1_3000_stage2_1000,LANG_VIEW=en1 run_en1_en2_knowledge_eval.slurm
#
# Outputs:
#   $MULTILINGUAL_PRETRAINING_ROOT/evals/knowledge_sharing/results/$EXPERIMENT_NAME/step-$STEP/$LANG_VIEW/results_*.json
#   $MULTILINGUAL_PRETRAINING_ROOT/evals/knowledge_sharing/results/$EXPERIMENT_NAME/step-$STEP/$LANG_VIEW/results_*.png

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

if [[ -z "${EXPERIMENT_NAME:-}" ]]; then
  echo "Set EXPERIMENT_NAME to the training output directory name." >&2
  exit 2
fi

LANG_VIEW="${LANG_VIEW:-}"
if [[ "${LANG_VIEW}" != "en1" && "${LANG_VIEW}" != "en2" ]]; then
  echo "Set LANG_VIEW to either en1 or en2." >&2
  exit 2
fi

if [[ -n "${LM_EVAL_BIN:-}" ]]; then
  :
elif command -v lm_eval >/dev/null 2>&1; then
  LM_EVAL_BIN="$(command -v lm_eval)"
elif [[ -x ".venv/bin/lm_eval" ]]; then
  LM_EVAL_BIN=".venv/bin/lm_eval"
elif command -v lm-eval >/dev/null 2>&1; then
  LM_EVAL_BIN="$(command -v lm-eval)"
elif [[ -x ".venv/bin/lm-eval" ]]; then
  LM_EVAL_BIN=".venv/bin/lm-eval"
else
  echo "lm_eval/lm-eval is not installed in the active environment." >&2
  echo "Install lm-evaluation-harness with the HF backend before submitting eval jobs." >&2
  exit 1
fi

lm_eval_cmd=("${LM_EVAL_BIN}")
if [[ "${LM_EVAL_SUBCOMMAND:-auto}" == "run" ]]; then
  lm_eval_cmd=("${LM_EVAL_BIN}" run)
elif [[ "${LM_EVAL_SUBCOMMAND:-auto}" == "none" ]]; then
  lm_eval_cmd=("${LM_EVAL_BIN}")
elif "${LM_EVAL_BIN}" run -h >/dev/null 2>&1; then
  lm_eval_cmd=("${LM_EVAL_BIN}" run)
fi

export MULTILINGUAL_PRETRAINING_ROOT="${MULTILINGUAL_PRETRAINING_ROOT:-$(cd "${TORCHTITAN_ROOT}/.." && pwd)}"
export HF_HOME="${HF_HOME:-${MULTILINGUAL_PRETRAINING_ROOT}/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${MULTILINGUAL_PRETRAINING_ROOT}/torch_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

STEP="${STEP:-4000}"
EVAL_ROOT="${EVAL_ROOT:-${MULTILINGUAL_PRETRAINING_ROOT}/evals/knowledge_sharing}"
TASK_DIR="${TASK_DIR:-${EVAL_ROOT}/lm_eval_tasks/en1_en2_mcq_en}"
TASK_GROUP="${TASK_GROUP:-fictive_entity_en1_en2_mcq_en_all_rates}"
MODEL_DIR="${MODEL_DIR:-${EVAL_ROOT}/hf_models/${EXPERIMENT_NAME}/step-${STEP}/${LANG_VIEW}}"
RESULTS_DIR="${RESULTS_DIR:-${EVAL_ROOT}/results/${EXPERIMENT_NAME}/step-${STEP}/${LANG_VIEW}}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
LIMIT="${LIMIT:-}"
GENERATE_TASKS="${GENERATE_TASKS:-0}"
PLOT_RESULTS="${PLOT_RESULTS:-1}"
PLOT_METRIC="${PLOT_METRIC:-acc}"
KT_METRIC="${KT_METRIC:-acc_norm}"
EXISTING_RESULTS_POLICY="${EXISTING_RESULTS_POLICY:-error}"

if [[ -d "${RESULTS_DIR}" ]]; then
  mapfile -d '' existing_outputs < <(
    find "${RESULTS_DIR}" -type f -print0
  )
else
  existing_outputs=()
fi

if (( ${#existing_outputs[@]} > 0 )); then
  case "${EXISTING_RESULTS_POLICY}" in
    skip-complete)
      json_count="$(
        find "${RESULTS_DIR}" -type f -name 'results_*.json' -print | wc -l
      )"
      png_count="$(
        find "${RESULTS_DIR}" -type f -name '*.png' -print | wc -l
      )"
      if (( json_count > 0 && (PLOT_RESULTS != 1 || png_count > 0) )); then
        echo "Skipping complete existing evaluation: ${RESULTS_DIR}"
        exit 0
      fi
      echo "Refusing to skip partial evaluation output: ${RESULTS_DIR}" >&2
      exit 1
      ;;
    error)
      echo "Refusing to overwrite existing evaluation output: ${RESULTS_DIR}" >&2
      printf '  %s\n' "${existing_outputs[@]}" >&2
      exit 1
      ;;
    *)
      echo "EXISTING_RESULTS_POLICY must be error or skip-complete." >&2
      exit 2
      ;;
  esac
fi

mkdir -p "${TASK_DIR}" "${RESULTS_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${HF_DATASETS_CACHE}"

if [[ "${GENERATE_TASKS}" == "1" ]]; then
  echo "Regenerating task YAMLs because GENERATE_TASKS=1"
  .venv/bin/python fictional_entity_data/eval_gen.py \
    --output-dir "${TASK_DIR}" \
    --group-name "${TASK_GROUP}" \
    --mcq-files mcq_en
fi

required_paths=(
  "${MODEL_DIR}/config.json"
  "${MODEL_DIR}/tokenizer.json"
  "${MODEL_DIR}/tokenizer_config.json"
  "${TASK_DIR}/_group.yaml"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

shopt -s nullglob
safetensors=("${MODEL_DIR}"/*.safetensors)
shopt -u nullglob
if (( ${#safetensors[@]} == 0 )); then
  echo "Missing safetensors weights under ${MODEL_DIR}" >&2
  exit 1
fi

MODEL_ARGS="pretrained=${MODEL_DIR},dtype=${MODEL_DTYPE}"

lm_eval_args=(
  --model hf
  --model_args "${MODEL_ARGS}"
  --tasks "${TASK_GROUP}"
  --include_path "${TASK_DIR}"
  --device cuda:0
  --batch_size "${BATCH_SIZE}"
  --output_path "${RESULTS_DIR}"
)

if [[ -n "${LIMIT}" ]]; then
  lm_eval_args+=(--limit "${LIMIT}")
fi

echo "Root: ${MULTILINGUAL_PRETRAINING_ROOT}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Language view: ${LANG_VIEW}"
echo "Model dir: ${MODEL_DIR}"
echo "Task dir: ${TASK_DIR}"
echo "Task group: ${TASK_GROUP}"
echo "Results dir: ${RESULTS_DIR}"
echo "lm_eval: ${lm_eval_cmd[*]}"

"${lm_eval_cmd[@]}" "${lm_eval_args[@]}"

echo "Evaluation completed: ${RESULTS_DIR}"

if [[ "${PLOT_RESULTS}" == "1" ]]; then
  echo "Plotting ${PLOT_METRIC} results in ${RESULTS_DIR} with KT from ${KT_METRIC}"
  .venv/bin/python scripts/plot_knowledge_sharing_results.py \
    --metric "${PLOT_METRIC}" \
    --kt-metric "${KT_METRIC}" \
    --view "${LANG_VIEW}" \
    "${RESULTS_DIR}"
fi
