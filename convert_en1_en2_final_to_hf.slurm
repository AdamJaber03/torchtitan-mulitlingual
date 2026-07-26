#!/usr/bin/env bash
#SBATCH --job-name=en1en2_hf_convert
#SBATCH --partition=cpu512G-48h
#SBATCH --account=ug_hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/knowledge_sharing_conversion/%x_%j.out
#SBATCH --error=logs/knowledge_sharing_conversion/%x_%j.err

set -euo pipefail

# Example:
#   sbatch --export=ALL,EXPERIMENT_NAME=smollm2_360m_en1_en2_sentence_wise_code_switching_mixed_data1pct_stage1_3000_stage2_1000 convert_en1_en2_final_to_hf.slurm
#
# Outputs:
#   $MULTILINGUAL_PRETRAINING_ROOT/evals/knowledge_sharing/hf_models/$EXPERIMENT_NAME/step-$STEP/en1
#   $MULTILINGUAL_PRETRAINING_ROOT/evals/knowledge_sharing/hf_models/$EXPERIMENT_NAME/step-$STEP/en2

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

export MULTILINGUAL_PRETRAINING_ROOT="${MULTILINGUAL_PRETRAINING_ROOT:-$(cd "${TORCHTITAN_ROOT}/.." && pwd)}"
export EN1_EN2_OUTPUT_ROOT="${EN1_EN2_OUTPUT_ROOT:-${MULTILINGUAL_PRETRAINING_ROOT}/outputs/torchtitan}"
export EN1_EN2_HF_ASSETS_PATH="${EN1_EN2_HF_ASSETS_PATH:-${MULTILINGUAL_PRETRAINING_ROOT}/assets/65k_paired}"
export HF_HOME="${HF_HOME:-${MULTILINGUAL_PRETRAINING_ROOT}/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${MULTILINGUAL_PRETRAINING_ROOT}/torch_home}"

STEP="${STEP:-4000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${EN1_EN2_OUTPUT_ROOT}/${EXPERIMENT_NAME}/step-${STEP}}"
EVAL_ROOT="${EVAL_ROOT:-${MULTILINGUAL_PRETRAINING_ROOT}/evals/knowledge_sharing}"
HF_MODEL_ROOT="${HF_MODEL_ROOT:-${EVAL_ROOT}/hf_models/${EXPERIMENT_NAME}/step-${STEP}}"
EXPORT_DTYPE="${EXPORT_DTYPE:-bfloat16}"
FORCE="${FORCE:-0}"

required_paths=(
  "${CHECKPOINT_DIR}/.metadata"
  "${EN1_EN2_HF_ASSETS_PATH}/tokenizer.json"
  "${EN1_EN2_HF_ASSETS_PATH}/tokenizer_config.json"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${HF_MODEL_ROOT}" "${HF_HOME}" "${TORCH_HOME}"

prepare_output_dir() {
  local out_dir="$1"
  if [[ -d "${out_dir}" && -n "$(find "${out_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    if [[ "${FORCE}" == "1" ]]; then
      rm -rf "${out_dir}"
    else
      echo "Output directory already exists and is non-empty: ${out_dir}" >&2
      echo "Set FORCE=1 to replace it." >&2
      exit 1
    fi
  fi
  mkdir -p "${out_dir}"
}

write_hf_metadata() {
  local out_dir="$1"
  cp -L "${EN1_EN2_HF_ASSETS_PATH}/tokenizer.json" "${out_dir}/tokenizer.json"
  cp -L "${EN1_EN2_HF_ASSETS_PATH}/tokenizer_config.json" "${out_dir}/tokenizer_config.json"

  .venv/bin/python - "${out_dir}" "${EXPORT_DTYPE}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
export_dtype = sys.argv[2]

config = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 960,
    "intermediate_size": 2560,
    "num_hidden_layers": 32,
    "num_attention_heads": 15,
    "num_key_value_heads": 5,
    "vocab_size": 65536,
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": True,
    "torch_dtype": export_dtype,
    "use_cache": True,
    "bos_token_id": None,
    "eos_token_id": 0,
    "pad_token_id": 0,
}

generation_config = {
    "_from_model_config": True,
    "eos_token_id": 0,
    "pad_token_id": 0,
}

tokenizer_config_path = out_dir / "tokenizer_config.json"
with tokenizer_config_path.open() as f:
    tokenizer_config = json.load(f)
tokenizer_config.setdefault("eos_token", "<|endoftext|>")
tokenizer_config["pad_token"] = tokenizer_config["eos_token"]
tokenizer_config["eos_token_id"] = 0
tokenizer_config["pad_token_id"] = 0

special_tokens_map = {
    "eos_token": tokenizer_config["eos_token"],
    "pad_token": tokenizer_config["pad_token"],
}

for filename, data in [
    ("config.json", config),
    ("generation_config.json", generation_config),
    ("tokenizer_config.json", tokenizer_config),
    ("special_tokens_map.json", special_tokens_map),
]:
    with (out_dir / filename).open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
PY
}

convert_view() {
  local view="$1"
  local extract_vocab="$2"
  local out_dir="${HF_MODEL_ROOT}/${view}"

  prepare_output_dir "${out_dir}"
  echo "Converting ${EXPERIMENT_NAME} step-${STEP} to ${view} HF view at ${out_dir}"

  .venv/bin/python scripts/checkpoint_conversion/convert_to_hf.py \
    "${CHECKPOINT_DIR}" \
    "${out_dir}" \
    --hf_assets_path "${EN1_EN2_HF_ASSETS_PATH}" \
    --model_name llama3 \
    --model_flavor smollm2_360m_2xvocab \
    --export_dtype "${EXPORT_DTYPE}" \
    --extract_vocab "${extract_vocab}"

  write_hf_metadata "${out_dir}"

  shopt -s nullglob
  local safetensors=("${out_dir}"/*.safetensors)
  shopt -u nullglob
  if (( ${#safetensors[@]} == 0 )); then
    echo "Conversion did not produce safetensors under ${out_dir}" >&2
    exit 1
  fi
}

echo "Root: ${MULTILINGUAL_PRETRAINING_ROOT}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "HF output root: ${HF_MODEL_ROOT}"
echo "Export dtype: ${EXPORT_DTYPE}"

convert_view "en1" "base"
convert_view "en2" "tagged"

echo "HF conversion completed:"
echo "  en1: ${HF_MODEL_ROOT}/en1"
echo "  en2: ${HF_MODEL_ROOT}/en2"
