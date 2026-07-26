#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUN_TAG="${RUN_TAG:-injcountfix_gemini_s43}"
FRACTIONS="${FRACTIONS:-0.01 0.10 0.60}"
VARIANTS="${VARIANTS:-sentence_wise_code_switching sentence_parallel_doc_order sentence_parallel_sentence_order}"
SUBMIT_FOLLOWUPS="${SUBMIT_FOLLOWUPS:-1}"

GPU_PARTITION="${GPU_PARTITION:-p_b200_schwartz}"
GPU_ACCOUNT="${GPU_ACCOUNT:-ug_schwartz}"
GPU_NODE="${GPU_NODE:-}"
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"
TRANSLATION_TIME="${TRANSLATION_TIME:-24:00:00}"

WANDB_PROJECT="${WANDB_PROJECT:-multilingual-pretraining}"
MANIFEST_ROOT="${MANIFEST_ROOT:-logs/en1en2_sentence/submissions}"
SUBMISSION_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST_DIR="${MANIFEST_ROOT}/${RUN_TAG}_${SUBMISSION_TAG}"
CONFIG_DIR="${MANIFEST_DIR}/configs"
JOB_MANIFEST="${MANIFEST_DIR}/jobs.tsv"
GIT_SHA="$(git rev-parse HEAD)"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to submit from a dirty working tree." >&2
  echo "Commit and verify the exact training code first." >&2
  exit 1
fi

mkdir -p \
  "${CONFIG_DIR}" \
  logs/en1en2_sentence \
  logs/en1en2_translation_val \
  logs/knowledge_sharing_conversion \
  logs/knowledge_sharing_eval

printf '%s\n' \
  $'experiment\tvariant\tfraction\tgit_sha\tgpu_partition\tgpu_account\tgpu_node\twandb_project\twandb_run_id\ttrain_job\ttranslation_job\tconvert_job\teval_en1\teval_en2' \
  > "${JOB_MANIFEST}"

read -r -a FRACTION_ARRAY <<< "${FRACTIONS}"
read -r -a VARIANT_ARRAY <<< "${VARIANTS}"

gpu_sbatch_args=(
  "--partition=${GPU_PARTITION}"
  "--account=${GPU_ACCOUNT}"
)
if [[ -n "${GPU_NODE}" ]]; then
  gpu_sbatch_args+=("--nodelist=${GPU_NODE}")
fi

fraction_tag() {
  case "$1" in
    0.01|.01) echo "1" ;;
    0.1|0.10|.1) echo "10" ;;
    0.6|0.60|.6) echo "60" ;;
    *)
      echo "Unsupported fraction '$1'; use 0.01, 0.10, or 0.60." >&2
      return 2
      ;;
  esac
}

variant_config() {
  case "$1" in
    sentence_wise_code_switching)
      echo "smollm2_360m_en1_en2_sentence_wise_code_switching"
      ;;
    sentence_parallel_doc_order)
      echo "smollm2_360m_en1_en2_sentence_parallel_doc_order"
      ;;
    sentence_parallel_sentence_order)
      echo "smollm2_360m_en1_en2_sentence_parallel_sentence_order"
      ;;
    *)
      echo "Unsupported variant '$1'." >&2
      return 2
      ;;
  esac
}

variant_short() {
  case "$1" in
    sentence_wise_code_switching) echo "cs" ;;
    sentence_parallel_doc_order) echo "doc" ;;
    sentence_parallel_sentence_order) echo "sent" ;;
  esac
}

job_id() {
  local submission="$1"
  echo "${submission%%;*}"
}

submit_index=0
for fraction in "${FRACTION_ARRAY[@]}"; do
  pct="$(fraction_tag "${fraction}")"
  for variant in "${VARIANT_ARRAY[@]}"; do
    config="$(variant_config "${variant}")"
    short="$(variant_short "${variant}")"
    experiment="smollm2_360m_en1_en2_${variant}_mixed_data${pct}pct_stage1_3000_stage2_1000_${RUN_TAG}"
    wandb_run_id="$(
      .venv/bin/python -c 'import wandb; print(wandb.util.generate_id())'
    )"
    train_port="$((29600 + submit_index))"
    translation_port="$((30600 + submit_index))"
    config_manifest="${CONFIG_DIR}/${experiment}.json"

    echo "Prechecking ${experiment}"
    EN1_EN2_MIXED_DATA_FRACTION="${fraction}" \
    EN1_EN2_EXPERIMENT_NAME="${experiment}" \
    EN1_EN2_INCLUDE_HUMAN_ENTITIES=0 \
      .venv/bin/python scripts/inspect_en1_en2_sentence_config.py \
        --config "${config}" \
        --output "${config_manifest}" \
        >/dev/null

    VARIANT="${variant}" \
    MIXED_DATA_FRACTION="${fraction}" \
    EN1_EN2_EXPERIMENT_NAME="${experiment}" \
    EN1_EN2_INCLUDE_HUMAN_ENTITIES=0 \
    PRECHECK_ONLY=1 \
      bash en1_en2_sentence_train.slurm >/dev/null

    train_submission="$(
      sbatch --parsable \
        "${gpu_sbatch_args[@]}" \
        "--time=${TRAIN_TIME}" \
        "--job-name=sp_${short}${pct}_train" \
        "--export=ALL,VARIANT=${variant},MIXED_DATA_FRACTION=${fraction},EN1_EN2_EXPERIMENT_NAME=${experiment},EN1_EN2_INCLUDE_HUMAN_ENTITIES=0,WANDB_PROJECT=${WANDB_PROJECT},WANDB_RUN_ID=${wandb_run_id},WANDB_RUN_NAME=${experiment},WANDB_RUN_GROUP=${RUN_TAG},MASTER_PORT=${train_port}" \
        en1_en2_sentence_train.slurm
    )"
    train_job="$(job_id "${train_submission}")"

    translation_job=""
    convert_job=""
    eval_en1=""
    eval_en2=""

    if [[ "${SUBMIT_FOLLOWUPS}" == "1" ]]; then
      translation_submission="$(
        sbatch --parsable \
          "${gpu_sbatch_args[@]}" \
          "--time=${TRANSLATION_TIME}" \
          "--dependency=afterok:${train_job}" \
          "--job-name=sp_${short}${pct}_trans" \
          "--export=ALL,VARIANTS=${variant},MIXED_DATA_FRACTIONS=${fraction},EXPERIMENT_NAME=${experiment},CHECKPOINT_STEPS=1 500 1000 1500 2000 2500 3000 3500 4000,WANDB_PROJECT=${WANDB_PROJECT},WANDB_RUN_ID=${wandb_run_id},MASTER_PORT=${translation_port},TRANSLATION_EXISTING_POLICY=skip-complete" \
          run_en1_en2_translation_validation.slurm
      )"
      translation_job="$(job_id "${translation_submission}")"

      convert_submission="$(
        sbatch --parsable \
          "--dependency=afterok:${train_job}" \
          "--job-name=sp_${short}${pct}_convert" \
          "--export=ALL,EXPERIMENT_NAME=${experiment},STEP=4000" \
          convert_en1_en2_final_to_hf.slurm
      )"
      convert_job="$(job_id "${convert_submission}")"

      eval_en1="$(job_id "$(
        sbatch --parsable \
          "${gpu_sbatch_args[@]}" \
          "--dependency=afterok:${convert_job}" \
          "--job-name=sp_${short}${pct}_eval1" \
          "--export=ALL,EXPERIMENT_NAME=${experiment},STEP=4000,LANG_VIEW=en1" \
          run_en1_en2_knowledge_eval.slurm
      )")"
      eval_en2="$(job_id "$(
        sbatch --parsable \
          "${gpu_sbatch_args[@]}" \
          "--dependency=afterok:${convert_job}" \
          "--job-name=sp_${short}${pct}_eval2" \
          "--export=ALL,EXPERIMENT_NAME=${experiment},STEP=4000,LANG_VIEW=en2" \
          run_en1_en2_knowledge_eval.slurm
      )")"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${experiment}" \
      "${variant}" \
      "${fraction}" \
      "${GIT_SHA}" \
      "${GPU_PARTITION}" \
      "${GPU_ACCOUNT}" \
      "${GPU_NODE}" \
      "${WANDB_PROJECT}" \
      "${wandb_run_id}" \
      "${train_job}" \
      "${translation_job}" \
      "${convert_job}" \
      "${eval_en1}" \
      "${eval_en2}" \
      >> "${JOB_MANIFEST}"

    echo "Submitted ${experiment}: train=${train_job} translation=${translation_job}"
    submit_index="$((submit_index + 1))"
  done
done

echo "Submission manifest: ${JOB_MANIFEST}"
