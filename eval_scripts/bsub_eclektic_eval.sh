#!/bin/bash
# Run ECLeKTic evaluation on the cluster.
#
# Pre-requisite: download the dataset once:
#   pip install kaggle
#   kaggle datasets download -d googleai/eclektic -p $ECLEKTIC_DATA --unzip
#
# Usage:
#   bash eval_scripts/bsub_eclektic_eval.sh <hf_model_dir> <eclektic_data_dir> <output_dir>
#
# Example (baseline model):
#   bash eval_scripts/bsub_eclektic_eval.sh \
#       /gpfs/.../outputs/hf_export/llama3_7b_en1_en2_baseline \
#       /gpfs/.../data/eclektic \
#       /gpfs/.../outputs/lm_eval_results/eclektic_baseline

set -euo pipefail

MODEL_DIR=${1:?Usage: $0 <model_dir> <data_dir> <output_dir>}
DATA_DIR=${2:?Usage: $0 <model_dir> <data_dir> <output_dir>}
OUTPUT_DIR=${3:?Usage: $0 <model_dir> <data_dir> <output_dir>}

PROJ=/gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
VENV=$PROJ/.venv
SCRIPT=$PROJ/eval_scripts/eclektic_eval.py

mkdir -p "$OUTPUT_DIR"

bsub \
  -q normal \
  -n 4 \
  -R "rusage[mem=64000] span[hosts=1]" \
  -gpu "num=1:mode=exclusive_process" \
  -o "$OUTPUT_DIR/eclektic_%J.log" \
  -J "eclektic_eval" \
  bash -c "
source $VENV/bin/activate
python $SCRIPT \
  --model $MODEL_DIR \
  --data $DATA_DIR \
  --output $OUTPUT_DIR \
  --batch_size 8 \
  --max_new_tokens 64
"

echo "Submitted ECLeKTic eval for model: $MODEL_DIR"
echo "Output: $OUTPUT_DIR"
