#!/bin/bash
# Submit bsub jobs to convert+upload each intermediate step checkpoint to HuggingFace.

PROJ=/gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
VENV=$PROJ/.venv
UPLOAD_DIR=$PROJ/upload_scripts
LOG_DIR=$PROJ/outputs/hf_upload_logs
mkdir -p "$LOG_DIR"

STEPS="5000 9500 14500 19000 24000 28000"
CONFIGS="en_ar en_translated_ar"

for CONFIG in $CONFIGS; do
  for STEP in $STEPS; do
    JOB_NAME="hf_upload_${CONFIG}_step${STEP}"
    LOG_FILE="$LOG_DIR/${JOB_NAME}.log"

    echo "Submitting: $JOB_NAME"
    bsub \
      -q preemptable \
      -G grp_preemptable \
      -J "$JOB_NAME" \
      -n 4 \
      -R "rusage[mem=60GB]" \
      -o "$LOG_FILE" \
      -e "$LOG_FILE" \
      "$VENV/bin/python" "$UPLOAD_DIR/convert_and_upload_step.py" "$CONFIG" "$STEP"
  done
done

echo ""
echo "All jobs submitted. Check logs in: $LOG_DIR"
echo "Monitor with: bjobs -noheader | grep hf_upload"
