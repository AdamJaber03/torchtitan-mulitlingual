#!/bin/bash
# Submit bsub jobs for:
#   - Main branch upload (2 jobs, one per config)
#   - Intermediate step conversion+upload (12 jobs, 6 steps x 2 configs)

PROJ=/gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
VENV=$PROJ/.venv
UPLOAD_DIR=$PROJ/upload_scripts
LOG_DIR=$PROJ/outputs/hf_upload_logs
mkdir -p "$LOG_DIR"

# Main branch jobs (network I/O only, no GPU needed)
for CONFIG in en_ar en_translated_ar; do
  JOB_NAME="hf_main_${CONFIG}"
  LOG_FILE="$LOG_DIR/${JOB_NAME}.log"
  echo "Submitting: $JOB_NAME"
  bsub \
    -q normal \
    -G grp_exploratory \
    -J "$JOB_NAME" \
    -n 1 \
    -R "rusage[mem=8GB]" \
    -o "$LOG_FILE" \
    -e "$LOG_FILE" \
    "$VENV/bin/python" "$UPLOAD_DIR/upload_main_branch.py" "$CONFIG"
done

# Intermediate step jobs (need CPU/RAM for conversion, no GPU)
STEPS="5000 9500 14500 19000 24000 28000"
for CONFIG in en_ar en_translated_ar; do
  for STEP in $STEPS; do
    JOB_NAME="hf_step_${CONFIG}_${STEP}"
    LOG_FILE="$LOG_DIR/${JOB_NAME}.log"
    echo "Submitting: $JOB_NAME"
    bsub \
      -q preemptable \
      -G grp_preemptable \
      -J "$JOB_NAME" \
      -n 4 \
      -R "rusage[mem=80GB]" \
      -o "$LOG_FILE" \
      -e "$LOG_FILE" \
      "$VENV/bin/python" "$UPLOAD_DIR/convert_and_upload_step.py" "$CONFIG" "$STEP"
  done
done

echo ""
echo "All 14 jobs submitted. Logs: $LOG_DIR"
echo "Monitor with: bjobs -noheader | grep hf_"
