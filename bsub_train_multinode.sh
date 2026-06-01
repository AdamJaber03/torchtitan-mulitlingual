#!/bin/bash
#BSUB -J torchtitan_multilingual_training  # Name of your job
#BSUB -q leshem.q                          # Your specific partition/queue
#BSUB -n 32                                # TOTAL CPU cores requested (32 cores * 1 node) CHANGE THIS TO 128 FOR 4 NODES...
#BSUB -R "span[ptile=32]"                  # Pack exactly 32 CPU cores per node
#BSUB -gpu "num=8:mode=exclusive_process"  # Request 8 GPUs per node
#BSUB -R "rusage[mem=400G]"                # Memory request per node
#BSUB -o logs/%J_train.out                 # Standard output log (%J is LSF's JobID)
#BSUB -e logs/%J_train.err                 # Error log

# 1. Navigate to your project directory
# cd /home/adamga/torchtitan

# 2. Activate your environment
source .venv/bin/activate  

# 3. Set your variables
export CONFIG="smollm2_360m_en1_en2"
export NGPU=8
export MODULE="llama3"

# ==========================================
# 4. MULTI-NODE DISTRIBUTED SETUP (LSF Style)
# ==========================================

# LSB_HOSTS contains a space-separated list of nodes. 
# If you ask for 32 cores, it prints the hostname 32 times. We filter for unique hosts.
HOSTS=($(echo $LSB_HOSTS | tr ' ' '\n' | sort -u))

export NNODES=${#HOSTS[@]}
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=29500
export JOB_ID=$LSB_JOBID

# Network debugging variables
export NCCL_DEBUG=WARN
export LOGLEVEL=INFO

# ==========================================
# 5. LAUNCH TRAINING
# ==========================================
# LSF does not have a direct equivalent to `srun` that cleanly drops 1 task per node 
# for torchrun without relying on MPI. The most robust native LSF method is to use 
# a simple loop with `blaunch` (LSF's remote execution tool).

for host in "${HOSTS[@]}"; do
    blaunch $host torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NGPU \
        --rdzv_id=$JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m torchtitan.train --module ${MODULE} --config ${CONFIG} &
done

# Wait for all background blaunch processes to finish
wait