#!/bin/bash
#BSUB -J torchtitan_multilingual_training  # Name of your job
#BSUB -q normal                            # Queue
#BSUB -n 512                               # TOTAL CPU cores: 32 per node x 16 nodes
#BSUB -R "span[ptile=32]"                  # Pack exactly 32 CPU cores per node
#BSUB -gpu "num=8:mode=exclusive_process"  # Request 8 GPUs per node
#BSUB -R "rusage[mem=400G]"                # Memory request per node
#BSUB -o logs/%J_train.out                 # Standard output log (%J is LSF's JobID)
#BSUB -e logs/%J_train.err                 # Error log

# USAGE:
#   Submit with default config:
#     bsub -G grp_exploratory < bsub_train_multinode.sh
#
#   Submit with custom config:
#     CONFIG=my_custom_config bsub -G grp_exploratory < bsub_train_multinode.sh
#
# NOTE: Use "bsub <" to submit so that LSF parses #BSUB directives above.

echo "Starting multi-node training job with LSF..."

DEFAULT_CONFIG="llama3_7B_en1_en2"
CONFIG=${CONFIG:-$DEFAULT_CONFIG}

source .venv/bin/activate

export CONFIG=$CONFIG
export NGPU=8
export MODULE="llama3"

echo "Starting training with CONFIG=$CONFIG"

# LSB_HOSTS is empty for large (>64 slot) jobs; use LSB_DJOB_HOSTFILE instead
if [ -f "$LSB_DJOB_HOSTFILE" ]; then
    HOSTS=($(sort -u "$LSB_DJOB_HOSTFILE"))
else
    HOSTS=($(echo $LSB_HOSTS | tr ' ' '\n' | sort -u))
fi

export NNODES=${#HOSTS[@]}
if [ "$NNODES" -eq 0 ]; then
    echo "ERROR: No hosts found (LSB_HOSTS='$LSB_HOSTS', LSB_DJOB_HOSTFILE='$LSB_DJOB_HOSTFILE')" >&2
    exit 1
fi

echo "NNODES=$NNODES, MASTER_ADDR=${HOSTS[0]}"
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=29500
export JOB_ID=$LSB_JOBID

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=^lo,^docker,^podman,^veth
export NCCL_DEBUG=WARN
export LOGLEVEL=INFO

for host in "${HOSTS[@]}"; do
    blaunch $host torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NGPU \
        --rdzv_id=$JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m torchtitan.train --module ${MODULE} --config ${CONFIG} \
        &> logs/${JOB_ID}_${host}.log &
done

wait
