#!/bin/bash
#BSUB -J torchtitan_debug
#BSUB -q preemptable
#BSUB -n 64                                # 32 per node x 2 nodes
#BSUB -R "span[ptile=32]"
#BSUB -gpu "num=8:mode=exclusive_process"
#BSUB -R "rusage[mem=400G]"
#BSUB -o logs/%J_debug.out
#BSUB -e logs/%J_debug.err

# USAGE:
#   bsub -G grp_preemptable < bsub_debug.sh
#   CONFIG=llama3_7B_en1_en2_codeswitching bsub -G grp_preemptable < bsub_debug.sh

DEFAULT_CONFIG="llama3_7B_en1_en2"
CONFIG=${CONFIG:-$DEFAULT_CONFIG}

source .venv/bin/activate

export NGPU=8
export MODULE="llama3"

echo "Debug run: CONFIG=$CONFIG"

HOSTS=($(echo $LSB_HOSTS | tr ' ' '\n' | sort -u))

export NNODES=${#HOSTS[@]}
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=29500
export JOB_ID=$LSB_JOBID

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=^lo,^docker,^podman,^veth
export NCCL_DEBUG=WARN

for host in "${HOSTS[@]}"; do
    blaunch $host torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NGPU \
        --rdzv_id=$JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m torchtitan.train --module ${MODULE} --config ${CONFIG} \
        --parallelism.data_parallel_replicate_degree ${NNODES} \
        --training.local_batch_size 1 \
        --training.global_batch_size $((NNODES * NGPU)) \
        --activation_checkpoint.mode selective \
        --training.steps 10 \
        --checkpoint.folder /tmp/torchtitan_debug_${JOB_ID} \
        &> logs/${JOB_ID}_${host}.log &
done

wait
