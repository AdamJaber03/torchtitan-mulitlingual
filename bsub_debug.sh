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

# Use the interface that routes to MASTER_ADDR — works on any cluster/hardware
# Resolve hostname to IP first since ip route get requires an IP address
MASTER_IP=$(getent ahosts "$MASTER_ADDR" 2>/dev/null | awk 'NR==1{print $1}')
MASTER_IFACE=$(ip route get "$MASTER_IP" 2>/dev/null | grep -oP 'dev \K\S+' | head -1)
if [ -n "$MASTER_IFACE" ]; then
    export NCCL_SOCKET_IFNAME="$MASTER_IFACE"
    echo "NCCL_SOCKET_IFNAME=$MASTER_IFACE (from ip route get $MASTER_IP for $MASTER_ADDR)"
else
    export NCCL_SOCKET_IFNAME=^lo,^docker,^podman,^veth
    echo "NCCL_SOCKET_IFNAME fallback (could not resolve route to $MASTER_ADDR / $MASTER_IP)"
fi

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

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
