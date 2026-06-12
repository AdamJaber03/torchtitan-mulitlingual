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
#   The "bsub < script" pipe does NOT work on BluVela because LSF spools the
#   script to ~/.lsbatch/ which is only on the login node's local filesystem
#   (not on GPFS) and therefore inaccessible from compute nodes.
#
#   Use the direct command form instead — all options as flags, script as arg.
#   Job name MUST match the CONFIG so each normal job kills only its own
#   preemptable sibling (not the other config's job).
#
#   Normal queue (production):
#     cd /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
#     export CONFIG=llama3_7B_en1_en2
#     bsub -G grp_exploratory -q normal \
#       -J ${CONFIG} -n 512 \
#       -R "span[ptile=32]" -R "rusage[mem=400G]" \
#       -gpu "num=8:mode=exclusive_process" \
#       -cwd /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual \
#       -o logs/%J_train.out -e logs/%J_train.err -env "all" \
#       bash /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual/bsub_train_multinode.sh
#
#     export CONFIG=llama3_7B_en1_en2_codeswitching
#     bsub -G grp_exploratory -q normal \
#       -J ${CONFIG} -n 512 \
#       -R "span[ptile=32]" -R "rusage[mem=400G]" \
#       -gpu "num=8:mode=exclusive_process" \
#       -cwd /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual \
#       -o logs/%J_train.out -e logs/%J_train.err -env "all" \
#       bash /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual/bsub_train_multinode.sh
#
#   Preemptable queue (runs while normal waits; killed automatically on normal start):
#     export CONFIG=llama3_7B_en1_en2
#     bsub -G grp_preemptable -q preemptable \
#       -J ${CONFIG} -n 512 \
#       -R "span[ptile=32]" -R "rusage[mem=400G]" \
#       -gpu "num=8:mode=exclusive_process" \
#       -cwd /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual \
#       -o logs/%J_train.out -e logs/%J_train.err -env "all" \
#       bash /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual/bsub_train_multinode.sh
#
#     export CONFIG=llama3_7B_en1_en2_codeswitching
#     bsub -G grp_preemptable -q preemptable \
#       -J ${CONFIG} -n 512 \
#       -R "span[ptile=32]" -R "rusage[mem=400G]" \
#       -gpu "num=8:mode=exclusive_process" \
#       -cwd /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual \
#       -o logs/%J_train.out -e logs/%J_train.err -env "all" \
#       bash /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual/bsub_train_multinode.sh

echo "Starting multi-node training job with LSF..."

DEFAULT_CONFIG="llama3_7B_en1_en2"
CONFIG=${CONFIG:-$DEFAULT_CONFIG}

source .venv/bin/activate

# When a normal-queue job starts, kill the preemptable sibling for the same
# config so they don't write to the same checkpoint directory simultaneously.
# The normal job resumes from wherever the preemptable job left off.
# Job names are set to $CONFIG at submission time, so -J filters precisely.
if [ "${LSB_QUEUE}" = "normal" ]; then
    SIBLINGS=$(bjobs -noheader -u "$USER" -q preemptable -J "$LSB_JOBNAME" 2>/dev/null | \
        grep " RUN " | awk '{print $1}' | grep -v "^${LSB_JOBID}$")
    for jid in $SIBLINGS; do
        echo "Normal queue job $LSB_JOBID starting — killing preemptable sibling $jid ($LSB_JOBNAME)"
        bkill "$jid" 2>/dev/null
    done
fi

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
export LOGLEVEL=INFO
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export PYTORCH_ALLOC_CONF=expandable_segments:True

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
