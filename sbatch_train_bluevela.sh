#!/bin/bash
#SBATCH --job-name=torchtitan_multilingual_training
#SBATCH --account=grp_exploratory
#SBATCH --partition=gpu-mid
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/%j_train.out
#SBATCH --error=logs/%j_train.err

# Slurm launcher for the Blue Vela pilot (login1). LSF equivalent:
# bsub_train_multinode.sh, still used on login2-5.
#
# USAGE -- CONFIG is mandatory and must match the -J name:
#   ssh leshem@login1.bluevela.rmf.ibm.com
#   cd /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
#   sbatch -J llama3_7B_en_ru --export=ALL,CONFIG=llama3_7B_en_ru \
#     sbatch_train_bluevela.sh
#
# Opportunistic lane (30-day limit, preempted by any normal/high job, and
# preemption requeues this script from the latest checkpoint):
#   sbatch -p gpu-low -t 30-00:00:00 -J llama3_7B_en_ru \
#     --export=ALL,CONFIG=llama3_7B_en_ru sbatch_train_bluevela.sh
#
# NEVER run two jobs for the same CONFIG at once, in any partition: they share
# one checkpoint directory and would corrupt each other. Unlike LSF there is no
# sibling-kill here -- check with `squeue -u $USER -o "%i %P %j %T"` first.
#
# Lane limits (scontrol show partition):
#   gpu-mid   default, <=16 nodes, <=7 days,  no group GPU cap
#   gpu-high  <=32 nodes, <=24h, grp_exploratory capped at 64 GPU (8 nodes)
#   gpu-long  <=4 nodes, <=14 days, 2 running jobs per user
#   gpu-low   <=128 nodes, <=30 days, preemptible
# A 16-node run is 128 GPUs, which exceeds the gpu-high entitlement, so
# gpu-mid is the only lane that fits it at full size.

set -u

if [ -z "${CONFIG:-}" ]; then
    echo "ERROR: CONFIG is not set. Pass --export=ALL,CONFIG=<config_name>." >&2
    echo "Refusing to fall back to a default: the wrong config trains the" >&2
    echo "wrong experiment into an existing checkpoint directory." >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
mkdir -p logs

export CONFIG
export NGPU=8
export MODULE="llama3"

HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
export NNODES=${#HOSTS[@]}
export MASTER_ADDR=${HOSTS[0]}
# Distinct port per job so a requeued or concurrent job cannot join a stale
# rendezvous on the same node.
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export JOB_ID=$SLURM_JOB_ID

echo "CONFIG=$CONFIG NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR:$MASTER_PORT"
echo "partition=$SLURM_JOB_PARTITION restart_count=${SLURM_RESTART_COUNT:-0}"

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_CROSS_NIC=1

# Pick the interface that carries traffic between nodes. This script runs ON
# the master, so routing to the master's own address yields "lo" and NCCL then
# bootstraps over 127.0.0.1 and hangs -- route to a peer instead.
PEER=${HOSTS[1]:-${HOSTS[0]}}
PEER_IP=$(getent ahosts "$PEER" 2>/dev/null | awk 'NR==1{print $1}')
MASTER_IFACE=$(ip route get "$PEER_IP" 2>/dev/null | grep -oP 'dev \K\S+' | head -1)
if [ -n "$MASTER_IFACE" ] && [ "$MASTER_IFACE" != "lo" ]; then
    export NCCL_SOCKET_IFNAME="$MASTER_IFACE"
    echo "NCCL_SOCKET_IFNAME=$MASTER_IFACE (route to peer $PEER / $PEER_IP)"
else
    export NCCL_SOCKET_IFNAME=^lo,^docker,^podman,^veth,^idrac
    echo "NCCL_SOCKET_IFNAME fallback (route to $PEER / $PEER_IP gave '${MASTER_IFACE:-none}')"
fi

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export LOGLEVEL=INFO
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 133_600 steps do not fit one 7-day allocation. Slurm sends USR1 to this
# script 15 min before the wall clock ends; requeue so it restarts from the
# most recent checkpoint (saved every 500 steps) with the same job ID.
requeue_self() {
    echo "$(date -Is) wall clock nearly up -- requeueing $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_self USR1

srun --kill-on-bad-exit=1 \
     --output=logs/%j_%N_train.log --error=logs/%j_%N_train.log \
     torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NGPU" \
        --rdzv_id="$JOB_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        -m torchtitan.train --module "$MODULE" --config "$CONFIG" &
wait
