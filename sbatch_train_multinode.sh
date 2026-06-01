#!/bin/bash
#SBATCH --job-name=torchtitan_multilingual_training         # Name of your job
#SBATCH --partition=leshem.q         # Your specific partition
#SBATCH --nodes=1                   # Request exactly 1 node
#SBATCH --ntasks-per-node=1         # 1 task per node (torchrun will spawn the 8 GPU processes)
#SBATCH --gres=gpu:8                # Request 8 GPUs per node
#SBATCH --mem=400G                  # Memory request per node
#SBATCH --cpus-per-task=32          # CPU cores per node
#SBATCH --output=logs/%j_train.out  # Standard output log (%j is JobID)
#SBATCH --error=logs/%j_train.err   # Error log

# 1. Navigate to your project directory
# cd /home/adamga/torchtitan

# 2. Activate your environment
source .venv/bin/activate  

# 3. Set your variables
export CONFIG="smollm2_360m_en1_en2"
export NGPU=8
export MODULE="llama3"

# ==========================================
# 4. MULTI-NODE DISTRIBUTED SETUP
# ==========================================

# Extract the IP/hostname of the first node to act as the Master
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Some useful network debugging variables for PyTorch multi-node
export NCCL_DEBUG=WARN
export LOGLEVEL=INFO

# 5. Launch the training using srun to distribute torchrun across all nodes
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NGPU \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m torchtitan.train --module ${MODULE} --config ${CONFIG}