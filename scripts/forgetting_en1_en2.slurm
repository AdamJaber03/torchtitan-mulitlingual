#!/bin/bash
#SBATCH --job-name=sm360_en1en2_forget_k500
#SBATCH --partition=leshem.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro_6000:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --output=/home/adamga/leshemg/adamga/train/torchtitan/logs/forget_k500_%j.out

set -x
cd /home/adamga/torchtitan
source /home/adamga/torchtitan/.venv/bin/activate

export NGPU=8
export MODULE=llama3
export CONFIG=smollm2_360m_en1_en2_forgetting

# Single node, 8 GPUs -> run_train.sh launches torchrun --nproc_per_node=8.
# Pass-through args ("$@") allow short smoke runs, e.g.:
#   sbatch scripts/forgetting_en1_en2.slurm --training.steps 12 \
#       --training.active-forgetting-interval 3 --compile.no-enable \
#       --checkpoint.no-enable --metrics.no-enable-wandb --validator.no-enable
bash run_train.sh "$@"
