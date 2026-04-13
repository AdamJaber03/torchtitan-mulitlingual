#!/bin/bash
#SBATCH --job-name=vllm-server
# SBATCH --nodes=1
#SBATCH --partition=leshem.q           # Your specific partition
#SBATCH --gres=gpu:pro_6000:8          # Request 8 GPUs
#SBATCH --mem=600G                     # Memory request
#SBATCH --cpus-per-task=32             # CPU cores
#SBATCH --output=/home/adamga/torchtitan/vLLM_hosting/logs/%j.out           # Standard output log (%j is JobID)
#SBATCH --error=/home/adamga/torchtitan/vLLM_hosting/logs/%j.err            # Error log
#SBATCH --time=24:00:00                # Optional: Set a max run time (e.g., 24h)

# Navigate to your project directory (update this path)
echo "init"
cd /home/adamga/torchtitan/vLLM_hosting
echo "part1"
# Activate your uv virtual environment
source .venv/bin/activate
echo "part2"
export HF_HOME=/home/adamga/leshemg/adamga/hf_home
# Get the internal IP address of the compute node
NODE_IP=$(hostname -i | awk '{print $1}')
echo "Starting vLLM server on $NODE_IP:8000"

# Boot the server
    # --model meta-llama/Llama-3.3-70B-Instruct \

python -m vllm.entrypoints.openai.api_server \
    --model kishizaki-sci/Llama-4-Maverick-17B-128E-Instruct-AWQ \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8000