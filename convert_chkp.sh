# 1. Point to your specific completed checkpoint
export CHKPT_DIR="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_03_02+weighttying/step-5000"

# 2. Define where you want the Hugging Face files to go
export HF_OUTPUT_DIR="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_03_02+weighttying/step-5000-hf"

# 3. Run the TorchTitan conversion script
python -m scripts.convert_to_hf \
    --torchtitan_checkpoint $CHKPT_DIR \
    --output_dir $HF_OUTPUT_DIR \
    --model_name "smollm2_360m_flex"