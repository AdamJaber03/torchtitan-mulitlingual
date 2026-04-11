import argparse
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp

# Import Torchtitan's checkpoint wrapper
from torchtitan.components.checkpoint import ModelWrapper

# Import your Torchtitan model registry
# (Adjust this import path if your file is named differently)
from torchtitan.models.multilingual_bottleneck import model_registry

# Import your custom HF wrapper
from torchtitan.models.multilingual_bottleneck.hf import MultilingualBottleneckForCausalLM
from torchtitan.models.multilingual_bottleneck.hf import MultilingualBottleneckConfig

@torch.inference_mode()
def convert_dcp_to_hf(input_dir: Path, output_dir: Path, flavor: str):
    print(f"Loading torchtitan Bottleneck model ({flavor})...")
    
    # 1. Build the empty PyTorch model exactly as torchtitan does
    model_spec = model_registry(flavor)
    model_config = model_spec.model
    with torch.device("cpu"):
        tt_model = model_config.build()

    # 2. Wrap it so the state_dict structure matches torchtitan's DCP saver
    wrapped_model = ModelWrapper(tt_model)
    state_dict = wrapped_model._get_state_dict()
    
    print(f"Reading DCP checkpoint from {input_dir}...")
    dcp.load(
        state_dict,
        checkpoint_id=str(input_dir),
    )

    # 3. Clean the state dict keys! 
    # Torchtitan adds 'model.' to the beginning of everything. We strip it here.
    clean_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    print("Building Hugging Face wrapper...")
    # 4. Create the HF config to match the torchtitan config
    hf_config = MultilingualBottleneckConfig(
        vocab_size=model_config.vocab_size,
        dim=model_config.dim,
        k_factor=model_config.k_factor,
        num_languages=model_config.num_languages,
        use_backbone_rope=model_config.use_backbone_rope,
        encoder_depth=model_config.encoder_depth,
        backbone_depth=model_config.backbone_depth,
        decoder_depth=model_config.decoder_depth,
        n_heads=model_config.layer.attention.n_heads,
        n_kv_heads=model_config.layer.attention.n_kv_heads,
        max_position_embeddings=model_config.rope.max_seq_len,
        norm_eps=model_config.layer.norm_eps,
    )
    
    # 5. Load the weights into the HF wrapper
    hf_model = MultilingualBottleneckForCausalLM(hf_config)
    
    # Use strict=False just in case the freqs_cis buffer behaves weirdly, 
    # but print any actual missing weights to verify.
    missing, unexpected = hf_model.load_state_dict(clean_state_dict, strict=False)
    if missing and not all("freqs_cis" in k for k in missing):
        print(f"Warning, missing keys: {missing}")
    if unexpected:
        print(f"Warning, unexpected keys: {unexpected}")

    # 6. Save natively using Hugging Face
    print(f"Saving HF format to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(output_dir, safe_serialization=True)
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True, help="DCP checkpoint folder")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output HF folder")
    parser.add_argument("--flavor", type=str, default="160M", help="Model flavor (e.g., 160M, 360M_bilingual)")
    args = parser.parse_args()

    convert_dcp_to_hf(args.input_dir, args.output_dir, args.flavor)