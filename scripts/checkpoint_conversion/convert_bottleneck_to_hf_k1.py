import argparse
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.checkpoint import ModelWrapper
# Adjust import path if needed based on your directory structure
from torchtitan.models.multilingual_bottleneck import model_registry

# Import your custom HF wrapper
from torchtitan.models.multilingual_bottleneck.hf import MultilingualBottleneckForCausalLM
from torchtitan.models.multilingual_bottleneck.hf import MultilingualBottleneckConfig
from transformers import LlamaConfig, LlamaForCausalLM

def permute(w, n_heads_arg, dim1, dim2):
    """Permute Q and K weights to convert from Meta's RoPE format to Hugging Face's."""
    return w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2).clone()

@torch.inference_mode()
def extract_language_model(input_dir: Path, output_dir: Path, flavor: str, lang_id: int):
    print(f"Loading torchtitan Bottleneck model ({flavor})...")
    
    # 1. Load torchtitan config
    model_spec = model_registry(flavor)
    tt_config = model_spec.model
    
    if tt_config.k_factor != 1:
        raise ValueError(f"This script only works for k_factor=1. Found k_factor={tt_config.k_factor}")

    # 2. Build empty PyTorch model and load DCP
    with torch.device("cpu"):
        tt_model = tt_config.build()

    wrapped_model = ModelWrapper(tt_model)
    state_dict = wrapped_model._get_state_dict()
    
    print(f"Reading DCP checkpoint from {input_dir}...")
    dcp.load(state_dict, checkpoint_id=str(input_dir))

    # Strip the 'model.' prefix Torchtitan adds
    raw_sd = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    # 3. Calculate architectural dimensions
    n_heads = tt_config.layer.attention.n_heads
    n_kv_heads = tt_config.layer.attention.n_kv_heads
    head_dim = tt_config.dim // n_heads
    
    enc_depth = tt_config.encoder_depth
    bb_depth = tt_config.backbone_depth
    dec_depth = tt_config.decoder_depth
    total_layers = enc_depth + bb_depth + dec_depth

    # Safely extract the FFN intermediate size from the actual loaded weights
    ffn_hidden_size = raw_sd["shared_backbone.0.feed_forward.w1.weight"].shape[0]

    print(f"Extracting path for Language ID: {lang_id}")
    print(f"Total Layers: {total_layers} (Enc: {enc_depth}, BB: {bb_depth}, Dec: {dec_depth})")

    # 4. Build standard Hugging Face LlamaConfig
    hf_config = LlamaConfig(
        vocab_size=tt_config.vocab_size,
        hidden_size=tt_config.dim,
        intermediate_size=ffn_hidden_size,
        num_hidden_layers=total_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=tt_config.rope.max_seq_len,
        rms_norm_eps=tt_config.layer.norm_eps,
        rope_theta=tt_config.rope.theta,
        bos_token_id=0,     #TODO: confirm these special token IDs with your tokenizer config
        eos_token_id=0,
    )

    hf_sd = {}
    
    # 5. Map global weights
    hf_sd["model.embed_tokens.weight"] = raw_sd[f"tok_embeddings.{lang_id}.weight"]
    hf_sd["model.norm.weight"] = raw_sd[f"norm.{lang_id}.weight"]
    hf_sd["lm_head.weight"] = raw_sd[f"output.{lang_id}.weight"]

    # 6. Map and concatenate the layers
    for target_layer in range(total_layers):
        # Determine which branched module this target layer comes from
        if target_layer < enc_depth:
            source_prefix = f"encoders.{lang_id}.{target_layer}"
        elif target_layer < enc_depth + bb_depth:
            source_prefix = f"shared_backbone.{target_layer - enc_depth}"
        else:
            source_prefix = f"decoders.{lang_id}.{target_layer - enc_depth - bb_depth}"

        target_prefix = f"model.layers.{target_layer}"

        # Permute Q and K weights for RoPE compatibility
        wq = raw_sd[f"{source_prefix}.attention.wq.weight"]
        wk = raw_sd[f"{source_prefix}.attention.wk.weight"]
        hf_sd[f"{target_prefix}.self_attn.q_proj.weight"] = permute(wq, n_heads, n_heads * head_dim, tt_config.dim)
        hf_sd[f"{target_prefix}.self_attn.k_proj.weight"] = permute(wk, n_kv_heads, n_kv_heads * head_dim, tt_config.dim)
        
        # Standard copies for everything else
        hf_sd[f"{target_prefix}.self_attn.v_proj.weight"] = raw_sd[f"{source_prefix}.attention.wv.weight"]
        hf_sd[f"{target_prefix}.self_attn.o_proj.weight"] = raw_sd[f"{source_prefix}.attention.wo.weight"]
        hf_sd[f"{target_prefix}.mlp.gate_proj.weight"] = raw_sd[f"{source_prefix}.feed_forward.w1.weight"]
        hf_sd[f"{target_prefix}.mlp.down_proj.weight"] = raw_sd[f"{source_prefix}.feed_forward.w2.weight"]
        hf_sd[f"{target_prefix}.mlp.up_proj.weight"] = raw_sd[f"{source_prefix}.feed_forward.w3.weight"]
        hf_sd[f"{target_prefix}.input_layernorm.weight"] = raw_sd[f"{source_prefix}.attention_norm.weight"]
        hf_sd[f"{target_prefix}.post_attention_layernorm.weight"] = raw_sd[f"{source_prefix}.ffn_norm.weight"]

    # 7. Load into a clean, standard HF model
    print("Initializing standard Hugging Face LlamaForCausalLM...")
    hf_model = LlamaForCausalLM(hf_config)
    
    print("Injecting mapped weights...")
    hf_model.load_state_dict(hf_sd, strict=True)

    # 8. Save
    print(f"Saving pristine model to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(output_dir, safe_serialization=True)
    print("Done! You can now evaluate this with standard --model hf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--flavor", type=str, default="360M_k1")
    parser.add_argument("--lang_id", type=int, required=True, help="Extract pathway for this specific language ID")
    args = parser.parse_args()

    extract_language_model(args.input_dir, args.output_dir, args.flavor, args.lang_id)