# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from torchtitan.components.loss import build_loss
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    FeedForward,
    GQAttention,
    RoPE,
)
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan.models.llama3.model import Llama3TransformerBlock

from .model import MultilingualBottleneckModel
from .parallelize_bottleneck import parallelize_bottleneck

__all__ = [
    "parallelize_bottleneck",
    "MultilingualBottleneckModel",
    "bottleneck_configs",
    "model_registry"
]

bottleneck_configs = {
    "160M_k4": MultilingualBottleneckModel.Config(
        dim=768,
        vocab_size=65536,       # Matching your previous baseline
        enable_weight_tying=True,
        enable_shared_embeddings=False,
        k_factor=4,
        num_languages=2,
        use_backbone_rope=False,
        encoder_depth=4,
        backbone_depth=12,
        decoder_depth=4,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    768, multiple_of=256, ffn_dim_multiplier=1.3
                )
            ),
            attention=GQAttention.Config(
                n_heads=12,
                n_kv_heads=12,        # 1:1 Ratio (Multi-Head Attention)
                attn_backend="sdpa",  # Fast optimized kernel
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=768 // 12,
            max_seq_len=1024,
            theta=10000,
            backend="complex",
            scaling="llama",
        ),
    ),    
    "360M_k1": MultilingualBottleneckModel.Config(
        dim=960,
        vocab_size=65536,       # Matching your previous baseline
        k_factor=1,
        num_languages=2,
        use_backbone_rope=True,
        encoder_depth=4,
        backbone_depth=24,
        decoder_depth=4,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    960, multiple_of=256, ffn_dim_multiplier=1.0
                )
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,        # 1:1 Ratio (Multi-Head Attention)
                attn_backend="sdpa",  # Fast optimized kernel
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=960 // 15,
            max_seq_len=2048,
            theta=10000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "360M_k1_sep_embeddings": MultilingualBottleneckModel.Config(
        dim=960,
        vocab_size=65536,       # Matching your previous baseline
        k_factor=1,
        num_languages=2,
        use_backbone_rope=True,
        encoder_depth=0,
        backbone_depth=32,
        decoder_depth=0,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    960, multiple_of=256, ffn_dim_multiplier=1.0
                )
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,        # 1:1 Ratio (Multi-Head Attention)
                attn_backend="sdpa",  # Fast optimized kernel
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=960 // 15,
            max_seq_len=2048,
            theta=10000,
            backend="complex",
            scaling="llama",
        ),
    ),

}

def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="multilingual_bottleneck",
        flavor=flavor,
        model=bottleneck_configs[flavor],
        parallelize_fn=parallelize_bottleneck,
        pipelining_fn=None, # Pipeline Parallelism not implemented for branched topology
        build_loss_fn=build_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None, # Will require a custom adapter for HF export later
    )