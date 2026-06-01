# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.loss import build_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    FeedForward,
    GQAttention,
    RoPE,
)
from torchtitan.protocols.model_spec import ModelSpec

from .model import Llama3Model, Llama3TransformerBlock
from .parallelize import parallelize_llama
from .state_dict_adapter import Llama3StateDictAdapter

__all__ = [
    "parallelize_llama",
    "Llama3Model",
    "llama3_configs",
]


llama3_configs = {
    "debugmodel": Llama3Model.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=16, attn_backend="sdpa", rope_backend="complex"
            ),
        ),
        rope=RoPE.Config(
            # TODO: find better ways to enforce dim = decoder dim // n_heads, for all models
            dim=256 // 16,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "debugmodel_flex_attn": Llama3Model.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=16,
                attn_backend="flex",
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=256 // 16,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "debugmodel_varlen_attn": Llama3Model.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=16,
                attn_backend="varlen",
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=256 // 16,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "160M_mha_baseline": Llama3Model.Config(
        dim=768,
        n_layers=12,
        # vocab_size=32768,
        vocab_size=65536,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    768, multiple_of=256, ffn_dim_multiplier=1.3
                )
            ),
            attention=GQAttention.Config(
                n_heads=12,
                n_kv_heads=12,        # 1:1 Ratio (Multi-Head Attention)
                attn_backend="sdpa",  # Fast optimized kernel for fixed seq_len
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=768 // 12,
            max_seq_len=512,
            theta=10000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "160M_mha_flex_baseline": Llama3Model.Config(
        dim=768,
        n_layers=12,
        # vocab_size=32768,
        vocab_size=65536,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    768, multiple_of=256, ffn_dim_multiplier=1.5
                )
            ),
            attention=GQAttention.Config(
                n_heads=12,
                n_kv_heads=12,        # 1:1 Ratio (Multi-Head Attention)
                attn_backend="flex",  # Fast optimized kernel for packed variable seq_len inputs
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=768 // 12,
            max_seq_len=2048,
            theta=10000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "160M_gqa_balanced": Llama3Model.Config(
        dim=768,
        n_layers=12,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    768, multiple_of=256, ffn_dim_multiplier=1.375 # Offset for KV parameter loss
                )
            ),
            attention=GQAttention.Config(
                n_heads=12,
                n_kv_heads=4,         # 3:1 Ratio (Grouped Query Attention)
                attn_backend="sdpa",  # Matches baseline for fair benchmark
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=768 // 12,
            max_seq_len=256,
            theta=10000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "500M_mha_baseline": Llama3Model.Config(
        dim=1280,
        n_layers=20,
        vocab_size=65536,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    1280, multiple_of=256, ffn_dim_multiplier=1#.3
                )
            ),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=16,        # 1:1 Ratio (Multi-Head Attention)
                attn_backend="sdpa",  # Fast optimized kernel for fixed seq_len
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=1280 // 16,
            max_seq_len=2048,
            theta=10000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "500M_mha_flex_baseline": Llama3Model.Config(
        dim=1280,
        n_layers=20,
        vocab_size=65536,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    1280, multiple_of=256, ffn_dim_multiplier=1#.3
                )
            ),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=16,        # 1:1 Ratio (Multi-Head Attention)
                attn_backend="flex",  # Fast optimized kernel for variable seq_len
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=1280 // 16,
            max_seq_len=2048,
            theta=10000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "smollm2_135m": Llama3Model.Config(
        dim=576,
        n_layers=30,
        vocab_size=49152,       # User override (Standard is 49152)
        enable_weight_tying=True,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    576, multiple_of=256  # Omitting multiplier defaults to standard 8/3 -> 1536
                )
            ),
            attention=GQAttention.Config(
                n_heads=9,
                n_kv_heads=3,         # 3:1 GQA Ratio
                attn_backend="flex",  # Fast optimized kernel for packed variable seq_len inputs
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=576 // 9,         # Head dim = 64
            max_seq_len=2048,
            theta=10000,         # SmolLM2 standard theta
            backend="complex",
            scaling="llama",
        ),
    ),

    "smollm2_360m": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536,    #*2 if tagging # User override (Standard is 49152)
        enable_weight_tying=True,
        # enable_contrastive_alignment=True,
        # contrastive_proj_dim=512,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    960, multiple_of=256  # Omitting multiplier defaults to standard 8/3 -> 2560
                )
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,         # 3:1 GQA Ratio
                attn_backend="flex",  # Fast optimized kernel for packed variable seq_len inputs
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=960 // 15,        # Head dim = 64
            max_seq_len=2048,
            theta=10000,         # SmolLM2 standard theta
            backend="complex",
            scaling="llama",
        ),
    ),
    "smollm2_360m_contrastive": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536,    #*2 if tagging # User override (Standard is 49152)
        enable_weight_tying=True,
        enable_contrastive_alignment=True,
        contrastive_proj_dim=512,
        contrastive_target_layer=4,  # Apply contrastive alignment at this layer index (0-based) -1 is embeddings, 0 is first layer, etc.
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    960, multiple_of=256  # Omitting multiplier defaults to standard 8/3 -> 2560
                )
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,         # 3:1 GQA Ratio
                attn_backend="flex",  # Fast optimized kernel for packed variable seq_len inputs
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=960 // 15,        # Head dim = 64
            max_seq_len=2048,
            theta=10000,         # SmolLM2 standard theta
            backend="complex",
            scaling="llama",
        ),
    ),
    "smollm2_360m_contrastive_2xvocab": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536*2,    #*2 if tagging # User override (Standard is 49152)
        enable_weight_tying=True,
        enable_contrastive_alignment=True,
        contrastive_proj_dim=512,
        contrastive_target_layer=4,  # Apply contrastive alignment at this layer index (0-based) -1 is embeddings, 0 is first layer, etc.
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    960, multiple_of=256  # Omitting multiplier defaults to standard 8/3 -> 2560
                )
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,         # 3:1 GQA Ratio
                attn_backend="flex",  # Fast optimized kernel for packed variable seq_len inputs
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=960 // 15,        # Head dim = 64
            max_seq_len=2048,
            theta=10000,         # SmolLM2 standard theta
            backend="complex",
            scaling="llama",
        ),
    ),
    "smollm2_360m_contrastive2_2xvocab": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536*2,    #*2 if tagging # User override (Standard is 49152)
        enable_weight_tying=True,
        enable_contrastive_alignment=True,
        contrastive_proj_dim=512,
        contrastive_target_layer=2,  # Apply contrastive alignment at this layer index (0-based) -1 is embeddings, 0 is first layer, etc.
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    960, multiple_of=256  # Omitting multiplier defaults to standard 8/3 -> 2560
                )
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,         # 3:1 GQA Ratio
                attn_backend="flex",  # Fast optimized kernel for packed variable seq_len inputs
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=960 // 15,        # Head dim = 64
            max_seq_len=2048,
            theta=10000,         # SmolLM2 standard theta
            backend="complex",
            scaling="llama",
        ),
    ),
    "smollm2_360m_contrastive8_2xvocab": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536*2,    #*2 if tagging # User override (Standard is 49152)
        enable_weight_tying=True,
        enable_contrastive_alignment=True,
        contrastive_proj_dim=512,
        contrastive_target_layer=8,  # Apply contrastive alignment at this layer index (0-based) -1 is embeddings, 0 is first layer, etc.
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    960, multiple_of=256  # Omitting multiplier defaults to standard 8/3 -> 2560
                )
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,         # 3:1 GQA Ratio
                attn_backend="flex",  # Fast optimized kernel for packed variable seq_len inputs
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=960 // 15,        # Head dim = 64
            max_seq_len=2048,
            theta=10000,         # SmolLM2 standard theta
            backend="complex",
            scaling="llama",
        ),
    ),
    "smollm2_360m_contrastive8": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536,    #*2 if tagging # User override (Standard is 49152)
        enable_weight_tying=True,
        enable_contrastive_alignment=True,
        contrastive_proj_dim=512,
        contrastive_target_layer=8,  # Apply contrastive alignment at this layer index (0-based) -1 is embeddings, 0 is first layer, etc.
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    960, multiple_of=256  # Omitting multiplier defaults to standard 8/3 -> 2560
                )
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,         # 3:1 GQA Ratio
                attn_backend="flex",  # Fast optimized kernel for packed variable seq_len inputs
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=960 // 15,        # Head dim = 64
            max_seq_len=2048,
            theta=10000,         # SmolLM2 standard theta
            backend="complex",
            scaling="llama",
        ),
    ),
    "smollm2_360m_contrastive16_2xvocab": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536*2,    #*2 if tagging # User override (Standard is 49152)
        enable_weight_tying=True,
        enable_contrastive_alignment=True,
        contrastive_proj_dim=512,
        contrastive_target_layer=16,  # Apply contrastive alignment at this layer index (0-based) -1 is embeddings, 0 is first layer, etc.
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    960, multiple_of=256  # Omitting multiplier defaults to standard 8/3 -> 2560
                )
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,         # 3:1 GQA Ratio
                attn_backend="flex",  # Fast optimized kernel for packed variable seq_len inputs
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=960 // 15,        # Head dim = 64
            max_seq_len=2048,
            theta=10000,         # SmolLM2 standard theta
            backend="complex",
            scaling="llama",
        ),
    ),
    "smollm2_360m_contrastive16": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536,    #*2 if tagging # User override (Standard is 49152)
        enable_weight_tying=True,
        enable_contrastive_alignment=True,
        contrastive_proj_dim=512,
        contrastive_target_layer=16,  # Apply contrastive alignment at this layer index (0-based) -1 is embeddings, 0 is first layer, etc.
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    960, multiple_of=256  # Omitting multiplier defaults to standard 8/3 -> 2560
                )
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,         # 3:1 GQA Ratio
                attn_backend="flex",  # Fast optimized kernel for packed variable seq_len inputs
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=960 // 15,        # Head dim = 64
            max_seq_len=2048,
            theta=10000,         # SmolLM2 standard theta
            backend="complex",
            scaling="llama",
        ),
    ),
    "smollm2_360m_2xvocab": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536*2,
        enable_weight_tying=True,
        # enable_contrastive_alignment=True,
        # contrastive_proj_dim=512,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    960, multiple_of=256  # Omitting multiplier defaults to standard 8/3 -> 2560
                )
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,         # 3:1 GQA Ratio
                attn_backend="flex",  # Fast optimized kernel for packed variable seq_len inputs
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=960 // 15,        # Head dim = 64
            max_seq_len=2048,
            theta=10000,         # SmolLM2 standard theta
            backend="complex",
            scaling="llama",
        ),
    ),
    "smollm2_360m_3xvocab": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536*3,
        enable_weight_tying=True,
        # enable_contrastive_alignment=True,
        # contrastive_proj_dim=512,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    960, multiple_of=256  # Omitting multiplier defaults to standard 8/3 -> 2560
                )
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,         # 3:1 GQA Ratio
                attn_backend="flex",  # Fast optimized kernel for packed variable seq_len inputs
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=960 // 15,        # Head dim = 64
            max_seq_len=2048,
            theta=10000,         # SmolLM2 standard theta
            backend="complex",
            scaling="llama",
        ),
    ),
    "8B": Llama3Model.Config(
        dim=4096,
        n_layers=32,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    4096, multiple_of=1024, ffn_dim_multiplier=1.3
                )
            ),
            attention=GQAttention.Config(
                n_heads=32, n_kv_heads=8, attn_backend="sdpa", rope_backend="complex"
            ),
        ),
        rope=RoPE.Config(
            dim=4096 // 32,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "8B_flex": Llama3Model.Config(
        dim=4096,
        n_layers=32,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    4096, multiple_of=1024, ffn_dim_multiplier=1.3
                )
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                attn_backend="flex",
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=4096 // 32,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "8B_varlen": Llama3Model.Config(
        dim=4096,
        n_layers=32,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    4096, multiple_of=1024, ffn_dim_multiplier=1.3
                )
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                attn_backend="varlen",
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=4096 // 32,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "70B": Llama3Model.Config(
        dim=8192,
        n_layers=80,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    8192, multiple_of=4096, ffn_dim_multiplier=1.3
                )
            ),
            attention=GQAttention.Config(
                n_heads=64, n_kv_heads=8, attn_backend="sdpa", rope_backend="complex"
            ),
        ),
        rope=RoPE.Config(
            dim=8192 // 64,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "405B": Llama3Model.Config(
        dim=16384,
        n_layers=126,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    16384, multiple_of=4096, ffn_dim_multiplier=1.2
                )
            ),
            attention=GQAttention.Config(
                n_heads=128, n_kv_heads=8, attn_backend="sdpa", rope_backend="complex"
            ),
        ),
        rope=RoPE.Config(
            dim=16384 // 128,
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="llama3",
        flavor=flavor,
        model=llama3_configs[flavor],
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        # build_loss_fn=build_cross_entropy_loss,
        build_loss_fn=build_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )
