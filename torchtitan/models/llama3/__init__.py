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
    "smollm2_360m_contrastive_linear": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536,    #*2 if tagging # User override (Standard is 49152)
        enable_weight_tying=True,
        enable_contrastive_alignment=True,
        contrastive_head_type="linear",  # Use a single linear layer for the contrastive head instead of a 2-layer MLP
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
    "smollm2_360m_contrastive_identity": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536,    #*2 if tagging # User override (Standard is 49152)
        enable_weight_tying=True,
        enable_contrastive_alignment=True,
        contrastive_head_type="identity",  # Use a single linear layer for the contrastive head instead of a 2-layer MLP
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
    # Same as smollm2_360m_contrastive_identity, but applies the (identity) contrastive loss at
    # MULTIPLE depths simultaneously to force Ar<->En alignment to propagate through the upper layers
    # (single layer-4 contrast does not propagate). identity head -> the per-layer heads are
    # nn.Identity (no params), so checkpoints carry no head weights.
    "smollm2_360m_contrastive_identity_multidepth": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536,
        enable_weight_tying=True,
        enable_contrastive_alignment=True,
        contrastive_head_type="identity",
        contrastive_proj_dim=512,  # ignored for identity
        contrastive_target_layers=[4, 8, 12, 16, 20, 24, 28],
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(960, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,
                attn_backend="flex",
                attn_mask_type="block_causal",
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
    # Same as smollm2_360m_contrastive but with early-exit drop enabled: the wordwise translation
    # member only reaches contrastive_target_layer, then the kept (non-translation) tokens are
    # compacted to `keep_len` for the remaining layers + output head. Pair with a training config
    # whose seq_len is ~1.5x keep_len (so the kept/CE budget matches a normal keep_len run) and a
    # source that sets contrastive_early_exit=True. keep_len is sized with headroom above
    # (2/3)*seq_len; tune via the logged kept-token distribution.
    "smollm2_360m_contrastive_shallow": Llama3Model.Config(
        dim=960,
        n_layers=32,
        vocab_size=65536,
        enable_weight_tying=True,
        enable_contrastive_alignment=True,
        contrastive_proj_dim=512,
        contrastive_target_layer=4,
        keep_len=2304,  # ~ (2/3)*3072 + ~12% headroom; see the new training config (seq_len=3072)
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    960, multiple_of=256
                )
            ),
            attention=GQAttention.Config(
                n_heads=15,
                n_kv_heads=5,
                attn_backend="flex",
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=960 // 15,
            max_seq_len=2048,  # auto-synced to training.seq_len (3072) via update_from_config
            theta=10000,
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

# Shallow-contrastive variants with alternative projection heads, derived from
# "smollm2_360m_contrastive_shallow" (same dim/layers/keep_len/target_layer, only the head differs).
# "linear" / "identity" put more of the contrastive pressure on the layer-`target` representation
# itself instead of letting a 2-layer MLP absorb it into a throwaway projection.
import dataclasses as _dc

for _head in ("linear", "identity"):
    llama3_configs[f"smollm2_360m_contrastive_shallow_{_head}"] = _dc.replace(
        llama3_configs["smollm2_360m_contrastive_shallow"],
        contrastive_head_type=_head,
    )

# Hybrid (partial) anchor embedding variant of "smollm2_360m" -- same architecture, but
# tok_embeddings is a HybridAnchorEmbedding (matched AR/EN 1-to-1 token pairs share the first
# half of the embedding dim). Registered here (not just wired into the
# smollm2_360m_flex_hybrid_anchor Trainer config in config_registry.py) so tools that resolve a
# model purely by flavor name -- e.g. scripts/checkpoint_conversion/convert_to_hf.py's
# `model_registry(model_flavor)` -- can build the matching architecture directly via
# --model_flavor=smollm2_360m_hybrid_anchor instead of accidentally building the stock
# (non-anchor) model and failing to load an anchor checkpoint into it.
llama3_configs["smollm2_360m_hybrid_anchor"] = _dc.replace(
    llama3_configs["smollm2_360m"],
    anchor_embedding_path="/home/adamga/leshemg/adamga/data/translations/ar_en_1to1_token_map.json",
    anchor_shared_dim_fraction=0.5,
)

# Tagged hybrid-anchor variant of "smollm2_360m_2xvocab" -- the doubled vocab is split into an
# EN-origin first half [0, 65536) and a "tagged" AR-origin second half [65536, 131072). Each
# second-half token V+i is partially tied (shared anchor subspace) to its first-half twin i via the
# synthetic "identity_shift:65536" anchor map (see HybridAnchorEmbedding._compute_group_id_list),
# so no data file is needed. Pair with a training config that translates Arabic word-wise to English
# and then tags the AR-origin tokens into the second half (stochastic_word_tagging, vocab_size
# 65536) -- see smollm2_360m_flex_en_TrAr_hybrid_anchor in config_registry.py. Registered here (not
# only wired into that Trainer config) so flavor-name resolvers -- e.g.
# scripts/checkpoint_conversion/convert_to_hf.py's model_registry(model_flavor) -- rebuild the exact
# architecture instead of the stock 2xvocab (non-anchor) model.
llama3_configs["smollm2_360m_tagged_hybrid_anchor"] = _dc.replace(
    llama3_configs["smollm2_360m_2xvocab"],
    anchor_embedding_path="identity_shift:65536",
    anchor_shared_dim_fraction=0.5,
)


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
