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
    "7B_flex": Llama3Model.Config(
        dim=4096,
        n_layers=28,
        vocab_size=65536,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    4096, multiple_of=1024, ffn_dim_multiplier=1.2
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
    "7B_flex_2xvocab": Llama3Model.Config(
        dim=4096,
        n_layers=28,
        vocab_size=65536*2,
        layer=Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    4096, multiple_of=1024, ffn_dim_multiplier=1.2
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

import dataclasses as _dc

# Tagged hybrid-anchor variant of "7B_flex_2xvocab" -- the doubled vocab is split into an
# EN-origin first half [0, 65536) and a "tagged" second half [65536, 131072). Each second-half
# token 65536+i is partially tied (shared anchor subspace) to its first-half twin i via the
# synthetic "identity_shift:65536" anchor map, so no data file is needed. Pair with a training
# config that tags one stream into the second half via stochastic_word_tagging with
# vocab_size=65536 -- see llama3_7B_en1_en2_hybrid_anchor in config_registry.py.
#
# enable_weight_tying is deliberately LEFT AT 7B_flex_2xvocab's False, unlike the smollm2_360m
# tagged-anchor flavors which tie. Only the INPUT embedding is anchored here; the LM head stays a
# stock nn.Linear. That keeps this a one-variable change against the llama3_7B_en1_en2 baseline,
# and avoids TiedAnchorOutput rebuilding a [131072, 4096] matrix every microbatch.
#
# At dim=4096 and fraction 0.999 the split is anchor_dim=4092 / residual_dim=4, i.e. a 4-dim
# independent sliver per token (the 360m runs at dim=960 get a 1-dim sliver). Use 4095/4096
# (~0.99976) instead if an exactly-1-dim sliver is wanted.
#
# Registered here (not only wired into the Trainer config) so flavor-name resolvers -- e.g.
# scripts/checkpoint_conversion/convert_to_hf.py's model_registry(model_flavor) -- rebuild the
# exact architecture instead of the stock 2xvocab (non-anchor) model.
llama3_configs["7B_flex_tagged_hybrid_anchor"] = _dc.replace(
    llama3_configs["7B_flex_2xvocab"],
    anchor_embedding_path="identity_shift:65536",
    anchor_shared_dim_fraction=0.999,
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
