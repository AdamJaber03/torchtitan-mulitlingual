# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# H-Net (Hierarchical Network with Dynamic Chunking), byte-level.
# Reference: goombalab/hnet, arXiv:2507.07955. See README.md.

from torchtitan.components.loss import build_loss
from torchtitan.protocols.model_spec import ModelSpec

from .model import HNetModel
from .parallelize import parallelize_hnet

__all__ = ["parallelize_hnet", "HNetModel", "hnet_configs", "model_registry"]


# ----------------------------------------------------------------------------
# Flavors. ``arch_layout`` encodes the U-Net hierarchy. For a 1-stage net:
#   [ "<encoder>", [ "<main>" ], "<decoder>" ]
# where each "<...>" is a layer-spec string of "<letter><count>" groups:
#   m = Mamba-2 (no FFN), M = Mamba-2 + SwiGLU,
#   t = attention (no FFN),  T = attention + SwiGLU.
#
# Per-stage list fields (d_model, d_intermediate, attn_*) are indexed by stage:
# index 0 = outer (encoder/decoder), index 1 = inner (main network).
# ----------------------------------------------------------------------------
hnet_configs = {
    # Tiny 1-stage net for forward/backward smoke testing.
    "debugmodel": HNetModel.Config(
        arch_layout=["m2", ["m2T2"], "m2"],
        d_model=[256, 512],
        d_intermediate=[0, 1024],
        vocab_size=256,
        tie_embeddings=False,
        ssm_d_conv=4,
        ssm_expand=2,
        ssm_d_state=128,
        ssm_chunk_size=256,
        attn_num_heads=[4, 8],
        attn_rotary_emb_dim=[0, 64],
        attn_window_size=[-1, -1],
        target_compression=[4.0],
    ),
    # ~347M 1-stage byte model whose main network reuses the smollm2_360m
    # backbone (d=960, 32 Transformer layers, SwiGLU FFN=2560, 15 query / 5 KV
    # heads = 3:1 GQA, head_dim=64, RoPE theta=10000), with H-Net's Mamba-2
    # encoder/decoder at the byte level (d=768). tie_embeddings on, like smollm2.
    "1stage_smollm": HNetModel.Config(
        arch_layout=["m4", ["T32"], "m4"],
        d_model=[768, 960],
        d_intermediate=[0, 2560],  # SwiGLU 8/3 * 960, rounded -> 2560
        vocab_size=256,
        # Authors use untied embeddings (tying at byte vocab=256 saves ~0.07%);
        # tie_embeddings=True now also works (init fixed) if you want it.
        tie_embeddings=False,
        ssm_d_conv=4,
        ssm_expand=2,
        ssm_d_state=128,
        ssm_chunk_size=256,
        attn_num_heads=[12, 15],  # stage 0 (Mamba enc/dec) unused; main = 15
        attn_num_kv_heads=[4, 5],  # 3:1 GQA in the main network
        attn_rotary_emb_dim=[32, 64],  # main: full head_dim (64) rotary
        attn_window_size=[-1, -1],  # global attention (smollm2 block-causal)
        target_compression=[6.0],
    ),
    # ~340M-param 1-stage byte model, faithful to the authors' `hnet_1stage_L`
    # design: pure-Mamba m4 encoder/decoder at d=1024, pure-Transformer main at
    # d=1536 (FFN 4096, 16 heads, RoPE=48, global attn), sliding-window 1023 on
    # the byte stage. Only the main depth is scaled down (22 -> 10 layers).
    "1stage_M": HNetModel.Config(
        arch_layout=["m4", ["T10"], "m4"],
        d_model=[1024, 1536],
        d_intermediate=[0, 4096],
        vocab_size=256,
        tie_embeddings=False,
        ssm_d_conv=4,
        ssm_expand=2,
        ssm_d_state=128,
        ssm_chunk_size=256,
        attn_num_heads=[16, 16],
        attn_rotary_emb_dim=[32, 48],
        attn_window_size=[1023, -1],
        target_compression=[6.0],
    ),
    # The authors' `hnet_1stage_L` (configs/hnet_1stage_L.json) — the SMALLEST
    # model scale in the paper ("Large", FLOP-matched to GPT-3 Large ~760M;
    # this 1-stage variant is ~679M). Exact arch from goombalab/hnet.
    "1stage_L": HNetModel.Config(
        arch_layout=["m4", ["T22"], "m4"],
        d_model=[1024, 1536],
        d_intermediate=[0, 4096],
        vocab_size=256,
        tie_embeddings=False,
        ssm_d_conv=4,
        ssm_expand=2,
        ssm_d_state=128,
        ssm_chunk_size=256,
        attn_num_heads=[16, 16],
        attn_rotary_emb_dim=[32, 48],
        attn_window_size=[1023, -1],  # byte stage: sliding window 1023; main: global
        target_compression=[6.0],
    ),
    # Small real 1-stage byte-level model (~100-200M params depending on data).
    "1stage_small": HNetModel.Config(
        arch_layout=["m4", ["m6T6"], "m4"],
        d_model=[768, 1024],
        d_intermediate=[0, 2816],
        vocab_size=256,
        tie_embeddings=False,
        ssm_d_conv=4,
        ssm_expand=2,
        ssm_d_state=128,
        ssm_chunk_size=256,
        attn_num_heads=[12, 16],
        attn_rotary_emb_dim=[0, 64],
        attn_window_size=[-1, -1],
        target_compression=[6.0],
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="hnet",
        flavor=flavor,
        model=hnet_configs[flavor],
        parallelize_fn=parallelize_hnet,
        pipelining_fn=None,  # PP unsupported (dynamic chunking)
        build_loss_fn=build_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
