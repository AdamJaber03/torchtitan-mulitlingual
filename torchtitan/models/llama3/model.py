# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import math
from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.hybrid_anchor_embedding import (
    HybridAnchorEmbedding,
    TiedAnchorOutput,
)
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.tools.logging import logger


class Llama3TransformerBlock(TransformerBlock):
    """
    Llama3 TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        dim (int): Model dimension.
        n_layers (int): Total number of layers.
        config (Llama3TransformerBlock.Config): Block configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        depth_init: bool = True

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__()
        self.attention = config.attention.build(dim=dim)
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build(dim=dim)
        self.attention_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.norm_eps)

        if config.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, **kwargs):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Llama3Model(Decoder):
    """
    Llama3Model Module

    Args:
        config (Llama3Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        n_layers: int = 32
        vocab_size: int = 128256
        enable_weight_tying: bool = False
        # --- Hybrid anchor embedding (partial cross-lingual tying) ---
        # Path to a token map JSON (must contain 'id_remap' or 'pairs', e.g.
        # ar_en_1to1_token_map.json), a prebuilt bundle DIRECTORY, or the synthetic
        # "identity_shift:<V>[:<token_fraction>]" form. When set, tok_embeddings is replaced by a
        # HybridAnchorEmbedding that splits the embedding dim into a shared/tied "anchor"
        # subspace (matched id pairs index the SAME row) and an independent "residual"
        # subspace. None (default) keeps the stock nn.Embedding -- no change from today.
        anchor_embedding_path: str | None = None
        # Fraction of the embedding dim allocated to the shared/tied anchor subspace when
        # anchor_embedding_path is set. 0.5 = half the dims tied, half independent. Sweepable
        # ablation knob: lower -> closer to fully independent embeddings (weaker anchor, less
        # interference); higher -> closer to full id-remap-style tying (stronger anchor, more
        # interference risk). Ignored when anchor_embedding_path is None.
        anchor_shared_dim_fraction: float = 0.5
        # --- Frozen off-the-shelf embedding (see scripts/build_frozen_anchor_tables.py) ---
        # Bundle directory holding tables.safetensors + group_id.json + meta.json. Normally
        # the same path is given as anchor_embedding_path so the group structure and the
        # table values come from one consistent bundle. Setting anchor_embedding_path to the
        # bundle while leaving this None keeps the bundle's groups but randomly initializes
        # the values -- the "does pretrained matter?" control arm.
        pretrained_bundle_path: str | None = None
        # Freeze the token embedding. Frozen params drop out of the optimizer automatically
        # (components/optimizer.py filters on requires_grad), so nothing else needs changing.
        # NOTE: this also means the embedding stops receiving weight decay, which is a real
        # difference from the trainable baseline beyond the frozen values themselves.
        freeze_tok_embeddings: bool = False
        # Freeze the LM head. Only meaningful when NOT weight-tying: under tying the head IS
        # the embedding, so freeze_tok_embeddings already covers it.
        freeze_output: bool = False
        # Learnable scalar softmax temperature on the tied anchor head. Strongly recommended
        # whenever the tied head is frozen -- see TiedAnchorOutput for why.
        learnable_logit_scale: bool = False
        init_logit_scale: float = 1.0
        # Cache the dense [vocab_size, dim] matrix once at init instead of rebuilding it every
        # microbatch. Requires freeze_tok_embeddings.
        cache_dense_embedding: bool = False
        cache_embedding_dtype: str | None = None
        layer: TransformerBlock.Config
        enable_contrastive_alignment: bool = False
        contrastive_proj_dim: int = 512
        contrastive_target_layer: int = 4  # -1 for embeddings, 0 for first layer, etc.

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            # Sync rope max_seq_len
            import dataclasses as _dc

            self.rope = _dc.replace(self.rope, max_seq_len=seq_len)

            if (
                parallelism.context_parallel_degree > 1
                and self.layer.attention.attn_backend == "varlen"
            ):
                raise NotImplementedError(
                    f"Context Parallel only supports SDPA and FlexAttention."
                    f"Got attn_backend='{self.layer.attention.attn_backend}'. "
                    f"Varlen attention is not supported with CP."
                )

            if (
                self.anchor_embedding_path is not None
                and parallelism.tensor_parallel_degree > 1
            ):
                raise NotImplementedError(
                    "anchor_embedding_path (HybridAnchorEmbedding) does not yet support "
                    "tensor parallelism: RowwiseParallel cannot shard a non-nn.Embedding "
                    "tok_embeddings module. Use FSDP-only (tensor_parallel_degree=1)."
                )

            # --- frozen / pretrained embedding guard rails ---
            if self.freeze_output and self.enable_weight_tying:
                raise ValueError(
                    "freeze_output is meaningless with enable_weight_tying=True -- the head IS "
                    "the embedding. Use freeze_tok_embeddings instead."
                )
            if self.learnable_logit_scale and not (
                self.enable_weight_tying and self.anchor_embedding_path is not None
            ):
                raise ValueError(
                    "learnable_logit_scale is only wired into TiedAnchorOutput, which requires "
                    "enable_weight_tying=True together with anchor_embedding_path."
                )
            if self.cache_dense_embedding and not self.freeze_tok_embeddings:
                raise ValueError(
                    "cache_dense_embedding requires freeze_tok_embeddings=True; otherwise the "
                    "cached matrix goes stale on the first optimizer step."
                )
            if self.pretrained_bundle_path is not None:
                if self.anchor_embedding_path is None:
                    raise ValueError(
                        "pretrained_bundle_path requires anchor_embedding_path (normally the "
                        "same bundle directory, so groups and values come from one bundle)."
                    )
                if abs(self.anchor_shared_dim_fraction - 0.5) > 1e-9:
                    raise ValueError(
                        "pretrained_bundle_path requires anchor_shared_dim_fraction == 0.5: the "
                        "structured embedding is concat(anchor, own) with equal halves, each of "
                        f"width source_dim. Got {self.anchor_shared_dim_fraction}."
                    )
                if training.dtype != "float32":
                    raise ValueError(
                        "pretrained_bundle_path requires training.dtype='float32': the bundle is "
                        "rescaled offline to a target RMS that a bf16 master copy would not "
                        f"preserve. Got {training.dtype}."
                    )
            elif self.freeze_tok_embeddings and self.anchor_embedding_path is not None:
                logger.warning(
                    "freeze_tok_embeddings=True with no pretrained_bundle_path: freezing a "
                    "RANDOMLY initialized embedding. This is a valid control arm, but make sure "
                    "it is the one you meant."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                2 * (self.dim // self.layer.attention.n_heads),
                seq_len,
            )
    def __init__(self, config: Config):
        super().__init__(config)
        self.enable_weight_tying = config.enable_weight_tying
        self.has_anchor_embedding = config.anchor_embedding_path is not None

        if self.has_anchor_embedding:
            self.tok_embeddings = HybridAnchorEmbedding.Config(
                vocab_size=config.vocab_size,
                dim=config.dim,
                anchor_map_path=config.anchor_embedding_path,
                shared_dim_fraction=config.anchor_shared_dim_fraction,
                pretrained_bundle_path=config.pretrained_bundle_path,
                freeze=config.freeze_tok_embeddings,
                cache_dense_matrix=config.cache_dense_embedding,
                cache_dtype=config.cache_embedding_dtype,
            ).build()
            if self.enable_weight_tying:
                # No .weight to alias (HybridAnchorEmbedding has none) -- the LM head instead
                # recomputes from the SAME anchor_table/residual_table params every forward.
                # Real gradient-level tying, never needs re-establishing.
                self.output = TiedAnchorOutput(
                    self.tok_embeddings,
                    learnable_logit_scale=config.learnable_logit_scale,
                    init_logit_scale=config.init_logit_scale,
                )
        else:
            if self.enable_weight_tying:
                self.tok_embeddings.weight = self.output.weight
            # Set here rather than in init_weights: parallelize_fn runs first, and
            # fully_shard() reads requires_grad when it wraps the module. Under weight tying
            # tok_embeddings.weight IS output.weight, so freezing one freezes both -- which is
            # why freeze_output is rejected in that case.
            if config.freeze_tok_embeddings:
                self.tok_embeddings.weight.requires_grad_(False)
            if config.freeze_output and not self.enable_weight_tying:
                self.output.weight.requires_grad_(False)

    def init_weights(
        self,
        *,
        buffer_device: torch.device | None = None,
        **kwargs,
    ):
        # The token embedding initialization produces weights with too large
        # standard deviation for the output layer. Under weight_tying, both should
        # use the output weights with a smaller, truncated normal distribution to
        # improve training stability.
        if self.enable_weight_tying and not self.has_anchor_embedding:
            # since when the model is initialized on meta device,
            # the tying in the __init__ may not have worked correctly
            # we ensure the weights are tied here
            assert self.tok_embeddings is not None and self.output is not None
            self.tok_embeddings.weight = self.output.weight

        super().init_weights(buffer_device=buffer_device, **kwargs)

        # Re-assert the freezes last: to_empty() and the model converters both rebuild
        # parameter storage between __init__ and here. Idempotent. (The anchor case
        # re-asserts its own freeze inside HybridAnchorEmbedding.init_weights.)
        if not self.has_anchor_embedding:
            if self.config.freeze_tok_embeddings:
                self.tok_embeddings.weight.requires_grad_(False)
            if self.config.freeze_output and not self.enable_weight_tying:
                self.output.weight.requires_grad_(False)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        logger.info(f"Llama3Model params: {n_train:,} trainable, {n_frozen:,} frozen")

    def _init_tok_embeddings(self):
        if self.has_anchor_embedding:
            # Tied case: this table is also read directly as unembedding logits (via
            # TiedAnchorOutput), so it needs the same small truncated-normal scale
            # Decoder._init_output() uses for a standalone output layer -- see
            # HybridAnchorEmbedding.init_weights() for why this can't just rely on the stock
            # _init_output() overwrite-after-the-fact trick used by the non-anchor tied case.
            final_out_std = self.config.dim**-0.5 if self.enable_weight_tying else None
            self.tok_embeddings.init_weights(final_out_std=final_out_std)
        else:
            super()._init_tok_embeddings()

    def _init_output(self):
        if self.has_anchor_embedding and self.enable_weight_tying:
            # TiedAnchorOutput has no independent weight, but it may own the scalar logit
            # scale -- and since we return early here, Decoder._init_output() never runs, so
            # this is the only place that initializes it.
            if getattr(self.output, "log_logit_scale", None) is not None:
                self.output.log_logit_scale.fill_(math.log(self.config.init_logit_scale))
            return
        super()._init_output()

