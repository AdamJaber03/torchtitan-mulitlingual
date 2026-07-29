# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

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
        en1en2_shared_embeddings_init: bool = False
        # --- Hybrid anchor embedding (partial cross-lingual tying) ---
        # Path to a token map JSON (must contain 'id_remap' or 'pairs', e.g.
        # ar_en_1to1_token_map.json). When set, tok_embeddings is replaced by a
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
        layer: TransformerBlock.Config
        enable_contrastive_alignment: bool = False
        contrastive_proj_dim: int = 512
        # Contrastive projection head: "mlp" (Linear-GELU-Linear, current default), "linear"
        # (single Linear), or "identity" (InfoNCE directly on the pooled layer embeddings).
        contrastive_head_type: str = "mlp"
        contrastive_target_layer: int = 4  # -1 for embeddings, 0 for first layer, etc.
        # When set (non-empty), apply the contrastive loss at EACH of these layers simultaneously
        # (each gets its own head; the per-layer InfoNCE losses are combined in the loss). When None,
        # falls back to the single contrastive_target_layer above.
        contrastive_target_layers: list[int] | None = None
        # Contrastive early-exit budget: number of kept (non-translation) tokens per row that run
        # through the layers AFTER contrastive_target_layer + the output head. When set (with the
        # dataloader emitting contrastive_only_mask), the trainer compacts the batch to this length
        # after the contrastive layer so the translation member only reaches contrastive_target_layer.
        # Typically ~ (2/3) * training.seq_len when training.seq_len is the 1.5x "raw" length.
        keep_len: int | None = None

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
        self.en1en2_shared_embeddings_init = config.en1en2_shared_embeddings_init
        self.has_anchor_embedding = config.anchor_embedding_path is not None
        if self.has_anchor_embedding:
            self.tok_embeddings = HybridAnchorEmbedding.Config(
                vocab_size=config.vocab_size,
                dim=config.dim,
                anchor_map_path=config.anchor_embedding_path,
                shared_dim_fraction=config.anchor_shared_dim_fraction,
            ).build()
            if self.enable_weight_tying:
                # No .weight to alias (HybridAnchorEmbedding has none) -- the LM head instead
                # recomputes from the SAME anchor_table/residual_table params every forward.
                # Real gradient-level tying, never needs re-establishing (see init_weights/
                # reinit_embeddings below: nothing drifts, so there's nothing to re-tie).
                self.output = TiedAnchorOutput(self.tok_embeddings)
        elif self.enable_weight_tying:
            self.tok_embeddings.weight = self.output.weight

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

        # Must run AFTER super().init_weights() -- _init_output() overwrites the full
        # output weight with a fresh truncated normal, which would otherwise clobber this copy.
        if self.en1en2_shared_embeddings_init:
            self._copy_first_half_to_second_half()

    def _copy_first_half_to_second_half(self):
        # Under FSDP2, output.weight/tok_embeddings.weight are sharded DTensors: a plain
        # weight[half:] = weight[:half] slice-assign does not perform a real cross-shard
        # copy. Gather to a full replicated tensor, mutate, and redistribute back.
        half = self.config.vocab_size // 2
        with torch.no_grad():
            for w in (self.output.weight, self.tok_embeddings.weight):
                if isinstance(w, DTensor):
                    full = w.full_tensor()
                    full[half:] = full[:half]
                    w.copy_(distribute_tensor(full, w.device_mesh, w.placements))
                else:
                    w[half:] = w[:half]

    def reinit_embeddings(self):
        # Re-establish tying before reinitializing so the shared input/output weight ends up with the
        # output's (smaller, truncated-normal) distribution, exactly as at initial init. Used by
        # active forgetting (periodic embedding reset) as well as initial init_weights.
        # (Anchor-embedding + tying needs no re-establishing: TiedAnchorOutput always recomputes
        # live from tok_embeddings, it never holds a snapshot that could drift apart.)
        if self.enable_weight_tying and not self.has_anchor_embedding:
            assert self.tok_embeddings is not None and self.output is not None
            self.tok_embeddings.weight = self.output.weight
        super().reinit_embeddings()

        # Must run AFTER super().reinit_embeddings() -- see init_weights() above.
        if self.en1en2_shared_embeddings_init:
            self._copy_first_half_to_second_half()

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
            return  # TiedAnchorOutput has no independent weight; nothing to reinit.
        super()._init_output()

