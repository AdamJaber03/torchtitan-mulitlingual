# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_attention_mask,
    create_varlen_metadata_for_document,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe.moe import MoE
from torchtitan.models.common.rope import RoPE
from torchtitan.models.common.utils import trunc_normal_
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


# TODO: we can unify the TransformerBlock impl across all models when
# there is no special logic for each model, including
# init_weights, ffn vs. moe naming and creation, rope vs. nope, etc.
class TransformerBlock(Module):
    """Base class for all language model transformer blocks.

    All language model TransformerBlocks share:
    - Attention module (from ``attention.build(dim=dim)``)
    - FFN or MoE (from ``feed_forward.build()`` / ``moe.build()``)
    - Two RMSNorms (``attention_norm``, ``ffn_norm``)
    - ``weight_init_std`` computed from ``layer_id``
    - Forward: ``x + attn(norm(x), ...); x + ffn(norm(x))``

    Children implement ``__init__``, ``forward``, and ``init_weights``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm_eps: float = 1e-5
        attention: BaseAttention.Config  # required, no default
        feed_forward: FeedForward.Config | None = None
        moe: MoE.Config | None = None


class Decoder(BaseModel):
    """Base class for autoregressive decoder-only language models.

    Provides shared ``__init__``, ``forward``, ``init_weights``, and
    ``get_attention_masks`` (flex/varlen dispatch) used by most models.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int
        n_layers: int
        vocab_size: int
        norm_eps: float = 1e-5
        # TODO: Right now RoPE config is not in each TransformerBlock / Attention,
        # so that rope cache, a.k.a. freqs_cis, is shared by all layers. However,
        # it causes redundantly passing backend (complex / cos_sin) to both RoPE
        # and Attention. Also RoPE itself as a standalone module requires PP special
        # handling, see below.
        rope: RoPE.Config
        layer: TransformerBlock.Config  # required, no default

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.rope = config.rope.build()
        self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = config.layer.build(
                layer_id=layer_id, dim=config.dim, n_layers=config.n_layers
            )

        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        # --- NEW: Contrastive Projection Head ---
        # Head operates on the concat of mean+max pooled layer features (dim * 2 -> proj_dim).
        #   "mlp"      : Linear -> GELU -> Linear  (SimCLR-style; decouples alignment from the
        #                backbone, so layer-`target` embeddings stay relatively unaligned)
        #   "linear"   : a single Linear (most pressure lands on the backbone)
        #   "identity" : no projection -- InfoNCE runs directly on the pooled layer-`target`
        #                embeddings (proj_dim is ignored; vectors are dim*2)
        self.enable_contrastive = getattr(config, "enable_contrastive_alignment", False)
        self.contrastive_proj = None
        if self.enable_contrastive:
            proj_dim = getattr(config, "contrastive_proj_dim", 512)
            head_type = getattr(config, "contrastive_head_type", "mlp")
            in_dim = config.dim * 2  # mean + max pooled concat
            # contrastive_target_layers (list) -> contrast at EACH layer; else single target layer.
            layers = getattr(config, "contrastive_target_layers", None) or [
                getattr(config, "contrastive_target_layer", 4)
            ]
            self.contrastive_layers = list(layers)
            self.contrastive_head_type = head_type
            # Drop point for the early-exit path (and single-layer back-compat) = last contrast layer.
            self.contrastive_target_layer = max(self.contrastive_layers)
            if len(self.contrastive_layers) == 1:
                # Single layer: keep the original attribute/keys so existing checkpoints load.
                self.contrastive_proj = self._build_contrastive_head(head_type, in_dim, proj_dim)
                self._head_for = {self.contrastive_layers[0]: self.contrastive_proj}
            else:
                # Multi-layer: one head per layer in a ModuleDict (separate attribute).
                self.contrastive_projs = nn.ModuleDict(
                    {
                        str(L): self._build_contrastive_head(head_type, in_dim, proj_dim)
                        for L in self.contrastive_layers
                    }
                )
                self._head_for = {L: self.contrastive_projs[str(L)] for L in self.contrastive_layers}

    def _build_contrastive_head(self, head_type: str, in_dim: int, proj_dim: int) -> nn.Module:
        if head_type == "mlp":
            return nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
            )
        if head_type == "linear":
            return nn.Linear(in_dim, proj_dim)
        if head_type == "identity":
            return nn.Identity()
        raise ValueError(
            f"Unknown contrastive_head_type {head_type!r}; expected 'mlp', 'linear', or 'identity'"
        )

    def init_weights(
        self,
        **kwargs,
    ):
        buffer_device: torch.device | None = kwargs.get("buffer_device")
        buffer_device = buffer_device or self.freqs_cis.device
        if self.rope is not None:
            self.rope.init_weights(buffer_device=buffer_device)
            self.freqs_cis = self.rope.cache
        else:
            # PP case: rope module was pruned, rebuild to get freqs_cis
            rope = self.config.rope.build()
            rope.init_weights(buffer_device=buffer_device)
            self.freqs_cis = rope.cache
        self._init_tok_embeddings()
        for layer in self.layers.values():
            # pyrefly: ignore [not-callable]
            layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        self._init_output()
        # --- NEW: Initialize contrastive head weights (single self.contrastive_proj or, for the
        # multi-depth case, every head in self.contrastive_projs). Identity heads have no Linears. ---
        if self.enable_contrastive:
            heads = []
            if self.contrastive_proj is not None:
                heads.append(self.contrastive_proj)
            if getattr(self, "contrastive_projs", None) is not None:
                heads.extend(self.contrastive_projs.values())
            for head in heads:
                for mod in head.modules():
                    if isinstance(mod, nn.Linear):
                        trunc_normal_(
                            mod.weight,
                            mean=0.0,
                            std=self.config.dim**-0.5,
                            a=-3 * (self.config.dim**-0.5),
                            b=3 * (self.config.dim**-0.5),
                        )
                        if mod.bias is not None:
                            nn.init.zeros_(mod.bias)

    def _init_tok_embeddings(self):
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)

    def _init_output(self):
        if self.output is not None:
            final_out_std = self.config.dim**-0.5
            cutoff_factor = 3
            trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def reinit_embeddings(self):
        """Reinitialize the token (input) embeddings and the output projection, using the same
        distributions as ``init_weights`` (so post-reset matches the original init). The transformer
        body is left untouched.

        Used by active forgetting (periodic embedding reset during training). Operates in place on the
        (possibly sharded DTensor) weights, so callers should wrap this in ``torch.no_grad()``.
        Subclasses with weight tying override this to re-tie first.
        """
        self._init_tok_embeddings()
        self._init_output()

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        # Shape strictly provided by Dataloader: [Batch, Max_Seqs, SeqLen]
        contrastive_masks: torch.Tensor | None = None,
        # Per-token early-exit flag from the dataloader. The Trainer consumes it (pops it and turns
        # it into contrastive_keep_index below); the model itself does NOT use it. It is accepted
        # here only so callers that forward dataloader inputs verbatim (e.g. the Validator, which
        # does not early-exit) don't error on the extra kwarg.
        contrastive_only_mask: torch.Tensor | None = None,
        # --- Contrastive early-exit (built by Trainer.post_dataloading_process) ---
        # When provided, after the contrastive layer the kept (non-translation) tokens are compacted
        # to [Batch, keep_len] (flat gather over the batch) and ONLY they run through the remaining
        # layers + output head, using ``reduced_attention_masks``. The translation tokens only ever
        # reach ``contrastive_target_layer`` (where their contrastive vectors are pooled) and get no
        # CE. ``contrastive_keep_index`` holds flat indices into [0, Batch*SeqLen); pad slots are
        # marked invalid in ``contrastive_keep_valid`` and are not scattered back.
        reduced_attention_masks: AttentionMasksType | None = None,
        contrastive_keep_index: torch.Tensor | None = None,
        contrastive_keep_valid: torch.Tensor | None = None,
    ):
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        compacting = (
            self.enable_contrastive
            and contrastive_keep_index is not None
            and reduced_attention_masks is not None
        )
        # `drop_at` is the early-exit drop point = the LAST (deepest) contrastive layer, so every
        # contrastive layer is pooled from the full hidden state before any compaction.
        drop_at = self.contrastive_target_layer if self.enable_contrastive else None
        contrast_layers = set(self.contrastive_layers) if self.enable_contrastive else set()
        multi = self.enable_contrastive and len(self.contrastive_layers) > 1

        contrastive_vectors = {} if multi else None
        valid_seq_mask = None
        h_full = None  # full-length hidden state captured at the drop layer (for scatter-back)

        def _store(lid_, vecs):
            nonlocal contrastive_vectors
            if multi:
                contrastive_vectors[lid_] = vecs
            else:
                contrastive_vectors = vecs

        # Special case: contrast at the embedding layer (-1), before any block.
        if self.enable_contrastive and contrastive_masks is not None and -1 in contrast_layers:
            vecs, valid_seq_mask = self._pool_contrastive(h, contrastive_masks, self._head_for[-1])
            _store(-1, vecs)
        if compacting and drop_at == -1:
            h_full = h
            h = self._gather_kept(h, contrastive_keep_index)

        for layer_id_str, layer in self.layers.items():
            lid = int(layer_id_str)
            mask_for_layer = (
                reduced_attention_masks if (compacting and lid > drop_at) else attention_masks
            )
            h = layer(h, self.freqs_cis, mask_for_layer, positions)

            # Pool the contrastive vectors from the FULL hidden state, before any compaction.
            if self.enable_contrastive and contrastive_masks is not None and lid in contrast_layers:
                vecs, valid_seq_mask = self._pool_contrastive(h, contrastive_masks, self._head_for[lid])
                _store(lid, vecs)

            # Drop the translation tokens for the deeper layers (after the last contrastive layer).
            if compacting and lid == drop_at:
                h_full = h
                h = self._gather_kept(h, contrastive_keep_index)

        # Scatter the deep representations of the kept tokens back to full length so the output head
        # + CE operate on [Batch, SeqLen]. Translation / pad positions retain their target-layer
        # state, but their labels are IGNORE_INDEX so CE skips them.
        if compacting and h_full is not None:
            h = self._scatter_kept(
                h_full, h, contrastive_keep_index, contrastive_keep_valid
            )

        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h

        if self.enable_contrastive:
            return {
                    "output": output,
                    "contrastive_vectors": contrastive_vectors,
                    "valid_seq_mask": valid_seq_mask
                    }
        return {"output": output}

    def _pool_contrastive(
        self, h: torch.Tensor, contrastive_masks: torch.Tensor, head: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean+max pool the hidden state over each contrastive sequence mask, then project with `head`.

        Args:
            h: [Batch, SeqLen, Dim] hidden state at the contrastive layer.
            contrastive_masks: [Batch, MaxSeqs, SeqLen] boolean per-sequence masks.
            head: the projection head to apply (per-layer for the multi-depth case).
        Returns:
            (contrastive_vectors [Batch, MaxSeqs, Proj_Dim], valid_seq_mask [Batch, MaxSeqs]).
        """
        h_expanded = h.unsqueeze(1)                                   # [B, 1, S, D]
        float_mask = contrastive_masks.unsqueeze(-1).to(h.dtype)      # [B, MaxSeqs, S, 1]
        bool_mask = contrastive_masks.unsqueeze(-1).bool()
        sum_embeddings = (h_expanded * float_mask).sum(dim=2)
        valid_counts = float_mask.sum(dim=2).clamp(min=1e-9).to(h.dtype)
        mean_pooled = sum_embeddings / valid_counts
        h_masked = h_expanded.masked_fill(~bool_mask, -1e9)
        max_pooled = h_masked.max(dim=2)[0]
        combined_pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
        contrastive_vectors = head(combined_pooled)                   # [B, MaxSeqs, Proj_Dim]
        valid_seq_mask = (contrastive_masks.sum(dim=-1) > 0)          # [B, MaxSeqs]
        return contrastive_vectors, valid_seq_mask

    def _gather_kept(self, h: torch.Tensor, keep_index: torch.Tensor) -> torch.Tensor:
        """Gather kept tokens (flat indices into [0, B*S)) into [B, keep_len, D]."""
        B, S, D = h.shape
        keep_len = keep_index.shape[1]
        gathered = h.reshape(B * S, D)[keep_index.reshape(-1)]        # [B*keep_len, D]
        return gathered.view(B, keep_len, D)

    def _scatter_kept(
        self,
        h_full: torch.Tensor,
        h_deep: torch.Tensor,
        keep_index: torch.Tensor,
        keep_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        """Write deep representations of kept tokens back to their original [B, S] positions.

        Padding slots are redirected to a throwaway scratch row, keeping this a fixed-shape,
        torch.compile-friendly ``index_copy_`` (no boolean-masked dynamic indexing).
        """
        B, S, D = h_full.shape
        out_ext = torch.cat(
            [h_full.reshape(B * S, D), h_full.new_zeros(1, D)], dim=0
        )  # [B*S + 1, D]; last row is the discarded scratch slot
        idx_flat = keep_index.reshape(-1).clone()
        if keep_valid is not None:
            scratch = torch.full_like(idx_flat, B * S)
            idx_flat = torch.where(keep_valid.reshape(-1), idx_flat, scratch)
        out_ext.index_copy_(0, idx_flat, h_deep.reshape(B * keep_index.shape[1], D))
        return out_ext[: B * S].view(B, S, D)

    def _get_flex_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]

        match self.attn_config.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                assert tokenizer.eos_id is not None
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.attn_config.attn_mask_type}"
                )

        return create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        match self.attn_config.attn_backend:
            case "flex":
                return self._get_flex_attention_masks(
                    input_batch, tokenizer, extra_inputs
                )
            case "varlen":
                if self.attn_config.attn_mask_type != "block_causal":
                    raise ValueError(
                        f"varlen attention is only supported with block_causal "
                        f"attention mask type, got {self.attn_config.attn_mask_type}"
                    )
                assert tokenizer.eos_id is not None
                return create_varlen_metadata_for_document(
                    input_batch, tokenizer.eos_id
                )
            case _:
                raise TypeError("Only varlen and flex attn masks are supported")

    @property
    def attn_config(self):
        """Convenience accessor for the attention config from layer."""
        return self.config.layer.attention
