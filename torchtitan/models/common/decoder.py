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

CONTRASTIVE_TARGET_LAYER = 4

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
        # --- NEW: Contrastive Projection Shield ---
        self.enable_contrastive = getattr(config, "enable_contrastive_alignment", False)
        if self.enable_contrastive:
            proj_dim = getattr(config, "contrastive_proj_dim", 512)
            self.contrastive_proj = nn.Sequential(
                nn.Linear(config.dim * 2, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim)
            )
        else:
            self.contrastive_proj = None

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
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            # pyrefly: ignore [not-callable]
            layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.config.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )
        # --- NEW: Initialize MLP weights ---
        if self.enable_contrastive and self.contrastive_proj is not None:
            buffer_device = kwargs.get("buffer_device") or self.freqs_cis.device
            for mod in self.contrastive_proj.modules():
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

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        # Shape strictly provided by Dataloader: [Batch, Max_Seqs, SeqLen]
        contrastive_masks: torch.Tensor | None = None, 
    ):
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        contrastive_vectors = None 
        valid_seq_mask = None

        for layer_id_str, layer in self.layers.items():
            h = layer(h, self.freqs_cis, attention_masks, positions)
            
            if self.enable_contrastive and contrastive_masks is not None:
                if int(layer_id_str) == CONTRASTIVE_TARGET_LAYER:
                    
                    # 1. EXPAND SHAPES FOR BATCHED POOLING
                    h_expanded = h.unsqueeze(1) # [B, 1, SeqLen, Dim]
                    float_mask = contrastive_masks.unsqueeze(-1).to(h.dtype) # [B, MaxSeqs, SeqLen, 1]
                    bool_mask = contrastive_masks.unsqueeze(-1).bool()   # [B, MaxSeqs, SeqLen, 1]

                    # 2. MEAN POOLING
                    sum_embeddings = (h_expanded * float_mask).sum(dim=2) 
                    valid_counts = float_mask.sum(dim=2).clamp(min=1e-9).to(h.dtype)  
                    mean_pooled = sum_embeddings / valid_counts           

                    # 3. MAX POOLING
                    h_masked = h_expanded.masked_fill(~bool_mask, -1e9)
                    max_pooled = h_masked.max(dim=2)[0]                   

                    # 4. COMBINE AND PROJECT
                    combined_pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
                    
                    # Output Shape: [Batch, MaxSeqs, Proj_Dim]
                    contrastive_vectors = self.contrastive_proj(combined_pooled)
                    
                    # Create a boolean mask of valid sequences to pass to the loss function
                    # (If a sequence mask had >0 valid tokens, it's a real sequence)
                    # Shape: [Batch, MaxSeqs]
                    valid_seq_mask = (contrastive_masks.sum(dim=-1) > 0)

        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        
        if self.enable_contrastive:
            return {
                    "output": output,
                    "contrastive_vectors": contrastive_vectors,
                    "valid_seq_mask": valid_seq_mask
                    }
        return {"output": output}

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
