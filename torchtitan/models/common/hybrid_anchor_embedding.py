# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.utils import trunc_normal_
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


class HybridAnchorEmbedding(Module):
    """Token embedding with a partially-tied "anchor" subspace for matched cross-lingual
    (e.g. Arabic/English) token pairs.

    Splits the embedding dim D into:
      - D_anchor = round(D * shared_dim_fraction): looked up via a small ``anchor_table`` of
        shape [num_groups, D_anchor]. Matched AR/EN pairs share a group id, so they index the
        IDENTICAL row -> real, persistent, gradient-level tying (no custom backward needed;
        this is just standard embedding-lookup semantics applied to a deduplicated table).
      - D_residual = D - D_anchor: looked up via a normal per-token ``residual_table`` of
        shape [vocab_size, D_residual]. Fully independent.

    forward(tokens) returns cat([anchor_table(group_id[tokens]), residual_table(tokens)], dim=-1).

    NOTE: this module intentionally has no ``.weight`` attribute (unlike nn.Embedding). Call
    sites that assume ``tok_embeddings.weight`` exists (Decoder._init_tok_embeddings, the active
    forgetting trainer hook) must be special-cased -- see Llama3Model's overrides.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        vocab_size: int
        dim: int
        anchor_map_path: str
        # Fraction of `dim` allocated to the shared/anchor subspace. The remaining
        # (1 - fraction) is the independent residual subspace. Sweepable ablation knob: lower ->
        # closer to fully independent embeddings (weaker anchor, less interference); higher ->
        # closer to full id-remap-style tying (stronger anchor, more interference risk).
        shared_dim_fraction: float = 0.5

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        assert 0.0 < config.shared_dim_fraction < 1.0, (
            "shared_dim_fraction must be in (0, 1) -- use a plain nn.Embedding for 0.0, and the "
            "data-level SharedAnchorRemap (full id-remap tying) for 1.0."
        )
        self.dim_anchor = round(config.dim * config.shared_dim_fraction)
        self.dim_residual = config.dim - self.dim_anchor
        assert self.dim_anchor > 0 and self.dim_residual > 0

        self._group_id_list, num_groups = self._compute_group_id_list(
            config.anchor_map_path, config.vocab_size
        )
        # Persistent buffer: saved/restored by DCP + HF round trip. Registered here with a
        # placeholder value only -- real values get filled in by init_weights() (see there for
        # why: this module is constructed under a meta device context, same as the rest of the
        # model, and to_empty(device=init_device) discards whatever a meta-time value held).
        self.register_buffer(
            "anchor_group_id", torch.zeros(config.vocab_size, dtype=torch.long), persistent=True
        )

        self.anchor_table = nn.Embedding(num_groups, self.dim_anchor)
        self.residual_table = nn.Embedding(config.vocab_size, self.dim_residual)

        logger.info(
            f"HybridAnchorEmbedding: vocab_size={config.vocab_size}, dim={config.dim} -> "
            f"anchor_dim={self.dim_anchor} (shared, {num_groups} groups), "
            f"residual_dim={self.dim_residual} (independent)."
        )

    @staticmethod
    def _compute_group_id_list(map_path: str, vocab_size: int) -> tuple[list[int], int]:
        """Every vocab id gets a group id. Matched AR/EN pairs (from the map's ``id_remap``:
        {arabic_id: english_id}) share a group id == the English id (so English ids are stable
        "canonical" anchors and Arabic ids piggyback onto them). Every other id gets its own
        unique group id, packed densely so anchor_table stays small (num_groups ==
        vocab_size - num_pairs, not vocab_size + num_pairs).

        Pure Python (no tensor ops): this runs from __init__, which executes under the model's
        meta-device construction context -- data-dependent tensor ops like .nonzero() have no
        meta implementation and would crash here.

        The synthetic (non-file) path "identity_shift:<V>" builds the map in memory as
        {V+i: i for i in range(V)} -- i.e. every second-half token V+i is tied to its first-half
        twin i. Used by the tagged-hybrid-anchor setup (translate AR->EN, then tag AR-origin tokens
        into the second vocab half), where the pairing is purely arithmetic and needs no data file.
        """
        if map_path.startswith("identity_shift:"):
            V = int(map_path.split(":", 1)[1])
            id_remap = {V + i: i for i in range(V)}
        else:
            with open(map_path) as f:
                data = json.load(f)
            if "id_remap" in data:
                id_remap = {int(k): int(v) for k, v in data["id_remap"].items()}
            elif "pairs" in data:
                id_remap = {int(p["arabic_id"]): int(p["english_id"]) for p in data["pairs"]}
            else:
                raise ValueError(f"{map_path} must contain 'id_remap' or 'pairs'")

        group_id = [-1] * vocab_size
        next_group = 0
        english_ids = sorted(set(eid for eid in id_remap.values() if eid < vocab_size))
        canon_group = {}
        for eid in english_ids:
            canon_group[eid] = next_group
            group_id[eid] = next_group
            next_group += 1
        for arabic_id, english_id in id_remap.items():
            if arabic_id >= vocab_size or english_id >= vocab_size:
                continue
            group_id[arabic_id] = canon_group[english_id]
        num_groups = next_group
        for i in range(vocab_size):
            if group_id[i] == -1:
                group_id[i] = num_groups
                num_groups += 1
        return group_id, num_groups

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        anchor_ids = self.anchor_group_id[tokens]
        return torch.cat(
            [self.anchor_table(anchor_ids), self.residual_table(tokens)], dim=-1
        )

    def init_weights(self, *, final_out_std: float | None = None, **kwargs) -> None:
        """final_out_std: set this when the embedding is ALSO serving as the tied LM head (via
        TiedAnchorOutput) -- its rows are then read directly as unembedding logits, so both
        sub-tables get the same small truncated-normal scale Decoder._init_output() uses for a
        standalone output layer. Leave it None (default, std=1 normal init) for the untied case,
        where this table is only ever an input embedding -- matching Decoder's stock
        _init_tok_embeddings() convention elsewhere in this codebase.

        This mirrors how the stock (non-anchor) tied case gets its small-std init: there,
        _init_tok_embeddings() (std=1) runs first and _init_output() OVERWRITES the same aliased
        .weight tensor with the small std afterward. TiedAnchorOutput has no independent weight
        for _init_output() to overwrite, so that correction has to happen here instead."""
        if final_out_std is not None:
            cutoff_factor = 3
            for table in (self.anchor_table, self.residual_table):
                trunc_normal_(
                    table.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )
        else:
            nn.init.normal_(self.anchor_table.weight)
            nn.init.normal_(self.residual_table.weight)
        # anchor_group_id is a deterministic function of the map file + vocab_size, but its
        # meta-time value (from __init__) gets discarded by to_empty(device=init_device) before
        # this runs -- so refill it here from the cached Python list, on the now-real device.
        self.anchor_group_id.copy_(
            torch.tensor(
                self._group_id_list, dtype=torch.long, device=self.anchor_group_id.device
            )
        )

    def materialize_full_matrix(self) -> torch.Tensor:
        """Build the full (vocab_size, dim) matrix for HF export / the tied-output forward.
        Uses ALL vocab ids (not just a batch) -- the "embedding table" in the conventional
        sense, post-tying.

        Under FSDP, anchor_table.weight/residual_table.weight are DTensor (sharded along dim 0).
        anchor_group_id (a buffer) and a freshly-built torch.arange() are plain, unsharded
        Tensors -- indexing or embedding-lookup with a plain-Tensor index against a DTensor
        weight is not a supported mixed-type op ("got mixed torch.Tensor and DTensor"). Calling
        .full_tensor() first all-gathers each DTensor into an ordinary, fully-materialized local
        tensor (exactly the "materialize the full matrix" semantics this function is named for),
        after which plain-Tensor indexing is unambiguous and safe."""
        anchor_weight = self.anchor_table.weight
        residual_weight = self.residual_table.weight
        if isinstance(anchor_weight, DTensor):
            anchor_weight = anchor_weight.full_tensor()
        if isinstance(residual_weight, DTensor):
            residual_weight = residual_weight.full_tensor()
        full_anchor = anchor_weight[self.anchor_group_id]  # [V, D_anchor]
        return torch.cat([full_anchor, residual_weight], dim=-1)  # [V, D]


class TiedAnchorOutput(nn.Module):
    """LM-head replacement used when enable_weight_tying + anchor_embedding_path are both set.

    output = h @ materialize_full_matrix().T, recomputed every forward from the SAME
    anchor_table/residual_table parameters tok_embeddings uses -- real gradient-level tying, not
    a snapshot. Holds no parameters of its own; references tok_embeddings as a submodule (so
    those params become reachable via two paths in the module tree -- harmless, see plan notes).
    """

    def __init__(self, tok_embeddings: HybridAnchorEmbedding):
        super().__init__()
        self.tok_embeddings = tok_embeddings

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # materialize_full_matrix() reads .weight directly (not via the module's __call__), which
        # bypasses FSDP's mixed-precision forward hook -- it returns the fp32 master DTensor's
        # all-gathered value, not the bf16 "compute dtype" copy FSDP would otherwise substitute.
        # Cast explicitly so this matches h's dtype, exactly what the hook would have done.
        weight = self.tok_embeddings.materialize_full_matrix()
        return F.linear(h, weight.to(h.dtype))
