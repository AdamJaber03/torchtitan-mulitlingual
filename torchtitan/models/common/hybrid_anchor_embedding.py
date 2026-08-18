# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

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
        # --- Frozen off-the-shelf embedding ---
        # Directory produced by scripts/build_frozen_anchor_tables.py holding
        # tables.safetensors ("anchor_table" [G, D_anchor], "residual_table" [V, D_residual])
        # and meta.json. When set, both tables are initialized from those pretrained values
        # instead of randomly. Normally the SAME directory is passed as anchor_map_path, so
        # the group assignment and the table values come from one consistent bundle; leaving
        # this None while anchor_map_path points at the bundle keeps the bundle's group
        # structure but randomly initializes the values (the "random control" arm).
        pretrained_bundle_path: str | None = None
        # requires_grad=False on both tables. They then drop out of the optimizer via
        # components/optimizer.py's `if p.requires_grad` filter -- no trainer change needed.
        freeze: bool = False
        # Frozen-only fast path: cache the dense [vocab_size, dim] matrix once at init as a
        # non-persistent buffer, so forward() is a single F.embedding and TiedAnchorOutput is
        # a single F.linear. Without it, a tied anchor head re-runs materialize_full_matrix()
        # EVERY microbatch: two DTensor all-gathers plus a [V, dim] fp32 cat that autograd
        # keeps alive for the backward. Illegal without freeze (the cache would go stale).
        cache_dense_matrix: bool = False
        cache_dtype: str | None = None  # None = param dtype; "bfloat16" halves the cache

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        assert 0.0 <= config.shared_dim_fraction <= 1.0, (
            "shared_dim_fraction must be in [0, 1]: 0.0 = fully independent per-token embeddings "
            "(no shared anchor subspace), 1.0 = fully tied (no independent residual subspace)."
        )
        self.dim_anchor = round(config.dim * config.shared_dim_fraction)
        self.dim_residual = config.dim - self.dim_anchor
        # The two extremes are supported as ablation endpoints by DROPPING the degenerate
        # (zero-width) table rather than creating a 0-dim nn.Embedding -- a 0-numel parameter is a
        # fragile / untested case under FSDP sharding. has_anchor False (fraction 0.0) -> this is
        # just a plain per-token embedding (no tying); has_residual False (fraction 1.0) -> matched
        # pairs share their ENTIRE embedding row (full tying, like the data-level SharedAnchorRemap
        # but at the parameter level).
        self.has_anchor = self.dim_anchor > 0
        self.has_residual = self.dim_residual > 0

        if self.has_anchor:
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
        else:
            self._group_id_list, num_groups = None, 0
            self.anchor_table = None

        self.residual_table = (
            nn.Embedding(config.vocab_size, self.dim_residual) if self.has_residual else None
        )

        if config.pretrained_bundle_path is not None:
            assert self.has_anchor and self.has_residual, (
                "pretrained_bundle_path needs both subspaces (0 < shared_dim_fraction < 1)"
            )
        assert not (config.cache_dense_matrix and not config.freeze), (
            "cache_dense_matrix requires freeze=True: the cache is built once at init and "
            "would silently go stale as soon as the tables received an optimizer update."
        )
        if config.freeze:
            # Set here, in __init__, NOT only in init_weights(): parallelize_fn runs before
            # init_weights, and fully_shard() copies requires_grad onto the sharded parameter
            # at wrap time. Setting it late would leave FSDP2 treating these as trainable.
            for table in (self.anchor_table, self.residual_table):
                if table is not None:
                    table.weight.requires_grad_(False)

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

        An optional third field "identity_shift:<V>:<token_fraction>" ties only the first
        `token_fraction` of the token pairs: {V+i: i for i in range(round(V*token_fraction))}. The
        remaining tagged tokens (and their untagged counterparts) stay UNTIED -- each gets its own
        independent group. token_fraction is the fraction of tokens that are anchored at all, which
        is orthogonal to shared_dim_fraction (the fraction of the embedding *dim* shared within an
        anchored pair). ":1.0" (or omitting it) == the plain full-tie form. E.g. with V=65536,
        ":0.5" ties V+i -> i only for i in [0, 32768); ids [32768, 65536) and [V+32768, 2V) are
        independent.

        A `map_path` that is a DIRECTORY is a prebuilt bundle from
        scripts/build_frozen_anchor_tables.py: its group_id.json already holds the final
        token->group assignment, so it is used verbatim. That script owns richer anchor
        semantics than an id_remap can express (Arabic tokens whose English translation is
        multi-token get a mean-of-those-embeddings anchor; unmatched Arabic tokens share one
        global-English-mean anchor), which is why the assignment is precomputed offline
        rather than derived here. JSON rather than a tensor file on purpose: this runs under
        the model's meta-device construction context, where building tensors is unsafe.
        """
        if os.path.isdir(map_path):
            with open(os.path.join(map_path, "group_id.json")) as f:
                bundle = json.load(f)
            group_id = [int(g) for g in bundle["group_id"]]
            num_groups = int(bundle["num_groups"])
            if len(group_id) != vocab_size:
                raise ValueError(
                    f"{map_path}/group_id.json has {len(group_id)} entries but the model's "
                    f"vocab_size is {vocab_size}"
                )
            if group_id and not (0 <= min(group_id) and max(group_id) < num_groups):
                raise ValueError(f"{map_path}/group_id.json has out-of-range group ids")
            return group_id, num_groups

        if map_path.startswith("identity_shift:"):
            # "identity_shift:<V>" or "identity_shift:<V>:<token_fraction>"
            fields = map_path.split(":")
            V = int(fields[1])
            token_fraction = float(fields[2]) if len(fields) > 2 else 1.0
            assert 0.0 <= token_fraction <= 1.0, (
                f"identity_shift token_fraction must be in [0, 1], got {token_fraction}"
            )
            num_tied = round(V * token_fraction)
            id_remap = {V + i: i for i in range(num_tied)}
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

    @staticmethod
    def _write_full_into_param(param: nn.Parameter, full: torch.Tensor) -> None:
        """Write a rank-identical CPU tensor into `param`, which is an FSDP2-sharded DTensor
        by the time init_weights runs. Caller holds torch.no_grad().

        src_data_rank=None is load-bearing: every rank has already read the same file, so
        each slices its own shard locally. The default (src_data_rank=0) would instead issue
        a scatter collective from rank 0.
        """
        if not isinstance(param, DTensor):
            param.data.copy_(full.to(device=param.device, dtype=param.dtype))
            return
        param.data.copy_(
            distribute_tensor(
                full.to(param.dtype), param.device_mesh, param.placements, src_data_rank=None
            )
        )

    def _load_pretrained_tables(self) -> None:
        """Fill both tables from the bundle's tables.safetensors.

        safe_open mmaps the file, so the 8 ranks on a node share one page-cache copy of it
        rather than each heap-allocating the whole matrix.
        """
        path = os.path.join(self.config.pretrained_bundle_path, "tables.safetensors")
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            for name, table in (
                ("anchor_table", self.anchor_table),
                ("residual_table", self.residual_table),
            ):
                if table is None:
                    continue
                if name not in f.keys():
                    raise KeyError(f"{path} is missing tensor '{name}' (has {list(f.keys())})")
                full = f.get_tensor(name).float()
                if tuple(full.shape) != tuple(table.weight.shape):
                    raise ValueError(
                        f"{path}:{name} has shape {tuple(full.shape)} but the model expects "
                        f"{tuple(table.weight.shape)} -- vocab_size/dim/bundle mismatch"
                    )
                self._write_full_into_param(table.weight, full)
                rms = full.pow(2).mean().sqrt().item()
                logger.info(f"HybridAnchorEmbedding: loaded {name} {tuple(full.shape)} rms={rms:.6f}")

    def _build_dense_cache(self) -> None:
        """Frozen-only: materialize [vocab_size, dim] once and hold it as a NON-persistent
        buffer. Non-persistent so DCP never writes this replicated copy into every checkpoint
        -- the sharded tables stay the single checkpointed source of truth."""
        dense = self.materialize_full_matrix().detach()
        if self.config.cache_dtype is not None:
            dense = dense.to(getattr(torch, self.config.cache_dtype))
        self.register_buffer("dense_matrix", dense, persistent=False)
        logger.info(
            f"HybridAnchorEmbedding: cached dense matrix {tuple(dense.shape)} ({dense.dtype}), "
            f"{dense.numel() * dense.element_size() / 2**20:.0f} MiB"
        )

    def _table_dtype(self) -> torch.dtype:
        """The dtype the two-table path would produce right now.

        Read live rather than cached: FSDP2's mixed-precision policy swaps in a bf16 copy of
        each parameter for the duration of the forward, so this is bf16 mid-forward under
        FSDP and fp32 outside it. The dense cache is a *buffer*, which FSDP does not cast, so
        without matching against this the cached path would hand the first transformer block a
        tensor in the wrong dtype (fp32 cache under a bf16 model, or vice versa).
        """
        table = self.residual_table if self.has_residual else self.anchor_table
        return table.weight.dtype

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        dense = getattr(self, "dense_matrix", None)
        if dense is not None:
            out = F.embedding(tokens, dense)
            target = self._table_dtype()
            return out if out.dtype == target else out.to(target)
        parts = []
        if self.has_anchor:
            parts.append(self.anchor_table(self.anchor_group_id[tokens]))
        if self.has_residual:
            parts.append(self.residual_table(tokens))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

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
        tables = [t for t in (self.anchor_table, self.residual_table) if t is not None]
        if self.config.pretrained_bundle_path is not None:
            # The bundle was already rescaled offline to the right RMS for its intended head
            # (dim**-0.5 when tied, 1.0 when untied), so final_out_std does not apply here.
            self._load_pretrained_tables()
        elif final_out_std is not None:
            cutoff_factor = 3
            for table in tables:
                trunc_normal_(
                    table.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )
        else:
            for table in tables:
                nn.init.normal_(table.weight)
        # anchor_group_id is a deterministic function of the map file + vocab_size, but its
        # meta-time value (from __init__) gets discarded by to_empty(device=init_device) before
        # this runs -- so refill it here from the cached Python list, on the now-real device.
        # (Only present when there is an anchor subspace; fraction 0.0 has no anchor_group_id.)
        if self.has_anchor:
            self.anchor_group_id.copy_(
                torch.tensor(
                    self._group_id_list, dtype=torch.long, device=self.anchor_group_id.device
                )
            )

        if self.config.freeze:
            # Re-assert: to_empty() and the model converters both rebuild parameter storage
            # between __init__ and here.
            for table in tables:
                table.weight.requires_grad_(False)
        if self.config.cache_dense_matrix:
            # After anchor_group_id is filled -- materialize_full_matrix() indexes with it.
            self._build_dense_cache()

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
        dense = getattr(self, "dense_matrix", None)
        if dense is not None:
            return dense  # frozen: the tables can no longer change, so the cache is exact
        parts = []
        if self.has_anchor:
            anchor_weight = self.anchor_table.weight
            if isinstance(anchor_weight, DTensor):
                anchor_weight = anchor_weight.full_tensor()
            parts.append(anchor_weight[self.anchor_group_id])  # [V, D_anchor]
        if self.has_residual:
            residual_weight = self.residual_table.weight
            if isinstance(residual_weight, DTensor):
                residual_weight = residual_weight.full_tensor()
            parts.append(residual_weight)  # [V, D_residual]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)  # [V, D]


class TiedAnchorOutput(nn.Module):
    """LM-head replacement used when enable_weight_tying + anchor_embedding_path are both set.

    output = h @ materialize_full_matrix().T, recomputed every forward from the SAME
    anchor_table/residual_table parameters tok_embeddings uses -- real gradient-level tying, not
    a snapshot. Holds no parameters of its own (other than the optional logit scale below);
    references tok_embeddings as a submodule (so those params become reachable via two paths in
    the module tree -- harmless, model.parameters() dedups by identity).

    learnable_logit_scale exists for the FROZEN embedding case. Normally a tied head grows or
    shrinks over the first few hundred steps to reach a workable softmax temperature; once the
    embedding is frozen that degree of freedom is gone entirely, and a scale that is too small
    gives a near-uniform softmax while one that is too large saturates it and floods the whole
    model with gradient that max_norm clipping then squashes. This adds the one scalar back.
    """

    def __init__(
        self,
        tok_embeddings: HybridAnchorEmbedding,
        *,
        learnable_logit_scale: bool = False,
        init_logit_scale: float = 1.0,
    ):
        super().__init__()
        self.tok_embeddings = tok_embeddings
        self.init_logit_scale = init_logit_scale
        # Shape (1,), never 0-dim: FSDP2 shards by torch.chunk along dim 0, which a 0-dim
        # parameter does not have. Ranks past the first get a 0-numel shard, which is handled.
        self.log_logit_scale = nn.Parameter(torch.empty(1)) if learnable_logit_scale else None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # materialize_full_matrix() reads .weight directly (not via the module's __call__), which
        # bypasses FSDP's mixed-precision forward hook -- it returns the fp32 master DTensor's
        # all-gathered value, not the bf16 "compute dtype" copy FSDP would otherwise substitute.
        # Cast explicitly so this matches h's dtype, exactly what the hook would have done.
        weight = self.tok_embeddings.materialize_full_matrix()
        out = F.linear(h, weight.to(h.dtype))
        if self.log_logit_scale is not None:
            out = out * self.log_logit_scale.to(out.dtype).exp()
        return out
