# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# H-Net (Hierarchical Network with Dynamic Chunking) torchtitan integration.
# Wraps the vendored goombalab/hnet reference model (see ._vendor) so that it
# plugs into torchtitan's ModelSpec / BaseModel contract like llama3 / qwen3.
#
# Reference: "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
# Hwang, Wang, Gu (arXiv:2507.07955).

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.protocols.model import BaseModel
from torchtitan.tools.logging import logger

# config_hnet has no CUDA-kernel dependencies, so it is safe to import eagerly.
from ._vendor.config_hnet import AttnConfig, HNetConfig, SSMConfig


def _load_hnet_impl():
    """Lazily import the kernel-dependent H-Net implementation.

    The vendored model depends on mamba_ssm / causal_conv1d / flash_attn (CUDA
    kernels). Importing them is deferred to model-build time so the rest of
    torchtitan can be imported on machines without those kernels.
    """
    try:
        from ._vendor.mixer_seq import HNetForCausalLM

        return HNetForCausalLM
    except ImportError as e:  # pragma: no cover - depends on GPU env
        raise ImportError(
            "H-Net requires the Mamba-2 / FlashAttention CUDA kernels "
            "(mamba_ssm, causal_conv1d, flash_attn), which are not importable. "
            "Install them on a GPU node — see torchtitan/models/hnet/README.md. "
            f"Original import error: {e!r}"
        ) from e


def hnet_load_balancing_loss(
    boundary_prob: torch.Tensor, boundary_mask: torch.Tensor, N: float
) -> torch.Tensor:
    """Faithful port of goombalab/hnet ``load_balancing_loss``.

    Encourages the realized chunking ratio to match the target downsampling
    factor ``N`` (> 1). Computed per micro-batch (matches upstream).
    """
    tokenized_prob = boundary_prob[..., -1]
    true_ratio = boundary_mask.float().mean()
    average_prob = tokenized_prob.float().mean()
    return (
        (1 - true_ratio) * (1 - average_prob) + true_ratio * average_prob * (N - 1)
    ) * N / (N - 1)


class HNetModel(BaseModel):
    """torchtitan model wrapper around the vendored ``HNetForCausalLM``.

    The forward returns ``{"output": logits, "ratio_loss": <scalar>}`` where
    ``ratio_loss`` is the summed chunking load-balancing loss across hierarchy
    stages. Combine it with cross-entropy via the ``hnet_ratio`` registered loss
    (see torchtitan/components/loss.py) and a ``LossConfig``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        # --- H-Net architecture (mirrors _vendor.config_hnet.HNetConfig) ---
        # arch_layout: nested list describing the hierarchy. For a 1-stage net:
        #   [ "<enc>", [ "<main>" ], "<dec>" ]  where each "<...>" is a layer
        #   spec string like "m4" (4 Mamba layers) or "m2T2" (2 Mamba + 2 attn+FFN).
        arch_layout: list = field(default_factory=list)
        d_model: list[int] = field(default_factory=list)
        # FFN intermediate dim per stage (0 => no FFN for lowercase arch letters)
        d_intermediate: list[int] = field(default_factory=list)
        vocab_size: int = 256  # byte-level
        tie_embeddings: bool = False
        initializer_range: float = 0.02

        # --- SSM (Mamba-2) hyperparameters (shared across stages) ---
        ssm_d_conv: int = 4
        ssm_expand: int = 2
        ssm_d_state: int = 128
        ssm_chunk_size: int = 256

        # --- Attention hyperparameters (one entry per stage) ---
        attn_num_heads: list[int] = field(default_factory=list)
        attn_rotary_emb_dim: list[int] = field(default_factory=list)
        attn_window_size: list[int] = field(default_factory=list)
        # KV heads per stage for grouped-query attention (GQA). Empty => each
        # stage defaults to attn_num_heads (standard MHA).
        attn_num_kv_heads: list[int] = field(default_factory=list)

        # --- Dynamic chunking ratio loss ---
        # Target downsampling factor per non-innermost stage (outer-first).
        target_compression: list[float] = field(default_factory=lambda: [4.0])

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            parallelism = trainer_config.parallelism
            unsupported = {
                "tensor_parallel_degree": "Tensor Parallel",
                "pipeline_parallel_degree": "Pipeline Parallel",
                "context_parallel_degree": "Context Parallel",
            }
            for attr, label in unsupported.items():
                degree = getattr(parallelism, attr, 1) or 1
                if degree > 1:
                    raise NotImplementedError(
                        f"{label} is not supported for H-Net: dynamic chunking "
                        f"produces variable-length inner sequences that are "
                        f"incompatible with {label}. Use FSDP / data parallelism "
                        f"(data_parallel_shard_degree / data_parallel_replicate_degree)."
                    )
            ep = getattr(parallelism, "expert_parallel_degree", 1) or 1
            if ep > 1:
                raise NotImplementedError("Expert Parallel is not supported for H-Net.")

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())
            nparams_embedding = sum(
                p.numel()
                for name, p in model.named_parameters()
                if "embeddings" in name
            )
            # Approximate dense fwd+bwd cost: 6 * (non-embedding params) per token.
            # NOTE: H-Net's realized FLOPs are data-dependent (the main network
            # runs on a chunked, compressed sequence), so this is an upper-ish
            # estimate that ignores the compression in the inner stages.
            nparams_no_embedding = nparams - nparams_embedding
            num_flops_per_token = 6 * nparams_no_embedding
            return nparams, num_flops_per_token

    def __init__(self, config: "HNetModel.Config"):
        super().__init__()
        self.config = config
        HNetForCausalLM = _load_hnet_impl()
        self.model = HNetForCausalLM(self._to_hnet_config(config))
        self._target_compression = list(config.target_compression) or [4.0]

    @staticmethod
    def _to_hnet_config(config: "HNetModel.Config") -> HNetConfig:
        return HNetConfig(
            arch_layout=config.arch_layout,
            d_model=list(config.d_model),
            d_intermediate=list(config.d_intermediate),
            vocab_size=config.vocab_size,
            ssm_cfg=SSMConfig(
                d_conv=config.ssm_d_conv,
                expand=config.ssm_expand,
                d_state=config.ssm_d_state,
                chunk_size=config.ssm_chunk_size,
            ),
            attn_cfg=AttnConfig(
                num_heads=list(config.attn_num_heads),
                rotary_emb_dim=list(config.attn_rotary_emb_dim),
                window_size=list(config.attn_window_size),
                # Default each stage's KV heads to its query heads (=> MHA).
                num_kv_heads=(
                    list(config.attn_num_kv_heads)
                    if config.attn_num_kv_heads
                    else list(config.attn_num_heads)
                ),
            ),
            tie_embeddings=config.tie_embeddings,
        )

    def forward(self, tokens: torch.Tensor, **kwargs):
        # Packed (training) mode: mask=None => the vendored model flattens
        # (B, L) -> (T,) with cu_seqlens internally. Extra dataloader kwargs
        # (e.g. attention_masks) are not used by H-Net and are ignored.
        out = self.model(input_ids=tokens, mask=None)
        logits = out.logits
        bpred_output = out.bpred_output  # list[RoutingModuleOutput], outer-first

        ratio_loss = logits.new_zeros(())
        for i, bpred in enumerate(bpred_output):
            N = (
                self._target_compression[i]
                if i < len(self._target_compression)
                else self._target_compression[-1]
            )
            ratio_loss = ratio_loss + hnet_load_balancing_loss(
                bpred.boundary_prob, bpred.boundary_mask, N
            )

        return {"output": logits, "ratio_loss": ratio_loss}

    # ------------------------------------------------------------------
    # Weight initialization
    #
    # torchtitan builds the model on the ``meta`` device, then (after sharding)
    # calls ``to_empty`` followed by ``init_weights``. ``to_empty`` allocates
    # uninitialized storage, so init_weights must initialize *every* parameter
    # and buffer — including Mamba-2's ``A_log`` / ``dt_bias`` / ``D``, conv1d,
    # RMSNorm weights, rotary ``inv_freq`` buffers, and the routing module's
    # identity init — not just the Linear layers that upstream re-inits.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_weights(self, *, buffer_device: torch.device | None = None, **kwargs):
        model = self.model
        ir = self.config.initializer_range

        # 1) Generic PyTorch defaults for every module that exposes
        #    reset_parameters (nn.Linear, nn.Conv1d, nn.Embedding, flash_attn
        #    RMSNorm). This restores conv1d / norm / linear defaults that
        #    to_empty() wiped.
        for m in model.modules():
            reset = getattr(m, "reset_parameters", None)
            if callable(reset) and m is not model:
                try:
                    reset()
                except Exception:  # pragma: no cover - some modules have none
                    pass

        # 2) Module-specific buffers / params not covered by reset_parameters.
        for m in model.modules():
            cls = type(m).__name__
            # Rotary embedding: recompute the inv_freq buffer + clear caches.
            if hasattr(m, "_compute_inv_freq") and hasattr(m, "inv_freq"):
                self._copy_into(m.inv_freq, m._compute_inv_freq())
                m._seq_len_cached = 0
                m._cos_cached = m._sin_cached = None
                m._cos_k_cached = m._sin_k_cached = None
            # Any RMSNorm flavor (incl. gated): weight->1, bias->0.
            if "RMSNorm" in cls:
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            # Mamba-2 special parameters (faithful to mamba_ssm defaults).
            if (
                hasattr(m, "A_log")
                and hasattr(m, "dt_bias")
                and hasattr(m, "D")
            ):
                self._init_mamba2_params(m)

        # 3) Faithful upstream init for embeddings / lm_head / scaled Linears.
        #    backbone._init_weights re-inits Linear weights (out_proj/fc2 scaled
        #    by residual depth) and skips those flagged _no_reinit.
        if self.config.tie_embeddings:
            # The shared weight is used as BOTH the input embedding and the
            # output projection, so it must use the (small) output-projection
            # std — a std=1.0 input-embedding init would make logits explode.
            nn.init.normal_(model.embeddings.weight, mean=0.0, std=ir)
            model.lm_head.weight = model.embeddings.weight
        else:
            nn.init.normal_(model.lm_head.weight, mean=0.0, std=ir)
            nn.init.normal_(model.embeddings.weight, mean=0.0, std=1.0)
        model.backbone._init_weights(ir)

        # 4) Re-apply the _no_reinit inits that to_empty() wiped: the routing
        #    module's identity projections and the per-stage residual_proj zeros.
        for name, m in model.named_modules():
            if type(m).__name__ == "RoutingModule":
                d = m.q_proj_layer.weight.shape[0]  # global dim (DTensor-aware)
                eye = torch.eye(d)
                self._copy_into(m.q_proj_layer.weight, eye)
                self._copy_into(m.k_proj_layer.weight, eye)
            if name.endswith("residual_proj") and isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _copy_into(param: torch.Tensor, full: torch.Tensor) -> None:
        """DTensor-safe in-place copy. ``full`` is the GLOBAL-shaped value; under
        FSDP the param is a DTensor, so distribute ``full`` to its sharding first
        (avoids the "mixed torch.Tensor and DTensor" error)."""
        full = full.to(param.device)
        if isinstance(param, DTensor):
            param.copy_(distribute_tensor(full, param.device_mesh, param.placements))
        else:
            param.copy_(full)

    @classmethod
    def _init_mamba2_params(cls, m: nn.Module) -> None:
        """Re-initialize Mamba-2 dt_bias / A_log / D using mamba_ssm defaults.

        Values are built at the parameter's GLOBAL shape and distributed to the
        local FSDP shard via ``_copy_into``. dt_bias/A_log are i.i.d. per head, so
        per-shard distribution is statistically equivalent to a global draw.
        """
        dt_min, dt_max = 1e-3, 0.1
        a_init_lo, a_init_hi = 1.0, 16.0
        nheads = m.dt_bias.shape[0]  # global (DTensor-aware)

        dt = torch.exp(
            torch.rand(nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        # inverse of softplus: dt = softplus(inv_dt)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        cls._copy_into(m.dt_bias, inv_dt)

        a = torch.empty(nheads).uniform_(a_init_lo, a_init_hi)
        cls._copy_into(m.A_log, torch.log(a))

        cls._copy_into(m.D, torch.ones(m.D.shape))  # global shape (DTensor-aware)
