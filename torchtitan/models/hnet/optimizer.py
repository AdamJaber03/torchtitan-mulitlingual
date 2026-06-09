# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Opt-in AdamW container implementing the H-Net authors' optimizer recipe:
#   1. Per-stage learning-rate multipliers — the outer, byte-level networks
#      (encoder/decoder/routing + embeddings/lm_head) train at a higher LR than
#      the inner main network, since they process much longer sequences at
#      smaller width. See goombalab/hnet `apply_lr_multiplier`
#      (vendored under ._vendor) and arXiv:2507.07955 Appendix C.
#   2. Weight-decay decoupling — parameters with ndim <= 1 (norm gains, biases,
#      and Mamba-2 `A_log`/`D`/`dt_bias`) get weight_decay=0, matching the
#      authors' `group_params` (which zeroes wd on biases + norms) and standard
#      Mamba practice.
#
# This is a SEPARATE subclass selected only by H-Net configs that opt in. The
# base ``OptimizersContainer`` (used by every other model) is not modified, so
# all other models' optimizer behavior — correctness and speed — is identical.

from collections import OrderedDict
from dataclasses import dataclass, field

import torch.nn as nn

from torchtitan.components.optimizer import OptimizersContainer


def _find_lr_multiplier_module(model: nn.Module):
    """Locate the submodule exposing ``apply_lr_multiplier`` (the vendored
    HNetForCausalLM). Returns None if absent."""
    if hasattr(model, "apply_lr_multiplier"):
        return model
    for m in model.modules():
        if hasattr(m, "apply_lr_multiplier"):
            return m
    return None


class HNetOptimizersContainer(OptimizersContainer):
    """AdamW container with H-Net per-stage LR multipliers + weight-decay
    decoupling. Opt-in; does not affect any other model's optimizer path."""

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        # Per-stage LR multipliers, OUTER-first (e.g. [1.5, 1.0] for a 1-stage
        # net: byte enc/dec at 1.5x, main net at 1x; [3.0, 1.7, 0.9] for 2-stage).
        # Length must equal the model's number of stages (1-stage => 2 entries).
        # Empty => uniform LR (all multipliers = 1.0); weight-decay decoupling
        # still applies if enabled below.
        lr_multipliers: list[float] = field(default_factory=list)
        # If True, params with ndim <= 1 (norms, biases, Mamba A_log/D/dt_bias)
        # use weight_decay=0.
        decouple_weight_decay: bool = True

    def __init__(self, config: "HNetOptimizersContainer.Config", *, model_parts):
        optimizer_cls = self._resolve_optimizer_cls(config.name)
        base_kwargs = self._build_optimizer_kwargs(config)
        base_lr = config.lr
        base_wd = config.weight_decay
        multipliers = list(config.lr_multipliers)

        # Globals shared by all groups (lr/weight_decay are per-group below, but
        # also passed as defaults).
        global_kwargs = dict(base_kwargs)

        self.optimizers = []
        self.model_parts = model_parts
        all_params = []
        for model in model_parts:
            if multipliers:
                inner = _find_lr_multiplier_module(model)
                if inner is None:
                    raise RuntimeError(
                        "HNetOptimizersContainer: lr_multipliers were given but no "
                        "module with `apply_lr_multiplier` was found. This container "
                        "is intended for H-Net models only."
                    )
                # Authors' own routine: annotates each param's `_optim` dict with
                # the multiplier for the stage it belongs to (handles the nested
                # hierarchy correctly).
                try:
                    inner.apply_lr_multiplier(multipliers)
                except IndexError as e:
                    raise ValueError(
                        f"lr_multipliers={multipliers} has too few entries for this "
                        f"H-Net (need one per stage; a 1-stage model has 2 stages). "
                        f"Original error: {e!r}"
                    ) from e

            groups = self._build_param_groups(
                model, base_lr, base_wd, bool(multipliers), config.decouple_weight_decay
            )
            self.optimizers.append(optimizer_cls(groups, **global_kwargs))
            all_params.extend(p for g in groups for p in g["params"])

        self._validate_length(len(self.model_parts))
        self._post_init(all_params, base_kwargs)

    @staticmethod
    def _build_param_groups(
        model: nn.Module,
        base_lr: float,
        base_wd: float,
        use_multipliers: bool,
        decouple_wd: bool,
    ) -> list[dict]:
        # Group params by (lr, weight_decay); LambdaLR scales each group's own
        # base lr by the schedule factor, so the multipliers are preserved.
        groups: "OrderedDict[tuple, list]" = OrderedDict()
        for _name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            mult = 1.0
            if use_multipliers:
                opt = getattr(p, "_optim", None)
                if opt is not None and "lr_multiplier" in opt:
                    mult = float(opt["lr_multiplier"])
            lr = base_lr * mult
            wd = 0.0 if (decouple_wd and p.ndim <= 1) else base_wd
            groups.setdefault((lr, wd), []).append(p)
        return [
            {"params": ps, "lr": lr, "weight_decay": wd}
            for (lr, wd), ps in groups.items()
        ]
