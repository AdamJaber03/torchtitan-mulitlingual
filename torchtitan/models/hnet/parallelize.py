# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Applies FSDP2 (data parallelism) and, optionally, activation checkpointing /
# torch.compile to the H-Net model.
#
# H-Net's dynamic chunking produces variable-length inner sequences, which are
# incompatible with Tensor / Pipeline / Context Parallel. This function therefore
# supports FSDP / HSDP / DDP only and raises on TP/PP/CP. torch.compile is OFF by
# default (the chunked sequence lengths are data-dependent / dynamic).

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._composable.replicate import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import _apply_ac_to_transformer_block
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def _isotropic_cls():
    """Lazily import the kernel-dependent Isotropic class (used as the FSDP /
    AC / compile wrapping unit). Deferred so importing this module doesn't pull
    in flash_attn / mamba_ssm on machines without those kernels."""
    from ._vendor.isotropic import Isotropic

    return Isotropic


def _isotropic_fqns(model: nn.Module) -> list[str]:
    """Fully-qualified names of the Isotropic sub-networks (encoder / main /
    decoder per stage) — the block-group units that AC, compile, and FSDP all
    wrap. Captured once, before any wrapping, so the three passes operate on the
    same units by name (FSDP ends up outermost, as in llama)."""
    Isotropic = _isotropic_cls()
    return [name for name, m in model.named_modules() if isinstance(m, Isotropic)]


def _register_at(model: nn.Module, fqn: str, module: nn.Module) -> None:
    parent, _, child = fqn.rpartition(".")
    parent_mod = model.get_submodule(parent) if parent else model
    parent_mod.register_module(child, module)


def parallelize_hnet(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply data parallelism (+ optional AC / compile) to H-Net.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "Tensor Parallel is not supported for H-Net (dynamic chunking yields "
            "variable-length inner sequences). Use FSDP / data parallelism."
        )
    if parallel_dims.pp_enabled:
        raise NotImplementedError("Pipeline Parallel is not supported for H-Net.")
    if parallel_dims.cp_enabled:
        raise NotImplementedError("Context Parallel is not supported for H-Net.")

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    # Capture the block-group units once, before wrapping, so AC/compile/FSDP
    # all target the same modules by name (FSDP applied last => outermost).
    block_fqns = _isotropic_fqns(model)

    if ac_config.mode != "none":
        # torchtitan's generic apply_ac assumes a llama-style ``model.layers``
        # ModuleDict, which H-Net does not have. Instead, checkpoint each
        # Isotropic sub-network (encoder / main / decoder per stage) — the
        # repeated, residual-stream-stacked block groups — using the same
        # per-block primitive apply_ac uses.
        apply_ac_hnet(model, ac_config, model_compile_enabled, block_fqns)

    if model_compile_enabled:
        logger.warning(
            "torch.compile is enabled for H-Net, but dynamic chunking produces "
            "data-dependent dynamic shapes; compilation may recompile frequently "
            "or fail. Disable compile if you hit issues."
        )
        apply_compile(model, block_fqns)

    if parallel_dims.fsdp_enabled:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            block_fqns=block_fqns,
        )
        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to H-Net")
        else:
            logger.info("Applied FSDP to H-Net")
        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to H-Net")
    elif parallel_dims.dp_replicate_enabled:
        dp_replicate_mesh = parallel_dims.get_mesh("dp_replicate")
        if parallel_dims.world_size != dp_replicate_mesh.size():
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(model, dp_replicate_mesh, enable_compile=model_compile_enabled)

    return model


# Selective-op activation-checkpoint save list (mirrors llama3; harmless extras
# are ignored if the op is absent in this build).
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops.aten.max.default,
}


def apply_ac_hnet(
    model: nn.Module,
    ac_config: ActivationCheckpointConfig,
    model_compile_enabled: bool,
    block_fqns: list[str],
) -> None:
    """Apply activation checkpointing to each Isotropic sub-network.

    H-Net has no llama-style ``model.layers``; its block groups are the Isotropic
    encoder/main/decoder modules. We wrap each with the same primitive
    torchtitan's apply_ac uses per transformer block, then re-register it.
    """
    if ac_config.mode not in ("full", "selective"):
        raise NotImplementedError(
            f"H-Net supports activation_checkpoint mode 'none', 'full', or "
            f"'selective'; got '{ac_config.mode}'."
        )
    for fqn in block_fqns:
        wrapped = _apply_ac_to_transformer_block(
            model.get_submodule(fqn),
            ac_config,
            base_fqn=fqn,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
        )
        _register_at(model, fqn, wrapped)
    logger.info(f"Applied {ac_config.mode} activation checkpointing to H-Net")


def apply_compile(model: nn.Module, block_fqns: list[str]) -> None:
    """Compile each Isotropic sub-network (repeated structure)."""
    for fqn in block_fqns:
        _register_at(
            model,
            fqn,
            torch.compile(model.get_submodule(fqn), fullgraph=False, dynamic=True),
        )
    logger.info("Compiled each H-Net Isotropic sub-network with torch.compile")


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """Disable FSDP's automatic gradient division (we normalize by the global
    token count in the training loop, like the other torchtitan models)."""
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    block_fqns: list[str] | None = None,
):
    """Apply FSDP2 to H-Net: shard each Isotropic sub-network individually
    (whatever module now sits at its fqn — possibly AC/compile-wrapped), then
    the whole model."""
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            reshard_after_forward = True
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    # Shard each Isotropic block group (encoder / main / decoder per stage).
    # These hold the bulk of the parameters and have repeated block structure.
    if block_fqns is None:
        block_fqns = _isotropic_fqns(model)
    for fqn in block_fqns:
        fully_shard(
            model.get_submodule(fqn),
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Shard the remaining top-level parameters (embeddings, lm_head, routing /
    # residual projections, pad_dimension).
    fully_shard(model, **fsdp_config)

    disable_fsdp_gradient_division(model)


def apply_ddp(model: nn.Module, dp_mesh: DeviceMesh, enable_compile: bool):
    if enable_compile:
        torch._dynamo.config.optimize_ddp = "ddp_optimizer"
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)
    logger.info("Applied DDP to H-Net")
