# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

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
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.tools.logging import logger

from .model import MultilingualBottleneckModel

# Standard Meta ops list for Selective Activation Checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
    torch.ops.torch_attn._varlen_attn.default,
    torch._higher_order_ops.inductor_compiled_code,
}

def parallelize_bottleneck(
    model: MultilingualBottleneckModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
    # Ignore model_converters as we are not using TP or float8 tensorwise TP here
    **kwargs, 
):
    """
    Apply Activation Checkpointing, torch.compile, and FSDP2/DDP to the custom branched model.
    """
    model_compile_enabled = compile_config.enable and "model" in compile_config.components

    # 1. Activation Checkpointing
    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
            base_folder=dump_folder,
        )

    # 2. Compile Blocks (Custom Traversal for Branched Topology)
    if model_compile_enabled:
        apply_compile(model, compile_config)

    # 3. Data Parallelism (FSDP2 or DDP)
    if parallel_dims.fsdp_enabled:
        # dp_mesh is the mesh for FSDP/HSDP
        names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        dp_mesh = parallel_dims.get_mesh(names)
        
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        )

        mode_str = "HSDP" if parallel_dims.dp_replicate_enabled else "FSDP2"
        logger.info(f"Applied {mode_str} to the Multilingual Bottleneck model")
        
        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")

    elif parallel_dims.dp_replicate_enabled:
        dp_replicate_mesh = parallel_dims.get_mesh("dp_replicate")
        apply_ddp(
            model,
            dp_replicate_mesh,
            enable_compile=model_compile_enabled,
        )

    return model

def apply_compile(model: MultilingualBottleneckModel, compile_config: CompileConfig):
    """Apply torch.compile to each block across encoders, backbone, and decoders."""
    # Encoders
    for lang_layers in model.encoders.values():
        for layer_id, block in lang_layers.named_children():
            compiled_block = torch.compile(block, backend=compile_config.backend, fullgraph=True)
            lang_layers.register_module(layer_id, compiled_block)
            
    # Backbone
    for layer_id, block in model.shared_backbone.named_children():
        compiled_block = torch.compile(block, backend=compile_config.backend, fullgraph=True)
        model.shared_backbone.register_module(layer_id, compiled_block)
        
    # Decoders
    for lang_layers in model.decoders.values():
        for layer_id, block in lang_layers.named_children():
            compiled_block = torch.compile(block, backend=compile_config.backend, fullgraph=True)
            lang_layers.register_module(layer_id, compiled_block)

    logger.info("Compiled each branched TransformerBlock with torch.compile")

def disable_fsdp_gradient_division(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)

def apply_fsdp(
    model: MultilingualBottleneckModel,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """Apply Fully Sharded Data Parallel (FSDP2) across the branched modules."""
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # Determine resharding policy
    reshard_after_forward = reshard_after_forward_policy != "never"

    # 1. Shard Embeddings
    fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=reshard_after_forward)
    
    # 2. Shard Branched Modules
    for lang_layers in model.encoders.values():
        for block in lang_layers.values():
            fully_shard(block, **fsdp_config, reshard_after_forward=reshard_after_forward)
            
    for block in model.shared_backbone.values():
        fully_shard(block, **fsdp_config, reshard_after_forward=reshard_after_forward)
        
    for lang_layers in model.decoders.values():
        for block in lang_layers.values():
            fully_shard(block, **fsdp_config, reshard_after_forward=reshard_after_forward)

    # 3. Shard Output Projection
    # Optimization: Do not reshard the last layers immediately
    fully_shard(
        [model.norm, model.output],
        **fsdp_config,
        reshard_after_forward=(reshard_after_forward_policy == "always"),
    )

    # 4. Shard the root module
    fully_shard(model, **fsdp_config)

    # Disable FSDP's automatic gradient division (handled manually in training loop)
    disable_fsdp_gradient_division(model)

def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
):
    if enable_compile:
        torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    # Add find_unused_parameters=True to support your routing topology
    replicate(
        model, 
        device_mesh=dp_mesh, 
        bucket_cap_mb=100,
        find_unused_parameters=True  # <--- ADD THIS LINE
    )

    logger.info("Applied DDP to the model")