# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Annotated, Any, cast

import torch
import torch.distributed.checkpoint.stateful
import tyro
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.loss import IGNORE_INDEX, LossFunction
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import ensure_pp_loss_visible, MetricsProcessor
from torchtitan.components.optimizer import (
    OptimizersContainer,
    OptimizersInBackwardContainer,
)
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.components.validate import BaseValidator, Validator
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
    LossConfig
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.models.common.decoder import Decoder
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
    ProfilingConfig,
)


class Trainer(torch.distributed.checkpoint.stateful.Stateful, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """
        Default container for training configuration.
        """

        # NOTE: model_spec is suppressed from tyro CLI parsing and is always
        # set programmatically by the model registry before Trainer construction.
        model_spec: Annotated[ModelSpec | None, tyro.conf.Suppress] = None

        hf_assets_path: str = "./tests/assets/tokenizer"
        """
        Path to HF assets folder. This folder contains local copies of Hugging Face assets,
        including model weights in .safetensors format, the model.safetensor.index.json file
        (fqn to file mapping), the config.json file, generation_config.json, and tokenizer files.
        """

        dump_folder: str = "./outputs"
        """Folder to dump job outputs"""

        profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
        metrics: MetricsProcessor.Config = field(
            default_factory=MetricsProcessor.Config
        )
        # TODO: remove the optional flag once Flux tokenizer is modeled properly
        tokenizer: BaseTokenizer.Config | None = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        dataloader: BaseDataLoader.Config = field(default_factory=BaseDataLoader.Config)
        model_converters: ModelConvertersContainer.Config = field(
            default_factory=ModelConvertersContainer.Config
        )
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        activation_checkpoint: ActivationCheckpointConfig = field(
            default_factory=ActivationCheckpointConfig
        )
        compile: CompileConfig = field(default_factory=CompileConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        validator: Validator.Config = field(default_factory=Validator.Config)
        debug: DebugConfig = field(default_factory=DebugConfig)
        loss: LossConfig = field(default_factory=LossConfig)

        def __post_init__(self):
            if isinstance(self.optimizer, OptimizersInBackwardContainer.Config):
                if self.parallelism.expert_parallel_degree > 1:
                    raise NotImplementedError(
                        "Optimizers in backward is not supported with Expert Parallel."
                    )
                if self.parallelism.pipeline_parallel_degree > 1:
                    raise NotImplementedError(
                        "Optimizers in backward is not supported with Pipeline Parallel."
                    )

        def to_dict(self) -> dict[str, Any]:
            d = {}
            for f in dataclasses.fields(self):
                if f.name == "model_spec":
                    assert self.model_spec is not None
                    # ModelSpec contains callables that can't be serialized
                    d["model_spec"] = {
                        "name": self.model_spec.name,
                        "flavor": self.model_spec.flavor,
                    }
                else:
                    d[f.name] = (
                        asdict(getattr(self, f.name))
                        if dataclasses.is_dataclass(getattr(self, f.name))
                        else getattr(self, f.name)
                    )
            return d

        def maybe_log(self) -> None:
            if self.debug.print_config:
                logger.info(
                    f"Running with configs: {json.dumps(self.to_dict(), indent=2, ensure_ascii=False)}"
                )

            if self.debug.save_config_file is not None:
                config_file = os.path.join(
                    self.dump_folder, self.debug.save_config_file
                )
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        os.makedirs(os.path.dirname(config_file), exist_ok=True)
                        with open(config_file, "w") as f:
                            json.dump(self.to_dict(), f, indent=2)
                    logger.info(f"Saved job configs to {config_file}")
                else:
                    logger.warning(
                        "Job configs logging is disabled due to torch.distributed not initialized."
                    )

    # core configs
    config: Config
    parallel_dims: ParallelDims

    # swappable training components
    tokenizer: BaseTokenizer | None
    dataloader: BaseDataLoader
    model_config: BaseModel.Config
    # TODO: we should make this list[BaseModel / Decoder] but this will affect many components.
    # will do this in a separate PR
    model_parts: list[torch.nn.Module]
    loss_fn: LossFunction
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    validator: BaseValidator
    metrics_processor: MetricsProcessor
    checkpointer: CheckpointManager

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: dist_utils.TrainContext
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # additional training states
    step: int
    ntokens_seen: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        assert (
            config.model_spec is not None
        ), "model_spec must be set before creating Trainer"
        model_spec = config.model_spec

        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        self.parallel_dims = parallel_dims = self.init_distributed()

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq, debug=config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

        # build tokenizer
        self.tokenizer = (
            config.tokenizer.build(tokenizer_path=config.hf_assets_path)
            if config.tokenizer is not None
            else None
        )

        # build model (using meta init)
        model_config = model_spec.model
        # set the model args from training job configs
        model_config.update_from_config(
            trainer_config=config,
        )
        self.model_config = model_config

        logger.info(
            f"Building {model_spec.name} {model_spec.flavor} "
            f"with {json.dumps(dataclasses.asdict(model_config), indent=2, ensure_ascii=False)}"
        )
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        # Build the collection of model converters. No-op if converters empty
        model_compile_enabled = (
            config.compile.enable and "model" in config.compile.components
        )
        model_converters = config.model_converters.build(
            parallel_dims=parallel_dims,
            model_compile_enabled=model_compile_enabled,
        )
        model_converters.convert(model)

        # metrics logging
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            config_dict=config.to_dict(),
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(model, config.training.seq_len)

        logger.info(
            f"{color.blue}Model {model_spec.name} {model_spec.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        buffer_device: torch.device | None
        if config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(device_type)
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = model_spec.build_loss_fn(
            config.loss, config.compile, parallel_dims=parallel_dims
        )

        # verify batch sizes
        global_batch_size = config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = config.training.local_batch_size * batch_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (config.training.local_batch_size * batch_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({config.training.local_batch_size} * {batch_degree}) != 0)"
        )

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            config.training.local_batch_size * batch_degree
        )
        assert self.gradient_accumulation_steps > 0

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            if not model_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {model_spec.name} "
                    f"does not support pipelining"
                )

            # apply both Pipeline Parallel and SPMD-style scaling techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = model_spec.pipelining_fn(
                model,
                parallel_dims=parallel_dims,
                training=config.training,
                model_converters=config.model_converters,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
                device=self.device,
                model_config=model_config,
                parallelize_fn=model_spec.parallelize_fn,
                loss_fn=self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    cast(Decoder, m).init_weights(buffer_device=buffer_device)
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(
                parallel_dims=parallel_dims,
                pp_schedule=config.parallelism.pipeline_parallel_schedule,
                color=color,
            )
        else:
            # apply Tensor/Context/Expert Parallel, activation checkpointing, torch.compile, Data Parallel
            model = model_spec.parallelize_fn(
                model,
                parallel_dims=parallel_dims,
                training=config.training,
                model_converters=config.model_converters,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
            )

            model.to_empty(device=init_device)
            with torch.no_grad():
                cast(BaseModel, model).init_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
        )
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0
        # Calculate initial stage based on starting step (useful if resuming from checkpoint)
        self.stage_idx = self._get_stage_idx(self.step)

        # build dataloader
        self.dataloader = config.dataloader.build(
            dp_world_size=batch_degree,
            dp_rank=batch_rank,
            tokenizer=self.tokenizer,
            seq_len=config.training.seq_len,
            local_batch_size=config.training.local_batch_size,
            stage_idx=self.stage_idx, # Pass the current stage
        )
        self.checkpointer = config.checkpoint.build(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=(
                model_spec.state_dict_adapter(model_config, config.hf_assets_path)
                if model_spec.state_dict_adapter
                else None
            ),
            base_folder=config.dump_folder,
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not config.parallelism.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(loss_parallel_enabled)
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            config.training.mixed_precision_param,
            device_type,
        )

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                dp_world_size=batch_degree,
                dp_rank=batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                maybe_enable_amp=self.maybe_enable_amp,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.seq_len,
                local_batch_size=config.training.local_batch_size,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {config.training.seq_len}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})"
        )

    def init_distributed(self) -> ParallelDims:
        config = self.config
        world_size = dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
        )

        parallelism_config = config.parallelism
        return ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )

    def _get_stage_idx(self, current_step: int) -> int:
        """Determines the current curriculum stage based on global steps."""
        if not hasattr(self.config.dataloader, "stages") or not self.config.dataloader.stages:
            return 0
            
        cumulative_steps = 0
        for idx, stage in enumerate(self.config.dataloader.stages):
            cumulative_steps += stage.get("steps", 0)
            if current_step < cumulative_steps:
                return idx
                
        # If we exceed all defined steps, stay on the last stage
        return len(self.config.dataloader.stages) - 1

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator.

        Note: Tensors are yielded on CPU. The caller is responsible for moving
        them to GPU when needed. This allows for more efficient memory usage
        when doing gradient accumulation.
        """
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            ntokens_batch = labels.numel()
            self.ntokens_seen += ntokens_batch
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Tensors stay on CPU; moved to GPU per-microbatch during training
            yield input_dict, labels

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """
        Post-processing hook after data loading and before model forward pass.

        This method processes the raw data from the dataloader and prepares it for
        the model's forward pass. It separates the main input tensor from auxiliary
        inputs and constructs additional keyword arguments (e.g., attention masks).

        This method can be overridden in subclasses to customize data processing
        for different training strategies (e.g., converting tensors to DTensors,
        applying custom transformations, etc.).

        Args:
            input_dict: Dictionary containing tensors from the dataloader. Must
                contain an "input" key with the main input tensor. May contain
                additional keys for auxiliary inputs (e.g., position ids).
            labels: Target labels for the batch.

        Returns:
            A tuple of (inputs, labels, extra_inputs, extra_kwargs) where:
                - inputs: Main input tensor extracted from input_dict["input"].
                - labels: Target labels (unchanged from input parameter).
                - extra_inputs: Dict of auxiliary input tensors (all keys except
                    "input" from input_dict). These are passed to the model forward
                    but are NOT forwarded across pipeline parallel stages.
                - extra_kwargs: Dict of additional keyword arguments for model forward.
                    These ARE forwarded across pipeline parallel stages. Contains
                    attention_masks if flex attention is enabled.

        Note:
            The distinction between extra_inputs and extra_kwargs is important for
            pipeline parallelism: extra_kwargs are forwarded to all pipeline stages,
            while extra_inputs are only available to the first stage.
        """
        inputs = input_dict["input"]
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        # For arguments, like attention_masks, we have to put them in a separate
        # dict as extra_inputs are not forwarded to other stages in PP, but
        # extra_kwargs are.
        extra_kwargs: dict[str, Any] = {}

        # --- NEW: Forward the contrastive mask to all pipeline stages ---
        if "contrastive_mask" in extra_inputs:
            extra_kwargs["contrastive_mask"] = extra_inputs.pop("contrastive_mask")

        # Pull the early-exit flag out of extra_inputs so it is not passed to the model forward
        # (the model only needs the derived keep_index / reduced mask, built below).
        contrastive_only = extra_inputs.pop("contrastive_only_mask", None)

        # TODO: improve the logic on obtaining attention masks
        layer = getattr(self.model_config, "layer", None)
        attn_config = getattr(layer, "attention", None) if layer else None
        attn_backend = getattr(attn_config, "attn_backend", "sdpa")
        if attn_backend in ["flex", "varlen"]:
            assert (
                self.tokenizer is not None
            ), "tokenizer is required for flex/varlen attention"
            model = cast(Decoder, self.model_parts[0])
            extra_kwargs["attention_masks"] = model.get_attention_masks(
                input_batch=inputs,
                tokenizer=self.tokenizer,
                extra_inputs=extra_inputs,
            )

        # --- Contrastive early-exit: batch-level compaction of the kept (non-translation) tokens ---
        # After the contrastive layer the model only needs to run the kept tokens through the deeper
        # layers. We build, on the host (eager), a fixed-shape gather index over the flattened batch
        # plus a reduced attention mask, and hand them to the model. Translation tokens were already
        # given label == IGNORE_INDEX by the dataloader; here we additionally drop CE on any kept
        # tokens that overflow the keep_len budget.
        keep_len = getattr(self.model_config, "keep_len", None)
        if (
            contrastive_only is not None
            and keep_len is not None
            and bool(contrastive_only.any())
        ):
            B, raw = inputs.shape
            device = inputs.device
            keep = (~contrastive_only.bool()).reshape(-1)              # [B*raw]
            flat_keep_idx = keep.nonzero(as_tuple=True)[0]             # [total_kept], ascending
            total_kept = int(flat_keep_idx.numel())
            budget = B * keep_len
            if total_kept >= budget:
                # Overflow: keep the first `budget` kept tokens row-major; the rest get no CE so the
                # shallow representation they retain after scatter-back is never trained on.
                overflow = flat_keep_idx[budget:]
                if overflow.numel() > 0:
                    labels.reshape(-1)[overflow] = IGNORE_INDEX
                keep_index_flat = flat_keep_idx[:budget]
                keep_valid_flat = torch.ones(budget, dtype=torch.bool, device=device)
            else:
                # Underflow: pad with a dummy index (0) marked invalid; those slots are computed but
                # never scattered back, and we set their reduced token to eos so they stay isolated.
                pad = budget - total_kept
                keep_index_flat = torch.cat(
                    [flat_keep_idx, torch.zeros(pad, dtype=flat_keep_idx.dtype, device=device)]
                )
                keep_valid_flat = torch.cat(
                    [
                        torch.ones(total_kept, dtype=torch.bool, device=device),
                        torch.zeros(pad, dtype=torch.bool, device=device),
                    ]
                )
            keep_index = keep_index_flat.view(B, keep_len)
            keep_valid = keep_valid_flat.view(B, keep_len)
            extra_kwargs["contrastive_keep_index"] = keep_index
            extra_kwargs["contrastive_keep_valid"] = keep_valid

            # --- keep_len budget monitoring ---
            # kept_frac < 1 => keep_len has headroom (you can shrink it for more savings);
            # kept_frac > 1 => overflow, some kept (Arabic) tokens are truncated out of CE (raise it).
            self._ee_kept = total_kept
            self._ee_budget = int(budget)
            overflow_tokens = max(0, total_kept - int(budget))
            self._ee_overflow = overflow_tokens
            if self.step % max(1, self.config.metrics.log_freq) == 0:
                logger.info(
                    f"[early-exit] kept {total_kept}/{int(budget)} "
                    f"= {total_kept / budget:.1%} of keep_len budget"
                    + (f"  -- OVERFLOW: {overflow_tokens} tokens truncated (no CE)"
                       if overflow_tokens > 0 else "")
                )

            # Reduced token ids (compacted, pad slots -> eos) drive the post-target-layer mask.
            reduced_tokens = inputs.reshape(-1)[keep_index_flat].view(B, keep_len)
            if self.tokenizer is not None and self.tokenizer.eos_id is not None:
                reduced_tokens = reduced_tokens.masked_fill(~keep_valid, self.tokenizer.eos_id)
            if attn_backend in ["flex", "varlen"]:
                model = cast(Decoder, self.model_parts[0])
                extra_kwargs["reduced_attention_masks"] = model.get_attention_masks(
                    input_batch=reduced_tokens,
                    tokenizer=self.tokenizer,
                    extra_inputs=extra_inputs,
                )

        if self.parallel_dims.cp_enabled:
            inputs, labels, extra_kwargs = prepare_context_parallel_input(
                inputs,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.device,
                self.config.parallelism.context_parallel_load_balancer,
            )

        return inputs, labels, extra_inputs, extra_kwargs

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]: # Updated signature
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context():
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs,
                        **extra_inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )
                else:
                    self.pp_schedule.step(
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )

            # accumulate losses across pipeline microbatches
            if self.pp_has_last_stage:
                real_losses = []
                breakdowns = []
                for l in losses:
                    if isinstance(l, tuple):
                        real_losses.append(l[0])
                        breakdowns.append(l[1]) # These are now dicts of Tensors
                    else:
                        real_losses.append(l)
                        breakdowns.append({})

                loss = (torch.sum(torch.stack(real_losses)) / global_valid_tokens).to(self.device)
                
                loss_breakdown = {}
                for b in breakdowns:
                    for k, v in b.items():
                        # CRITICAL FIX: Safely extract .item() here
                        val = (v.detach() / global_valid_tokens).item() if isinstance(v, torch.Tensor) else (v / global_valid_tokens.item())
                        loss_breakdown[k] = loss_breakdown.get(k, 0.0) + val
            else:
                loss = torch.tensor([-1.0], device=self.device)
                loss_breakdown = {}
        else:
            # Non-PP forward / backward
            assert len(model_parts) == 1
            with self.train_context():
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                    
                    # Compute loss sum. This is the boundary of the torch.compile graph!
                    loss_out = self.loss_fn(pred, labels)

                    # Handle MixedLoss returning a tuple
                    if isinstance(loss_out, tuple):
                        loss_sum, loss_breakdown_tensors = loss_out
                    else:
                        loss_sum, loss_breakdown_tensors = loss_out, {}

                    # Scale the total loss
                    loss = loss_sum / global_valid_tokens
                    
                    # CRITICAL FIX: We are now outside the compiled graph. 
                    # We can safely detach and extract the floats for W&B.
                    loss_breakdown = {}
                    for k, v in loss_breakdown_tensors.items():
                        if isinstance(v, torch.Tensor):
                            loss_breakdown[k] = (v.detach() / global_valid_tokens).item()
                        else:
                            loss_breakdown[k] = v / global_valid_tokens.item()

                # need to free pred before bwd to avoid peaking memory
                del pred
                loss.backward()

        return loss, loss_breakdown

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        # Collect all microbatches on CPU and count total valid tokens
        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _microbatch in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()
            microbatches.append((input_dict, labels))

        # All-reduce to get global token count across DP ranks
        # Move to GPU for distributed communication
        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens.float()

        # Process each microbatch: move to GPU, forward/backward, then free
        accumulated_losses = []
        accumulated_breakdowns = {} # NEW: Track breakdowns across micro-batches
        
        for input_dict, labels in microbatches:
            # Move tensors to GPU
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)

            # NEW: Safely unpack the tuple
            loss, loss_breakdown = self.forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )
            accumulated_losses.append(loss.detach())
            
            # NEW: Accumulate the breakdown items
            for k, v in loss_breakdown.items():
                accumulated_breakdowns[k] = accumulated_breakdowns.get(k, 0.0) + v

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        global_breakdown = {} # NEW: Dict to hold synchronized breakdown metrics

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            local_avg_loss = loss * global_valid_tokens / local_valid_tokens
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(loss, loss_mesh),
                dist_utils.dist_max(local_avg_loss, loss_mesh),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                ),
            )
            
            # NEW: Sync the individual loss breakdown components across DP ranks
            # Because we already divided by global_valid_tokens in forward_backward_step,
            # summing them across ranks yields the true global average!
            for k, v in accumulated_breakdowns.items():
                v_tensor = torch.tensor(v, dtype=torch.float32, device=self.device)
                v_sync = dist_utils.dist_sum(v_tensor, loss_mesh)
                global_breakdown[f"Losses/{k}"] = v_sync
                
        else:
            global_avg_loss = global_max_loss = loss.detach().item()
            global_ntokens_seen = self.ntokens_seen
            
            # Add standalone breakdown metrics if DP is disabled
            for k, v in accumulated_breakdowns.items():
                global_breakdown[f"Losses/{k}"] = v

        injection_metrics = {}
        # Track the number of contrastive pairs
        contrastive_metrics = {}
        # Safely drill down into the dataloader hierarchy to find our custom datasets
        if hasattr(self.dataloader, "dataset") and hasattr(self.dataloader.dataset, "datasets"):
            for i, ds in enumerate(self.dataloader.dataset.datasets):
                if hasattr(ds, "injection_counts") and hasattr(ds, "injection_paths"):
                    for idx, path in enumerate(ds.injection_paths):
                        # indicates which injection set this is eg. "gemini_seeds"
                        parent_parent_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
                        # Use the filename as the label (e.g., "synthetic_entities.jsonl")
                        parent_dir = os.path.basename(os.path.dirname(path))

                        # Extract the file name without the extension (e.g., "en_data")
                        file_stem = os.path.splitext(os.path.basename(path))[0]

                        # Combine them for W&B (e.g., "42_en_data")
                        file_name = f"{parent_parent_dir}_{parent_dir}_{file_stem}"
                        # Grab the local count from this specific GPU
                        local_count = torch.tensor(
                            ds.injection_counts[idx], dtype=torch.int64, device=self.device
                        )
                        
                        # Sum the counts across all Data Parallel ranks to get the global total
                        if parallel_dims.dp_enabled:
                            batch_mesh = parallel_dims.get_mesh("batch")
                            global_count = dist_utils.dist_sum(local_count, batch_mesh)
                        else:
                            global_count = local_count.item()  # If not distributed, just take the local count
                        
                        # Prefix with "Injections/" to automatically group them in W&B
                        injection_metrics[f"Injections/{file_name}"] = global_count
                if hasattr(ds, "contrastive_pair_counter"):
                    local_contrastive_count = torch.tensor(
                        ds.contrastive_pair_counter, dtype=torch.int64, device=self.device
                    )
                    if parallel_dims.dp_enabled:
                        batch_mesh = parallel_dims.get_mesh("batch")
                        global_contrastive_count = dist_utils.dist_sum(local_contrastive_count, batch_mesh)
                    else:
                        global_contrastive_count = local_contrastive_count.item()
                    contrastive_metrics[f"contrastive_pairs/{i}"] = global_contrastive_count
            

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
            **global_breakdown, # NEW: Inject our aggregated loss breakdowns!
            **injection_metrics,
            **contrastive_metrics,  # Inject new metrics here!
        }
        # Contrastive early-exit keep_len budget usage (from the last microbatch of this step).
        if hasattr(self, "_ee_budget") and self._ee_budget:
            extra_metrics["contrastive/kept_frac_of_keeplen"] = self._ee_kept / self._ee_budget
            extra_metrics["contrastive/overflow_tokens"] = float(self._ee_overflow)
        
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics=extra_metrics,
        )

    def _maybe_active_forget(self) -> None:
        """Active forgetting (arXiv:2410.16168 / Chen et al. 2023): every
        `training.active_forgetting_interval` steps, reinitialize the token embeddings and reset
        their optimizer (Adam) state. The transformer body and the LR schedule are left untouched.
        Skipped on the final step so the converged model is not perturbed."""
        k = self.config.training.active_forgetting_interval
        if not k or self.step % k != 0 or self.step == self.config.training.steps:
            return

        emb_params = set()
        with torch.no_grad():
            for mp in self.model_parts:
                if hasattr(mp, "reinit_embeddings"):
                    mp.reinit_embeddings()
                for name in ("tok_embeddings", "output"):
                    mod = getattr(mp, name, None)
                    weight = getattr(mod, "weight", None) if mod is not None else None
                    if weight is not None:
                        emb_params.add(weight)  # tied input/output -> same tensor, set dedups

        # Reset Adam moments for the embedding params so fresh weights start from a clean state.
        for optimizer in self.optimizers.optimizers:
            for param in emb_params:
                optimizer.state.pop(param, None)

        logger.info(
            f"[active-forgetting] reinitialized {len(emb_params)} embedding tensor(s) and reset "
            f"their optimizer state at step {self.step} (interval={k})"
        )

    @record
    def train(self):
        config = self.config

        self.checkpointer.load(step=config.checkpoint.load_step)
        
        # Ensure stage is correct after a checkpoint load
        loaded_stage = self._get_stage_idx(self.step)
        if loaded_stage != self.stage_idx:
            self.stage_idx = loaded_stage
            # Rebuild dataloader if we resumed into a different stage
            dp_mesh = self.parallel_dims.get_mesh("batch") if self.parallel_dims.dp_enabled else None
            self.dataloader = config.dataloader.build(
                dp_world_size=dp_mesh.size() if dp_mesh else 1,
                dp_rank=dp_mesh.get_local_rank() if dp_mesh else 0,
                tokenizer=self.tokenizer,
                seq_len=config.training.seq_len,
                local_batch_size=config.training.local_batch_size,
                stage_idx=self.stage_idx,
            )
        
        if hasattr(self.dataloader, "step"):
            self.dataloader.step(self.step)

        logger.info(f"Training starts at step {self.step + 1} (Stage {self.stage_idx})")

        with (
            maybe_enable_profiling(
                config.profiling,
                global_step=self.step,
                base_folder=config.dump_folder,
            ) as torch_profiler,
            maybe_enable_memory_snapshot(
                config.profiling,
                global_step=self.step,
                base_folder=config.dump_folder,
            ) as memory_profiler,
        ):
            data_iterator = self.batch_generator(self.dataloader)
            
            while self.should_continue_training():
                
                # --- CURRICULUM TRANSITION CHECK ---
                expected_stage = self._get_stage_idx(self.step)
                if expected_stage != self.stage_idx:
                    logger.info(f"Transitioning from Stage {self.stage_idx} to Stage {expected_stage} at step {self.step}")
                    self.stage_idx = expected_stage
                    
                    # Rebuild dataloader for the new stage
                    del self.dataloader
                    del data_iterator
                    
                    dp_mesh = self.parallel_dims.get_mesh("batch") if self.parallel_dims.dp_enabled else None
                    self.dataloader = config.dataloader.build(
                        dp_world_size=dp_mesh.size() if dp_mesh else 1,
                        dp_rank=dp_mesh.get_local_rank() if dp_mesh else 0,
                        tokenizer=self.tokenizer,
                        seq_len=config.training.seq_len,
                        local_batch_size=config.training.local_batch_size,
                        stage_idx=self.stage_idx,
                    )
                    if hasattr(self.dataloader, "step"):
                        self.dataloader.step(self.step)
                    data_iterator = self.batch_generator(self.dataloader)
                # -----------------------------------

                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break
        
                if hasattr(self.dataloader, "step"):
                    self.dataloader.step(self.step)
                
                self.checkpointer.save(
                    self.step, last_step=(self.step == config.training.steps)
                )

                # Run validation if validator is available
                if self.config.validator.enable and self.validator.should_validate(
                    self.step
                ):
                    self.validator.validate(self.model_parts, self.step)

                # Active forgetting: periodically reinitialize the token embeddings. Done after
                # checkpoint/validation so those reflect the trained (pre-reset) model.
                self._maybe_active_forget()

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(seconds=config.comm.train_timeout_seconds),
                        parallel_dims=self.parallel_dims,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.step < self.config.training.steps

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]

    def close(self) -> None:
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.close()
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()
