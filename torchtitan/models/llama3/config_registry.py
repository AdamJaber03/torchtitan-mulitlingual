# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import (
    OptimizersContainer,
    OptimizersInBackwardContainer,
)
from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.components.validate import Validator
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
    LossConfig,
)
from torchtitan.hf_datasets.text_datasets import (
    En1En2TranslationValidationDataLoader,
    HuggingFaceTextDataLoader,
)
from torchtitan.experiments.en1_en2 import (
    EntityCorpusSpec,
    assign_rate_grid,
    build_entity_records,
    get_injection_probabilities,
    get_injection_probability_plan,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer

from . import model_registry
import os
from pathlib import Path
import random
import re

import numpy as np

def llama3_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B"),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
        validator=Validator.Config(
            freq=5,
            steps=10,
        ),
    )


def llama3_debugmodel_flex_attn() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_spec = model_registry("debugmodel_flex_attn")
    return config


def llama3_debugmodel_varlen_attn() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_spec = model_registry("debugmodel_varlen_attn")
    return config


def llama3_debugmodel_opt_in_bwd() -> Trainer.Config:
    config = llama3_debugmodel()
    config.optimizer = OptimizersInBackwardContainer.Config(lr=8e-4)
    return config


def llama3_debugmodel_float8() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(
                enable_fsdp_float8_all_gather=True,
                precompute_float8_dynamic_scale_for_fsdp=True,
            ),
        ],
    )
    return config


def llama3_debugmodel_float8_emulate() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(
                enable_fsdp_float8_all_gather=True,
                precompute_float8_dynamic_scale_for_fsdp=True,
                emulate=True,
            ),
        ],
    )
    return config


def llama3_8b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        profiling=ProfilingConfig(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_registry("8B"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def llama3_70b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Llama-3.1-70B",
        profiling=ProfilingConfig(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_registry("70B"),
        optimizer=OptimizersContainer.Config(lr=1.5e-4),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=8192,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def llama3_405b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Llama-3.1-405B",
        profiling=ProfilingConfig(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_registry("405B"),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                Float8LinearConverter.Config(
                    enable_fsdp_float8_all_gather=True,
                    precompute_float8_dynamic_scale_for_fsdp=True,
                    filter_fqns=["output"],
                ),
            ],
        ),
        optimizer=OptimizersContainer.Config(lr=8e-5),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=8192,
            steps=3000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=8,
            enable_async_tensor_parallel=True,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=True),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def llama3_160m_mha() -> Trainer.Config:
    return Trainer.Config(
        # Point to the default Llama-3 tokenizer (or your custom one)
        hf_assets_path="./tests/assets/tokenizer", 
        
        # Reference the model shape from model_registry
        model_spec=model_registry("160M_mha_baseline"), 
        
        # Turn on Blackwell FP8 Linear layers
        # model_converters=ModelConvertersContainer.Config(
        #     converters=[
        #         Float8LinearConverter.Config(
        #             enable_fsdp_float8_all_gather=False,
        #             precompute_float8_dynamic_scale_for_fsdp=False,
        #         ),
        #     ],
        # ),
        # --- THE FIX 1: DDP OVER FSDP ---
        # This tells TorchTitan's DeviceMesh to replicate the entire 160M model 
        # on all 4 GPUs (DDP) instead of chopping it up (FSDP).
        # parallelism=ParallelismConfig(
        #     data_parallel_replicate_degree=4,  
        #     data_parallel_shard_degree=1,      
        # ),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        # Learning Rate for a 160M model
        optimizer=OptimizersContainer.Config(lr=1e-3),
        # lr_scheduler=LRSchedulersContainer.Config(warmup_steps=400),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=150,        # ~5% of your 4000 steps
            decay_ratio=1.0,         # Decay over the full duration
            decay_type="cosine",     # Standard for Llama-style training
            min_lr_factor=0.1,       # Keep a small "tail" LR for final convergence
        ),
        # The batch configuration
        training=TrainingConfig(
            local_batch_size=240, # 192 local * 4 GPUs = 768 Global Batch Size
            seq_len=512,
            steps=9000,
        ),
        
        # # Dataset
        # dataloader=MixedPretokenizedDataLoader.Config(
        #     data_paths=[r"/home/adamga/gpt-neox/data/raw_multilingual_jsonl/en.bin", r"/home/adamga/gpt-neox/data/raw_multilingual_jsonl/arb_Arab.bin"],
        #     mix_rates=[0.5, 0.5],
        #     num_workers=24, # Push data loading to background CPU threads
        # ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
            num_workers=8
        ),
        
        # Enable torch.compile for maximum speed
        compile=CompileConfig(enable=True),
        
        # Tracking via WandB
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
            # config_dict={"project_name": "160M_Architecture_Tests"}
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/160m_mha_11_10+512ctx",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=True,
            async_mode="async"
        ),
    )
def llama3_160m_mha_flex() -> Trainer.Config:
    return Trainer.Config(
        # Point to the default Llama-3 tokenizer (or your custom one)
        hf_assets_path="./tests/assets/tokenizer", 
        
        # Reference the model shape from model_registry
        model_spec=model_registry("160M_mha_flex_baseline"), 
        
        # Turn on Blackwell FP8 Linear layers
        # model_converters=ModelConvertersContainer.Config(
        #     converters=[
        #         Float8LinearConverter.Config(
        #             enable_fsdp_float8_all_gather=False,
        #             precompute_float8_dynamic_scale_for_fsdp=False,
        #         ),
        #     ],
        # ),
        # --- THE FIX 1: DDP OVER FSDP ---
        # This tells TorchTitan's DeviceMesh to replicate the entire 160M model 
        # on all 4 GPUs (DDP) instead of chopping it up (FSDP).
        # parallelism=ParallelismConfig(
        #     data_parallel_replicate_degree=2,  
        #     data_parallel_shard_degree=1,      
        # ),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        # Learning Rate for a 160M model
        optimizer=OptimizersContainer.Config(lr=1e-3),
        # lr_scheduler=LRSchedulersContainer.Config(warmup_steps=400),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=400,        # ~5% of your 4000 steps
            decay_ratio=1.0,         # Decay over the full duration
            decay_type="cosine",     # Standard for Llama-style training
            min_lr_factor=0.03,       # Keep a small "tail" LR for final convergence
        ),
        # The batch configuration
        training=TrainingConfig(
            local_batch_size=60, # 192 local * 4 GPUs = 768 Global Batch Size
            seq_len=2048,
            steps=18000,
        ),
        
        # # Dataset
        # dataloader=MixedPretokenizedDataLoader.Config(
        #     data_paths=[r"/home/adamga/gpt-neox/data/raw_multilingual_jsonl/en.bin", r"/home/adamga/gpt-neox/data/raw_multilingual_jsonl/arb_Arab.bin"],
        #     mix_rates=[0.5, 0.5],
        #     num_workers=24, # Push data loading to background CPU threads
        # ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
            num_workers=8
        ),
        
        # Enable torch.compile for maximum speed
        compile=CompileConfig(enable=True),
        
        # Tracking via WandB
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
            # config_dict={"project_name": "160M_Architecture_Tests"}
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/160m_mha_14_flex_2048ctx_ffnmult1.5",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=True,
            async_mode="async"
        ),
    )

def llama3_160m_gqa() -> Trainer.Config:
    # We can just copy the MHA config and change the model shape and dump folder
    config = llama3_160m_mha()
    config.model_spec = model_registry("160m_gqa_balanced")
    config.checkpoint.folder_path = "./outputs/160m_gqa"
    return config

def llama3_500m_mha() -> Trainer.Config:
    return Trainer.Config(
        # Point to the default Llama-3 tokenizer (or your custom one)
        hf_assets_path="./tests/assets/tokenizer", 
        
        # Reference the model shape from model_registry
        model_spec=model_registry("500M_mha_baseline"), 
        
        # Turn on Blackwell FP8 Linear layers
        # model_converters=ModelConvertersContainer.Config(
        #     converters=[
        #         Float8LinearConverter.Config(
        #             enable_fsdp_float8_all_gather=True,
        #             precompute_float8_dynamic_scale_for_fsdp=True,
        #         ),
        #     ],
        # ),
        # --- THE FIX 1: DDP OVER FSDP ---
        # This tells TorchTitan's DeviceMesh to replicate the entire 160M model 
        # on all 4 GPUs (DDP) instead of chopping it up (FSDP).
        # parallelism=ParallelismConfig(
        #     data_parallel_replicate_degree=4,  
        #     data_parallel_shard_degree=1,      
        # ),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        # Learning Rate for a 500M model
        optimizer=OptimizersContainer.Config(lr=6e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 4000 steps
            decay_ratio=1.0,         # Decay over the full duration
            decay_type="cosine",     # Standard for Llama-style training
            min_lr_factor=0.1,       # Keep a small "tail" LR for final convergence
        ),        
        # The 12-hour batch configuration
        training=TrainingConfig(
            local_batch_size=36, # 36 local * 4 GPUs = 144 Global Batch Size
            seq_len=2048,
            steps=20000,
        ),
        
        # # Dataset
        # dataloader=MixedPretokenizedDataLoader.Config(
        #     data_paths=[r"/home/adamga/gpt-neox/data/raw_multilingual_jsonl/en.bin", r"/home/adamga/gpt-neox/data/raw_multilingual_jsonl/arb_Arab.bin"],
        #     mix_rates=[0.5, 0.5],
        #     num_workers=24, # Push data loading to background CPU threads
        # ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
            num_workers=8
        ),
        
        # Enable torch.compile for maximum speed
        compile=CompileConfig(enable=True),
        
        # Tracking via WandB
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
            # config_dict={"project_name": "160M_Architecture_Tests"}
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/500m_mha_03_65kVocab_6e-4lr",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=True,
            async_mode="async"
        ),
    )
def llama3_500m_mha_flex() -> Trainer.Config:
    return Trainer.Config(
        # Point to the default Llama-3 tokenizer (or your custom one)
        hf_assets_path="./tests/assets/tokenizer", 
        
        # Reference the model shape from model_registry
        model_spec=model_registry("500M_mha_flex_baseline"), 

        # Turn on Blackwell FP8 Linear layers
        # model_converters=ModelConvertersContainer.Config(
        #     converters=[
        #         Float8LinearConverter.Config(
        #             enable_fsdp_float8_all_gather=True,
        #             precompute_float8_dynamic_scale_for_fsdp=True,
        #         ),
        #     ],
        # ),
        # --- THE FIX 1: DDP OVER FSDP ---
        # This tells TorchTitan's DeviceMesh to replicate the entire 160M model 
        # on all 4 GPUs (DDP) instead of chopping it up (FSDP).
        # parallelism=ParallelismConfig(
        #     data_parallel_replicate_degree=4,  
        #     data_parallel_shard_degree=1,      
        # ),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        # Learning Rate for a 500M model
        optimizer=OptimizersContainer.Config(lr=6e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 4000 steps
            decay_ratio=1.0,         # Decay over the full duration
            decay_type="cosine",     # Standard for Llama-style training
            min_lr_factor=0.1,       # Keep a small "tail" LR for final convergence
        ),        
        # The 12-hour batch configuration
        training=TrainingConfig(
            local_batch_size=36, # 36 local * 4 GPUs = 144 Global Batch Size
            seq_len=2048,
            steps=20000,
        ),
        
        # # Dataset
        # dataloader=MixedPretokenizedDataLoader.Config(
        #     data_paths=[r"/home/adamga/gpt-neox/data/raw_multilingual_jsonl/en.bin", r"/home/adamga/gpt-neox/data/raw_multilingual_jsonl/arb_Arab.bin"],
        #     mix_rates=[0.5, 0.5],
        #     num_workers=24, # Push data loading to background CPU threads
        # ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
            num_workers=8
        ),
        
        # Enable torch.compile for maximum speed
        compile=CompileConfig(enable=True),
        
        # Tracking via WandB
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
            # config_dict={"project_name": "160M_Architecture_Tests"}
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/500m_mha_04_flex_2048ctx_6e-4lr",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=True,
            async_mode="async"
        ),
    )
def smollm2_360m_flex() -> Trainer.Config:
    return Trainer.Config(      
        # hf_assets_path="./tests/assets/Yi-1.5-9B-Tokenizer",  # Using Yi-1.5 tokenizer for 64k vocab compatibility
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=16,
            sources=[
                # Source 1: The main educational dataset with a high weight
                {
                    "name": "fineweb-edu-ar-ar",
                    "weight": 0.3,
                    "injection_paths": [
                        "/home/adamga/leshemg/adamga/data/fictive_entities_gemini/1_ar.jsonl",
                        "/home/adamga/leshemg/adamga/data/fictive_entities_gemini/2_ar.jsonl",
                        "/home/adamga/leshemg/adamga/data/fictive_entities_gemini/3_ar.jsonl"
                    ],
                    "injection_probs": [0.00005, 0.00005, 0.00005]
                },
                # Source 2: Standard C4 to maintain general knowledge, with no injections
                {
                    "name": "fineweb-edu-ar-en",
                    "weight": 0.7,
                    "injection_paths": [
                        "/home/adamga/leshemg/adamga/data/fictive_entities_gemini/1.jsonl",
                        "/home/adamga/leshemg/adamga/data/fictive_entities_gemini/2.jsonl",
                        "/home/adamga/leshemg/adamga/data/fictive_entities_gemini/3.jsonl"
                    ],
                    "injection_probs": [0.000023, 0.000023, 0.000023]
                }
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m"), 
        
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        
        # Matched to official SmolLM2 360M hyperparams
        optimizer=OptimizersContainer.Config(
            lr=5e-4,              # HF increased peak LR for 360M compared to 135M
            weight_decay=0.1,     # Mandatory for AdamW (Loshchilov & Hutter)
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.05,       # Decays down to 5e-5 at step 6000
        ),
        
        # The Official 1.57M Token Batch Setup
        training=TrainingConfig(
            local_batch_size=32,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4000,                 # 6000 steps aligns perfectly with Chinchilla
            max_norm=1.0,               # Gradient clipping 
        ),
        
        compile=CompileConfig(enable=True),
        
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        
        checkpoint=CheckpointManager.Config(
            interval=1000, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_25_13_en_inject_50_ar_inject_50",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
            # initial_load_path="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_19_13_en_inject_0_ar_inject_200/step-1000"
        ),
    )
def smollm2_135m_flex() -> Trainer.Config:
    return Trainer.Config(      
        hf_assets_path="./tests/assets/smallm2_tokenizer",  # Using Yi-1.5 tokenizer for 64k vocab compatibility
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="fineweb-edu-100b-shuffle",
            num_workers=16
        ),  
        # Reference your 360M model shape
        model_spec=model_registry("smollm2_135m"), 
        
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        
        # Matched to official SmolLM2 360M hyperparams
        optimizer=OptimizersContainer.Config(
            lr=3e-4,              # Lowered to HF's exact peak LR
            weight_decay=0.1,     # Mandatory for AdamW (Loshchilov & Hutter)
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.01,       # Decays down to 5e-5 at step 6000
        ),
        
        # The Official 1.57M Token Batch Setup
        training=TrainingConfig(
            local_batch_size=48,       # 48 * 4 GPUs * 2048 seq_len = 393,216 tokens
            global_batch_size=384,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=7000,
            max_norm=1.0,               # Gradient clipping to prevent explosion
        ),
        
        compile=CompileConfig(enable=True),
        
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        
        checkpoint=CheckpointManager.Config(
            interval=1000, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_135m_flex_02_weighttying_0.01minfactor_49kvocab",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async"
        ),
    )
def smollm2_360m_flex_curriculum() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    # file_order_shuffler = random.Random(43)
    file_order_shuffler = random.Random(43)
    # file_order = []
    # for start in range(0, 1408, 32):
    #     chunk = list(range(start, start + 32))
    #     file_order_shuffler.shuffle(chunk)
    #     file_order.extend(chunk)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_data.jsonl" for i in file_order]
    # en_files = [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in range(2080)] + [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in range(2080)]
    # en_files = [en_files[i] for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    # unique_rates = [5]*468 + [20]*468 + [80]*468

    return Trainer.Config(      
        # hf_assets_path="./tests/assets/Yi-1.5-9B-Tokenizer",  # Using Yi-1.5 tokenizer for 64k vocab compatibility
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                # --- STAGE 0: wordwise codeswitching Phase (Steps 0 to 1300) ---
                # {
                #     "steps": 2000,
                #     "sources": [
                #         {
                #             "name": "fineweb-edu-ar-ar",
                #             "weight": 0.5,
                #             "injection_paths": ar_files,
                #             "injection_probs": ar_probs,
                #         },
                #         {
                #             "name": "fineweb-edu-ar-en",
                #             "weight": 0.5,
                #             "injection_paths": en_files,
                #             "injection_probs": en_probs,
                #         }
                #     ],
                #     "augmentations": [
                #         {
                #             "name": "wordwise_codeswitching",
                #             "prob": 0.5,  # 50% of the text will undergo wordwise code-switching
                #             "dict_paths": {
                #                 "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated.json",
                #                 "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_english_translated.json"
                #             }
                #         }
                #     ]
                # },
                # # --- STAGE 1: Sentence-level Code-Switching with Entity Injection (Steps 1300 to 2600) ---
                # {
                #     "steps": 1300,
                #     "sources": [
                #         {
                #             "name": "fineweb-edu-ar-paired",
                #             "weight": 1.0,
                #             "start_idx": 2_000_000,
                #             "injection_paths": ar_files + en_files,
                #             "injection_probs": [p for p in ar_probs + en_probs],    # originally devided prob by 2 but since documents are now twice as large it evens out
                #         }
                #     ],
                #     "augmentations": [
                #         # {
                #         #     "name": "document_translation"
                #         # }
                #     ]

                # },

                # --- STAGE 2: Clean Phase ---
                # {
                #     "steps": 2000,
                #     "sources": [
                #         {
                #             "name": "fineweb-edu-ar-ar",
                #             "weight": 0.5,
                #             "start_idx": 4_000_000,
                #             "injection_paths": ar_files,
                #             "injection_probs": ar_probs,
                #             # "unique_rates": unique_rates
                #         },
                #         {
                #             "name": "fineweb-edu-ar-en",
                #             "weight": 0.5,
                #             "start_idx": 4_000_000,
                #             "injection_paths": en_files,
                #             "injection_probs": en_probs,
                #             # "unique_rates": unique_rates
                #         }
                #     ],
                    
                # },
                #en1_en2_stage1
                {
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    # "prob": 0.8,  # 50% of the text will undergo wordwise code-switching
                                    "prob_schedualer": {
                                        "name": "linear",
                                        "start_val": 1.0,
                                        "end_val": 0.0,
                                        "duration_steps": 2000
                                    },
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex.json",
                                    }
                                }
                            ],

                            # "post_token_augmentations": [
                            #     {
                            #         # "name": "stochastic_token_tagging",
                            #         "name": "stochastic_word_tagging",
                            #         "prob": 0.5,
                            #         "vocab_size": 65536,
                            #     }
                            # ],
                        },
                        # {
                        #     "name": "fineweb-edu-ar-tr2en",
                        #     "weight": 0.2,
                        #     "start_idx": 3_000_000,
                        #     "injection_paths": tr2en_files,
                        #     "injection_probs": [prob *5/2 for prob in en_probs],
                        #     "augmentations": [
                        #         # {
                        #         #     "name": "wordwise_unigram_codeswitching",
                        #         #     "prob": 1.0,  # 50% of the text will undergo wordwise code-switching
                        #         #     "dict_paths": {
                        #         #         "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_11M.json",
                        #         #     }
                        #         # }
                        #     ],
                        # },
                        {
                            "name": "fineweb-edu-ar-tr2en",
                            "weight": 0.5,
                            "start_idx": 1_800_000,
                            # "injection_paths": ar_files,
                            # "injection_probs": [prob *5/2 for prob in ar_probs],
                            # "post_token_augmentations": [
                            #     {
                            #         # "name": "stochastic_token_tagging",
                            #         "name": "stochastic_word_tagging",
                            #         "prob": 1.0,
                            #         "vocab_size": 65536,
                            #     }
                            # ],
                        }

                    ],
                },
                #en1_en2_stage2
                {
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-tr2en",
                            "weight": 0.5,
                            "start_idx": 3_600_000,
                            "injection_paths": tr2en_files,
                            "injection_probs": [en_prob*2 for en_prob in en_probs],
                            # "augmentations": [
                            #     {
                            #         "name": "wordwise_unigram_codeswitching",
                            #         "prob": 1.0,  # 50% of the text will undergo wordwise code-switching
                            #         "dict_paths": {
                            #             "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_11M.json",
                            #         }
                            #     }
                            # ],

                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "start_idx": 4_800_000,
                            "injection_paths": ar_files,
                            "injection_probs": [ar_prob*2 for ar_prob in ar_probs],
                            # "augmentations": [
                            #     {
                            #         "name": "wordwise_unigram_codeswitching",
                            #         "prob": 1.0,  # 50% of the text will undergo wordwise code-switching
                            #         "dict_paths": {
                            #             "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_11M.json",
                            #         }
                            #     }
                            # ],

                            # "unique_rates": unique_rates
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m"), 
        
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        
        # Matched to official SmolLM2 360M hyperparams
        optimizer=OptimizersContainer.Config(
            lr=5e-4,              # HF increased peak LR for 360M compared to 135M
            weight_decay=0.1,     # Mandatory for AdamW (Loshchilov & Hutter)
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.05,       # Decays down to 5e-5 at step 6000
        ),
        
        # The Official 1.57M Token Batch Setup
        training=TrainingConfig(
            local_batch_size=24,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4000,                 # 6000 steps aligns perfectly with Chinchilla
            max_norm=1.0,               # Gradient clipping 
        ),
        
        compile=CompileConfig(enable=True),
        
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_84_ar_tr2en_1xvocab_stage1_2k_0.5tr2en_0.5wordwise_p1-0_linear_no_injection_stage2_2k_clean_injection_0_20_100_1000_20800entities_rerun",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
            # initial_load_path="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_01_wordwise_codeswitching_baseline_en_0.7_ar_0.3/step-2000",
            # load_step=2000
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={"arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "tr2en": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-tr2en",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                    # "augmentations": [
                                    #     {
                                    #         "name": "wordwise_unigram_codeswitching",
                                    #         "prob": 1.0,  # 50% of the text will undergo wordwise code-switching
                                    #         "dict_paths": {
                                    #             "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_11M.json",
                                    #         }
                                    #     }
                                    # ],
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
            },
        ),
        loss=LossConfig(
            losses=[
                {
                    "name": "cross_entropy",
                    "weight": 1.0
                },
            ]
        )
    )
def smollm2_360m_flex_curriculum1() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    tr2en_1to1map_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_1to1map_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage2
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 3_300_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-tr2en_1to1map",
                            "weight": 0.5,
                            "injection_paths": tr2en_1to1map_files,
                            "injection_probs": ar_probs,
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=5e-4,              # HF increased peak LR for 360M compared to 135M
            weight_decay=0.1,     # Mandatory for AdamW (Loshchilov & Hutter)
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.05,       # Decays down to 5e-5 at step 6000
        ),
        training=TrainingConfig(
            local_batch_size=24,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4000,                 # 6000 steps aligns perfectly with Chinchilla
            max_norm=1.0,               # Gradient clipping 
        ),
        compile=CompileConfig(enable=True),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_85_en_tr2en_1to1map_1xvocab_stage1_4k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={"en": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "tr2en_1to1map": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-tr2en_1to1map",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
            },
        ),
        loss=LossConfig(
            losses=[
                {
                    "name": "cross_entropy",
                    "weight": 1.0
                },
            ]
        )
    )
def smollm2_360m_flex_curriculum2() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 3000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.01,
                                    "symmetric": True,
                                    "vocab_size": 65536,
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "start_idx": 3_000_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/2 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "start_idx": 4_000_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/2 for prob in ar_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        }

                    ],
                },
                #en1_en2_stage2
                {
                    "steps": 1000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_800_000,
                            "injection_paths": en_files,
                            "injection_probs": ar_probs,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_2xvocab"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=5e-4,              # HF increased peak LR for 360M compared to 135M
            weight_decay=0.1,     # Mandatory for AdamW (Loshchilov & Hutter)
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.05,       # Decays down to 5e-5 at step 6000
        ),
        training=TrainingConfig(
            local_batch_size=24,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4000,                 # 6000 steps aligns perfectly with Chinchilla
            max_norm=1.0,               # Gradient clipping 
        ),
        compile=CompileConfig(enable=True),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_79_en1_en2_2xvocab_stage1_3k_0.6_wordwise0.01_stage2_1k_clean_injection_0_20_100_1000_20800entities_rerun",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={"en1": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "en2": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
            },
        ),
        loss=LossConfig(
            losses=[
                {
                    "name": "cross_entropy",
                    "weight": 1.0
                },
            ]
        )
    )
def smollm2_360m_flex_curriculum3() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 3000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.5,
                                    "symmetric": True,
                                    "vocab_size": 65536,
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.45,
                            "start_idx": 3_000_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/4.5 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.45,
                            "start_idx": 4_000_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/4.5 for prob in ar_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        }

                    ],
                },
                #en1_en2_stage2
                {
                    "steps": 1000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_800_000,
                            "injection_paths": en_files,
                            "injection_probs": ar_probs,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_2xvocab"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=5e-4,              # HF increased peak LR for 360M compared to 135M
            weight_decay=0.1,     # Mandatory for AdamW (Loshchilov & Hutter)
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.05,       # Decays down to 5e-5 at step 6000
        ),
        training=TrainingConfig(
            local_batch_size=24,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4000,                 # 6000 steps aligns perfectly with Chinchilla
            max_norm=1.0,               # Gradient clipping 
        ),
        compile=CompileConfig(enable=True),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_80_en1_en2_2xvocab_stage1_3k_0.1_wordwise0.5_stage2_1k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={"en1": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "en2": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
            },
        ),
        loss=LossConfig(
            losses=[
                {
                    "name": "cross_entropy",
                    "weight": 1.0
                },
            ]
        )
    )
def smollm2_360m_flex_curriculum4() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 3000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.01,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.5,
                                    "symmetric": True,
                                    "vocab_size": 65536,
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.495,
                            "start_idx": 3_000_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/4.95 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.495,
                            "start_idx": 4_000_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/4.95 for prob in ar_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        }

                    ],
                },
                #en1_en2_stage2
                {
                    "steps": 1000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_800_000,
                            "injection_paths": en_files,
                            "injection_probs": ar_probs,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_2xvocab"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=5e-4,              # HF increased peak LR for 360M compared to 135M
            weight_decay=0.1,     # Mandatory for AdamW (Loshchilov & Hutter)
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.05,       # Decays down to 5e-5 at step 6000
        ),
        training=TrainingConfig(
            local_batch_size=24,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4000,                 # 6000 steps aligns perfectly with Chinchilla
            max_norm=1.0,               # Gradient clipping 
        ),
        compile=CompileConfig(enable=True),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_81_en1_en2_2xvocab_stage1_3k_0.01_wordwise0.5_stage2_1k_clean_injection_0_20_100_1000_20800entities_rerun",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={"en1": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "en2": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
            },
        ),
        loss=LossConfig(
            losses=[
                {
                    "name": "cross_entropy",
                    "weight": 1.0
                },
            ]
        )
    )
def smollm2_360m_flex_curriculum5() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 3000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.1,
                                    "symmetric": True,
                                    "vocab_size": 65536,
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.45,
                            "start_idx": 3_000_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/4.5 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.45,
                            "start_idx": 4_000_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/4.5 for prob in ar_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        }

                    ],
                },
                #en1_en2_stage2
                {
                    "steps": 1000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_800_000,
                            "injection_paths": en_files,
                            "injection_probs": ar_probs,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_2xvocab"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=5e-4,              # HF increased peak LR for 360M compared to 135M
            weight_decay=0.1,     # Mandatory for AdamW (Loshchilov & Hutter)
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.05,       # Decays down to 5e-5 at step 6000
        ),
        training=TrainingConfig(
            local_batch_size=24,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4000,                 # 6000 steps aligns perfectly with Chinchilla
            max_norm=1.0,               # Gradient clipping 
        ),
        compile=CompileConfig(enable=True),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_82_en1_en2_2xvocab_stage1_3k_0.1_wordwise0.1_stage2_1k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={"en1": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "en2": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
            },
        ),
        loss=LossConfig(
            losses=[
                {
                    "name": "cross_entropy",
                    "weight": 1.0
                },
            ]
        )
    )
def smollm2_360m_flex_curriculum6() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 3000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.01,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.01,
                                    "symmetric": True,
                                    "vocab_size": 65536,
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.495,
                            "start_idx": 3_000_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/4.95 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.495,
                            "start_idx": 4_000_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/4.95 for prob in ar_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        }

                    ],
                },
                #en1_en2_stage2
                {
                    "steps": 1000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_800_000,
                            "injection_paths": en_files,
                            "injection_probs": ar_probs,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_2xvocab"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=5e-4,              # HF increased peak LR for 360M compared to 135M
            weight_decay=0.1,     # Mandatory for AdamW (Loshchilov & Hutter)
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.05,       # Decays down to 5e-5 at step 6000
        ),
        training=TrainingConfig(
            local_batch_size=24,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4000,                 # 6000 steps aligns perfectly with Chinchilla
            max_norm=1.0,               # Gradient clipping 
        ),
        compile=CompileConfig(enable=True),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_83_en1_en2_2xvocab_stage1_3k_0.01_wordwise0.01_stage2_1k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={"en1": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "en2": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
            },
        ),
        loss=LossConfig(
            losses=[
                {
                    "name": "cross_entropy",
                    "weight": 1.0
                },
            ]
        )
    )
def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default
    return float(raw_value)

def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default
    return int(raw_value)

def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default
    return raw_value.lower() in {"1", "true", "yes", "on"}

def _mixed_data_fraction() -> float:
    raw_value = os.environ.get(
        "EN1_EN2_MIXED_DATA_FRACTION",
        os.environ.get("MIXED_DATA_FRACTION", "0.6"),
    )
    value = float(raw_value)
    if not 0.0 <= value < 1.0:
        raise ValueError(
            "EN1_EN2_MIXED_DATA_FRACTION must be a fraction in [0, 1), "
            "because stage 1 includes nonzero clean-source injection targets; "
            f"got {raw_value}."
        )
    return value

def _multilingual_pretraining_root() -> Path:
    root = os.environ.get("MULTILINGUAL_PRETRAINING_ROOT")
    if root:
        return Path(root).expanduser()

    torchtitan_root = Path(os.environ.get("TORCHTITAN_ROOT", Path.cwd())).resolve()
    if torchtitan_root.name == "multilingual-pretraining":
        return torchtitan_root
    return torchtitan_root.parent

def _under_multilingual_root(*parts: str) -> str:
    return str(_multilingual_pretraining_root().joinpath(*parts))

def _en1_en2_entity_corpus_specs() -> list[EntityCorpusSpec]:
    specs = [
        EntityCorpusSpec(
            name="gemini_seeds",
            root=Path(
                os.environ.get(
                    "FICTIONAL_ENTITY_DATA_ROOT",
                    _under_multilingual_root(
                        "fictional_entity_data", "gemini_seeds"
                    ),
                )
            ).expanduser(),
            count=_env_int("EN1_EN2_GEMINI_ENTITY_COUNT", 2080),
            rng="python",
            seed=_env_int("EN1_EN2_GEMINI_SHUFFLE_SEED", 43),
            token_stats_key="gemini_seeds_en",
        )
    ]
    if _env_bool("EN1_EN2_INCLUDE_HUMAN_ENTITIES", False):
        specs.append(
            EntityCorpusSpec(
                name="from_domains_humans",
                root=Path(
                    os.environ.get(
                        "EN1_EN2_HUMAN_ENTITY_DATA_ROOT",
                        _under_multilingual_root(
                            "fictional_entity_data", "from_domains_humans"
                        ),
                    )
                ).expanduser(),
                count=_env_int("EN1_EN2_HUMAN_ENTITY_COUNT", 2080),
                rng="numpy",
                seed=_env_int("EN1_EN2_HUMAN_SHUFFLE_SEED", 48),
                token_stats_key="from_domains_humans_en",
            )
        )
    return specs


def _en1_en2_fictional_entities(
    target_counts: tuple[float, ...],
) -> tuple[list[EntityCorpusSpec], list[str], list[float], list[float]]:
    corpus_specs = _en1_en2_entity_corpus_specs()
    records = build_entity_records(
        corpus_specs,
        data_filename="en_data.jsonl",
        rate_count=len(target_counts),
    )
    en1_targets, en2_targets = assign_rate_grid(records, target_counts)
    return (
        corpus_specs,
        [str(record.data_path) for record in records],
        en1_targets,
        en2_targets,
    )


def _probabilities_for_entities(
    entity_targets: list[float],
    target_counts: tuple[float, ...],
    probabilities: list[float],
) -> list[float]:
    probability_by_target = dict(zip(target_counts, probabilities, strict=True))
    return [probability_by_target[target] for target in entity_targets]


def _en1_en2_experiment_name(
    mode: str,
    mixed_data_fraction: float,
    stage1_steps: int,
    clean_steps: int,
) -> str:
    def validate_name(name: str, variable: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ValueError(
                f"{variable} must contain only letters, digits, '.', '_', and '-'."
            )
        return name

    explicit_name = os.environ.get("EN1_EN2_EXPERIMENT_NAME")
    if explicit_name:
        return validate_name(explicit_name, "EN1_EN2_EXPERIMENT_NAME")

    mixed_data_tag = f"{mixed_data_fraction * 100:g}".replace(".", "p")
    name = (
        f"smollm2_360m_en1_en2_{mode}"
        f"_mixed_data{mixed_data_tag}pct"
        f"_stage1_{stage1_steps}_stage2_{clean_steps}"
    )
    run_tag = os.environ.get("EN1_EN2_RUN_TAG", "").strip()
    if run_tag:
        validate_name(run_tag, "EN1_EN2_RUN_TAG")
        name = f"{name}_{run_tag}"
    return name

def _en2_post_token_aug(vocab_size: int) -> list[dict]:
    return [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": vocab_size,
        }
    ]

def _span_post_token_aug(vocab_size: int) -> list[dict]:
    return [
        {
            "name": "language_span_token_tagging",
            "vocab_size": vocab_size,
        }
    ]

def _append_source_if_positive(sources: list[dict], source: dict) -> None:
    if source.get("weight", 0.0) > 0.0:
        sources.append(source)

def _en1_en2_sentence_validation_dataloaders(
    vocab_size: int, translation_validation_enabled: bool
) -> dict:
    num_workers = _env_int("EN1_EN2_NUM_WORKERS", 3)

    dataloaders = {
        "en1": HuggingFaceTextDataLoader.Config(
            num_workers=num_workers,
            stages=[
                {
                    "steps": 300,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 1.0,
                            "start_idx": 6_600_000,
                        },
                    ],
                }
            ],
            eos_token_id=0,
        ),
        "en2": HuggingFaceTextDataLoader.Config(
            num_workers=num_workers,
            stages=[
                {
                    "steps": 300,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 1.0,
                            "start_idx": 6_600_000,
                            "post_token_augmentations": _en2_post_token_aug(
                                vocab_size
                            ),
                        },
                    ],
                }
            ],
            eos_token_id=0,
        ),
    }

    if translation_validation_enabled:
        val_start_idx = _env_int("EN1_EN2_TRANSLATION_VAL_START_IDX", 6_600_000)
        translation_steps = _env_int(
            "EN1_EN2_TRANSLATION_VAL_STEPS",
            _env_int("EN1_EN2_VALIDATOR_STEPS", 10),
        )
        translation_workers = _env_int(
            "EN1_EN2_TRANSLATION_VAL_NUM_WORKERS", num_workers
        )
        dataloaders.update(
            {
                "translation_en1_to_en2": En1En2TranslationValidationDataLoader.Config(
                    num_workers=translation_workers,
                    direction="en1_to_en2",
                    start_idx=val_start_idx,
                    vocab_size=vocab_size,
                    eos_token_id=0,
                    validation_steps=translation_steps,
                ),
                "translation_en2_to_en1": En1En2TranslationValidationDataLoader.Config(
                    num_workers=translation_workers,
                    direction="en2_to_en1",
                    start_idx=val_start_idx,
                    vocab_size=vocab_size,
                    eos_token_id=0,
                    validation_steps=translation_steps,
                ),
            }
        )

    return dataloaders

def _smollm2_360m_en1_en2_sentence_config(mode: str) -> Trainer.Config:
    mixed_data_fraction = _mixed_data_fraction()
    clean_fraction = 1.0 - mixed_data_fraction
    stage1_steps = _env_int("EN1_EN2_STAGE1_STEPS", 3000)
    clean_steps = _env_int("EN1_EN2_CLEAN_STEPS", 1000)
    if stage1_steps <= 0 or clean_steps <= 0:
        raise ValueError("Both en1/en2 curriculum stages must contain steps.")
    total_steps = stage1_steps + clean_steps
    stage1_target_share = stage1_steps / total_steps
    clean_target_share = clean_steps / total_steps
    global_batch_size = _env_int("EN1_EN2_GLOBAL_BATCH_SIZE", 768)
    seq_len = _env_int("EN1_EN2_SEQ_LEN", 2048)
    vocab_size = _env_int("EN1_EN2_BASE_VOCAB_SIZE", 65536)
    translation_validation_enabled = _env_bool(
        "EN1_EN2_TRANSLATION_VALIDATION_ENABLE", False
    )
    validator_steps = _env_int("EN1_EN2_VALIDATOR_STEPS", 10)

    target_counts = (0.0, 20.0, 100.0, 1000.0)
    corpus_specs, en_files, en1_targets, en2_targets = (
        _en1_en2_fictional_entities(target_counts)
    )
    injection_datasets = [spec.token_stats_key for spec in corpus_specs]
    entity_counts = [spec.count for spec in corpus_specs]

    stage1_targets = [
        target * stage1_target_share for target in target_counts
    ]
    clean_targets = [
        target * clean_target_share for target in target_counts
    ]
    stage1_source_weight = clean_fraction / 2
    stage1_token_budget = (
        stage1_steps * global_batch_size * seq_len * stage1_source_weight
    )
    clean_token_budget = clean_steps * global_batch_size * seq_len * 0.5
    stage1_plan = get_injection_probability_plan(
        stage1_targets,
        stage1_token_budget,
        "fineweb-edu-ar-en",
        injection_datasets,
        entity_counts=entity_counts,
    )
    clean_plan = get_injection_probability_plan(
        clean_targets,
        clean_token_budget,
        "fineweb-edu-ar-en",
        injection_datasets,
        entity_counts=entity_counts,
    )
    stage1_en1_probs = _probabilities_for_entities(
        en1_targets,
        target_counts,
        list(stage1_plan.probabilities),
    )
    stage1_en2_probs = _probabilities_for_entities(
        en2_targets,
        target_counts,
        list(stage1_plan.probabilities),
    )
    clean_en1_probs = _probabilities_for_entities(
        en1_targets,
        target_counts,
        list(clean_plan.probabilities),
    )
    clean_en2_probs = _probabilities_for_entities(
        en2_targets,
        target_counts,
        list(clean_plan.probabilities),
    )

    logger.info(
        "en1/en2 injection plan: mode=%s mix=%g corpora=%s "
        "stage1_probabilities=%s clean_probabilities=%s",
        mode,
        mixed_data_fraction,
        [spec.name for spec in corpus_specs],
        list(stage1_plan.probabilities),
        list(clean_plan.probabilities),
    )

    stage1_sources = []
    _append_source_if_positive(
        stage1_sources,
        {
            "name": "fineweb-edu-ar-en",
            "weight": mixed_data_fraction,
            "augmentations": [
                {
                    "name": "synthetic_sentence_language_mixing",
                    "mode": mode,
                    "lang2_prob": 0.5,
                }
            ],
            "post_token_augmentations": _span_post_token_aug(vocab_size),
        },
    )
    _append_source_if_positive(
        stage1_sources,
        {
            "name": "fineweb-edu-ar-en",
            "weight": clean_fraction / 2,
            "start_idx": 3_000_000,
            "injection_paths": en_files,
            "injection_probs": stage1_en1_probs,
            "injection_target_counts": en1_targets,
            "injection_expected_counts": [
                target * stage1_target_share for target in en1_targets
            ],
            "injection_summary_name": "en1",
        },
    )
    _append_source_if_positive(
        stage1_sources,
        {
            "name": "fineweb-edu-ar-en",
            "weight": clean_fraction / 2,
            "start_idx": 4_000_000,
            "injection_paths": en_files,
            "injection_probs": stage1_en2_probs,
            "injection_target_counts": en2_targets,
            "injection_expected_counts": [
                target * stage1_target_share for target in en2_targets
            ],
            "injection_summary_name": "en2",
            "post_token_augmentations": _en2_post_token_aug(vocab_size),
        },
    )

    output_root = os.environ.get(
        "EN1_EN2_OUTPUT_ROOT",
        _under_multilingual_root("outputs", "torchtitan"),
    )
    experiment_name = _en1_en2_experiment_name(
        mode, mixed_data_fraction, stage1_steps, clean_steps
    )

    return Trainer.Config(
        hf_assets_path=os.environ.get(
            "EN1_EN2_HF_ASSETS_PATH",
            _under_multilingual_root("assets", "65k_paired"),
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=_env_int("EN1_EN2_NUM_WORKERS", 3),
            stages=[
                {
                    "steps": stage1_steps,
                    "sources": stage1_sources,
                },
                {
                    "steps": clean_steps,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": en_files,
                            "injection_probs": clean_en1_probs,
                            "injection_target_counts": en1_targets,
                            "injection_expected_counts": [
                                target * clean_target_share
                                for target in en1_targets
                            ],
                            "injection_summary_name": "en1",
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_800_000,
                            "injection_paths": en_files,
                            "injection_probs": clean_en2_probs,
                            "injection_target_counts": en2_targets,
                            "injection_expected_counts": [
                                target * clean_target_share
                                for target in en2_targets
                            ],
                            "injection_summary_name": "en2",
                            "post_token_augmentations": _en2_post_token_aug(
                                vocab_size
                            ),
                        },
                    ],
                },
            ],
            eos_token_id=0,
        ),
        model_spec=model_registry("smollm2_360m_2xvocab"),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=_env_float("EN1_EN2_LR", 5e-4),
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=_env_int("EN1_EN2_WARMUP_STEPS", 300),
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.05,
        ),
        training=TrainingConfig(
            local_batch_size=_env_int("EN1_EN2_LOCAL_BATCH_SIZE", 24),
            global_batch_size=global_batch_size,
            seq_len=seq_len,
            steps=stage1_steps + clean_steps,
            max_norm=1.0,
        ),
        compile=CompileConfig(enable=True),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=_env_int("EN1_EN2_LOG_FREQ", 10),
        ),
        checkpoint=CheckpointManager.Config(
            interval=_env_int("EN1_EN2_CHECKPOINT_INTERVAL", 500),
            folder=f"{output_root}/{experiment_name}",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=_env_int("EN1_EN2_VALIDATOR_FREQ", 500),
            steps=validator_steps,
            enable=True,
            dataloader=_en1_en2_sentence_validation_dataloaders(
                vocab_size, translation_validation_enabled
            ),
        ),
        loss=LossConfig(
            losses=[
                {
                    "name": "cross_entropy",
                    "weight": 1.0,
                },
            ]
        ),
    )

def smollm2_360m_en1_en2_sentence_wise_code_switching() -> Trainer.Config:
    return _smollm2_360m_en1_en2_sentence_config("sentence_wise_code_switching")

def smollm2_360m_en1_en2_sentence_parallel_doc_order() -> Trainer.Config:
    return _smollm2_360m_en1_en2_sentence_config("sentence_parallel_doc_order")

def smollm2_360m_en1_en2_sentence_parallel_sentence_order() -> Trainer.Config:
    return _smollm2_360m_en1_en2_sentence_config("sentence_parallel_sentence_order")

def smollm2_360m_flex_en1_en2() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_eng1, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    logger.info("Base probabilities for english1: %s", base_probs_en1)
    target_counts_eng2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_eng2, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    logger.info("Base probabilities for english2: %s", base_probs_en2)
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en2_probs,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_2xvocab"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=5e-4,              # HF increased peak LR for 360M compared to 135M
            weight_decay=0.1,     # Mandatory for AdamW (Loshchilov & Hutter)
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.05,       # Decays down to 5e-5 at step 6000
        ),
        training=TrainingConfig(
            local_batch_size=24,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4600,                 # 6000 steps aligns perfectly with Chinchilla
            max_norm=1.0,               # Gradient clipping 
        ),
        compile=CompileConfig(enable=True),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_203_en1_en2_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=30,
            enable=True,
            dataloader={
                "english1": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "english2": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
            },
        ),
        loss=LossConfig(
            losses=[
                {
                    "name": "cross_entropy",
                    "weight": 1.0
                },
            ]
        )
    )

def smollm2_360m_flex_en1_en2_codeswitching() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    pt1_base_probs_en1 = get_injection_probabilities([t*(3.6/4.6) for t in target_counts_eng1], tot_tokens=3600*768*2048/5, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    pt2_base_probs_en1 = get_injection_probabilities([t*(1.0/4.6) for t in target_counts_eng1], tot_tokens=1000*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    
    logger.info("Base probabilities for english1 (part 1): %s", pt1_base_probs_en1)
    logger.info("Base probabilities for english1 (part 2): %s", pt2_base_probs_en1)
    target_counts_eng2 = [0, 20, 100, 1000]
    pt1_base_probs_en2 = get_injection_probabilities([t*(3.6/4.6) for t in target_counts_eng2], tot_tokens=3600*768*2048/5, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    pt2_base_probs_en2 = get_injection_probabilities([t*(1.0/4.6) for t in target_counts_eng2], tot_tokens=1000*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    
    logger.info("Base probabilities for english2 (part 1): %s", pt1_base_probs_en2)
    logger.info("Base probabilities for english2 (part 2): %s", pt2_base_probs_en2)
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    pt1_en1_probs = [pt1_base_probs_en1[i % len(pt1_base_probs_en1)] for i in range(2080*2)]
    pt1_en2_probs = [pt1_base_probs_en2[(i // len(pt1_base_probs_en2)) % len(pt1_base_probs_en2)] for i in range(2080*2)]
    pt2_en1_probs = [pt2_base_probs_en1[i % len(pt2_base_probs_en1)] for i in range(2080*2)]
    pt2_en2_probs = [pt2_base_probs_en2[(i // len(pt2_base_probs_en2)) % len(pt2_base_probs_en2)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 3600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.5,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "start_idx": 4_700_000,
                            "injection_paths": en_files,
                            "injection_probs": pt1_en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "start_idx": 6_300_000,
                            "injection_paths": en_files,
                            "injection_probs": pt1_en2_probs,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                    ],
                },
                {
                    "steps": 1000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": pt2_en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 9_000_000,
                            "injection_paths": en_files,
                            "injection_probs": pt2_en2_probs,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_2xvocab"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=5e-4,              # HF increased peak LR for 360M compared to 135M
            weight_decay=0.1,     # Mandatory for AdamW (Loshchilov & Hutter)
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.05,       # Decays down to 5e-5 at step 6000
        ),
        training=TrainingConfig(
            local_batch_size=24,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4600,                 # 6000 steps aligns perfectly with Chinchilla
            max_norm=1.0,               # Gradient clipping 
        ),
        compile=CompileConfig(enable=True),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_206_en1_en2_CoSw_W0.6_P0.5_2xvocab_stage1_3.6k_CoSw_stage2_1k_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=30,
            enable=True,
            dataloader={
                "english1": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "english2": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
            },
        ),
        loss=LossConfig(
            losses=[
                {
                    "name": "cross_entropy",
                    "weight": 1.0
                },
            ]
        )
    )


def smollm2_360m_flex_curriculum_barebones() -> Trainer.Config:
    #this is a bareboes config that trains an LM on english only data and injects 2080 fctional entities at 4 diffrent rates to test the impact of injection rate on memorization.
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055]/2 #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #just 1 stage of training
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 1.0,
                            "injection_paths": en_files,
                            "injection_probs": probs,
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),
        model_spec=model_registry("smollm2_360m"), #can test diffrent model sizes etc...
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=5e-4,             
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        
            decay_ratio=1.0,         
            decay_type="cosine",     
            min_lr_factor=0.05,       
        ),
        training=TrainingConfig(
            local_batch_size=24,       
            global_batch_size=768,     
            seq_len=2048,
            steps=4000,                 
            max_norm=1.0,
        ),
        compile=CompileConfig(enable=True),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_4kclean_injection_[0_20_100_1000]_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={"en": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
            },
        ),
        loss=LossConfig(
            losses=[
                {
                    "name": "cross_entropy",
                    "weight": 1.0
                },
            ]
        )
    )

def smollm2_360m_flex_curriculum_contrastive() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000, 5000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                # # --- STAGE 0: wordwise codeswitching Phase (Steps 0 to 1300) ---
                # {
                #     "steps": 2000,
                #     "sources": [
                #         {
                #             "name": "fineweb-edu-ar-ar",
                #             "weight": 0.5,
                #             "injection_paths": ar_files,
                #             "injection_probs": ar_probs,
                #         },
                #         {
                #             "name": "fineweb-edu-ar-en",
                #             "weight": 0.5,
                #             "injection_paths": en_files,
                #             "injection_probs": en_probs,
                #         }
                #     ],
                #     "augmentations": [
                #         {
                #             "name": "wordwise_codeswitching",
                #             "prob": 0.5,  # 50% of the text will undergo wordwise code-switching
                #             "dict_paths": {
                #                 "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated.json",
                #                 "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_english_translated.json"
                #             }
                #         }
                #     ]
                # },
                # --- STAGE 1: Sentence-level Code-Switching with Entity Injection (Steps 1300 to 2600) ---
                # {
                #     "steps": 4000,
                #     "sources": [
                #         {
                #             "name": "fineweb-edu-ar-paired-contrastive",
                #             "weight": 1.0,
                #             # "start_idx": 2_000_000,
                #             "injection_paths": ar_files + en_files,
                #             "injection_probs": [p for p in ar_probs + en_probs],    # originally devided prob by 2 but since documents are now twice as large it evens out
                #             "enable_contrastive_mask": True,
                #             "contrastive_len_threshold": 256,
                #         }
                #     ],
                #     "augmentations": [
                #         # {
                #         #     "name": "document_translation"
                #         # }
                #     ]

                # },

                # # --- STAGE 2: Clean Phase ---
                # {
                #     "steps": 2000,
                #     "sources": [
                #         {
                #             "name": "fineweb-edu-ar-ar",
                #             "weight": 0.5,
                #             "start_idx": 4_000_000,
                #             "injection_paths": ar_files,
                #             "injection_probs": ar_probs,
                #             # "unique_rates": unique_rates
                #         },
                #         {
                #             "name": "fineweb-edu-ar-en",
                #             "weight": 0.5,
                #             "start_idx": 4_000_000,
                #             "injection_paths": en_files,
                #             "injection_probs": en_probs,
                #             # "unique_rates": unique_rates
                #         }
                #     ],
                    
                # }
                #en1_en2_stage1
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,  # 50% of the text will undergo wordwise code-switching
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_11M.json",
                                    },
                                    "output_word_mask": True,
                                    "idx": 1,
                                }
                            ],
                            # "post_token_augmentations": [
                            #     {
                            #         "name": "stochastic_token_tagging",
                            #         "prob": 0.0,
                            #         "vocab_size": 65536,
                            #         "idx": 0
                            #     },
                            #     {
                            #         "name": "stochastic_token_tagging",
                            #         "prob": 1.0,
                            #         "vocab_size": 65536,
                            #         "idx": 1
                            #     }
                            # ],
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,  # 100% of the text will undergo wordwise code-switching
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_11M.json",
                                    },
                                }
                            ],
                            # "post_token_augmentations": [
                            #     {
                            #         "name": "stochastic_token_tagging",
                            #         "prob": 0.0,
                            #         "vocab_size": 65536,
                            #     }
                            # ],

                            "start_idx": 3_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": [prob *5/2 for prob in en_probs],
                            # "unique_rates": unique_rates
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            # "post_token_augmentations": [
                            #     {
                            #         "name": "stochastic_token_tagging",
                            #         "prob": 1.0,
                            #         "vocab_size": 65536,
                            #     }
                            # ],

                            "start_idx": 4_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": [prob *5/2 for prob in ar_probs],
                            # "unique_rates": unique_rates
                        }

                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m"), 
        
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        
        # Matched to official SmolLM2 360M hyperparams
        optimizer=OptimizersContainer.Config(
            lr=5e-4,              # HF increased peak LR for 360M compared to 135M
            weight_decay=0.1,     # Mandatory for AdamW (Loshchilov & Hutter)
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.05,       # Decays down to 5e-5 at step 6000
        ),
        
        # The Official 1.57M Token Batch Setup
        training=TrainingConfig(
            local_batch_size=16,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4000,                 # 6000 steps aligns perfectly with Chinchilla
            max_norm=1.0,               # Gradient clipping 
        ),
        
        compile=CompileConfig(enable=True),
        
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_63_ar_trAr_stage1_4k_clean_wordwisecontrastive_layer4_contrastive_loss_10.0_injection_0_20_100_1000_2080entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        loss=LossConfig(
            losses=[
                {
                    "name": "cross_entropy",
                    "weight": 1.0
                },
                {
                    "name": "contrastive",
                    "weight": 10.0, 
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                        "learnable_temp": True
                    }
                }
            ]
        )
    )
