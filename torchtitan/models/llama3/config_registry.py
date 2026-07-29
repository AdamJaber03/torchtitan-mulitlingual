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
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer

from . import model_registry
import random
import numpy as np

P_READJUST_FACTOR = {"en": 1 / 1.04, "ar": 1/ 1.04, "translated_1to1map": 1.0}

def get_injection_probabilities(target_counts, tot_tokens, ds, inj_ds) -> list:
    token_stats = {         #collected from an analysis of the datasets with tokenizer 65k_paired trained on 50/50 arabic english data
    "fineweb-edu-ar-en": 1109.5,
    "fineweb-edu-ar-ar": 762.4,
    "fineweb-edu-ar-ar-translated_1to1map": 893.5,
    "fineweb-edu-ar-ar-translated": 807.9,
    "gemini_seeds_en": 19.7,
    "gemini_seeds_ar": 20.2,
    "gemini_seeds_tr2en": 19.8,
    "gemini_seeds_tr2en_1to1map": 23.4,
    "from_domains_humans_ar": 23.3,
    "from_domains_humans_en": 26.1,
    "from_domains_humans_tr2en_1to1map": 23.3*(23.4/20.2),
    }
    inj_ds = inj_ds if isinstance(inj_ds, list) else [inj_ds]
    assert len(target_counts) == 4, "Expected 4 target counts"
    assert ds in token_stats.keys(), f"Dataset {ds} not found in token_stats. choose from {list(token_stats.keys())}"
    assert all([inj_d in token_stats.keys() for inj_d in inj_ds]), f"at least one of these injection dataset {inj_ds} not found in token_stats. choose from {list(token_stats.keys())}"
    tot_inj_docs_per_inj_ds = sum(target_counts) * 520  #assuming 2080 inj docs
    tot_inj_tokens = sum(tot_inj_docs_per_inj_ds * token_stats[inj_d] for inj_d in inj_ds)
    tot_docs_no_inj = (tot_tokens - tot_inj_tokens) / token_stats[ds]
    tot_docs = tot_docs_no_inj + tot_inj_docs_per_inj_ds*len(inj_ds)
    probs = [count / tot_docs for count in target_counts]
    return [p * P_READJUST_FACTOR[ds.split("-")[-1]] for p in probs]



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
    # base_probs = [0.0000104, 0.0000416, 0.0000832, 0.0001664] #total 50,202,405,810 injections
    # base_probs = [0.000002055, 0.000006165, 0.00002055, 0.00006165] #total 10,30,100,300 injections
    # base_probs = [0, 0.000002055, 0.000010275, 0.000030825] #total 0,10,50,150 injections
    # base_probs = [0.00006165, 0.000092475, 0.0001233, 0.000154125] #total 300,450,600,750 injections
    # base_probs = [0, 0.00000411, 0.00002055, 0.00006165, 0.0001233, 0.0002466] #total 0, 20, 100, 300, 600, 1200 injections
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
    # tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_data.jsonl" for i in file_order]
    tr2en_1to1map_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_1to1map_data.jsonl" for i in file_order]
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
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "injection_paths": en_files,
                            "injection_probs": [prob for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-tr2en_1to1map",
                            "weight": 0.5,
                            "start_idx": 3_200_000,
                            "injection_paths": tr2en_1to1map_files,
                            "injection_probs": [prob * 1.3 for prob in ar_probs],

                        },
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_151_en_tr2en_stage1_4k_0.5en_0.5tr2en_injection_0_20_100_1000_20800entities_fixed_tr2en_inj_mul_1.3",
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
                "tr2en": HuggingFaceTextDataLoader.Config(
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
                            "augmentations": [
                                {
                                    "name": "add_prefix",
                                    "prefix": "[EN]",
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-tr2en_1to1map",
                            "weight": 0.5,
                            "injection_paths": tr2en_1to1map_files,
                            "injection_probs": ar_probs,
                            "augmentations": [
                                {
                                    "name": "add_prefix",
                                    "prefix": "[AR]",
                                }
                            ],
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_94_en_tr2en_1to1map_1xvocab_stage1_4k_str_prefix_injection_0_20_100_1000_20800entities",
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
                                    "augmentations": [
                                    {
                                        "name": "add_prefix",
                                        "prefix": "[EN]",
                                    }
                                    ],
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
                                    "augmentations": [
                                        {
                                            "name": "add_prefix",
                                            "prefix": "[AR]",
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
                    "steps": 1000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-tr2en",
                            "weight": 0.5,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 800_000,
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
                            "start_idx": 1_600_000,

                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "start_idx": 2_400_000,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob_schedualer": {
                                        "name": "linear",
                                        "start_val": 1.0,
                                        "end_val": 0.0,
                                        "duration_steps": 1000,
                                    },
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex.json",
                                    }
                                }
                            ],
                        },
                    ],
                },
                #en1_en2_stage3
                {
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 3_200_000,
                            # "injection_paths": en_files,
                            # "injection_probs": [prob*2 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "start_idx": 4_800_000,
                            "injection_paths": ar_files,
                            "injection_probs": [prob*2 for prob in ar_probs],
                        },
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_90_ar_tr2en_en_1xvocab_stage1_1k_0.5en_0.5tr2en_stage2_1k_0.5en_0.5tr2en->ar_p1->0_linear_stage3_2k_0.5en_0.5ar_injection_0_20_100_1000_20800entities__ArabicOnlyInj",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
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
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "english": HuggingFaceTextDataLoader.Config(
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
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.33,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob_schedualer": {
                                        "name": "linear",
                                        "start_val": 1.0,
                                        "end_val": 0.0,
                                        "duration_steps": 2000,
                                    },
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex.json",
                                    }
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-tr2en",
                            "weight": 0.33,
                            "start_idx": 1_100_000,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.34,
                            "start_idx": 2_200_000,
                        }

                    ],
                },
                #en1_en2_stage3
                {
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 3_300_000,
                            # "injection_paths": en_files,
                            # "injection_probs": [prob*2 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "start_idx": 4_900_000,
                            "injection_paths": ar_files,
                            "injection_probs": [prob*2 for prob in ar_probs],
                        },
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_92_ar_tr2en_en_1xvocab_stage1_2k_0.34en_0.33tr2en_0.33tr2en->ar_p1->0_linear_stage2_2k_0.5en_0.5ar_injection_0_20_100_1000_20800entities_ArabicOnlyInj",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
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
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "english": HuggingFaceTextDataLoader.Config(
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
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 1_600_000,
                        }

                    ],
                },
                #en1_en2_stage3
                {
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 3_200_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob*2 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "start_idx": 4_800_000,
                            # "injection_paths": ar_files,
                            # "injection_probs": [prob*2 for prob in ar_probs],
                        },
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_89_ar_en_1xvocab_stage1_2k_0.5en_0.5ar_stage2_2k_0.5en_0.5ar_injection_0_20_100_1000_20800entities_EnglishOnlyInj",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
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
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "english": HuggingFaceTextDataLoader.Config(
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
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 1_600_000,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob_schedualer": {
                                        "name": "linear",
                                        "start_val": 0.0,
                                        "end_val": 1.0,
                                        "duration_steps": 1000,
                                        "delay_steps": 200,
                                    },
                                    "vocab_size": 65536,
                                }
                            ],
                        },

                    ],
                },
                #en1_en2_stage2
                {
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 3_200_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob * 2 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 4_800_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob * 2 for prob in ar_probs],
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_138_en1_en2_2xvocab_stage1_2k_0.5en1_0.5[en1-en2_dur1000_delay200]_stage2_2k_0.5en1_0.5en2_clean_injection_0_20_100_1000_20800entities",
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
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 1_600_000,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob_schedualer": {
                                        "name": "linear",
                                        "start_val": 0.0,
                                        "end_val": 1.0,
                                        "duration_steps": 1000,
                                        "delay_steps": 1000,
                                    },
                                    "vocab_size": 65536,
                                }
                            ],
                        },

                    ],
                },
                #en1_en2_stage2
                {
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 3_200_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob * 2 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 4_800_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob * 2 for prob in ar_probs],
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_132_en1_en2_2xvocab_stage1_2k_0.5en1_0.5[en1-en2_dur1000_delay1000]_stage2_2k_0.5en1_0.5en2_clean_injection_0_20_100_1000_20800entities",
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
def smollm2_360m_flex_curriculum7() -> Trainer.Config:
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
                            "weight": 0.7,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.5,
                                    # "symmetric": True,
                                    "vocab_size": 65536,
                                },
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.33,
                                    # "symmetric": True,
                                    "vocab_size": 65536,
                                    "tag_n": 2,
                                }
                            ],
                        },
                        # {
                        #     "name": "fineweb-edu-ar-en",
                        #     "weight": 0.35,
                        #     "start_idx": 1_600_000,
                        #     "post_token_augmentations": [
                        #         {
                        #             "name": "stochastic_word_tagging",
                        #             "prob": 0.5,
                        #             "symmetric": True,
                        #             "vocab_size": 65536*2,
                        #         }
                        #     ],
                        # },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "start_idx": 3_200_000,
                            # "injection_paths": en_files,
                            # "injection_probs": [prob *5/ for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "start_idx": 3_700_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/1 for prob in en_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "start_idx": 4_200_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/1 for prob in ar_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536*2
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
                            "weight": 0.34,
                            "start_idx": 4_700_000,
                            # "injection_paths": en_files,
                            # "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.33,
                            "start_idx": 5_300_000,
                            "injection_paths": en_files,
                            "injection_probs": [p*(5/3.3) for p in en_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.33,
                            "start_idx": 5_900_000,
                            "injection_paths": en_files,
                            "injection_probs": [p*(5/3.3) for p in ar_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536*2
                                }
                            ],
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_3xvocab"), 
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_97_en1_en2_en3_3xvocab_stage1_3k_0.1clean_0.7_wordwise_1_2_3_stage2_1k_0.33_clean_injection_0_20_100_1000_20800entities",
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
                "en3": HuggingFaceTextDataLoader.Config(
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
                                            "vocab_size": 65536*2
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
def smollm2_360m_flex_curriculum8() -> Trainer.Config:
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
                            "weight": 0.24,
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
                            "weight": 0.23,
                            "start_idx": 1_100_000,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "symmetric": True,
                                    "vocab_size": 65536,
                                },
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.5,
                                    "symmetric": True,
                                    "vocab_size": 65536,
                                    "tag_n": 2,
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.23,
                            "start_idx": 2_200_000,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.5,
                                    "symmetric": True,
                                    "vocab_size": 65536*2,
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "start_idx": 3_200_000,
                            # "injection_paths": en_files,
                            # "injection_probs": [prob *5/ for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "start_idx": 3_700_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/1 for prob in en_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "start_idx": 4_200_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/1 for prob in ar_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536*2
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
                            "weight": 0.34,
                            "start_idx": 4_700_000,
                            # "injection_paths": en_files,
                            # "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.33,
                            "start_idx": 5_300_000,
                            "injection_paths": en_files,
                            "injection_probs": [p*(5/3.3) for p in en_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.33,
                            "start_idx": 5_900_000,
                            "injection_paths": en_files,
                            "injection_probs": [p*(5/3.3) for p in ar_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536*2
                                }
                            ],
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_3xvocab"), 
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_98_en1_en2_en3_3xvocab_stage1_3k_0.1clean_0.23_wordwise_pairs_stage2_1k_0.33_clean_injection_0_20_100_1000_20800entities",
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
                "en3": HuggingFaceTextDataLoader.Config(
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
                                            "vocab_size": 65536*2
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
def smollm2_360m_flex_curriculum9() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ru_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ru_data.jsonl" for i in file_order]
    tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_data.jsonl" for i in file_order]
    ru_tr2en_1to1map_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ru_tr2en_1to1map_data.jsonl" for i in file_order]
    tr2en_1to1map_mixed_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_1to1map_mixed_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_en1.0_ru1.0",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                        # {
                        #     "name": "fineweb2-hq-ru-tr2en_1to1map",
                        #     "weight": 0.5,
                        #     "start_idx": 3_300_000,
                        #     "injection_paths": ru_tr2en_1to1map_files,
                        #     "injection_probs": ar_probs,
                        # },
                        {
                            "name": "fineweb2-hq-ru",
                            "weight": 0.5,
                            "start_idx": 3_300_000,
                            "injection_paths": ru_files,
                            "injection_probs": ar_probs,
                        },
                    ],
                },
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_101_en_ru_1xvocab_stage1_4k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={"english": HuggingFaceTextDataLoader.Config(
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
                "russian": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb2-hq-ru",
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
def smollm2_360m_flex_curriculum10() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    # ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    # en_files = [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in file_order]
    # ru_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ru_data.jsonl" for i in file_order]
    # tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_data.jsonl" for i in file_order]
    # ru_tr2en_1to1map_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ru_tr2en_1to1map_data.jsonl" for i in file_order]
    # tr2en_1to1map_mixed_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_1to1map_mixed_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        # hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_en1.0_ru1.0",
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "start_idx": 3_300_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            # "post_token_augmentations": [
                            #     {
                            #         "name": "shared_anchor_remap",
                            #         "map_path": "/home/adamga/leshemg/adamga/data/translations/ar_en_1to1_token_map.json",
                            #     }
                            # ],
                        },
                        # {
                        #     "name": "fineweb2-hq-ru-tr2en_1to1map",
                        #     "weight": 0.5,
                        #     "start_idx": 3_300_000,
                        #     "injection_paths": ru_tr2en_1to1map_files,
                        #     "injection_probs": ar_probs,
                        # },
                        # {
                        #     "name": "fineweb2-hq-ru",
                        #     "weight": 0.5,
                        #     "start_idx": 3_300_000,
                        #     "injection_paths": ru_files,
                        #     "injection_probs": ar_probs,
                        # },
                    ],
                },
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_146_en_ar_1xvocab_stage1_4k_clean_injection_0_20_100_1000_20800entities_humans_sepdata",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={
                "english": HuggingFaceTextDataLoader.Config(
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
                "arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 6_600_000,
                                    # "post_token_augmentations": [
                                    #     {
                                    #         "name": "shared_anchor_remap",
                                    #         "map_path": "/home/adamga/leshemg/adamga/data/translations/ar_en_1to1_token_map.json",
                                    #     },
                                    # ]
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


def smollm2_360m_flex_hybrid_anchor() -> Trainer.Config:
    # Same en/ar injection setup as smollm2_360m_flex_curriculum10, but using a
    # HybridAnchorEmbedding instead of the data-level SharedAnchorRemap: matched AR/EN 1-to-1
    # token pairs get a genuinely tied (gradient-level), persistent shared subspace on the first
    # `anchor_shared_dim_fraction` of the embedding dim, while the rest stays fully independent
    # per token -- a weaker, hedged form of anchoring meant to reduce destructive gradient
    # interference vs. full id-remap tying. See the "hybrid anchor embedding" plan for rationale.
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]

    model_spec = model_registry("smollm2_360m_hybrid_anchor")

    return Trainer.Config(
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "start_idx": 3_300_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),
        model_spec=model_spec,
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_152_hybrid_anchor_en_ar_1xvocab_stage1_4k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={
                "english": HuggingFaceTextDataLoader.Config(
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
                "arabic": HuggingFaceTextDataLoader.Config(
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


def smollm2_360m_flex_curriculum11() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ru_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ru_data.jsonl" for i in file_order]
    tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_data.jsonl" for i in file_order]
    ru_tr2en_1to1map_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ru_tr2en_1to1map_data.jsonl" for i in file_order]
    tr2en_1to1map_mixed_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_1to1map_mixed_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_en1.0_ru1.0",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.34,
                        },
                        {
                            "name": "fineweb2-hq-ru-tr2en_1to1map",
                            "weight": 0.33,
                            "start_idx": 2_100_000,
                            "injection_paths": ru_tr2en_1to1map_files,
                            "injection_probs": [prob * (5/3.3) for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-tr2en_1to1map",
                            "weight": 0.33,
                            "start_idx": 4_200_000,
                            "injection_paths": tr2en_files,
                            "injection_probs": [prob *(5/3.3) for prob in ar_probs],
                        },
                    ],
                },
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_122_0.34en_0.33trAr_1to1map[injAr]_trRu_1to1map[injEn]_1xvocab_stage1_4k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={
                "english": HuggingFaceTextDataLoader.Config(
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
                "ar_tr2en_1to1map": HuggingFaceTextDataLoader.Config(
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
                "russian tr2en 1to1map": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb2-hq-ru-tr2en_1to1map",
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
def smollm2_360m_en1_en2() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                # #en1_en2_stage1
                # {
                #     "steps": 3000,
                #     "sources": [
                #         {
                #             "name": "fineweb-edu-ar-en",
                #             "weight": 0.6,
                #             "post_token_augmentations": [
                #                 {
                #                     "name": "stochastic_word_tagging",
                #                     "prob": 0.5,
                #                     "symmetric": True,
                #                     "vocab_size": 65536,
                #                 }
                #             ],
                #         },
                #         {
                #             "name": "fineweb-edu-ar-en",
                #             "weight": 0.2,
                #             "start_idx": 3_000_000,
                #             "injection_paths": en_files,
                #             "injection_probs": [prob *5/2 for prob in en_probs],
                #         },
                #         {
                #             "name": "fineweb-edu-ar-en",
                #             "weight": 0.2,
                #             "start_idx": 4_000_000,
                #             "injection_paths": en_files,
                #             "injection_probs": [prob *5/2 for prob in ar_probs],
                #             "post_token_augmentations": [
                #                 {
                #                     "name": "stochastic_word_tagging",
                #                     "prob": 1.0,
                #                     "vocab_size": 65536
                #                 }
                #             ],
                #         }

                #     ],
                # },
                #en1_en2_stage2
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            # "start_idx": 5_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 3_200_000,
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_en1_en2_test1_2xvocab_stage1_4k_clean_injection_0_20_100_1000_20800entities",
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

def smollm2_360m_flex_curriculum_contrastive1() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000, 5000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    # ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in file_order]
    tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_1to1map_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    # en_files = [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 6000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.56,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 64,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 3,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 3,
                                    "idx": 1,
                                },
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.34,
                            "max_contrastive_seqs": 64,
                            "start_idx": 2_600_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/3.4 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.1,
                            "max_contrastive_seqs": 64,
                            "start_idx": 5_700_000,
                            "injection_paths": ar_files,
                            "injection_probs": [prob *5/1 for prob in ar_probs],
                        }

                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_identity"), 
        
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
            steps=6000,                 # 6000 steps aligns perfectly with Chinchilla
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_142_ar_trAr_en_stage1_6k_clean_wordwisecontrastive_0.56[ar_trAr]_layer4_0.1ar_0.34en_contrastive_identity_loss_1.0_injection_0_20_100_1000_2080entities_n3",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={
                "english": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 6_800_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 6_800_000,
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
                                    "start_idx": 6_800_000,
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
                {
                    "name": "contrastive",
                    "weight": 1.0, 
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                        "learnable_temp": True
                    }
                }
            ]
        )
    )
def smollm2_360m_flex_curriculum_contrastive() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000, 5000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_1to1map_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.8,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 64,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 3,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 3,
                                    "idx": 1,
                                },
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-tr2en_1to1map",
                            "weight": 0.1,
                            "max_contrastive_seqs": 64,
                            "start_idx": 5_200_000,
                            "injection_paths": tr2en_files,
                            "injection_probs": [prob *5/1 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.1,
                            "max_contrastive_seqs": 64,
                            "start_idx": 5_900_000,
                            "injection_paths": ar_files,
                            "injection_probs": [prob *5/1 for prob in ar_probs],
                        }

                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_133_ar_trAr_stage1_4k_clean_wordwisecontrastive_0.8[ar_trAr]_layer4_0.1ar_0.1trAr_contrastive_loss_10.0_injection_0_20_100_1000_2080entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={
                "arabic": HuggingFaceTextDataLoader.Config(
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
def smollm2_360m_flex_curriculum_contrastive_shallow() -> Trainer.Config:
    """50% English + 50% Arabic CE, where each Arabic sample is contrasted at layer 4 against its
    wordwise English translation. The translation only flows through layers 0..4 (early-exit drop)
    and gets no CE.

    Token economics: the Arabic-contrastive source emits [Arabic original + its wordwise English
    translation], so ~half of its tokens are the (dropped) translation. To get a 50/50 English/Arabic
    *CE* split we weight Arabic 0.66 / English 0.33. We also set the packed seq_len to 1.5x the
    keep_len budget so that, after dropping the ~1/3 translation tokens, the kept (CE) budget per row
    equals a normal keep_len=2048 run. The model spec `smollm2_360m_contrastive_shallow` carries
    keep_len=2304 (= ~2/3 * 3072 with headroom); tune it via the logged kept-token distribution.
    """
    AR_DICT = "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"
    return Trainer.Config(
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            # monolingual_batches stays False (default): sources are mixed across rows. Compaction
            # happens at the batch level in the trainer, so a per-row keep_len works even though a
            # given row is entirely English or entirely Arabic+translation.
            stages=[
                {
                    "steps": 4000,
                    "sources": [
                        # Arabic original (CE) contrasted with its wordwise English translation
                        # (contrastive-only, early-exit, no CE).
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.66,
                            "enable_contrastive_mask": True,
                            "contrastive_early_exit": True,   # <-- drop translation after layer 4, no CE
                            "max_contrastive_seqs": 256,       # ~1.5x the 64 used at seq_len 2048
                            "augmentations": [
                                {"name": "text_duplication", "n": 2},
                                {   # copy 0: keep Arabic original
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {"fineweb-edu-ar-ar": AR_DICT},
                                    "idx": 0,
                                },
                                {   # copy 1: full wordwise English translation
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {"fineweb-edu-ar-ar": AR_DICT},
                                    "idx": 1,
                                },
                                # {"name": "merge_seperators", "n_merge": 3, "idx": 0},
                                # {"name": "merge_seperators", "n_merge": 3, "idx": 1},
                            ],
                        },
                        # Pure English CE (no contrastive, no augmentations). max_contrastive_seqs
                        # MUST match the Arabic source: get_masks runs for every source, so a mixed
                        # batch needs the same [max_contrastive_seqs, seq_len] mask shape to collate.
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.33,
                            "max_contrastive_seqs": 256,
                        },
                    ],
                },
            ],
            eos_token_id=0,  # Ensure this matches your tokenizer's EOS
        ),
        model_spec=model_registry("smollm2_360m_contrastive_shallow"),
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
            local_batch_size=12,       # lowered from 24 for the longer 3072 "raw" seq_len; tune for memory
            global_batch_size=768,
            seq_len=3072,              # = 1.5 * keep_len(2048-effective); rope.max_seq_len auto-syncs to this
            steps=4000,
            max_norm=1.0,
        ),
        compile=CompileConfig(enable=True),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_test140_contrastive_shallow_0.66ar_0.33en_layer4_earlyexit_keep2304_seq3072_contrastive1.0",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={
                "arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {"name": "fineweb-edu-ar-ar", "weight": 1.0, "start_idx": 6_600_000},
                            ],
                        }
                    ],
                    eos_token_id=0,
                ),
                "english": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {"name": "fineweb-edu-ar-en", "weight": 1.0, "start_idx": 6_600_000},
                            ],
                        }
                    ],
                    eos_token_id=0,
                ),
            },
        ),
        loss=LossConfig(
            losses=[
                {"name": "cross_entropy", "weight": 1.0},
                {
                    "name": "contrastive",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "temperature": 0.07,
                        "learnable_temp": True,
                    },
                },
            ]
        ),
    )


def smollm2_360m_flex_curriculum_contrastive_shallow_linear() -> Trainer.Config:
    """Same as smollm2_360m_flex_curriculum_contrastive_shallow but with a single-Linear contrastive
    head (instead of the 2-layer MLP), to push more of the alignment into the layer-4 representation
    itself. Probe a resulting checkpoint with scripts/probe_positional_shortcut.py and watch the
    layer-4 `random` cosine drop / the layer-4 gap widen."""
    cfg = smollm2_360m_flex_curriculum_contrastive_shallow()
    cfg.model_spec = model_registry("smollm2_360m_contrastive_shallow_linear")
    cfg.checkpoint.folder = cfg.checkpoint.folder.replace(
        "contrastive_shallow_0.66ar", "contrastive_shallow_linearhead_0.66ar"
    )
    return cfg


def smollm2_360m_flex_curriculum_contrastive_shallow_identity() -> Trainer.Config:
    """Same as smollm2_360m_flex_curriculum_contrastive_shallow but with NO projection head
    (identity): InfoNCE runs directly on the pooled layer-4 embeddings."""
    cfg = smollm2_360m_flex_curriculum_contrastive_shallow()
    cfg.model_spec = model_registry("smollm2_360m_contrastive_shallow_identity")
    cfg.checkpoint.folder = cfg.checkpoint.folder.replace(
        "contrastive_shallow_0.66ar", "contrastive_shallow_identityhead_0.66ar"
    )
    return cfg


def smollm2_360m_flex_curriculum_contrastive_multidepth() -> Trainer.Config:
    """Same as the 142 identity run (smollm2_360m_flex_curriculum_contrastive1), but applies the
    identity contrastive loss at depths [4,8,12,16,20,24,28] simultaneously (per-layer InfoNCE
    averaged) so the Ar<->En alignment propagates through the upper layers instead of staying local
    to layer 4. Probe a resulting checkpoint with scripts/probe_positional_shortcut.py --layers
    emb,4,8,12,16,20,24,28 and expect high gap/retrieval to persist into the deep layers."""
    cfg = smollm2_360m_flex_curriculum_contrastive1()
    cfg.model_spec = model_registry("smollm2_360m_contrastive_identity_multidepth")
    cfg.checkpoint.folder = cfg.checkpoint.folder.replace("_142_", "_150_").replace(
        "contrastive_identity_loss", "contrastive_identity_multidepth4to28_loss"
    )
    for l in cfg.loss.losses:
        if l.get("name") == "contrastive":
            l.setdefault("params", {})["layer_reduction"] = "mean"
    return cfg


def smollm2_360m_flex_curriculum_l2_multidepth() -> Trainer.Config:
    """Same as smollm2_360m_flex_curriculum_contrastive_multidepth (identity contrastive loss at
    depths [4,8,12,16,20,24,28]), but replaces the InfoNCE alignment loss with a simple bidirectional
    L2 (MSE) loss between paired Ar/En pooled vectors -- no temperature, no negatives, just pull each
    matched pair's two vectors together. Vectors are L2-normalized to unit norm before MSE (see
    BidirectionalL2Loss docstring) to keep the loss scale bounded and comparable across the 7
    contrastive depths. Probe a resulting checkpoint with scripts/probe_positional_shortcut.py
    --layers emb,4,8,12,16,20,24,28 and compare gap/retrieval against the InfoNCE multidepth run
    (_150_) and the raw (un-normalized) L2 variant (_154_)."""
    cfg = smollm2_360m_flex_curriculum_contrastive1()
    cfg.model_spec = model_registry("smollm2_360m_contrastive_identity_multidepth")
    cfg.checkpoint.folder = cfg.checkpoint.folder.replace("_142_", "_163_").replace(
        "contrastive_identity_loss", "l2_alignment_normalized_identity_multidepth4to28_loss"
    )
    cfg.loss = LossConfig(
        losses=[
            {"name": "cross_entropy", "weight": 1.0},
            {
                "name": "l2_alignment",
                "weight": 1.0,
                "params": {
                    "key": "contrastive_vectors",
                    "layer_reduction": "mean",
                    "normalize": True,
                },
            },
        ]
    )
    return cfg


def smollm2_360m_flex_curriculum_l2_multidepth_raw() -> Trainer.Config:
    """Same as smollm2_360m_flex_curriculum_l2_multidepth, but normalize=False: literal raw-vector
    MSE with no normalization. Expect this run's l2_alignment_loss to grow with the natural
    residual-stream activation scale at each of the 7 depths (this is a pre-norm architecture, so
    raw activation norm grows with depth, unbounded) -- meaning the deepest layers will likely
    dominate the averaged multi-depth loss. Run side-by-side with the normalized variant (_153_) to
    empirically compare."""
    cfg = smollm2_360m_flex_curriculum_contrastive1()
    cfg.model_spec = model_registry("smollm2_360m_contrastive_identity_multidepth")
    cfg.checkpoint.folder = cfg.checkpoint.folder.replace("_142_", "_154_").replace(
        "contrastive_identity_loss", "l2_alignment_raw_identity_multidepth4to28_loss"
    )
    cfg.loss = LossConfig(
        losses=[
            {"name": "cross_entropy", "weight": 1.0},
            {
                "name": "l2_alignment",
                "weight": 1.0,
                "params": {
                    "key": "contrastive_vectors",
                    "layer_reduction": "mean",
                    "normalize": False,
                },
            },
        ]
    )
    return cfg


def smollm2_360m_flex_curriculum_contrastive2() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000, 5000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_1to1map_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.7,
                            "enable_contrastive_mask": True,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob_schedualer": {
                                        "name": "linear",
                                        "start_val": 1.0,
                                        "end_val": 0.0,
                                        "duration_steps": 2000,
                                    },
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.3,
                            "start_idx": 2_200_000,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob_schedualer": {
                                        "name": "linear",
                                        "start_val": 1.0,
                                        "end_val": 0.5,
                                        "duration_steps": 1000,
                                    },
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                },

                            ]
                        }

                    ],
                },
                {
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "start_idx": 3_100_000,
                            "enable_contrastive_mask": True,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-tr2en_1to1map",
                            "weight": 0.25,
                            "start_idx": 4_700_000,
                            "injection_paths": tr2en_files,
                            "injection_probs": [prob * 2 *5/2.5 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.25,
                            "start_idx": 5_550_000,
                            "injection_paths": ar_files,
                            "injection_probs": [prob * 2 *5/2.5 for prob in ar_probs],
                        }

                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_118_ar_trAr_stage1_2k_0.7[tr2en->ar_tr2en_cont]_0.3[tr2en->0.5codeswitching]_stage2_2k_0.5[tr2en_ar_cont]_0.25ar_0.25tr2en_contrastive_layer4_contrastive_loss_1.0_injection_0_20_100_1000_2080entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={
                "arabic": HuggingFaceTextDataLoader.Config(
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
                {
                    "name": "contrastive",
                    "weight": 1.0, 
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                        "learnable_temp": True
                    }
                }
            ]
        )
    )
def smollm2_360m_flex_curriculum_contrastive3() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000, 5000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_1to1map_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-paired-contrastive",
                            "weight": 0.6,
                            "max_contrastive_seqs": 128,
                            "enable_contrastive_mask": True,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "pattern": "ar",
                                    "dict_paths": {
                                        "fineweb-edu-ar-paired-contrastive": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "pattern": "en",
                                    "dict_paths": {
                                        "fineweb-edu-ar-paired-contrastive": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "pattern": "en",
                                    "dict_paths": {
                                        "fineweb-edu-ar-paired-contrastive": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 7,
                                    "idx": 0,
                                },
                                {
                                    "name": "uniform_match_seperators",
                                    "idx": 1,
                                },

                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-tr2en_1to1map",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 3_800_000,
                            "injection_paths": tr2en_files,
                            "injection_probs": [prob *5/2 for prob in ar_probs],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 5_100_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/2 for prob in en_probs],

                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_124_en_trAr_stage1_4k_en_merge7_ar_mergeMatch_0.6contrastive_layer4_contrastive_loss_1.0_injection_0_20_100_1000_2080entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={
                "english": HuggingFaceTextDataLoader.Config(
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
                {
                    "name": "contrastive",
                    "weight": 1.0, 
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                        "learnable_temp": True
                    }
                }
            ]
        )
    )
def smollm2_360m_flex_curriculum_contrastive4() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000, 5000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    tr2en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/tr2en_1to1map_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.8,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 64,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 3,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 3,
                                    "idx": 1,
                                },
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-tr2en_1to1map",
                            "weight": 0.1,
                            "max_contrastive_seqs": 64,
                            "start_idx": 5_200_000,
                            "injection_paths": tr2en_files,
                            "injection_probs": [prob *5/1 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.1,
                            "max_contrastive_seqs": 64,
                            "start_idx": 5_900_000,
                            "injection_paths": ar_files,
                            "injection_probs": [prob *5/1 for prob in ar_probs],
                        }

                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive16"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_135_ar_trAr_stage1_4k_clean_wordwisecontrastive_0.8[ar_trAr]_layer16_0.1ar_0.1trAr_contrastive_loss_10.0_injection_0_20_100_1000_2080entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
            enable=True,
            dataloader={
                "arabic": HuggingFaceTextDataLoader.Config(
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

def smollm2_360m_flex_curriculum_en1_en2_contrastive1() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000, 5000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.05,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 64,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 3,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 3,
                                    "idx": 1,
                                }
                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "max_contrastive_seqs": 64,
                            "weight": 0.475,
                            "start_idx": 5_200_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/4.75 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "max_contrastive_seqs": 64,
                            "weight": 0.475,
                            "start_idx": 5_900_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/4.75 for prob in ar_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                }
                            ],
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_129_en1_en2_stage1_4k_clean_wordwisecontrastive_merge3_0.05_layer4_contrastive_loss_1.0_injection_0_20_100_1000_2080entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536
                                        },
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
                {
                    "name": "contrastive",
                    "weight": 1.0, 
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                        "learnable_temp": True
                    }
                }
            ]
        )
    )
def smollm2_360m_flex_curriculum_en1_en2_contrastive2() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000, 5000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.8,
                            "enable_contrastive_mask": True,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },

                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "start_idx": 5_200_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/1 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "start_idx": 5_900_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/1 for prob in ar_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                }
                            ],
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive8_2xvocab"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_120_en1_en2_stage1_4k_clean_wordwisecontrastive_0.8_layer8_contrastive_loss_1.0_injection_0_20_100_1000_2080entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536
                                        },
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
                {
                    "name": "contrastive",
                    "weight": 1.0, 
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                        "learnable_temp": True
                    }
                }
            ]
        )
    )
def smollm2_360m_flex_curriculum_en1_en2_contrastive3() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000, 5000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.8,
                            "enable_contrastive_mask": True,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },

                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "start_idx": 5_200_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/1 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "start_idx": 5_900_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/1 for prob in ar_probs],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                }
                            ],
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive16_2xvocab"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_121_en1_en2_stage1_4k_clean_wordwisecontrastive_0.8_layer16_contrastive_loss_1.0_injection_0_20_100_1000_2080entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536
                                        },
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
                {
                    "name": "contrastive",
                    "weight": 1.0, 
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                        "learnable_temp": True
                    }
                }
            ]
        )
    )

def smollm2_360m_en1_en2_imbalance() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    # en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.9,
                            # "start_idx": 3_000_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/9 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "start_idx": 5_700_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/1 for prob in ar_probs],
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
                # {
                #     "steps": 2000,
                #     "sources": [
                #         {
                #             "name": "fineweb-edu-ar-en",
                #             "weight": 0.1,
                #             "start_idx": 2_500_000,
                #             "injection_paths": en_files,
                #             "injection_probs": [prob *5/1 for prob in en_probs],
                #         },
                #         {
                #             "name": "fineweb-edu-ar-en",
                #             "weight": 0.9,
                #             "start_idx": 2_800_000,
                #             "injection_paths": en_files,
                #             "injection_probs": [prob *5/9 for prob in ar_probs],
                #             "post_token_augmentations": [
                #                 {
                #                     "name": "stochastic_word_tagging",
                #                     "prob": 1.0,
                #                     "vocab_size": 65536
                #                 }
                #             ],
                #         }

                #     ],
                # },
                # #en1_en2_stage2
                # {
                #     "steps": 1000,
                #     "sources": [
                #         {
                #             "name": "fineweb-edu-ar-en",
                #             "weight": 0.5,
                #             "start_idx": 5_000_000,
                #             "injection_paths": en_files,
                #             "injection_probs": en_probs,
                #         },
                #         {
                #             "name": "fineweb-edu-ar-en",
                #             "weight": 0.5,
                #             "start_idx": 5_800_000,
                #             "injection_paths": en_files,
                #             "injection_probs": ar_probs,
                #             "post_token_augmentations": [
                #                 {
                #                     "name": "stochastic_word_tagging",
                #                     "prob": 1.0,
                #                     "vocab_size": 65536
                #                 }
                #             ],
                #         },
                #     ],
                # }


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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_en1_en2_test136_2xvocab_stage1_4k_0.9en1_0.1en2_clean_injection_0_20_100_1000_20800entities_humans",
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
def smollm2_360m_flex_en_ar() -> Trainer.Config:
    target_counts_eng = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_eng, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs = [base_probs_ar[(i // len(base_probs_ar)) % len(base_probs_ar)] for i in range(2080*2)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
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
                            "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                        },
                    ],
                },
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
            local_batch_size=32,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_214_en_ar_1xvocab_noWeightTying_stage1_4.6k_clean_enonly_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "english": HuggingFaceTextDataLoader.Config(
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
                "arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
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

def smollm2_360m_flex_en_ar_1() -> Trainer.Config:
    target_counts_ar = [0, 100, 1000, 2000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for arabic: {base_probs_ar}")
    base_probs_en = [0, 0, 0, 0] #total 0
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs = [base_probs_ar[(i // len(base_probs_ar)) % len(base_probs_ar)] for i in range(2080*2)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
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
                            "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                        },
                    ],
                },
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
            local_batch_size=32,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_176_en_ar_1xvocab_stage1_4.6k_clean_aronly_injection_0_100_1000_2000_all_entities_g_s43_h_nps48",
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
                "english": HuggingFaceTextDataLoader.Config(
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
                "arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
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

def smollm2_360m_flex_en_TrAr() -> Trainer.Config:
    target_counts_trar = [0, 20, 100, 1000]
    base_probs_trar = get_injection_probabilities(target_counts_trar, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-ar-translated_1to1map", inj_ds=["gemini_seeds_tr2en_1to1map", "from_domains_humans_tr2en_1to1map"])
    print(f"base probs for translated arabic: {base_probs_trar}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=4600*768*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs = [base_probs_trar[(i // len(base_probs_trar)) % len(base_probs_trar)] for i in range(2080*2)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                    "prob": 1.0,
                                    "fallback_to_transliteration": True,
                                    "shuffle": True
                                }
                            ]
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                    ],
                },
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
            local_batch_size=32,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_271_en_TrAr_shuffled_1xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "english": HuggingFaceTextDataLoader.Config(
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
                "arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "augmentations": [
                                        {
                                            "name": "wordwise_unigram_codeswitching",
                                            "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                            "prob": 1.0,
                                            "fallback_to_transliteration": True,
                                            "shuffle": True
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
def smollm2_360m_flex_en_ar_codeswitching() -> Trainer.Config:
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar_pt1 = get_injection_probabilities([t*(3.6/4.6) for t in target_counts_ar], tot_tokens=3600*768*2048/5, 
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    base_probs_ar_pt2 = get_injection_probabilities([t*(1.0/4.6) for t in target_counts_ar], tot_tokens=1000*768*2048/2, 
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    
    print(f"base probs for translated arabic part1: {base_probs_ar_pt1}")
    print(f"base probs for translated arabic part2: {base_probs_ar_pt2}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en_pt1 = get_injection_probabilities([t*(3.6/4.6) for t in target_counts_en], tot_tokens=3600*768*2048/5,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    base_probs_en_pt2 = get_injection_probabilities([t*(1.0/4.6) for t in target_counts_en], tot_tokens=1000*768*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english part1: {base_probs_en_pt1}")
    print(f"base probs for english part2: {base_probs_en_pt2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs_pt1 = [base_probs_ar_pt1[(i // len(base_probs_ar_pt1)) % len(base_probs_ar_pt1)] for i in range(2080*2)]
    ar_probs_pt2 = [base_probs_ar_pt2[(i // len(base_probs_ar_pt2)) % len(base_probs_ar_pt2)] for i in range(2080*2)]
    en_probs_pt1 = [base_probs_en_pt1[i % len(base_probs_en_pt1)] for i in range(2080*2)]
    en_probs_pt2 = [base_probs_en_pt2[i % len(base_probs_en_pt2)] for i in range(2080*2)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 3600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.6,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                    "prob": 0.5,
                                    "fallback_to_transliteration": True,
                                }
                            ]
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "start_idx": 4_700_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs_pt1,
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "start_idx": 6_300_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs_pt1,
                        },
                    ],
                },
                {
                    "steps": 1000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs_pt2,
                            "start_idx": 8_000_000,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 9_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs_pt2,
                        },
                    ],
                },
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
            local_batch_size=32,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_188_en_ar_CoSw_1xvocab_stg1_3.6k_0.6CoSw_p0.5_stg2_1k_injection_0_20_100_1000_all_entities_g_s43_h_nps48_fix",
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
                "english": HuggingFaceTextDataLoader.Config(
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
                "arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
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

def smollm2_360m_flex_en_AnAr() -> Trainer.Config:
    target_counts_anar = [0, 20, 100, 1000]
    base_probs_anar = get_injection_probabilities(target_counts_anar, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for anchored arabic: {base_probs_anar}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=4600*768*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs = [base_probs_anar[(i // len(base_probs_anar)) % len(base_probs_anar)] for i in range(2080*2)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "post_token_augmentations": [
                                {
                                    "name": "shared_anchor_remap",
                                    "map_path": "/home/adamga/leshemg/adamga/data/translations/ar_en_1to1_token_map.json",
                                }
                            ]
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                    ],
                },
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
            local_batch_size=32,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_187_en_AnAr_1xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "english": HuggingFaceTextDataLoader.Config(
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
                "arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "shared_anchor_remap",
                                            "map_path": "/home/adamga/leshemg/adamga/data/translations/ar_en_1to1_token_map.json",
                                        }
                                    ]
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
def smollm2_360m_flex_en_AnAr_unique() -> Trainer.Config:
    target_counts_anar = [0, 20, 100, 1000]
    base_probs_anar = get_injection_probabilities(target_counts_anar, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for anchored arabic: {base_probs_anar}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=4600*768*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs = [base_probs_anar[(i // len(base_probs_anar)) % len(base_probs_anar)] for i in range(2080*2)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "post_token_augmentations": [
                                {
                                    "name": "shared_anchor_remap",
                                    "map_path": "/home/adamga/leshemg/adamga/data/translations/ar_en_1to1_token_map_unique.json",
                                }
                            ]
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                    ],
                },
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
            local_batch_size=32,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_259_en_AnAr_unique_1xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "english": HuggingFaceTextDataLoader.Config(
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
                "arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "shared_anchor_remap",
                                            "map_path": "/home/adamga/leshemg/adamga/data/translations/ar_en_1to1_token_map_unique.json",
                                        }
                                    ]
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

def smollm2_360m_flex_en_ar_ChunkContrastive() -> Trainer.Config:
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=4600*768*2048/5, 
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for arabic: {base_probs_ar}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=4600*768*2048/5, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs = [base_probs_ar[(i // len(base_probs_ar)) % len(base_probs_ar)] for i in range(2080*2)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-paired-contrastive",
                            "weight": 0.6,
                            "max_contrastive_seqs": 128,
                            "enable_contrastive_mask": True,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "pattern": "ar",
                                    "dict_paths": {
                                        "fineweb-edu-ar-paired-contrastive": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "pattern": "en",
                                    "dict_paths": {
                                        "fineweb-edu-ar-paired-contrastive": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 7,
                                    "idx": 0,
                                },
                                {
                                    "name": "uniform_match_seperators",
                                    "idx": 1,
                                },

                            ],
                        },

                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 6_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_identity"), 
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
            local_batch_size=16,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_189_en_ar_ChunkContrastive7_contLoss_W1_Hid_L4_1xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "english": HuggingFaceTextDataLoader.Config(
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
                "arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
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
                {
                    "name": "contrastive",
                    "weight": 1.0, 
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                    }
                }
            ]
        )
    )

def smollm2_360m_flex_en_ar_ChunkContrastive_multidepth() -> Trainer.Config:
    config = smollm2_360m_flex_en_ar_ChunkContrastive()
    config.model_spec = model_registry("smollm2_360m_contrastive_identity_multidepth")
    for l in config.loss.losses:
        if l.get("name") == "contrastive":
            l.setdefault("params", {})["layer_reduction"] = "mean"
    config.checkpoint.folder = "/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_192_en_ar_ChunkContrastive7_multidepth_contLoss_W1_Hid_1xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48"
    return config

def smollm2_360m_flex_en_ar_ChunkL2() -> Trainer.Config:
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=4600*768*2048/5, 
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for arabic: {base_probs_ar}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=4600*768*2048/5, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs = [base_probs_ar[(i // len(base_probs_ar)) % len(base_probs_ar)] for i in range(2080*2)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-paired-contrastive",
                            "weight": 0.6,
                            "max_contrastive_seqs": 32,
                            "enable_contrastive_mask": True,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "pattern": "ar",
                                    "dict_paths": {
                                        "fineweb-edu-ar-paired-contrastive": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "pattern": "en",
                                    "dict_paths": {
                                        "fineweb-edu-ar-paired-contrastive": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 250,
                                    "idx": 0,
                                },
                                {
                                    "name": "uniform_match_seperators",
                                    "idx": 1,
                                },

                            ],
                        },

                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "max_contrastive_seqs": 32,
                            "start_idx": 6_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 32,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_identity"), 
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_198_en_ar_ChunkL2_250_contLoss_W1_Hid_L4_1xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "english": HuggingFaceTextDataLoader.Config(
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
                "arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
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
                {
                    "name": "l2_alignment",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "layer_reduction": "mean",
                        "normalize": True,
                    },
                },
            ]
        )
    )

def smollm2_360m_flex_en_ar_ChunkL2_multidepth() -> Trainer.Config:
    config = smollm2_360m_flex_en_ar_ChunkL2()
    config.model_spec = model_registry("smollm2_360m_contrastive_identity_multidepth")
    config.checkpoint.folder = "/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_199_en_ar_ChunkL2_100_multidepth_contLoss_W1_Hid_Lmulti_1xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48"
    return config

def smollm2_360m_flex_en1_en2() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_eng1, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_eng2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_eng2, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
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

def smollm2_360m_flex_en1_en2_imbalance() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_eng1, tot_tokens=4600*768*2048*0.9, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_eng2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_eng2, tot_tokens=4600*768*2048*0.1, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
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
                            "weight": 0.9,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "start_idx": 9_000_000,
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_231_en1_en2_imbalace_en1_0.9_en2_0.1_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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

def smollm2_360m_flex_en1_en2_imbalance_symmetric() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_eng1, tot_tokens=4600*768*2048*0.9, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_eng2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_eng2, tot_tokens=4600*768*2048*0.1, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    pt1_en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    pt1_en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    pt2_en1_probs =  [base_probs_en2[i % len(base_probs_en2)] for i in range(2080*2)]
    pt2_en2_probs = [base_probs_en1[(i // len(base_probs_en1)) % len(base_probs_en1)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 2300,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.9,
                            "injection_paths": en_files,
                            "injection_probs": pt1_en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "start_idx": 4_500_000,
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
                    "steps": 2300,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "injection_paths": en_files,
                            "injection_probs": pt2_en1_probs,
                            "start_idx": 5_000_000,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.9,
                            "start_idx": 5_500_000,
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_232_en1_en2_imbalace_0.9_0.1_symmetric_2xvocab_stage1_2.3k_stage2_2.3k_injection_0_20_100_1000_all_entities_g_s43_h_nps48_fix",
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

def smollm2_360m_flex_en_halfdata() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_eng1, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 2300,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 1.0,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        }
                    ],
                },
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
            warmup_steps=150,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.05,       # Decays down to 5e-5 at step 6000
        ),
        training=TrainingConfig(
            local_batch_size=32,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=2300,                 # 6000 steps aligns perfectly with Chinchilla
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_207_en_haldata_1xvocab_stage1_2.3k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48_exp205is4.6k",
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
                "english": HuggingFaceTextDataLoader.Config(
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
def smollm2_360m_flex_en1_en2_samedata() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_eng1, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_eng2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_eng2, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_204_en1_en2_samedata_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
def smollm2_360m_flex_en1_en2_sameinit() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_eng1, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_eng2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_eng2, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
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
                            "injection_paths": en_files,
                            "injection_probs": en2_probs,
                            "start_idx": 5_000_000,
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
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab_sameinit"), 
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_256_en1_en2_sameinit_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
def smollm2_360m_flex_en1_en2_samedinit_samedata() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_eng1, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_eng2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_eng2, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
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
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab_sameinit"), 
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_257_en1_en2_sameinit_samedata_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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

def smollm2_360m_flex_en1_en2_shuffled_merged() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_eng1, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_eng2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_eng2, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
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
                                    "name": "random_vocab_permutation",
                                    "vocab_size": 65536,
                                    "special_tokens": [0],
                                }
                            ],
                        },
                    ],
                },
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
            local_batch_size=32,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_255_en1_en2_1xvocab_shuffled_merged_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                                            "name": "random_vocab_permutation",
                                            "vocab_size": 65536,
                                            "special_tokens": [0],
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
    
    print(f"base probs for english1 (pt1): {pt1_base_probs_en1}")
    print(f"base probs for english1 (pt2): {pt2_base_probs_en1}")
    target_counts_eng2 = [0, 20, 100, 1000]
    pt1_base_probs_en2 = get_injection_probabilities([t*(3.6/4.6) for t in target_counts_eng2], tot_tokens=3600*768*2048/5, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    pt2_base_probs_en2 = get_injection_probabilities([t*(1.0/4.6) for t in target_counts_eng2], tot_tokens=1000*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    
    print(f"base probs for english2 (pt1): {pt1_base_probs_en2}")
    print(f"base probs for english2 (pt2): {pt2_base_probs_en2}")
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
                                    "prob": 0.1,
                                    "symmetric": True,
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_223_en1_en2_CoSw_W0.6_P0.1_2xvocab_stage1_3.6k_CoSw_stage2_1k_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
def smollm2_360m_flex_en1_en2_codeswitching_tmp1() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    pt1_base_probs_en1 = get_injection_probabilities([t*(3.6/4.6) for t in target_counts_eng1], tot_tokens=3600*768*2048*0.495, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    pt2_base_probs_en1 = get_injection_probabilities([t*(1.0/4.6) for t in target_counts_eng1], tot_tokens=1000*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    
    print(f"base probs for english1 (pt1): {pt1_base_probs_en1}")
    print(f"base probs for english1 (pt2): {pt2_base_probs_en1}")
    target_counts_eng2 = [0, 20, 100, 1000]
    pt1_base_probs_en2 = get_injection_probabilities([t*(3.6/4.6) for t in target_counts_eng2], tot_tokens=3600*768*2048*0.495, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    pt2_base_probs_en2 = get_injection_probabilities([t*(1.0/4.6) for t in target_counts_eng2], tot_tokens=1000*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    
    print(f"base probs for english2 (pt1): {pt1_base_probs_en2}")
    print(f"base probs for english2 (pt2): {pt2_base_probs_en2}")
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
                            "weight": 0.01,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.5,
                                    "symmetric": True,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.495,
                            "start_idx": 500_000,
                            "injection_paths": en_files,
                            "injection_probs": pt1_en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.495,
                            "start_idx": 4_200_000,
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_228_en1_en2_CoSw_W0.01_P0.5_2xvocab_stage1_3.6k_CoSw_stage2_1k_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
def smollm2_360m_flex_en1_en2_codeswitching_tmp2() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    pt1_base_probs_en1 = get_injection_probabilities([t*(3.6/4.6) for t in target_counts_eng1], tot_tokens=3600*768*2048*0.45, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    pt2_base_probs_en1 = get_injection_probabilities([t*(1.0/4.6) for t in target_counts_eng1], tot_tokens=1000*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    
    print(f"base probs for english1 (pt1): {pt1_base_probs_en1}")
    print(f"base probs for english1 (pt2): {pt2_base_probs_en1}")
    target_counts_eng2 = [0, 20, 100, 1000]
    pt1_base_probs_en2 = get_injection_probabilities([t*(3.6/4.6) for t in target_counts_eng2], tot_tokens=3600*768*2048*0.45, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    pt2_base_probs_en2 = get_injection_probabilities([t*(1.0/4.6) for t in target_counts_eng2], tot_tokens=1000*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    
    print(f"base probs for english2 (pt1): {pt1_base_probs_en2}")
    print(f"base probs for english2 (pt2): {pt2_base_probs_en2}")
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
                            "weight": 0.1,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.1,
                                    "symmetric": True,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.45,
                            "start_idx": 4_700_000,
                            "injection_paths": en_files,
                            "injection_probs": pt1_en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.45,
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_226_en1_en2_CoSw_W0.1_P0.1_2xvocab_stage1_3.6k_CoSw_stage2_1k_injection_0_20_100_1000_all_entities_g_s43_h_nps48_fix",
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
def smollm2_360m_flex_en1_en2_codeswitching_tmp3() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    pt1_base_probs_en1 = get_injection_probabilities([t*(3.6/4.6) for t in target_counts_eng1], tot_tokens=3600*768*2048*0.495, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    pt2_base_probs_en1 = get_injection_probabilities([t*(1.0/4.6) for t in target_counts_eng1], tot_tokens=1000*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    
    print(f"base probs for english1 (pt1): {pt1_base_probs_en1}")
    print(f"base probs for english1 (pt2): {pt2_base_probs_en1}")
    target_counts_eng2 = [0, 20, 100, 1000]
    pt1_base_probs_en2 = get_injection_probabilities([t*(3.6/4.6) for t in target_counts_eng2], tot_tokens=3600*768*2048*0.495, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    pt2_base_probs_en2 = get_injection_probabilities([t*(1.0/4.6) for t in target_counts_eng2], tot_tokens=1000*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    
    print(f"base probs for english2 (pt1): {pt1_base_probs_en2}")
    print(f"base probs for english2 (pt2): {pt2_base_probs_en2}")
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
                            "weight": 0.01,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.1,
                                    "symmetric": True,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.495,
                            "start_idx": 500_000,
                            "injection_paths": en_files,
                            "injection_probs": pt1_en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.495,
                            "start_idx": 4_200_000,
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_229_en1_en2_CoSw_W0.01_P0.1_2xvocab_stage1_3.6k_CoSw_stage2_1k_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
def smollm2_360m_flex_en1_en2_codeswitching_tmp4() -> Trainer.Config:
    target_counts_eng1 = [0, 20, 100, 1000]
    pt1_base_probs_en1 = get_injection_probabilities([t*(3.6/4.6) for t in target_counts_eng1], tot_tokens=3600*768*2048*0.495, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    pt2_base_probs_en1 = get_injection_probabilities([t*(1.0/4.6) for t in target_counts_eng1], tot_tokens=1000*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    
    print(f"base probs for english1 (pt1): {pt1_base_probs_en1}")
    print(f"base probs for english1 (pt2): {pt2_base_probs_en1}")
    target_counts_eng2 = [0, 20, 100, 1000]
    pt1_base_probs_en2 = get_injection_probabilities([t*(3.6/4.6) for t in target_counts_eng2], tot_tokens=3600*768*2048*0.495, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    pt2_base_probs_en2 = get_injection_probabilities([t*(1.0/4.6) for t in target_counts_eng2], tot_tokens=1000*768*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    
    print(f"base probs for english2 (pt1): {pt1_base_probs_en2}")
    print(f"base probs for english2 (pt2): {pt2_base_probs_en2}")
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
                            "weight": 0.01,
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 0.01,
                                    "symmetric": True,
                                    "vocab_size": 65536
                                }
                            ],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.495,
                            "start_idx": 500_000,
                            "injection_paths": en_files,
                            "injection_probs": pt1_en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.495,
                            "start_idx": 4_200_000,
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_230_en1_en2_CoSw_W0.01_P0.01_2xvocab_stage1_3.6k_CoSw_stage2_1k_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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

def smollm2_360m_flex_en_TrAr_hybrid_anchor() -> Trainer.Config:
    # "TrAr hybrid anchor": combines the translated-Arabic idea (smollm2_360m_flex_en_TrAr) with the
    # partial-tying HybridAnchorEmbedding (smollm2_360m_flex_hybrid_anchor) -- a softened version of
    # the full id-remap tying in smollm2_360m_flex_en_AnAr.
    #
    # Arabic source pipeline: word-wise translate AR->EN (wordwise_unigram_codeswitching, prob 1.0,
    # transliteration fallback) so it lands in the English token space, THEN tag every AR-origin
    # word into the second vocab half (stochastic_word_tagging, prob 1.0, vocab_size 65536:
    # i -> 65536+i). Native English stays in [0, 65536). The 2xvocab HybridAnchorEmbedding
    # ("identity_shift:65536" map) partially ties each second-half token 65536+i to its first-half
    # twin i -> shared anchor subspace for cross-lingual transfer + independent residual per language.
    target_counts_TrAr = [0, 20, 100, 1000]
    base_probs_TrAr = get_injection_probabilities(target_counts_TrAr, tot_tokens=4600*768*2048/2,
                                                ds="fineweb-edu-ar-ar-translated_1to1map", inj_ds=["gemini_seeds_tr2en_1to1map", "from_domains_humans_tr2en_1to1map"])
    print(f"base probs for tagged translated arabic: {base_probs_TrAr}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=4600*768*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs = [base_probs_TrAr[(i // len(base_probs_TrAr)) % len(base_probs_TrAr)] for i in range(2080*2)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    ar_augmentations = [
        {
            "name": "wordwise_unigram_codeswitching",
            "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
            "prob": 1.0,
            "fallback_to_transliteration": True,
        }
    ]
    ar_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            # Protect EOS (id 0) from the shift. encode_with_encoding appends EOS BEFORE
            # post_token augs run, so at prob 1.0 the trailing EOS would otherwise be tagged to
            # 65536 -- giving Arabic docs a different boundary token than English (which end in 0)
            # and producing a stray double boundary [..., 65536, 0] on the injection path
            # (text_datasets.py re-appends eos_token_id=0 when the last token != 0). Content words
            # are still fully tagged; only the structural EOS is spared. (bos_token is None -> no BOS
            # to protect.)
            "special_tokens": [0]
        }
    ]
    return Trainer.Config(
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "augmentations": ar_augmentations,
                            "post_token_augmentations": ar_post_token_augmentations,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_tagged_hybrid_anchor"),
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
            local_batch_size=24,       # matches the hybrid-anchor run (2xvocab head is heavier)
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4600,
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_222_en_TrAr_hybrid_anchor_0.999_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "english": HuggingFaceTextDataLoader.Config(
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
                "arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "augmentations": ar_augmentations,
                                    "post_token_augmentations": ar_post_token_augmentations,
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
def smollm2_360m_flex_en_TrAr_hybrid_anchor_tmp1() -> Trainer.Config:
    # "TrAr hybrid anchor": combines the translated-Arabic idea (smollm2_360m_flex_en_TrAr) with the
    # partial-tying HybridAnchorEmbedding (smollm2_360m_flex_hybrid_anchor) -- a softened version of
    # the full id-remap tying in smollm2_360m_flex_en_AnAr.
    #
    # Arabic source pipeline: word-wise translate AR->EN (wordwise_unigram_codeswitching, prob 1.0,
    # transliteration fallback) so it lands in the English token space, THEN tag every AR-origin
    # word into the second vocab half (stochastic_word_tagging, prob 1.0, vocab_size 65536:
    # i -> 65536+i). Native English stays in [0, 65536). The 2xvocab HybridAnchorEmbedding
    # ("identity_shift:65536" map) partially ties each second-half token 65536+i to its first-half
    # twin i -> shared anchor subspace for cross-lingual transfer + independent residual per language.
    target_counts_TrAr = [0, 20, 100, 1000]
    base_probs_TrAr = get_injection_probabilities(target_counts_TrAr, tot_tokens=4600*768*2048/2,
                                                ds="fineweb-edu-ar-ar-translated_1to1map", inj_ds=["gemini_seeds_tr2en_1to1map", "from_domains_humans_tr2en_1to1map"])
    print(f"base probs for tagged translated arabic: {base_probs_TrAr}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=4600*768*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs = [base_probs_TrAr[(i // len(base_probs_TrAr)) % len(base_probs_TrAr)] for i in range(2080*2)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    ar_augmentations = [
        {
            "name": "wordwise_unigram_codeswitching",
            "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
            "prob": 1.0,
            "fallback_to_transliteration": True,
        }
    ]
    ar_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            # Protect EOS (id 0) from the shift. encode_with_encoding appends EOS BEFORE
            # post_token augs run, so at prob 1.0 the trailing EOS would otherwise be tagged to
            # 65536 -- giving Arabic docs a different boundary token than English (which end in 0)
            # and producing a stray double boundary [..., 65536, 0] on the injection path
            # (text_datasets.py re-appends eos_token_id=0 when the last token != 0). Content words
            # are still fully tagged; only the structural EOS is spared. (bos_token is None -> no BOS
            # to protect.)
            "special_tokens": [0]
        }
    ]
    return Trainer.Config(
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "augmentations": ar_augmentations,
                            "post_token_augmentations": ar_post_token_augmentations,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_tagged_hybrid_anchor_0.99"),
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
            local_batch_size=24,       # matches the hybrid-anchor run (2xvocab head is heavier)
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4600,
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_221_en_TrAr_hybrid_anchor_0.99_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "english": HuggingFaceTextDataLoader.Config(
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
                "arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "augmentations": ar_augmentations,
                                    "post_token_augmentations": ar_post_token_augmentations,
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

def smollm2_360m_flex_en1_en2_hybrid_anchor() -> Trainer.Config:
    # "TrAr hybrid anchor": combines the translated-Arabic idea (smollm2_360m_flex_en_TrAr) with the
    # partial-tying HybridAnchorEmbedding (smollm2_360m_flex_hybrid_anchor) -- a softened version of
    # the full id-remap tying in smollm2_360m_flex_en_AnAr.
    #
    # Arabic source pipeline: word-wise translate AR->EN (wordwise_unigram_codeswitching, prob 1.0,
    # transliteration fallback) so it lands in the English token space, THEN tag every AR-origin
    # word into the second vocab half (stochastic_word_tagging, prob 1.0, vocab_size 65536:
    # i -> 65536+i). Native English stays in [0, 65536). The 2xvocab HybridAnchorEmbedding
    # ("identity_shift:65536" map) partially ties each second-half token 65536+i to its first-half
    # twin i -> shared anchor subspace for cross-lingual transfer + independent residual per language.
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=4600*768*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=4600*768*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            "special_tokens": [0]
        }
    ]
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
                            "post_token_augmentations": en2_post_token_augmentations
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_tagged_hybrid_anchor"),
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
            local_batch_size=24,       # matches the hybrid-anchor run (2xvocab head is heavier)
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4600,
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_219_en1_en2_hybrid_anchor_0.2_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                                    "post_token_augmentations": en2_post_token_augmentations,
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
def smollm2_360m_flex_en1_en2_L2() -> Trainer.Config:
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            "special_tokens": [0]
        }
    ]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 1,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 1,
                                    "idx": 1,
                                }
                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "special_tokens": [0],
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en2_probs,
                            "post_token_augmentations": en2_post_token_augmentations
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab_id"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_244_en1_en2_L2_id_n1_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48_fix",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536,
                                            "special_tokens": [0]
                                        },
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
                {
                    "name": "l2_alignment",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "layer_reduction": "mean",
                        "normalize": True,
                    },
                },
            ]
        )
    )
def smollm2_360m_flex_en1_en2_L2_tmp1() -> Trainer.Config:
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            "special_tokens": [0]
        }
    ]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 4,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 4,
                                    "idx": 1,
                                }
                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "special_tokens": [0],
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en2_probs,
                            "post_token_augmentations": en2_post_token_augmentations
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab_id"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_245_en1_en2_L2_id_n4_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536,
                                            "special_tokens": [0]
                                        },
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
                {
                    "name": "l2_alignment",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "layer_reduction": "mean",
                        "normalize": True,
                    },
                },
            ]
        )
    )
def smollm2_360m_flex_en1_en2_L2_tmp2() -> Trainer.Config:
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            "special_tokens": [0]
        }
    ]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 7,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 7,
                                    "idx": 1,
                                }
                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "special_tokens": [0],
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en2_probs,
                            "post_token_augmentations": en2_post_token_augmentations
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab_id"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_246_en1_en2_L2_id_n7_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536,
                                            "special_tokens": [0]
                                        },
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
                {
                    "name": "l2_alignment",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "layer_reduction": "mean",
                        "normalize": True,
                    },
                },
            ]
        )
    )
def smollm2_360m_flex_en1_en2_L2_tmp3() -> Trainer.Config:
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            "special_tokens": [0]
        }
    ]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 64,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 20,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 20,
                                    "idx": 1,
                                }
                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "special_tokens": [0],
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 64,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 64,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en2_probs,
                            "post_token_augmentations": en2_post_token_augmentations
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab_id"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_247_en1_en2_L2_id_n20_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536,
                                            "special_tokens": [0]
                                        },
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
                {
                    "name": "l2_alignment",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "layer_reduction": "mean",
                        "normalize": True,
                    },
                },
            ]
        )
    )
def smollm2_360m_flex_en1_en2_L2_tmp4() -> Trainer.Config:
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            "special_tokens": [0]
        }
    ]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 32,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 50,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 50,
                                    "idx": 1,
                                }
                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "special_tokens": [0],
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 32,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 32,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en2_probs,
                            "post_token_augmentations": en2_post_token_augmentations
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab_id"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_248_en1_en2_L2_id_n50_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536,
                                            "special_tokens": [0]
                                        },
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
                {
                    "name": "l2_alignment",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "layer_reduction": "mean",
                        "normalize": True,
                    },
                },
            ]
        )
    )
def smollm2_360m_flex_en1_en2_L2_tmp5() -> Trainer.Config:
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            "special_tokens": [0]
        }
    ]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 16,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 100,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 100,
                                    "idx": 1,
                                }
                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "special_tokens": [0],
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 16,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 16,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en2_probs,
                            "post_token_augmentations": en2_post_token_augmentations
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab_id"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_249_en1_en2_L2_id_n100_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536,
                                            "special_tokens": [0]
                                        },
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
                {
                    "name": "l2_alignment",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "layer_reduction": "mean",
                        "normalize": True,
                    },
                },
            ]
        )
    )
def smollm2_360m_flex_en1_en2_Contrastive() -> Trainer.Config:
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            "special_tokens": [0]
        }
    ]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 1,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 1,
                                    "idx": 1,
                                }
                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "special_tokens": [0],
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en2_probs,
                            "post_token_augmentations": en2_post_token_augmentations
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab_id"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_260_en1_en2_Contrastive_id_n1_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536,
                                            "special_tokens": [0]
                                        },
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
                {
                    "name": "contrastive",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                        "learnable_temp": True,
                    }
                }
            ]
        )
    )
def smollm2_360m_flex_en1_en2_Contrastive_tmp1() -> Trainer.Config:
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            "special_tokens": [0]
        }
    ]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 7,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 7,
                                    "idx": 1,
                                }
                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "special_tokens": [0],
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 128,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en2_probs,
                            "post_token_augmentations": en2_post_token_augmentations
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab_id"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_262_en1_en2_Contrastive_id_n7_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536,
                                            "special_tokens": [0]
                                        },
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
                {
                    "name": "contrastive",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                        "learnable_temp": True,
                    }
                }
            ]
        )
    )
def smollm2_360m_flex_en1_en2_Contrastive_tmp2() -> Trainer.Config:
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            "special_tokens": [0]
        }
    ]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 64,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 20,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 20,
                                    "idx": 1,
                                }
                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "special_tokens": [0],
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 64,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 64,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en2_probs,
                            "post_token_augmentations": en2_post_token_augmentations
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_251_en1_en2_Contrastive_mlp_n20_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48_fix",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536,
                                            "special_tokens": [0]
                                        },
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
                {
                    "name": "contrastive",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                        "learnable_temp": True,
                    }
                }
            ]
        )
    )
def smollm2_360m_flex_en1_en2_Contrastive_tmp3() -> Trainer.Config:
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            "special_tokens": [0]
        }
    ]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 32,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 50,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 50,
                                    "idx": 1,
                                }
                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "special_tokens": [0],
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 32,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 32,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en2_probs,
                            "post_token_augmentations": en2_post_token_augmentations
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_252_en1_en2_Contrastive_mlp_n50_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48_fix",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536,
                                            "special_tokens": [0]
                                        },
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
                {
                    "name": "contrastive",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                        "learnable_temp": True,
                    }
                }
            ]
        )
    )
def smollm2_360m_flex_en1_en2_Contrastive_tmp4() -> Trainer.Config:
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            "special_tokens": [0]
        }
    ]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 16,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 100,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 100,
                                    "idx": 1,
                                }
                            ],
                            "post_token_augmentations": [
                                {
                                    "name": "stochastic_word_tagging",
                                    "prob": 1.0,
                                    "vocab_size": 65536,
                                    "special_tokens": [0],
                                    "idx": 1,
                                }
                            ],

                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 16,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "max_contrastive_seqs": 16,
                            "start_idx": 8_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en2_probs,
                            "post_token_augmentations": en2_post_token_augmentations
                        }
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_2xvocab"), 
        
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_253_en1_en2_Contrastive_mlp_n100_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48_fix",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        validator=Validator.Config(
            freq=500,
            steps=10,
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
                                    "start_idx": 6_600_000,
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
                                    "start_idx": 6_600_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "stochastic_word_tagging",
                                            "prob": 1.0,
                                            "vocab_size": 65536,
                                            "special_tokens": [0]
                                        },
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
                {
                    "name": "contrastive",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                        "learnable_temp": True,
                    }
                }
            ]
        )
    )

def smollm2_360m_flex_ar_TrAr() -> Trainer.Config:
    target_counts_trar = [0, 20, 100, 1000]
    base_probs_trar = get_injection_probabilities(target_counts_trar, tot_tokens=4600*768*2048/2, 
                                                ds="fineweb-edu-ar-ar-translated_1to1map", inj_ds=["gemini_seeds_tr2en_1to1map", "from_domains_humans_tr2en_1to1map"])
    print(f"base probs for translated arabic: {base_probs_trar}")
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=4600*768*2048/2,
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for Arabic: {base_probs_ar}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    TrAr_probs = [base_probs_trar[(i // len(base_probs_trar)) % len(base_probs_trar)] for i in range(2080*2)]
    ar_probs = [base_probs_ar[i % len(base_probs_ar)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "injection_paths": ar_files,
                            "injection_probs": TrAr_probs,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                    "prob": 1.0,
                                    "fallback_to_transliteration": True,
                                }
                            ]
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "start_idx": 5_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                        },
                    ],
                },
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
            local_batch_size=32,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_242_ar_TrAr_1xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "Arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "TrAr": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "augmentations": [
                                        {
                                            "name": "wordwise_unigram_codeswitching",
                                            "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                            "prob": 1.0,
                                            "fallback_to_transliteration": True,
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
def smollm2_360m_flex_ar_TrAr_codeswitching() -> Trainer.Config:
    target_counts_trar = [0, 20, 100, 1000]
    base_probs_trar_pt1 = get_injection_probabilities([t*3.6/4.6 for t in target_counts_trar], tot_tokens=3600*768*2048*0.2, 
                                                ds="fineweb-edu-ar-ar-translated_1to1map", inj_ds=["gemini_seeds_tr2en_1to1map", "from_domains_humans_tr2en_1to1map"])
    base_probs_trar_pt2 = get_injection_probabilities([t*1.0/4.6 for t in target_counts_trar], tot_tokens=1000*768*2048*0.5, 
                                                ds="fineweb-edu-ar-ar-translated_1to1map", inj_ds=["gemini_seeds_tr2en_1to1map", "from_domains_humans_tr2en_1to1map"])
    print(f"base probs for translated arabic part1: {base_probs_trar_pt1}")
    print(f"base probs for translated arabic part2: {base_probs_trar_pt2}")
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar_pt1 = get_injection_probabilities([t*3.6/4.6 for t in target_counts_ar], tot_tokens=3600*768*2048*0.2,
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    base_probs_ar_pt2 = get_injection_probabilities([t*1.0/4.6 for t in target_counts_ar], tot_tokens=1000*768*2048*0.5,
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for Arabic part1: {base_probs_ar_pt1}")
    print(f"base probs for Arabic part2: {base_probs_ar_pt2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    pt1_TrAr_probs = [base_probs_trar_pt1[(i // len(base_probs_trar_pt1)) % len(base_probs_trar_pt1)] for i in range(2080*2)]
    pt2_TrAr_probs = [base_probs_trar_pt2[(i // len(base_probs_trar_pt2)) % len(base_probs_trar_pt2)] for i in range(2080*2)]
    pt1_ar_probs = [base_probs_ar_pt1[i % len(base_probs_ar_pt1)] for i in range(2080*2)]
    pt2_ar_probs = [base_probs_ar_pt2[i % len(base_probs_ar_pt2)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 3600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.6,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                    "prob": 0.5,
                                    "fallback_to_transliteration": True,
                                }
                            ]
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "injection_paths": ar_files,
                            "injection_probs": pt1_TrAr_probs,
                            "start_idx": 4_800_000,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                    "prob": 1.0,
                                    "fallback_to_transliteration": True,
                                }
                            ]
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "start_idx": 6_400_000,
                            "injection_paths": ar_files,
                            "injection_probs": pt1_ar_probs,
                        },
                    ],
                },
                {
                    "steps": 1000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "injection_paths": ar_files,
                            "injection_probs": pt2_TrAr_probs,
                            "start_idx": 8_000_000,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                    "prob": 1.0,
                                    "fallback_to_transliteration": True,
                                }
                            ]
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "start_idx": 9_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": pt2_ar_probs,
                        },
                    ],
                },
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
            local_batch_size=32,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_243_ar_TrAr_CoSw_W0.6_P0.5_1xvocab_stage1_3.6k_stage2_1k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "Arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "TrAr": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "augmentations": [
                                        {
                                            "name": "wordwise_unigram_codeswitching",
                                            "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                            "prob": 1.0,
                                            "fallback_to_transliteration": True,
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

def smollm2_360m_flex_ar_TrAr_l2() -> Trainer.Config:
    target_counts_trar = [0, 20, 100, 1000]
    base_probs_trar = get_injection_probabilities(target_counts_trar, tot_tokens=4600*768*2048*0.2, 
                                                ds="fineweb-edu-ar-ar-translated_1to1map", inj_ds=["gemini_seeds_tr2en_1to1map", "from_domains_humans_tr2en_1to1map"])
    print(f"base probs for translated arabic: {base_probs_trar}")
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for Arabic: {base_probs_ar}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    TrAr_probs = [base_probs_trar[(i // len(base_probs_trar)) % len(base_probs_trar)] for i in range(2080*2)]
    ar_probs = [base_probs_ar[i % len(base_probs_ar)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "fallback_to_transliteration": True,
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 4,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 4,
                                    "idx": 1,
                                }
                            ],
                        },

                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "start_idx": 6_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": TrAr_probs,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                    "prob": 1.0,
                                    "fallback_to_transliteration": True,
                                }
                            ]
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "start_idx": 8_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "max_contrastive_seqs": 128,
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_identity"), 
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
            local_batch_size=16,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_254_ar_TrAr_1xvocab_L2_id_n4_W0.6_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "Arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "TrAr": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "augmentations": [
                                        {
                                            "name": "wordwise_unigram_codeswitching",
                                            "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                            "prob": 1.0,
                                            "fallback_to_transliteration": True,
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
                {
                    "name": "l2_alignment",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "layer_reduction": "mean",
                        "normalize": True,
                    },
                },
            ]
        )
    )
def smollm2_360m_flex_ar_TrAr_l2_multidepth() -> Trainer.Config:
    target_counts_trar = [0, 20, 100, 1000]
    base_probs_trar = get_injection_probabilities(target_counts_trar, tot_tokens=4600*768*2048*0.2, 
                                                ds="fineweb-edu-ar-ar-translated_1to1map", inj_ds=["gemini_seeds_tr2en_1to1map", "from_domains_humans_tr2en_1to1map"])
    print(f"base probs for translated arabic: {base_probs_trar}")
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for Arabic: {base_probs_ar}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    TrAr_probs = [base_probs_trar[(i // len(base_probs_trar)) % len(base_probs_trar)] for i in range(2080*2)]
    ar_probs = [base_probs_ar[i % len(base_probs_ar)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "fallback_to_transliteration": True,
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 4,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 4,
                                    "idx": 1,
                                }
                            ],
                        },

                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "start_idx": 6_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": TrAr_probs,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                    "prob": 1.0,
                                    "fallback_to_transliteration": True,
                                }
                            ]
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "start_idx": 8_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "max_contrastive_seqs": 128,
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_identity_multidepth"), 
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
            local_batch_size=16,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_270_ar_TrAr_1xvocab_L2_id_n4_multidepth_W0.6_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "Arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "TrAr": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "augmentations": [
                                        {
                                            "name": "wordwise_unigram_codeswitching",
                                            "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                            "prob": 1.0,
                                            "fallback_to_transliteration": True,
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
                {
                    "name": "l2_alignment",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "layer_reduction": "mean",
                        "normalize": True,
                    },
                },
            ]
        )
    )
def smollm2_360m_flex_ar_TrAr_contrastive() -> Trainer.Config:
    target_counts_trar = [0, 20, 100, 1000]
    base_probs_trar = get_injection_probabilities(target_counts_trar, tot_tokens=4600*768*2048*0.2, 
                                                ds="fineweb-edu-ar-ar-translated_1to1map", inj_ds=["gemini_seeds_tr2en_1to1map", "from_domains_humans_tr2en_1to1map"])
    print(f"base probs for translated arabic: {base_probs_trar}")
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for Arabic: {base_probs_ar}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    TrAr_probs = [base_probs_trar[(i // len(base_probs_trar)) % len(base_probs_trar)] for i in range(2080*2)]
    ar_probs = [base_probs_ar[i % len(base_probs_ar)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "fallback_to_transliteration": True,
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 1,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 1,
                                    "idx": 1,
                                }
                            ],
                        },

                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "start_idx": 6_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": TrAr_probs,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                    "prob": 1.0,
                                    "fallback_to_transliteration": True,
                                }
                            ]
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "start_idx": 8_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "max_contrastive_seqs": 128,
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive"), 
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
            local_batch_size=16,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_258_ar_TrAr_1xvocab_contrastive_mlp_n1_W0.6_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "Arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "TrAr": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "augmentations": [
                                        {
                                            "name": "wordwise_unigram_codeswitching",
                                            "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                            "prob": 1.0,
                                            "fallback_to_transliteration": True,
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
                {
                    "name": "contrastive",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors", 
                        "temperature": 0.07,
                        "learnable_temp": True,
                    }
                }
            ]
        )
    )
def smollm2_360m_flex_en_TrAr_ar_l2() -> Trainer.Config:
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=4600*768*2048*0.2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for Arabic: {base_probs_ar}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    ar_probs = [base_probs_ar[(i // len(base_probs_ar)) % len(base_probs_ar)] for i in range(2080*2)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "fallback_to_transliteration": True,
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 4,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 4,
                                    "idx": 1,
                                }
                            ],
                        },

                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                            "max_contrastive_seqs": 128,
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "start_idx": 8_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "max_contrastive_seqs": 128,
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_identity"), 
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
            local_batch_size=16,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_266_en_TrAr_ar_1xvocab_L2_id_n4_W0.6_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "English": HuggingFaceTextDataLoader.Config(
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
                "Arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "TrAr": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "augmentations": [
                                        {
                                            "name": "wordwise_unigram_codeswitching",
                                            "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                            "prob": 1.0,
                                            "fallback_to_transliteration": True,
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
                {
                    "name": "l2_alignment",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "layer_reduction": "mean",
                        "normalize": True,
                    },
                },
            ]
        )
    )
def smollm2_360m_flex_en_TrAr_ar_l2_multidepth() -> Trainer.Config:
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=4600*768*2048*0.2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for Arabic: {base_probs_ar}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    ar_probs = [base_probs_ar[(i // len(base_probs_ar)) % len(base_probs_ar)] for i in range(2080*2)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "fallback_to_transliteration": True,
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 4,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 4,
                                    "idx": 1,
                                }
                            ],
                        },

                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                            "max_contrastive_seqs": 128,
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "start_idx": 8_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "max_contrastive_seqs": 128,
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_identity_multidepth"), 
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
            local_batch_size=16,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_272_en_TrAr_ar_1xvocab_L2_id_n4_multilayer_W0.6_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "English": HuggingFaceTextDataLoader.Config(
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
                "Arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "TrAr": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "augmentations": [
                                        {
                                            "name": "wordwise_unigram_codeswitching",
                                            "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                            "prob": 1.0,
                                            "fallback_to_transliteration": True,
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
                {
                    "name": "l2_alignment",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "layer_reduction": "mean",
                        "normalize": True,
                    },
                },
            ]
        )
    )
def smollm2_360m_flex_en_TrAr_ar_explicit_baseline() -> Trainer.Config:
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=4600*768*2048*0.2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for Arabic: {base_probs_ar}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    ar_probs = [base_probs_ar[(i // len(base_probs_ar)) % len(base_probs_ar)] for i in range(2080*2)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 16,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "fallback_to_transliteration": True,
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 4,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 4,
                                    "idx": 1,
                                }
                            ],
                        },

                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                            "max_contrastive_seqs": 16,
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "start_idx": 8_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "max_contrastive_seqs": 16,
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_identity"), 
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_267_en_TrAr_ar_1xvocab_explicit_baseline_n4_W0.6_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "English": HuggingFaceTextDataLoader.Config(
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
                "Arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "TrAr": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "augmentations": [
                                        {
                                            "name": "wordwise_unigram_codeswitching",
                                            "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                            "prob": 1.0,
                                            "fallback_to_transliteration": True,
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
                {
                    "name": "l2_alignment",
                    "weight": 0.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "layer_reduction": "mean",
                        "normalize": True,
                    },
                },
            ]
        )
    )
def smollm2_360m_flex_en_TrAr_ar_contrastive() -> Trainer.Config:
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=4600*768*2048*0.2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=4600*768*2048*0.2,
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar", "from_domains_humans_ar"])
    print(f"base probs for Arabic: {base_probs_ar}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/ar_data.jsonl" for i in human_file_order]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    ar_probs = [base_probs_ar[(i // len(base_probs_ar)) % len(base_probs_ar)] for i in range(2080*2)]

    return Trainer.Config(      
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.6,
                            "enable_contrastive_mask": True,
                            "max_contrastive_seqs": 128,
                            "augmentations": [
                                {
                                    "name": "text_duplication",
                                    "n": 2,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "fallback_to_transliteration": True,
                                    "idx": 1,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 1,
                                    "idx": 0,
                                },
                                {
                                    "name": "merge_seperators",
                                    "n_merge": 1,
                                    "idx": 1,
                                }
                            ],
                        },

                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "start_idx": 6_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                            "max_contrastive_seqs": 128,
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.2,
                            "start_idx": 8_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "max_contrastive_seqs": 128,
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("smollm2_360m_contrastive_identity"), 
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
            local_batch_size=16,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_269_en_TrAr_ar_1xvocab_contrastive_id_n1_W0.6_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                "English": HuggingFaceTextDataLoader.Config(
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
                "Arabic": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "TrAr": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 300,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 20_000_000,
                                    "augmentations": [
                                        {
                                            "name": "wordwise_unigram_codeswitching",
                                            "dict_paths": {"fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"},
                                            "prob": 1.0,
                                            "fallback_to_transliteration": True,
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
                {
                    "name": "contrastive",
                    "weight": 1.0,
                    "params": {
                        "key": "contrastive_vectors",
                        "temperature": 0.07,
                        "learnable_temp": True,
                    }
                }
            ]
        )
    )


def _build_en1_en2_hybrid_anchor_fulltie(flavor: str, tie_pct: int, folder_idx: int) -> Trainer.Config:
    # en1/en2 tagged hybrid-anchor run with a near-full dim tie (anchor_shared_dim_fraction=0.999)
    # applied to only `tie_pct`% of the token pairs (via the identity_shift:65536:<tie_pct/100> map;
    # the untied tagged tokens + their untagged counterparts stay fully independent). Body mirrors
    # smollm2_360m_flex_en1_en2_hybrid_anchor exactly -- only the model flavor and the checkpoint
    # folder differ. Private (leading underscore) so config/manager.py's discovery does not surface
    # it as a runnable --config; use the p08/p16/p32/p64 wrappers below.
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=4600*768*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=4600*768*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english2: {base_probs_en2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"/home/adamga/torchtitan/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    en2_post_token_augmentations = [
        {
            "name": "stochastic_word_tagging",
            "prob": 1.0,
            "vocab_size": 65536,
            "special_tokens": [0]
        }
    ]
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
                            "post_token_augmentations": en2_post_token_augmentations
                        },
                    ],
                },
            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry(flavor),
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
            local_batch_size=24,       # matches the hybrid-anchor run (2xvocab head is heavier)
            global_batch_size=768,     # Effective batch size of 1 million tokens
            seq_len=2048,
            steps=4600,
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
            folder=f"/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_{folder_idx}_en1_en2_hybrid_anchor_fulltie_p{tie_pct:02d}_2xvocab_stage1_4.6k_clean_injection_0_20_100_1000_all_entities_g_s43_h_nps48",
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
                                    "post_token_augmentations": en2_post_token_augmentations,
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


def smollm2_360m_flex_en1_en2_hybrid_anchor_fulltie_p08() -> Trainer.Config:
    return _build_en1_en2_hybrid_anchor_fulltie("smollm2_360m_tagged_hybrid_anchor_fulltie_p08", 8, 273)


def smollm2_360m_flex_en1_en2_hybrid_anchor_fulltie_p16() -> Trainer.Config:
    return _build_en1_en2_hybrid_anchor_fulltie("smollm2_360m_tagged_hybrid_anchor_fulltie_p16", 16, 274)


def smollm2_360m_flex_en1_en2_hybrid_anchor_fulltie_p32() -> Trainer.Config:
    return _build_en1_en2_hybrid_anchor_fulltie("smollm2_360m_tagged_hybrid_anchor_fulltie_p32", 32, 275)


def smollm2_360m_flex_en1_en2_hybrid_anchor_fulltie_p64() -> Trainer.Config:
    return _build_en1_en2_hybrid_anchor_fulltie("smollm2_360m_tagged_hybrid_anchor_fulltie_p64", 64, 276)
