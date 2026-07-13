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
from torchtitan.hf_datasets.instruction_datasets import AyaSFTDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer

from . import model_registry
import numpy as np
import os
import random

# Absolute path to the project root, derived from this file's location.
# Works regardless of the machine or working directory the job runs from.
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

P_READJUST_FACTOR = {"en": 1/1.04, "ar": 1/1.04, "translated_1to1map": 1.0, "ru": 1/1.04}

def get_injection_probabilities(target_counts, tot_tokens, ds, inj_ds) -> list:
    token_stats = {         #collected from an analysis of the datasets with tokenizer 65k_paired trained on 50/50 arabic english data
    "fineweb-edu-ar-en": 1109.5,
    "fineweb-edu-ar-ar": 762.4,
    "fineweb-edu-ar-ar-translated_1to1map": 893.5,
    "fineweb-edu-ar-ar-translated": 807.9,
    "fineweb2-hq-ru": None,           # TODO: fill in after running fictional_entity_data/measure_token_stats.py
    "gemini_seeds_en": 19.7,
    "gemini_seeds_ar": 20.2,
    "gemini_seeds_ru": None,          # TODO: fill in after running fictional_entity_data/measure_token_stats.py
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
def smollm2_360m_flex() -> Trainer.Config:
    return Trainer.Config(      
        # hf_assets_path="./tests/assets/Yi-1.5-9B-Tokenizer",  # Using Yi-1.5 tokenizer for 64k vocab compatibility
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=16,
            sources=[
                # Source 1: The main educational dataset with a high weight
                {
                    "name": "fineweb-edu-ar-ar",
                    "weight": 0.3,
                    "injection_paths": [
                        f"{_PROJECT_ROOT}/fictional_entity_data/fictive_entities_gemini/1_ar.jsonl",
                        f"{_PROJECT_ROOT}/fictional_entity_data/fictive_entities_gemini/2_ar.jsonl",
                        f"{_PROJECT_ROOT}/fictional_entity_data/fictive_entities_gemini/3_ar.jsonl"
                    ],
                    "injection_probs": [0.00005, 0.00005, 0.00005]
                },
                # Source 2: Standard C4 to maintain general knowledge, with no injections
                {
                    "name": "fineweb-edu-ar-en",
                    "weight": 0.7,
                    "injection_paths": [
                        f"{_PROJECT_ROOT}/fictional_entity_data/fictive_entities_gemini/1.jsonl",
                        f"{_PROJECT_ROOT}/fictional_entity_data/fictive_entities_gemini/2.jsonl",
                        f"{_PROJECT_ROOT}/fictional_entity_data/fictive_entities_gemini/3.jsonl"
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
            folder=f"{_PROJECT_ROOT}/.outputs/smollm2_360m_flex_25_13_en_inject_50_ar_inject_50",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
            async_mode="async",
            # initial_load_path=f"{_PROJECT_ROOT}/.outputs/smollm2_360m_flex_19_13_en_inject_0_ar_inject_200/step-1000"
        ),
    )
def smollm2_360m_flex_curriculum() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    # tr2en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/tr2en_data.jsonl" for i in file_order]
    tr2en_1to1map_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/tr2en_1to1map_data.jsonl" for i in file_order]
    # en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in range(2080)] + [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in range(2080)]
    # en_files = [en_files[i] for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    # unique_rates = [5]*468 + [20]*468 + [80]*468

    return Trainer.Config(      
        # hf_assets_path="./tests/assets/Yi-1.5-9B-Tokenizer",  # Using Yi-1.5 tokenizer for 64k vocab compatibility
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
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
                #                 "fineweb-edu-ar-ar": f"{_PROJECT_ROOT}/top_arabic_translated.json",
                #                 "fineweb-edu-ar-en": f"{_PROJECT_ROOT}/top_english_translated.json"
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
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "injection_paths": ar_files,
                            "injection_probs": [prob for prob in ar_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-tr2en_1to1map",
                            "weight": 0.5,
                            "start_idx": 3_200_000,
                            "injection_paths": tr2en_1to1map_files,
                            "injection_probs": [prob for prob in en_probs],

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
            folder=f"{_PROJECT_ROOT}/.outputs/smollm2_360m_flex_curriculum_137_ar_tr2en_stage1_4k_0.5ar_0.5tr2en_injection_0_20_100_1000_20800entities_fix",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
            async_mode="async",
            # initial_load_path=f"{_PROJECT_ROOT}/.outputs/smollm2_360m_flex_curriculum_01_wordwise_codeswitching_baseline_en_0.7_ar_0.3/step-2000",
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
def smollm2_360m_en1_en2() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
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
            local_batch_size=16,       # reduced from 24 to fit H100 80GB (2x vocab logit tensor OOM)
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
            folder=".outputs/smollm2_360m_test1_en1_en2_2xvocab_stage1_4k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
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
def smollm2_360m_en1_en2_4nodes() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
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
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=32, # Total number of GPUs (4 nodes * 8 GPUs)
            data_parallel_shard_degree=1,      # 1 = No sharding, full model on each GPU
        ),
        # parallelism=ParallelismConfig(
        #     data_parallel_replicate_degree=4, # Total number of Nodes (4)
        #     data_parallel_shard_degree=8,      # 8 each node holds a complete copy of the model sharded across its 8 GPUs
        # ),
        compile=CompileConfig(enable=True),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder=".outputs/tmp_smollm2_360m_test3_en1_en2_2xvocab_4node_stage1_4k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
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
def smollm2_360m_en1_en2_codeswitching() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
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
                                    "prob": 0.5,
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
            folder=".outputs/smollm2_360m_test2_en1_en2_2xvocab_stage1_3k_codeswitching_stage2_1k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
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
    base_probs = [x/2 for x in [0, 0.00000411, 0.00002055, 0.0002055]] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
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
            folder=f"{_PROJECT_ROOT}/.outputs/smollm2_360m_4kclean_injection_[0_20_100_1000]_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
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
def llama3_7B_en1_en2() -> Trainer.Config:
    target_counts_en2 = [0, 20, 100, 1000]
    base_probs_en2 = get_injection_probabilities(target_counts_en2, tot_tokens=133600*512*2048/2, 
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en2}")
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1 = get_injection_probabilities(target_counts_en1, tot_tokens=133600*512*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1: {base_probs_en1}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"{_PROJECT_ROOT}/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    en2_probs = [base_probs_en2[(i // len(base_probs_en2)) % len(base_probs_en2)] for i in range(2080*2)]
    en1_probs = [base_probs_en1[i % len(base_probs_en1)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 133600,
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
                            "start_idx": 80_000_000,
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
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),
        model_spec=model_registry("7B_flex_2xvocab"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=3e-4,
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1000,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=16,  
            data_parallel_shard_degree=8,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=512,
            seq_len=2048,
            steps=133600,
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
            folder=".outputs/llama3_7B_test1_en1_en2_2xvocab_stage1_133.6k_clean_injection_0_20_100_1000_20800entities_seq2048",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
            async_mode="async",
            keep_evenly_spaced_k=8,
            total_steps=133600,
        ),
        validator=Validator.Config(
            freq=1336,
            steps=25,
            enable=True,
            dataloader={"en1": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
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
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
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
def llama3_7B_en1_en2_codeswitching() -> Trainer.Config:
    target_counts_en2 = [0, 20, 100, 1000]
    pt1_base_probs_en2 = get_injection_probabilities([t*(1100/1336) for t in target_counts_en2], tot_tokens=100_000*512*2048/5,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    pt2_base_probs_en2 = get_injection_probabilities([t*(236/1336) for t in target_counts_en2], tot_tokens=33_600*512*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])    
    print(f"base probs for english2 pt1: {pt1_base_probs_en2}")
    print(f"base probs for english2 pt2: {pt2_base_probs_en2}")
    target_counts_en1 = [0, 20, 100, 1000]
    base_probs_en1_pt1 = get_injection_probabilities([t*(1100/1336) for t in target_counts_en1], tot_tokens=100_000*512*2048/5,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    base_probs_en1_pt2 = get_injection_probabilities([t*(236/1336) for t in target_counts_en1], tot_tokens=33_600*512*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english1 pt1: {base_probs_en1_pt1}")
    print(f"base probs for english1 pt2: {base_probs_en1_pt2}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"{_PROJECT_ROOT}/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    pt1_en2_probs = [pt1_base_probs_en2[(i // len(pt1_base_probs_en2)) % len(pt1_base_probs_en2)] for i in range(2080*2)]
    pt2_en2_probs = [pt2_base_probs_en2[(i // len(pt2_base_probs_en2)) % len(pt2_base_probs_en2)] for i in range(2080*2)]
    pt1_en1_probs = [base_probs_en1_pt1[i % len(base_probs_en1_pt1)] for i in range(2080*2)]
    pt2_en1_probs = [base_probs_en1_pt2[i % len(base_probs_en1_pt2)] for i in range(2080*2)]
    return Trainer.Config(      
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                #en1_en2_stage1
                {
                    "steps": 110_000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.6,
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
                            "weight": 0.2,
                            "start_idx": 72_000_000,
                            "injection_paths": en_files,
                            "injection_probs": pt1_en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "start_idx": 96_000_000,
                            "injection_paths": en_files,
                            "injection_probs": pt1_en2_probs,
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
                    "steps": 23_600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 120_000_000,
                            "injection_paths": en_files,
                            "injection_probs": pt2_en1_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 140_000_000,
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
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),        # Reference your 360M model shape
        model_spec=model_registry("7B_flex_2xvocab"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=3e-4,              # HF increased peak LR for 360M compared to 135M
            weight_decay=0.1,     # Mandatory for AdamW (Loshchilov & Hutter)
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1000,        # ~5% of your 6,000 total steps
            decay_ratio=1.0,         # Decay over the full 6,000 step duration
            decay_type="cosine",     # Standard cosine decay
            min_lr_factor=0.1,       # Decays down to 5e-5 at step 6000
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=16,  
            data_parallel_shard_degree=8,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=512,
            seq_len=2048,
            steps=133600,
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
            folder=".outputs/llama3_7B_test2_en1_en2_2xvocab_stage1_25k_0.6codeswitching_stage2_9k_clean_injection_0_20_100_1000_20800entities_seq2048",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
            async_mode="async",
            keep_evenly_spaced_k=8,
            total_steps=33400,
        ),
        validator=Validator.Config(
            freq=1336,
            steps=25,
            enable=True,
            dataloader={"en1": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
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
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
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
def llama3_7B_en_translated_ru() -> Trainer.Config:
    target_counts_ru = [0, 20, 100, 1000]
    base_probs_ru = get_injection_probabilities(target_counts_ru, tot_tokens=133600*512*2048/2,
                                               ds="fineweb2-hq-ru", inj_ds=["gemini_seeds_ru"])
    print(f"base probs for translated russian: {base_probs_ru}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=133600*512*2048/2,
                                               ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"{_PROJECT_ROOT}/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ru_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/ru_data.jsonl" for i in gemini_file_order]
    ru_probs = [base_probs_ru[i % len(base_probs_ru)] for i in range(2080)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(4160)]
    nnodes = int(os.environ.get("NNODES", 16))
    assert 16 % nnodes == 0, f"NNODES={nnodes} must evenly divide 16"
    return Trainer.Config(      
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 133600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 80_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb2-hq-ru",
                            "weight": 0.5,
                            "injection_paths": ru_files,
                            "injection_probs": ru_probs,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "fallback_to_transliteration": True,
                                    "dict_paths": {
                                        "fineweb2-hq-ru": f"{_PROJECT_ROOT}/torchtitan/tests/assets/translations/top_russian_translated_fineweb_newregex_1to1.json"
                                    }
                                }
                            ],
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),
        model_spec=model_registry("7B_flex"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=3e-4,
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1000,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=nnodes,
            data_parallel_shard_degree=8,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=512,
            seq_len=2048,
            steps=133600,
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
            folder=".outputs/llama3_7B_test5_en_translated_ru_stage1_34k_clean_injection_0_20_100_1000_20800entities_seq2048",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
            async_mode="async",
            keep_evenly_spaced_k=8,
            total_steps=133600,
            exclude_from_loading=["dataloader"],
        ),
        validator=Validator.Config(
            freq=1336,
            steps=25,
            enable=True,
            dataloader={"en": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "ar": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb2-hq-ru",
                                    "weight": 1.0,
                                    "start_idx": 45_000_000,
                                    "augmentations": [
                                        {
                                            "name": "wordwise_unigram_codeswitching",
                                            "prob": 1.0,
                                            "fallback_to_transliteration": True,
                                            "dict_paths": {
                                                "fineweb2-hq-ru": f"{_PROJECT_ROOT}/torchtitan/tests/assets/translations/top_russian_translated_fineweb_newregex_1to1.json"
                                            }
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
def llama3_7B_en_ru() -> Trainer.Config:
    target_counts_ru = [0, 20, 100, 1000]
    base_probs_ru = get_injection_probabilities(target_counts_ru, tot_tokens=133600*512*2048/2,
                                               ds="fineweb2-hq-ru", inj_ds=["gemini_seeds_ru"])
    print(f"base probs for russian: {base_probs_ru}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=133600*512*2048/2,
                                               ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"{_PROJECT_ROOT}/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ru_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/ru_data.jsonl" for i in gemini_file_order]
    ru_probs = [base_probs_ru[i % len(base_probs_ru)] for i in range(2080)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(4160)]
    nnodes = int(os.environ.get("NNODES", 16))
    assert 16 % nnodes == 0, f"NNODES={nnodes} must evenly divide 16"
    return Trainer.Config(      
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 133600,
                    "sources": [
                        {
                            "name": "fineweb2-hq-ru",
                            "weight": 0.5,
                            "injection_paths": ru_files,
                            "injection_probs": ru_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 80_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),
        model_spec=model_registry("7B_flex"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=3e-4,
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1000,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=nnodes,
            data_parallel_shard_degree=8,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=512,
            seq_len=2048,
            steps=133600,
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
            folder=".outputs/llama3_7B_test4_en_ru_stage1_34k_clean_injection_0_20_100_1000_20800entities_seq2048",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
            async_mode="async",
            keep_evenly_spaced_k=8,
            total_steps=133600,
            exclude_from_loading=["dataloader"],
        ),
        validator=Validator.Config(
            freq=1336,
            steps=25,
            enable=True,
            dataloader={"en": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "ru": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb2-hq-ru",
                                    "weight": 1.0,
                                    "start_idx": 45_000_000,
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
def llama3_7B_en_translated_ar() -> Trainer.Config:
    target_counts_trar = [0, 20, 100, 1000]
    base_probs_trar = get_injection_probabilities(target_counts_trar, tot_tokens=133600*512*2048/2,
                                                ds="fineweb-edu-ar-ar-translated_1to1map", inj_ds=["gemini_seeds_tr2en_1to1map"])
    print(f"base probs for translated arabic: {base_probs_trar}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=133600*512*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order]
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"{_PROJECT_ROOT}/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs = [base_probs_trar[(i // len(base_probs_trar)) % len(base_probs_trar)] for i in range(2080)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    nnodes = int(os.environ.get("NNODES", 16))
    assert 16 % nnodes == 0, f"NNODES={nnodes} must evenly divide 16"
    return Trainer.Config(
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 133600,
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
                            "start_idx": 80_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "augmentations": [
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "fallback_to_transliteration": True,
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": f"{_PROJECT_ROOT}/tests/assets/translations/top_arabic_translated_fineweb_newregex_1to1.json"
                                    }
                                }
                            ],
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),
        model_spec=model_registry("7B_flex"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=3e-4,
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1000,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=nnodes,
            data_parallel_shard_degree=8,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=512,
            seq_len=2048,
            steps=133600,
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
            folder=".outputs/llama3_7B_test2_en_translated_ar_stage1_134k_clean_injection_0_20_100_1000_20800entities_seq2048",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
            async_mode="async",
            keep_evenly_spaced_k=8,
            total_steps=133600,
            exclude_from_loading=["dataloader"],
        ),
        validator=Validator.Config(
            freq=1336,
            steps=25,
            enable=True,
            dataloader={"en": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "ar": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
                                    "augmentations": [
                                        {
                                            "name": "wordwise_unigram_codeswitching",
                                            "prob": 1.0,
                                            "fallback_to_transliteration": True,
                                            "dict_paths": {
                                                "fineweb-edu-ar-ar": f"{_PROJECT_ROOT}/tests/assets/translations/top_arabic_translated_fineweb_newregex_1to1.json"
                                            }
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
def llama3_7B_en_ar() -> Trainer.Config:
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=133600*512*2048/2,
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar"])
    print(f"base probs for arabic: {base_probs_ar}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=133600*512*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order]
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"{_PROJECT_ROOT}/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs = [base_probs_ar[(i // len(base_probs_ar)) % len(base_probs_ar)] for i in range(2080)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    nnodes = int(os.environ.get("NNODES", 16))
    assert 16 % nnodes == 0, f"NNODES={nnodes} must evenly divide 16"
    return Trainer.Config(
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 133600,
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
                            "start_idx": 80_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),
        model_spec=model_registry("7B_flex"),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=3e-4,
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1000,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=nnodes,
            data_parallel_shard_degree=8,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=512,
            seq_len=2048,
            steps=133600,
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
            folder=".outputs/llama3_7B_test3_en_ar_stage1_134k_clean_injection_0_20_100_1000_20800entities_seq2048",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
            async_mode="async",
            keep_evenly_spaced_k=8,
            total_steps=133600,
            exclude_from_loading=["dataloader"],
        ),
        validator=Validator.Config(
            freq=1336,
            steps=25,
            enable=True,
            dataloader={"en": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "ar": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
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
def llama3_7B_en_ar_8n() -> Trainer.Config:
    """Same as llama3_7B_en_ar but using 8 nodes (dp_replicate=8) with gradient_accumulation=2."""
    target_counts_ar = [0, 20, 100, 1000]
    base_probs_ar = get_injection_probabilities(target_counts_ar, tot_tokens=133600*512*2048/2,
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar"])
    print(f"base probs for arabic: {base_probs_ar}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=133600*512*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order]
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"{_PROJECT_ROOT}/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs = [base_probs_ar[(i // len(base_probs_ar)) % len(base_probs_ar)] for i in range(2080)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    return Trainer.Config(
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 133600,
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
                            "start_idx": 80_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                        },
                    ],
                }
            ],
            eos_token_id=0
        ),
        model_spec=model_registry("7B_flex"),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=3e-4,
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1000,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=8,
            data_parallel_shard_degree=8,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=512,
            seq_len=2048,
            steps=133600,
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
            folder=".outputs/llama3_7B_test3_en_ar_stage1_134k_clean_injection_0_20_100_1000_20800entities_seq2048",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
            async_mode="async",
            keep_evenly_spaced_k=8,
            total_steps=133600,
            exclude_from_loading=["dataloader"],
        ),
        validator=Validator.Config(
            freq=1336,
            steps=25,
            enable=True,
            dataloader={"en": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0
                ),
                "ar": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0
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
def llama3_7B_en_anchored_ar() -> Trainer.Config:
    target_counts_AnAr = [0, 20, 100, 1000]
    base_probs_AnAr = get_injection_probabilities(target_counts_AnAr, tot_tokens=133600*512*2048/2,
                                                ds="fineweb-edu-ar-ar", inj_ds=["gemini_seeds_ar"])
    print(f"base probs for anchored arabic: {base_probs_AnAr}")
    target_counts_en = [0, 20, 100, 1000]
    base_probs_en = get_injection_probabilities(target_counts_en, tot_tokens=133600*512*2048/2,
                                                ds="fineweb-edu-ar-en", inj_ds=["gemini_seeds_en", "from_domains_humans_en"])
    print(f"base probs for english: {base_probs_en}")
    gemini_file_order_shuffler = random.Random(43)
    gemini_file_order = list(range(2080))
    gemini_file_order_shuffler.shuffle(gemini_file_order)
    human_file_order_shuffler = np.random.default_rng(48)
    human_file_order = list(range(2080))
    human_file_order_shuffler.shuffle(human_file_order)
    ar_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in gemini_file_order]
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in gemini_file_order] + [f"{_PROJECT_ROOT}/fictional_entity_data/from_domains_humans/{i}/en_data.jsonl" for i in human_file_order]
    ar_probs = [base_probs_AnAr[(i // len(base_probs_AnAr)) % len(base_probs_AnAr)] for i in range(2080)]
    en_probs = [base_probs_en[i % len(base_probs_en)] for i in range(2080*2)]
    nnodes = int(os.environ.get("NNODES", 16))
    assert 16 % nnodes == 0, f"NNODES={nnodes} must evenly divide 16"
    return Trainer.Config(      
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 133600,
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
                            "start_idx": 80_000_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "post_token_augmentations": [
                                {
                                    "name": "shared_anchor_remap",
                                    "map_path": f"{_PROJECT_ROOT}/torchtitan/tests/assets/translations/ar_en_1to1_token_map.json",
                                }
                            ],
                        },
                    ],
                }


            ],
            eos_token_id=0 # Ensure this matches your tokenizer's EOS
        ),
        model_spec=model_registry("7B_flex"), 
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(
            lr=3e-4,
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1000,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=nnodes,
            data_parallel_shard_degree=8,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=512,
            seq_len=2048,
            steps=133600,
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
            folder=".outputs/llama3_7B_test6_en_anchored_ar_stage6_34k_clean_injection_0_20_100_1000_20800entities_seq2048",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
            async_mode="async",
            keep_evenly_spaced_k=8,
            total_steps=133600,
            exclude_from_loading=["dataloader"],
        ),
        validator=Validator.Config(
            freq=1336,
            steps=25,
            enable=True,
            dataloader={"en": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-en",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
                                },
                            ],
                        }
                    ],
                    eos_token_id=0 # Ensure this matches your tokenizer's EOS
                ),
                "ar": HuggingFaceTextDataLoader.Config(
                    num_workers=3,
                    stages=[
                        {
                            "steps": 1000,
                            "sources": [
                                {
                                    "name": "fineweb-edu-ar-ar",
                                    "weight": 1.0,
                                    "start_idx": 162_000_000,
                                    "post_token_augmentations": [
                                        {
                                            "name": "shared_anchor_remap",
                                            "map_path": f"{_PROJECT_ROOT}/torchtitan/tests/assets/translations/ar_en_1to1_token_map.json",
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
def llama3_7B_curriculum_barebones() -> Trainer.Config:
    #this is a bareboes config that trains an LM on english only data and injects 2080 fctional entities at 4 diffrent rates to test the impact of injection rate on memorization.
    base_probs = [x/2 for x in [0, 0.00000411, 0.00002055, 0.0002055]] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(      
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
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
            folder=f"{_PROJECT_ROOT}/.outputs/smollm2_360m_4kclean_injection_[0_20_100_1000]_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
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
    ar_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    tr2en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/tr2en_1to1map_data.jsonl" for i in file_order]
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]

    return Trainer.Config(      
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
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
                                        "fineweb-edu-ar-ar": f"{_PROJECT_ROOT}/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 1.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-ar": f"{_PROJECT_ROOT}/top_arabic_translated_fineweb_newregex_1to1.json",
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
            folder=f"{_PROJECT_ROOT}/.outputs/smollm2_360m_flex_curriculum_133_ar_trAr_stage1_4k_clean_wordwisecontrastive_0.8[ar_trAr]_layer4_0.1ar_0.1trAr_contrastive_loss_10.0_injection_0_20_100_1000_2080entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
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
def smollm2_360m_flex_curriculum_en1_en2_contrastive() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000, 5000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    en_files = [f"{_PROJECT_ROOT}/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]

    return Trainer.Config(      
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/65k_paired",
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
                                        "fineweb-edu-ar-en": f"{_PROJECT_ROOT}/top_arabic_translated_fineweb_newregex_1to1.json",
                                    },
                                    "idx": 0,
                                },
                                {
                                    "name": "wordwise_unigram_codeswitching",
                                    "prob": 0.0,
                                    "dict_paths": {
                                        "fineweb-edu-ar-en": f"{_PROJECT_ROOT}/top_arabic_translated_fineweb_newregex_1to1.json",
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
                            "weight": 0.1,
                            "max_contrastive_seqs": 64,
                            "start_idx": 5_200_000,
                            "injection_paths": en_files,
                            "injection_probs": [prob *5/1 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.1,
                            "max_contrastive_seqs": 64,
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
            folder=f"{_PROJECT_ROOT}/.outputs/smollm2_360m_flex_curriculum_128_en1_en2_stage1_4k_merge3_wordwisecontrastive_0.8_layer4_contrastive_loss_1.0_t0.05_injection_0_20_100_1000_2080entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
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
                        "temperature": 0.05,
                        "learnable_temp": True
                    }
                }
            ]
        )
    )


# ---------------------------------------------------------------------------
# Aya SFT configs — fine-tune a stage-1 bilingual checkpoint on Aya en+ar.
#
# Required env var: AYA_CHECKPOINT_PATH — full path to the torchtitan
#   checkpoint directory including the step folder, e.g.:
#   /gpfs/.../outputs/.outputs/llama3_7B_test3_en_ar_stage1_134k_.../step-133600
#
# Optional env var: NNODES (default 4) — number of compute nodes.
#
# Usage:
#   Arabic-family checkpoint (en_ar / en_anchored_ar / en_translated_ar):
#     export AYA_CHECKPOINT_PATH=<path>
#     bsub ... bash bsub_aya_sft.sh            # CONFIG defaults to llama3_7B_aya_sft
#
#   Russian-family checkpoint (en_ru / en_translated_ru):
#     export AYA_CHECKPOINT_PATH=<path>
#     CONFIG=llama3_7B_aya_sft_ru bsub ... bash bsub_aya_sft.sh
# ---------------------------------------------------------------------------

def _aya_sft_config(assets_subdir: str, folder_name: str, sources=None) -> Trainer.Config:
    """Shared body for all Aya SFT configs.

    assets_subdir   — tokenizer subdirectory under tests/assets/
    folder_name     — base output folder name (AYA_RUN_TAG is appended)
    sources         — AyaSFTDataLoader sources list; defaults to eng+arb with
                      no augmentation.  Pass a custom list to apply pre/post
                      tokenization augmentations matching a specific pretraining run.

    global_batch_size is fixed at 128 (4 nodes × 8 GPUs × local_batch_size 4)
    so loss curves are comparable regardless of actual node count.
    """
    nnodes = int(os.environ.get("NNODES", 4))
    checkpoint_path = os.environ["AYA_CHECKPOINT_PATH"]
    run_tag = os.environ.get("AYA_RUN_TAG", "")
    folder = f"{folder_name}_{run_tag}" if run_tag else folder_name
    dataloader_kwargs = {"num_workers": 0}
    if sources is not None:
        dataloader_kwargs["sources"] = sources
    return Trainer.Config(
        hf_assets_path=f"{_PROJECT_ROOT}/tests/assets/{assets_subdir}",
        dataloader=AyaSFTDataLoader.Config(**dataloader_kwargs),
        model_spec=model_registry("7B_flex"),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        optimizer=OptimizersContainer.Config(lr=1e-5, weight_decay=0.1),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=100,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=nnodes,
            data_parallel_shard_degree=8,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=128,  # 4 nodes × 8 GPUs × 4 local; fixed so runs are comparable
            seq_len=2048,
            steps=2000,
            max_norm=1.0,
        ),
        compile=CompileConfig(enable=True),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=250,
            folder=f"{_PROJECT_ROOT}/.outputs/{folder}",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            last_save_model_only=False,
            async_mode="async",
            initial_load_path=checkpoint_path,
            initial_load_model_only=True,
            exclude_from_loading=["dataloader"],
        ),
        loss=LossConfig(losses=[{"name": "cross_entropy", "weight": 1.0}]),
    )


def llama3_7B_aya_sft() -> Trainer.Config:
    """test3 (en_ar): no augmentation, English + Arabic Aya data, 65k_paired tokenizer."""
    return _aya_sft_config("65k_paired", "llama3_7b_aya_sft")


def llama3_7B_aya_sft_en_translated_ar() -> Trainer.Config:
    """test2 (en_translated_ar): Arabic words mapped to English before tokenizing (prob=1.0).

    Matches pretraining: WordwiseUnigramCodeSwitching with the same dictionary and
    dataset_name key used during pretraining of the en_translated_ar checkpoint.
    """
    return _aya_sft_config("65k_paired", "llama3_7b_aya_sft_en_translated_ar", sources=[
        {"dataset": "aya_dataset", "language_code": "eng", "weight": 1.0},
        {"dataset": "aya_dataset", "language_code": "arb", "weight": 1.0,
         "augmentations": [{
             "name": "wordwise_unigram_codeswitching",
             "prob": 1.0,
             "fallback_to_transliteration": True,
             "dict_paths": {
                 "fineweb-edu-ar-ar": f"{_PROJECT_ROOT}/tests/assets/translations/top_arabic_translated_fineweb_newregex_1to1.json",
             },
         }]},
    ])


def llama3_7B_aya_sft_en_anchored_ar() -> Trainer.Config:
    """test6 (en_anchored_ar): Arabic token IDs remapped to shared English anchor IDs.

    Matches pretraining: SharedAnchorRemap applied after tokenization using the
    same 1-to-1 token map used during pretraining of the en_anchored_ar checkpoint.
    """
    return _aya_sft_config("65k_paired", "llama3_7b_aya_sft_en_anchored_ar", sources=[
        {"dataset": "aya_dataset", "language_code": "eng", "weight": 1.0},
        {"dataset": "aya_dataset", "language_code": "arb", "weight": 1.0,
         "post_token_augmentations": [{
             "name": "shared_anchor_remap",
             "map_path": f"{_PROJECT_ROOT}/torchtitan/tests/assets/translations/ar_en_1to1_token_map.json",
         }]},
    ])


def llama3_7B_aya_sft_ru() -> Trainer.Config:
    """test4 (en_ru): no augmentation, English + Russian Aya data, 65k_en1.0_ru1.0 tokenizer."""
    return _aya_sft_config("65k_en1.0_ru1.0", "llama3_7b_aya_sft_ru", sources=[
        {"dataset": "aya_dataset", "language_code": "eng", "weight": 1.0},
        {"dataset": "aya_dataset", "language_code": "rus", "weight": 1.0},
    ])


def llama3_7B_aya_sft_en_translated_ru() -> Trainer.Config:
    """test5 (en_translated_ru): Russian words mapped to English before tokenizing (prob=1.0).

    Matches pretraining: WordwiseUnigramCodeSwitching with the same dictionary and
    dataset_name key used during pretraining of the en_translated_ru checkpoint.
    """
    return _aya_sft_config("65k_en1.0_ru1.0", "llama3_7b_aya_sft_en_translated_ru", sources=[
        {"dataset": "aya_dataset", "language_code": "eng", "weight": 1.0},
        {"dataset": "aya_dataset", "language_code": "rus", "weight": 1.0,
         "augmentations": [{
             "name": "wordwise_unigram_codeswitching",
             "prob": 1.0,
             "fallback_to_transliteration": True,
             "dict_paths": {
                 "fineweb2-hq-ru": f"{_PROJECT_ROOT}/torchtitan/tests/assets/translations/top_russian_translated_fineweb_newregex_1to1.json",
             },
         }]},
    ])
