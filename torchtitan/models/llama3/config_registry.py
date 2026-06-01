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
def smollm2_360m_flex_curriculum() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_137_ar_tr2en_stage1_4k_0.5ar_0.5tr2en_injection_0_20_100_1000_20800entities_fix",
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
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
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
            folder=".outputs/smollm2_360m_test1_en1_en2_2xvocab_stage1_4k_clean_injection_0_20_100_1000_20800entities",
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
def smollm2_360m_en1_en2_4nodes() -> Trainer.Config:
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
def llama3_7B_en1_en2() -> Trainer.Config:
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] / 24 #total 0, 20, 100, 1000 injections
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
                {
                    "steps": 33400,
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
                            "start_idx": 80_000_000,
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
            seq_len=8192,
            steps=33400,
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
            folder=".outputs/llama3_7B_test1_en1_en2_2xvocab_stage1_34k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
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
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] / 24 #total 0, 20, 100, 1000 injections
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
                    "steps": 25050,
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
                            "injection_probs": [prob *5/2 for prob in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.2,
                            "start_idx": 96_000_000,
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
                    "steps": 8350,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 120_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "start_idx": 140_000_000,
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
            local_batch_size=4,       # 32 * 4 GPUs * 2048 seq_len = 262,144 tokens
            global_batch_size=512,     # Effective batch size of 1 million tokens
            seq_len=8192,
            steps=33400,                 # 6000 steps aligns perfectly with Chinchilla
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
            folder=".outputs/llama3_7B_test2_en1_en2_2xvocab_stage1_25k_0.6codeswitching_stage2_9k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False,
            async_mode="async",
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

def llama3_7B_curriculum_barebones() -> Trainer.Config:
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
def smollm2_360m_flex_curriculum_en1_en2_contrastive() -> Trainer.Config:
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
            folder="/home/adamga/leshemg/adamga/train/torchtitan/smollm2_360m_flex_curriculum_128_en1_en2_stage1_4k_merge3_wordwisecontrastive_0.8_layer4_contrastive_loss_1.0_t0.05_injection_0_20_100_1000_2080entities",
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
                        "temperature": 0.05,
                        "learnable_temp": True
                    }
                }
            ]
        )
    )