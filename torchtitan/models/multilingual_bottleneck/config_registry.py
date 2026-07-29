# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
    LossConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry
import random

def bottleneck_160m() -> Trainer.Config:
    return Trainer.Config(
        # Point to your custom tokenizer
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired", 
        
        # Reference the model shape from __init__.py
        model_spec=model_registry("160M"), 
        
        # DDP OVER FSDP: Replicate the entire 160M model on all 4 GPUs
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=4,  
            data_parallel_shard_degree=1,      
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        
        # Learning Rate for a 160M model
        optimizer=OptimizersContainer.Config(lr=6e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
        
        training=TrainingConfig(
            local_batch_size=108, # 108 local * 4 GPUs = 432 Global Batch Size
            global_batch_size=864,
            seq_len=1024,
            steps=3000,
        ),
        
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=16,
            stages=[
                {
                    "steps": 2000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "lang_id": 0,
                            # "start_idx": 700_000
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "lang_id": 1,
                            # "start_idx": 700_000
                        },
                    ],
                },
            ],
        ),

        # Enable torch.compile for maximum speed via your parallelize_bottleneck.py
        compile=CompileConfig(enable=True),
        
        # Tracking via WandB
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        
        checkpoint=CheckpointManager.Config(
            interval=1000, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/bottleneck_160m_01_test",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False, # Must be False until a custom StateDictAdapter is written
            async_mode="async"
        ),
    )
def bottleneck_360m_k1() -> Trainer.Config:
    # base_probs = [0.00002055, 0.00006165, 0.0001233] #total 100,300,600 injections
    base_probs = [0, 0.00000274, 0.0000137, 0.0000548, 0.000137] # 6k steps - total 0, 20, 100, 400, 1000 injections
    file_order_shuffler = random.Random(43)
    # file_order = []
    # for start in range(0, 1408, 32):
    #     chunk = list(range(start, start + 32))
    #     file_order_shuffler.shuffle(chunk)
    #     file_order.extend(chunk)
    file_order = list(range(1632))
    file_order_shuffler.shuffle(file_order)
    ar_files = [f"/home/adamga/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    en_files = [f"/home/adamga/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(1632)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(1632)]
    
    ar_tokenizer_path = "/home/adamga/torchtitan/tests/assets/65k_en0.0_ar1.0_paired_data"
    en_tokenizer_path = "/home/adamga/torchtitan/tests/assets/65k_en1.0_ar0.0_paired_data"
    # ar_tokenizer_path = "/home/adamga/torchtitan/tests/assets/65k_paired"
    # en_tokenizer_path = "/home/adamga/torchtitan/tests/assets/65k_paired"
    return Trainer.Config(
        # Point to your custom tokenizer
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired", 
        
        # Reference the model shape from __init__.py
        model_spec=model_registry("360M_k1"), 

        # DDP OVER FSDP: Replicate the entire 360M model on all 8 GPUs
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=8,  
            data_parallel_shard_degree=1,      
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),

        # Learning Rate for a 360M model
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
        
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=16,
            monolingual_batches=True, 
            stages=[
                # {
                    # "steps": 2000,
                    # "sources": [
                    #     {
                    #         "name": "fineweb-edu-ar-ar",
                    #         "weight": 0.5,
                    #         "lang_id": 1,
                    #         "injection_paths": ar_files,
                    #         "injection_probs": ar_probs,
                    #     },
                    #     {
                    #         "name": "fineweb-edu-ar-en",
                    #         "weight": 0.5,
                    #         "lang_id": 0,
                    #         "injection_paths": en_files,
                    #         "injection_probs": en_probs,
                    #     }
                    # ],
                    # "augmentations": [
                    #     {
                    #         "name": "wordwise_codeswitching",
                    #         "prob": 0.3,  # 30% of the time, apply
                    #         "dict_paths": {
                    #             "fineweb-edu-ar-ar": "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated.json",
                    #             "fineweb-edu-ar-en": "/home/adamga/leshemg/adamga/data/translations/top_english_translated.json"
                    #         }
                    #     }
                    # ]
                # },
                {
                    "steps": 6000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            "lang_id": 1,
                            # "start_idx": 1_400_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                            "tokenizer": ar_tokenizer_path,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "lang_id": 0,
                            # "start_idx": 1_400_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                            "tokenizer": en_tokenizer_path,
                        }
                    ], 
                },
            ],
        ),

        # Enable torch.compile for maximum speed via your parallelize_bottleneck.py
        compile=CompileConfig(enable=True),
        
        # Tracking via WandB
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/bottleneck_360m_10_stage1_6k_seperate_tokenizer_injection_0_20_100_400_1000_rand_shuffle_1632_entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False, # Must be False until a custom StateDictAdapter is written
            async_mode="async"
        ),
    )

def bottleneck_360m_k1_sep_embeddings() -> Trainer.Config:
    # base_probs = [0.00002055, 0.00006165, 0.0001233] #total 100,300,600 injections
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections (for 4k)

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
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    
    ar_tokenizer_path = "/home/adamga/torchtitan/tests/assets/65k_en0.0_ar1.0_paired_data"
    en_tokenizer_path = "/home/adamga/torchtitan/tests/assets/65k_en1.0_ar0.0_paired_data"
    # ar_tokenizer_path = "/home/adamga/torchtitan/tests/assets/65k_paired"
    # en_tokenizer_path = "/home/adamga/torchtitan/tests/assets/65k_paired"
    return Trainer.Config(
        # Point to your custom tokenizer
        hf_assets_path="/home/adamga/torchtitan/tests/assets/65k_paired", 
        
        # Reference the model shape from __init__.py
        model_spec=model_registry("360M_k1_sep_embeddings"), 

        # DDP OVER FSDP: Replicate the entire 360M model on all 8 GPUs
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=2,  
            data_parallel_shard_degree=1,      
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),

        # Learning Rate for a 360M model
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
            steps=4000,                 # 8000 steps aligns perfectly with Chinchilla
            max_norm=1.0,               # Gradient clipping 
        ),
        
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=16,
            monolingual_batches=True, 
            stages=[
                {
                    "steps": 4600,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "lang_id": 1,
                            # "start_idx": 1_400_000,
                            "injection_paths": en_files,
                            "injection_probs": ar_probs,
                            "tokenizer": en_tokenizer_path,
                        },
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "lang_id": 0,
                            "start_idx": 4_000_000,
                            "injection_paths": en_files,
                            "injection_probs": en_probs,
                            "tokenizer": en_tokenizer_path,
                        }
                    ], 
                },
            ],
        ),

        # Enable torch.compile for maximum speed via your parallelize_bottleneck.py
        compile=CompileConfig(enable=True),
        
        # Tracking via WandB
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        
        checkpoint=CheckpointManager.Config(
            interval=500, 
            folder="/home/adamga/leshemg/adamga/train/torchtitan/bottleneck_360m_14_stage1_4k_en_en_sep_embeddings_sameinit_injection_0_20_100_1000_2080_entities",
            enable=True,
            enable_first_step_checkpoint=True,
            last_save_in_hf=False, # Must be False until a custom StateDictAdapter is written
            async_mode="async"
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