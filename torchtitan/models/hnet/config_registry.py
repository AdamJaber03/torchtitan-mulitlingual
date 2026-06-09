# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Training configs for byte-level H-Net. Launch with, e.g.:
#   MODULE=hnet CONFIG=hnet_debugmodel NGPU=1 ./run_train.sh

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.tokenizer import ByteTokenizer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    LossConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry
from .optimizer import HNetOptimizersContainer
import random

# Byte-level: cross-entropy on the next byte + H-Net dynamic-chunking ratio loss.
_HNET_LOSS = LossConfig(
    losses=[
        {"name": "cross_entropy", "weight": 1.0},
        {"name": "hnet_ratio", "weight": 0.03},
    ]
)


def hnet_debugmodel() -> Trainer.Config:
    """Tiny 1-stage byte H-Net for forward/backward smoke testing (FSDP-only)."""
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",  # unused by ByteTokenizer
        tokenizer=ByteTokenizer.Config(vocab_size=256, eos_id=255),
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
            seq_len=1024,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            # This fork's loader is a multi-dataset loader driven by stages/sources.
            num_workers=2,  # must be > 0 (loader divides start_idx by num_workers)
            stages=[{"steps": 10, "sources": [{"name": "c4", "weight": 1.0}]}],
            eos_token_id=255,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        # H-Net does not support PP/TP/CP; keep parallelism on data-parallel.
        parallelism=ParallelismConfig(),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
        loss=_HNET_LOSS,
    )


def hnet_1stage_small() -> Trainer.Config:
    """Small real 1-stage byte H-Net on C4 (FSDP / HSDP)."""
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",  # unused by ByteTokenizer
        tokenizer=ByteTokenizer.Config(vocab_size=256, eos_id=255),
        model_spec=model_registry("1stage_small"),
        optimizer=OptimizersContainer.Config(lr=3e-4, weight_decay=0.1),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=500,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=8192,  # bytes — ~8KB of text per sequence
            steps=20000,
            max_norm=1.0,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=8,
            stages=[{"steps": 20000, "sources": [{"name": "c4", "weight": 1.0}]}],
            eos_token_id=255,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10,
        ),
        compile=CompileConfig(enable=False),  # dynamic chunking -> dynamic shapes
        activation_checkpoint=ActivationCheckpointConfig(mode="selective", selective_ac_option="op"),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            folder=".outputs/hnet_1stage_small_test1_en",
            enable=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        loss=_HNET_LOSS,
    )


def hnet_1stage_m() -> Trainer.Config:
    """~340M-param 1-stage byte H-Net, faithful to the authors' hnet_1stage_L
    design (m4 Mamba enc/dec + pure-Transformer main, d=1024/1536), scaled to
    a T10 main. Trained byte-level on FineWeb-Edu (the paper's data)."""
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",  # unused by ByteTokenizer
        tokenizer=ByteTokenizer.Config(vocab_size=256, eos_id=255),
        model_spec=model_registry("1stage_M"),
        optimizer=OptimizersContainer.Config(lr=3e-4, weight_decay=0.1),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1000,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=8192,  # bytes
            steps=50000,
            max_norm=1.0,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=8,
            # The paper trains on FineWeb-Edu; swap to "c4" if that source is
            # unavailable in your environment.
            stages=[{"steps": 50000, "sources": [{"name": "fineweb-edu-100b-shuffle", "weight": 1.0}]}],
            eos_token_id=255,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10,
        ),
        compile=CompileConfig(enable=False),  # dynamic chunking -> dynamic shapes
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective", selective_ac_option="op"
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            folder=".outputs/hnet_1stage_M",
            enable=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        loss=_HNET_LOSS,
    )


def hnet_1stage_smollm() -> Trainer.Config:
    """~347M-param 1-stage byte H-Net whose main network reuses the
    smollm2_360m backbone (d=960, 32 layers, FFN 2560, 3:1 GQA, RoPE theta
    10000), with Mamba-2 byte-level encoder/decoder (d=768)."""
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",  # unused by ByteTokenizer
        tokenizer=ByteTokenizer.Config(vocab_size=256, eos_id=255),
        model_spec=model_registry("1stage_smollm"),
        optimizer=OptimizersContainer.Config(lr=5e-4, weight_decay=0.1),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=300,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=0.05,
        ),
        training=TrainingConfig(
            local_batch_size=12,
            global_batch_size=768,
            seq_len=8192,  # bytes
            steps=4000,
            max_norm=1.0,
        ),
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
                            "start_idx": 3_200_000,
                            "injection_paths": ar_files,
                            "injection_probs": ar_probs,
                        },
                    ],
                }
            ],
            eos_token_id=255
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        compile=CompileConfig(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective", selective_ac_option="op"
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            folder="/home/adamga/leshemg/adamga/train/torchtitan/hnet_1stage_smollm2_test1_en_ar_stage1_4k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        loss=_HNET_LOSS,
    )


def hnet_1stage_smollm_lropt() -> Trainer.Config:
    """Same as `hnet_1stage_smollm` but with the H-Net-aware optimizer:
    per-stage LR multipliers (byte-level enc/dec/routing + embeddings/lm_head at
    1.5x the main-network LR) and weight-decay decoupling (no wd on norms,
    biases, and Mamba A_log/D/dt_bias), following arXiv:2507.07955 and the
    authors' `apply_lr_multiplier` / `group_params`.

    Additive: reuses `hnet_1stage_smollm` and only swaps the optimizer + the
    checkpoint folder, so the original config and all other models/configs are
    unaffected. Tune `lr_multipliers` (outer-first; paper uses notably higher
    outer LR, ~2-3x for 2-stage) and consider bumping `lr` toward ~6.25e-4.
    """
    config = hnet_1stage_smollm()
    config.optimizer = HNetOptimizersContainer.Config(
        lr=6.25e-4,
        weight_decay=0.1,
        lr_multipliers=[2.0, 1.0],  # [outer byte stage, inner main]; 2 entries for 1-stage
        decouple_weight_decay=True,
    )
    config.compile = CompileConfig(enable=True)
    config.checkpoint.folder = "/home/adamga/leshemg/adamga/train/torchtitan/hnet_1stage_smollm2_test5_lropt_en_ar_stage1_4k_clean_injection_0_20_100_1000_20800entities"
    return config


def hnet_1stage_L_paper() -> Trainer.Config:
    """The paper's SMALLEST model — 1-stage Large (~679M, the "Large" scale,
    FLOP-matched to GPT-3 Large) — trained with the H-Net paper's hyperparameters
    (arXiv:2507.07955), on the SAME dataset as `hnet_1stage_smollm`.

    Paper hyperparameters reproduced directly:
      - peak LR 6.25e-4 (Large scale)
      - AdamW (betas 0.9/0.95) with weight-decay decoupling on biases + norms
        (+ Mamba A_log/D/dt_bias), matching the authors' `group_params`
      - WSD schedule: 10% linear warmup, then 20% inverse-sqrt (1-sqrt) decay
      - batch size 256 sequences x 8192 utf-8 bytes
      - gradient clipping (max_norm) 1.0
      - dynamic-chunking ratio-loss alpha = 0.03 (in _HNET_LOSS)
    """
    base_probs = [0, 0.00000411, 0.00002055, 0.0002055] #total 0, 20, 100, 1000 injections
    file_order_shuffler = random.Random(43)
    file_order = list(range(2080))
    file_order_shuffler.shuffle(file_order)
    en_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/en_data.jsonl" for i in file_order]
    ar_files = [f"/home/adamga/torchtitan/fictional_entity_data/gemini_seeds/{i}/ar_data.jsonl" for i in file_order]
    ar_probs = [base_probs[(i // len(base_probs)) % len(base_probs)] for i in range(2080)]
    en_probs = [base_probs[i % len(base_probs)] for i in range(2080)]
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",  # unused by ByteTokenizer
        tokenizer=ByteTokenizer.Config(vocab_size=256, eos_id=255),
        model_spec=model_registry("1stage_L"),
        optimizer=HNetOptimizersContainer.Config(
            lr=6.25e-4,
            weight_decay=0.1,
            lr_multipliers=[2.0, 1.0],
            decouple_weight_decay=True,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=800,
            decay_ratio=0.20,
            decay_type="sqrt",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            global_batch_size=256,
            seq_len=8192,  # bytes
            steps=24000,
            max_norm=1.0,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            num_workers=3,
            stages=[
                {
                    "steps": 24000,
                    "sources": [
                        {
                            "name": "fineweb-edu-ar-en",
                            "weight": 0.5,
                            "injection_paths": en_files,
                            "injection_probs": [p/2 for p in en_probs],
                        },
                        {
                            "name": "fineweb-edu-ar-ar",
                            "weight": 0.5,
                            # "start_idx": 3_200_000,
                            "injection_paths": ar_files,
                            "injection_probs": [p/2 for p in ar_probs],
                        },
                    ],
                }
            ],
            eos_token_id=255
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=False,
            enable_wandb=True,
            log_freq=10
        ),
        compile=CompileConfig(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective", selective_ac_option="op"
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            folder="/home/adamga/leshemg/adamga/train/torchtitan/hnet_1stage_L_test6_en_ar_stage1_8k_clean_injection_0_20_100_1000_20800entities",
            enable=True,
            last_save_in_hf=False,
            async_mode="async",
        ),
        loss=_HNET_LOSS,
    )