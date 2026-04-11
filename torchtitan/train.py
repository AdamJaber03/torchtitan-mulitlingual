# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer

# --- ADD THIS IMPORT ---
from torchtitan.datasets import build_mixed_dataloader
from torchtitan.datasets import build_mixed_sbatch_dataloader
from torchtitan.datasets import build_injection_dataloader
from torchtitan.datasets import build_mixed_preload_dataloader
from torchtitan.datasets import build_aligned_bilingual_dataloader
from torchtitan.datasets import build_aligned_multilingual_packed_dataloader

def main() -> None:
    """Main entry point for training."""
    init_logger()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Trainer | None = None

    try:
        # pyrefly: ignore [missing-attribute]
        if config.comm.mode == "local_tensor":
            logger.info("Local tensor mode enabled - skipping training execution")
            return

        # 1. Build the default Trainer
        # pyrefly: ignore [missing-attribute]
        trainer = config.build()
        
        # --- THE DATALOADER INJECTION ---
        # 2. Overwrite the slow HuggingFace dataloader with our fast binary loader
        logger.info("Injecting custom binary Multi-Dataset DataLoader...")
        # trainer.dataloader = build_mixed_dataloader(
        #     batch_size=trainer.config.training.local_batch_size,
        #     seq_len=trainer.config.training.seq_len
        # )
        # trainer.dataloader = build_aligned_multilingual_packed_dataloader(
        #     batch_size=trainer.config.training.local_batch_size,
        #     seq_len=trainer.config.training.seq_len
        # )
        # trainer.dataloader = build_aligned_bilingual_dataloader(
        #     batch_size=trainer.config.training.local_batch_size,
        #     seq_len=trainer.config.training.seq_len
        # )
        # trainer.dataloader = build_mixed_preload_dataloader(
        #     batch_size=trainer.config.training.local_batch_size,
        #     seq_len=trainer.config.training.seq_len
        # )
        # trainer.dataloader = build_mixed_sbatch_dataloader(
        #     batch_size=trainer.config.training.local_batch_size,
        #     seq_len=trainer.config.training.seq_len
        # )
        # trainer.dataloader = build_injection_dataloader(
        #     batch_size=trainer.config.training.local_batch_size,
        #     seq_len=trainer.config.training.seq_len,
        #     total_training_sequences=trainer.config.training.steps * trainer.config.training.local_batch_size * int(os.environ["WORLD_SIZE"])
        # )
        # --------------------------------

        # pyrefly: ignore [missing-attribute]
        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                # pyrefly: ignore [missing-attribute]
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            # 3. Start training!
            trainer.train()
            
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()