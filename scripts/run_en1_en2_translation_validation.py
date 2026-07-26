#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch

from torchtitan.components.metrics import BaseLogger, LoggerContainer
from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger, logger


class LocalValidationLogger(BaseLogger):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output_dir / "validation_metrics.jsonl"
        self.csv_path = output_dir / "validation_metrics.csv"

    def log(self, metrics: dict[str, Any], step: int) -> None:
        validation_metrics = {
            key: value
            for key, value in metrics.items()
            if key.startswith("validation_metrics/")
        }
        if not validation_metrics:
            return

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "metrics": validation_metrics}) + "\n")

        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "metric", "value"])
            if write_header:
                writer.writeheader()
            for metric, value in sorted(validation_metrics.items()):
                writer.writerow({"step": step, "metric": metric, "value": value})


def _attach_local_logger(trainer, output_dir: Path) -> None:
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    current_logger = trainer.metrics_processor.logger
    local_logger = LocalValidationLogger(output_dir)
    if isinstance(current_logger, LoggerContainer):
        current_logger.add_logger(local_logger)
    else:
        container = LoggerContainer()
        if type(current_logger) is not BaseLogger:
            container.add_logger(current_logger)
        container.add_logger(local_logger)
        trainer.metrics_processor.logger = container


def main() -> None:
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-output-dir",
        default=os.environ.get("EN1_EN2_TRANSLATION_VAL_OUTPUT_DIR"),
    )
    args, config_args = parser.parse_known_args()

    os.environ.setdefault("EN1_EN2_TRANSLATION_VALIDATION_ENABLE", "1")
    if (
        os.environ.get("EN1_EN2_REQUIRE_WANDB_RUN_ID") == "1"
        and not os.environ.get("WANDB_RUN_ID")
    ):
        raise RuntimeError(
            "WANDB_RUN_ID is required for translation validation backfill."
        )

    config = ConfigManager().parse_args(config_args)
    config.validator.enable = True
    if os.environ.get("EN1_EN2_TRANSLATION_ONLY_BACKFILL", "1") == "1":
        if not isinstance(config.validator.dataloader, dict):
            raise RuntimeError("translation-only backfill requires named dataloaders.")
        config.validator.dataloader = {
            name: dataloader
            for name, dataloader in config.validator.dataloader.items()
            if name.startswith("translation_")
        }
        if not config.validator.dataloader:
            raise RuntimeError("no translation validation dataloaders are configured.")
    config.checkpoint.load_only = True
    config.checkpoint.exclude_from_loading = sorted(
        set(config.checkpoint.exclude_from_loading)
        | {"optimizer", "lr_scheduler", "dataloader", "train_state"}
    )

    trainer = None
    try:
        trainer = config.build()
        output_dir = Path(
            args.local_output_dir
            or Path(config.checkpoint.folder) / "translation_validation_eval"
        )
        _attach_local_logger(trainer, output_dir)

        checkpoint_step = config.checkpoint.load_step
        loaded = trainer.checkpointer.load(step=checkpoint_step)
        if not loaded:
            raise RuntimeError("No checkpoint was loaded; set --checkpoint.load_step.")

        if checkpoint_step == -1:
            checkpoint_step = trainer.checkpointer._find_load_step()
        logger.info("Running translation validation at checkpoint step %s", checkpoint_step)
        trainer.validator.validate(trainer.model_parts, checkpoint_step)
        logger.info("Wrote local validation metrics under %s", output_dir)
    finally:
        if trainer is not None:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
