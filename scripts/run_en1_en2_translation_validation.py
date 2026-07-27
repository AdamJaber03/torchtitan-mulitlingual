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
from torchtitan.experiments.en1_en2_translation import (
    EXPECTED_TRANSLATION_LOSSES,
    prepare_local_step,
    read_step_metrics,
)
from torchtitan.tools.logging import init_logger, logger


_STEP_ERROR = -1
_STEP_SKIP = 0
_STEP_RUN = 1


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


def _broadcast_rank0_status(status: int, device: torch.device) -> int:
    if not torch.distributed.is_initialized():
        return status

    status_tensor = torch.tensor(
        [status],
        dtype=torch.int64,
        device=device,
    )
    torch.distributed.broadcast(status_tensor, src=0)
    return int(status_tensor.item())


def main() -> None:
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-output-dir",
        default=os.environ.get("EN1_EN2_TRANSLATION_VAL_OUTPUT_DIR"),
    )
    parser.add_argument(
        "--existing-policy",
        choices=("skip-complete", "error", "overwrite"),
        default=os.environ.get(
            "TRANSLATION_EXISTING_POLICY", "skip-complete"
        ),
    )
    args, config_args = parser.parse_known_args()

    os.environ.setdefault("EN1_EN2_TRANSLATION_VALIDATION_ENABLE", "1")
    if (
        os.environ.get("EN1_EN2_REQUIRE_WANDB_RUN_ID") == "1"
        and not os.environ.get("WANDB_RUN_ID")
    ):
        raise RuntimeError(
            "WANDB_RUN_ID is required for post-training translation validation."
        )

    config = ConfigManager().parse_args(config_args)
    config.validator.enable = True
    translation_only = os.environ.get("EN1_EN2_TRANSLATION_ONLY", "1")
    if translation_only == "1":
        if not isinstance(config.validator.dataloader, dict):
            raise RuntimeError(
                "Translation-only evaluation requires named dataloaders."
            )
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

    output_dir = Path(
        args.local_output_dir
        or Path(config.checkpoint.folder) / "translation_validation_eval"
    )
    checkpoint_step = config.checkpoint.load_step

    trainer = None
    try:
        trainer = config.build()
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )

        step_status = _STEP_RUN
        if rank == 0 and checkpoint_step != -1:
            try:
                if not prepare_local_step(
                    output_dir, checkpoint_step, args.existing_policy
                ):
                    step_status = _STEP_SKIP
            except Exception:
                logger.exception(
                    "Failed to prepare local translation output for step %s",
                    checkpoint_step,
                )
                step_status = _STEP_ERROR

        step_status = _broadcast_rank0_status(step_status, trainer.device)
        if step_status == _STEP_ERROR:
            raise RuntimeError(
                f"Rank 0 could not prepare translation output for step "
                f"{checkpoint_step}."
            )
        if step_status == _STEP_SKIP:
            if rank == 0:
                logger.info(
                    "Skipping complete translation validation step %s",
                    checkpoint_step,
                )
            return

        _attach_local_logger(trainer, output_dir)

        loaded = trainer.checkpointer.load(step=checkpoint_step)
        if not loaded:
            raise RuntimeError("No checkpoint was loaded; set --checkpoint.load_step.")

        if checkpoint_step == -1:
            checkpoint_step = trainer.checkpointer._find_load_step()
        logger.info("Running translation validation at checkpoint step %s", checkpoint_step)
        trainer.validator.validate(trainer.model_parts, checkpoint_step)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        step_status = _STEP_RUN
        if rank == 0:
            written_losses = (
                EXPECTED_TRANSLATION_LOSSES
                & set(read_step_metrics(output_dir, checkpoint_step))
            )
            if written_losses != EXPECTED_TRANSLATION_LOSSES:
                logger.error(
                    "Translation validation step %s is incomplete; found %s.",
                    checkpoint_step,
                    sorted(written_losses),
                )
                step_status = _STEP_ERROR

        step_status = _broadcast_rank0_status(step_status, trainer.device)
        if step_status == _STEP_ERROR:
            raise RuntimeError(
                f"Translation validation step {checkpoint_step} did not write "
                "the complete local metric set."
            )
        if rank == 0:
            logger.info("Wrote local validation metrics under %s", output_dir)
    finally:
        if trainer is not None:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
