"""Retry-safe local result handling for en1/en2 translation validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from torchtitan.tools.logging import logger


EXPECTED_TRANSLATION_LOSSES = {
    "validation_metrics/translation_en1_to_en2/loss",
    "validation_metrics/translation_en2_to_en1/loss",
}


def read_step_metrics(output_dir: Path, step: int) -> dict[str, float]:
    csv_path = output_dir / "validation_metrics.csv"
    if not csv_path.exists():
        return {}

    metrics: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["step"]) != step:
                continue
            metric = row["metric"]
            if metric in metrics:
                raise RuntimeError(
                    f"Duplicate local validation row for step={step}, metric={metric!r}."
                )
            metrics[metric] = float(row["value"])
    return metrics


def remove_step(output_dir: Path, step: int) -> None:
    csv_path = output_dir / "validation_metrics.csv"
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = [
                row
                for row in csv.DictReader(f)
                if int(row["step"]) != step
            ]
        temporary_path = csv_path.with_suffix(".csv.tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "metric", "value"])
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(csv_path)

    jsonl_path = output_dir / "validation_metrics.jsonl"
    if jsonl_path.exists():
        records = []
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record["step"]) != step:
                    records.append(record)
        temporary_path = jsonl_path.with_suffix(".jsonl.tmp")
        with temporary_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        temporary_path.replace(jsonl_path)


def prepare_local_step(
    output_dir: Path,
    step: int,
    existing_policy: str,
) -> bool:
    metrics = read_step_metrics(output_dir, step)
    existing_losses = EXPECTED_TRANSLATION_LOSSES & set(metrics)
    if not metrics:
        return True
    if existing_policy == "overwrite":
        remove_step(output_dir, step)
        return True
    if (
        existing_policy == "skip-complete"
        and existing_losses == EXPECTED_TRANSLATION_LOSSES
    ):
        logger.info(
            "Translation validation step %s is already complete; skipping.",
            step,
        )
        return False
    if existing_policy == "error":
        raise RuntimeError(
            f"Local translation output already contains checkpoint step {step}."
        )
    raise RuntimeError(
        f"Local translation output for step {step} is partial: "
        f"found losses {sorted(existing_losses)}. Use "
        "TRANSLATION_EXISTING_POLICY=overwrite to recompute this step."
    )
