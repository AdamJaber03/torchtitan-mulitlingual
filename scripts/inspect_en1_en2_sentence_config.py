#!/usr/bin/env python3
"""Validate and serialize the effective en1/en2 sentence experiment config."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from torchtitan.config import ConfigManager


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _probability_table(source: dict[str, Any]) -> list[dict[str, float]]:
    grouped: dict[float, dict[str, set[float]]] = {}
    for target, expected, probability in zip(
        source["injection_target_counts"],
        source["injection_expected_counts"],
        source["injection_probs"],
        strict=True,
    ):
        bucket = grouped.setdefault(
            float(target),
            {"expected": set(), "probability": set()},
        )
        bucket["expected"].add(float(expected))
        bucket["probability"].add(float(probability))

    table = []
    for target, values in sorted(grouped.items()):
        if len(values["expected"]) != 1 or len(values["probability"]) != 1:
            raise RuntimeError(
                f"Target bucket {target:g} does not have one expected count "
                "and one probability."
            )
        table.append(
            {
                "target_count": target,
                "expected_stage_count": next(iter(values["expected"])),
                "probability": next(iter(values["probability"])),
            }
        )
    return table


def _validate_source(source: dict[str, Any], check_paths: bool) -> dict[str, Any]:
    paths = source.get("injection_paths", [])
    probabilities = source.get("injection_probs", [])
    targets = source.get("injection_target_counts", [])
    expected = source.get("injection_expected_counts", [])
    if not paths:
        raise RuntimeError("Expected an injected clean source.")
    if not (len(paths) == len(probabilities) == len(targets) == len(expected)):
        raise RuntimeError("Injection source arrays have inconsistent lengths.")
    total_probability = sum(probabilities)
    if not 0 <= total_probability < 1:
        raise RuntimeError(
            f"Invalid total injection probability {total_probability:g}."
        )

    if check_paths:
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            shown = "\n".join(missing[:20])
            extra = (
                ""
                if len(missing) <= 20
                else f"\n... and {len(missing) - 20} more"
            )
            raise FileNotFoundError(f"Missing injection files:\n{shown}{extra}")

    return {
        "language": source["injection_summary_name"],
        "weight": source["weight"],
        "entity_count": len(paths),
        "total_injection_probability": total_probability,
        "probability_table": _probability_table(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-path-check", action="store_true")
    args = parser.parse_args()

    config = ConfigManager().parse_args(
        ["--module", "llama3", "--config", args.config]
    )
    stages = []
    for stage_index, stage in enumerate(config.dataloader.stages):
        source_weight = sum(source["weight"] for source in stage["sources"])
        if abs(source_weight - 1.0) > 1e-9:
            raise RuntimeError(
                f"Stage {stage_index} source weights sum to {source_weight:g}."
            )
        injected_sources = [
            _validate_source(source, not args.skip_path_check)
            for source in stage["sources"]
            if source.get("injection_paths")
        ]
        if len(injected_sources) != 2:
            raise RuntimeError(
                f"Stage {stage_index} must contain en1 and en2 injection sources."
            )
        stages.append(
            {
                "stage": stage_index,
                "steps": stage["steps"],
                "sources": injected_sources,
            }
        )

    payload = {
        "git_sha": _git_sha(),
        "config": args.config,
        "experiment_name": Path(config.checkpoint.folder).name,
        "checkpoint_folder": config.checkpoint.folder,
        "global_batch_size": config.training.global_batch_size,
        "local_batch_size": config.training.local_batch_size,
        "sequence_length": config.training.seq_len,
        "training_steps": config.training.steps,
        "stages": stages,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
