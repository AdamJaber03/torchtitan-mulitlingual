import argparse
import os
import random
from pathlib import Path

import yaml


DEFAULT_RATES = [0, 20, 100, 1000]
DEFAULT_ENTITY_COUNT = 2080
DEFAULT_SEED = 43
DEFAULT_MCQ_FILES = ["mcq_en"]
DEFAULT_GROUP_NAME = "fictive_entity_en1_en2_mcq_en_all_rates"


def default_multilingual_root() -> Path:
    root = os.environ.get("MULTILINGUAL_PRETRAINING_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = default_multilingual_root()
    default_data_dir = root / "fictional_entity_data" / "gemini_seeds"
    default_output_dir = (
        root
        / "evals"
        / "knowledge_sharing"
        / "lm_eval_tasks"
        / "en1_en2_mcq_en"
    )

    parser = argparse.ArgumentParser(
        description="Generate lm-eval MCQ tasks for en1/en2 fictional-entity recall."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing numbered fictional-entity folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where lm-eval task YAML files will be written.",
    )
    parser.add_argument(
        "--group-name",
        default=DEFAULT_GROUP_NAME,
        help="Name of the generated lm-eval task group.",
    )
    parser.add_argument(
        "--mcq-files",
        nargs="+",
        default=DEFAULT_MCQ_FILES,
        help="MCQ jsonl basenames to include, for example: mcq_en mcq_ar.",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=int,
        default=DEFAULT_RATES,
        help="Injection-rate labels used to form the en1/en2 task grid.",
    )
    parser.add_argument(
        "--entity-count",
        type=int,
        default=DEFAULT_ENTITY_COUNT,
        help="Number of shuffled entity folders to use.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for shuffling entity folders; keep at 43 to match training.",
    )
    parser.add_argument(
        "--doc-to-text",
        default="{{question}} ",
        help="lm-eval doc_to_text template.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing MCQ files instead of failing.",
    )
    return parser.parse_args()


def mcq_filename(name: str) -> str:
    return name if name.endswith(".jsonl") else f"{name}.jsonl"


def task_safe_name(name: str) -> str:
    return name.removesuffix(".jsonl").replace("-", "_")


def build_entity_paths(base_dir: Path, entity_count: int, seed: int) -> list[Path]:
    order = list(range(entity_count))
    random.Random(seed).shuffle(order)
    return [base_dir / str(idx) for idx in order]


def main() -> None:
    args = parse_args()
    rates = args.rates
    grid_size = len(rates) ** 2
    if args.entity_count % grid_size != 0:
        raise ValueError(
            f"--entity-count must be divisible by len(--rates)^2; "
            f"got {args.entity_count} and grid size {grid_size}."
        )

    entity_paths = build_entity_paths(
        args.base_dir.expanduser().resolve(),
        args.entity_count,
        args.shuffle_seed,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_task_names: list[str] = []
    missing_paths: list[Path] = []

    for group_idx in range(grid_size):
        en1_rate = rates[group_idx % len(rates)]
        en2_rate = rates[group_idx // len(rates)]
        combo_name = f"en1_{en1_rate}_en2_{en2_rate}"
        group_entity_paths = [
            entity_paths[group_idx + (grid_size * offset)]
            for offset in range(args.entity_count // grid_size)
        ]

        for mcq_name in args.mcq_files:
            filename = mcq_filename(mcq_name)
            data_files = [path / filename for path in group_entity_paths]
            missing_for_task = [path for path in data_files if not path.exists()]
            if missing_for_task:
                missing_paths.extend(missing_for_task)
                if args.allow_missing:
                    data_files = [path for path in data_files if path.exists()]
                else:
                    continue

            task_name = f"fictive_{combo_name}_{task_safe_name(mcq_name)}"
            all_task_names.append(task_name)

            config = {
                "task": task_name,
                "dataset_path": "json",
                "dataset_kwargs": {
                    "data_files": {
                        "test": [str(path) for path in data_files],
                    }
                },
                "output_type": "multiple_choice",
                "training_split": None,
                "validation_split": None,
                "test_split": "test",
                "doc_to_text": args.doc_to_text,
                "doc_to_target": "answer_index",
                "doc_to_choice": "{{choices}}",
                "metric_list": [
                    {
                        "metric": "acc",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    },
                    {
                        "metric": "acc_norm",
                        "aggregation": "mean",
                        "higher_is_better": True,
                    },
                ],
            }

            with (output_dir / f"{task_name}.yaml").open("w") as f:
                yaml.dump(config, f, sort_keys=False)

    if missing_paths and not args.allow_missing:
        shown = "\n".join(str(path) for path in missing_paths[:20])
        extra = "" if len(missing_paths) <= 20 else f"\n... and {len(missing_paths) - 20} more"
        raise FileNotFoundError(f"Missing MCQ files:\n{shown}{extra}")

    group_config = {
        "group": args.group_name,
        "task": all_task_names,
    }
    with (output_dir / "_group.yaml").open("w") as f:
        yaml.dump(group_config, f, sort_keys=False)

    print(
        f"Generated {len(all_task_names)} tasks and group '{args.group_name}' "
        f"in {output_dir}"
    )


if __name__ == "__main__":
    main()
