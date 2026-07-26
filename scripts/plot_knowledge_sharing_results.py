#!/usr/bin/env python3
"""Plot knowledge-sharing lm-eval results next to each result JSON.

The plot also reports the directional knowledge-transfer ratio used by the
fictive-entity eval: after fitting score ~ en1_rate + en2_rate + intercept,
KT is own-language slope divided by other-language slope.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"),
)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "text.antialiased": True,
        "lines.antialiased": True,
        "patch.antialiased": True,
    }
)


TASK_RE = re.compile(r"^fictive_en1_(\d+)_en2_(\d+)_mcq_en$")

LINE_COLORS = {
    0: "#4285F4",
    20: "#EA4335",
    100: "#FBBC05",
    1000: "#34A853",
}
AVG_COLOR = "#FF7F0E"


@dataclass(frozen=True)
class KTFit:
    ratio: float
    en1_slope: float
    en2_slope: float
    intercept: float
    own_label: str
    other_label: str
    n_points: int


def default_root() -> Path:
    env_root = os.environ.get("MULTILINGUAL_PRETRAINING_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def result_jsons(paths: list[Path], results_root: Path) -> list[Path]:
    if not paths:
        return sorted(results_root.glob("**/results*.json"))

    found: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_dir():
            found.extend(sorted(path.glob("**/results*.json")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(path)
    return found


def load_points(path: Path, metric: str) -> dict[int, dict[int, float]]:
    with path.open() as handle:
        data = json.load(handle)

    metric_key = f"{metric},none"
    points: dict[int, dict[int, float]] = {}
    for task_name, task_result in data.get("results", {}).items():
        match = TASK_RE.match(task_name)
        if not match:
            continue
        en1_rate = int(match.group(1))
        en2_rate = int(match.group(2))
        if metric_key not in task_result:
            raise KeyError(f"{path}: missing metric {metric_key!r} for {task_name}")
        points.setdefault(en1_rate, {})[en2_rate] = float(task_result[metric_key])

    if not points:
        raise ValueError(f"{path}: no fictive_en1_*_en2_*_mcq_en task results found")
    return points


def fit_knowledge_transfer(points: dict[int, dict[int, float]], view: str) -> KTFit:
    triples = [
        (en1_rate, en2_rate, score)
        for en1_rate, by_en2 in points.items()
        for en2_rate, score in by_en2.items()
    ]
    if len(triples) < 3:
        raise ValueError("At least three rate cells are required to fit KT")

    x = np.array([[en1_rate, en2_rate, 1.0] for en1_rate, en2_rate, _ in triples])
    y = np.array([score for _, _, score in triples])
    en1_slope, en2_slope, intercept = np.linalg.lstsq(x, y, rcond=None)[0]

    if view == "en1":
        own_label, other_label = "en1", "en2"
        own_slope, other_slope = en1_slope, en2_slope
    else:
        own_label, other_label = "en2", "en1"
        own_slope, other_slope = en2_slope, en1_slope

    ratio = own_slope / other_slope if other_slope != 0 else math.nan
    return KTFit(
        ratio=float(ratio),
        en1_slope=float(en1_slope),
        en2_slope=float(en2_slope),
        intercept=float(intercept),
        own_label=own_label,
        other_label=other_label,
        n_points=len(triples),
    )


def format_kt_ratio(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.1f}x"
    if abs_value >= 10:
        return f"{value:.2f}x"
    return f"{value:.3f}x"


def eval_view(path: Path, requested_view: str) -> str:
    if requested_view != "auto":
        return requested_view
    for part in reversed(path.parts):
        if part in {"en1", "en2"}:
            return part
    return "en1"


def plot_result(path: Path, metric: str, kt_metric: str, requested_view: str) -> tuple[Path, KTFit]:
    points = load_points(path, metric)
    en1_rates = sorted(points)
    en2_rates = sorted({rate for values in points.values() for rate in values})
    view = eval_view(path, requested_view)
    kt_points = points if kt_metric == metric else load_points(path, kt_metric)
    kt_fit = fit_knowledge_transfer(kt_points, view)

    label = "Accuracy" if metric == "acc" else "Normalized Accuracy"
    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=180)

    if view == "en1":
        x_rates = en2_rates
        line_rates = en1_rates
        x_label = "En2 Rate"
        line_label = "En1 Rates"

        def y_values(line_rate: int) -> list[float | None]:
            return [points[line_rate].get(x_rate) for x_rate in x_rates]

        def value_at(line_rate: int, x_rate: int) -> float:
            return points[line_rate][x_rate]

        def has_value(line_rate: int, x_rate: int) -> bool:
            return x_rate in points[line_rate]

    else:
        x_rates = en1_rates
        line_rates = en2_rates
        x_label = "En1 Rate"
        line_label = "En2 Rates"

        def y_values(line_rate: int) -> list[float | None]:
            return [points[x_rate].get(line_rate) for x_rate in x_rates]

        def value_at(line_rate: int, x_rate: int) -> float:
            return points[x_rate][line_rate]

        def has_value(line_rate: int, x_rate: int) -> bool:
            return line_rate in points[x_rate]

    for line_rate in line_rates:
        ax.plot(
            x_rates,
            y_values(line_rate),
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=LINE_COLORS.get(line_rate),
            label=str(line_rate),
        )

    avg_values = [
        sum(
            value_at(line_rate, x_rate)
            for line_rate in line_rates
            if has_value(line_rate, x_rate)
        )
        / sum(1 for line_rate in line_rates if has_value(line_rate, x_rate))
        for x_rate in x_rates
    ]
    ax.plot(
        x_rates,
        avg_values,
        marker="o",
        markersize=3,
        linewidth=1.8,
        color=AVG_COLOR,
        label="avg",
    )

    ax.set_title(f"{label} vs {x_label} by {line_label}", fontsize=11, pad=10)
    ax.set_xlabel(x_label, fontsize=8.5)
    ax.set_ylabel(label, fontsize=8.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(min(x_rates), max(x_rates))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.grid(True, color="#B7BEC8", linewidth=0.7, alpha=0.8)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.text(
        0.965,
        0.055,
        f"KT ({kt_metric})\n{kt_fit.own_label}/{kt_fit.other_label}: {format_kt_ratio(kt_fit.ratio)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "#8FA3BF",
            "linewidth": 0.6,
            "alpha": 0.92,
        },
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=min(len(line_rates) + 1, 5),
        frameon=False,
        fontsize=7.5,
        handlelength=1.4,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.36)

    output_path = path.with_suffix(".png")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.07)
    plt.close(fig)
    return output_path, kt_fit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one knowledge-sharing plot PNG next to each lm-eval result JSON."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Result JSON files or directories to scan. Defaults to the knowledge-sharing results tree.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_root() / "evals" / "knowledge_sharing" / "results",
        help="Default results tree to scan when no paths are provided.",
    )
    parser.add_argument(
        "--metric",
        choices=("acc", "acc_norm"),
        default="acc",
        help="lm-eval metric to plot.",
    )
    parser.add_argument(
        "--kt-metric",
        choices=("acc", "acc_norm"),
        default="acc_norm",
        help="lm-eval metric to use for the KT regression annotation.",
    )
    parser.add_argument(
        "--view",
        choices=("auto", "en1", "en2"),
        default="auto",
        help="Evaluation view. Auto infers en1/en2 from the result path.",
    )
    args = parser.parse_args()

    paths = result_jsons(args.paths, args.results_root.expanduser().resolve())
    if not paths:
        raise SystemExit(f"No results*.json files found under {args.results_root}")

    for path in paths:
        output_path, kt_fit = plot_result(path, args.metric, args.kt_metric, args.view)
        print(
            f"{output_path}\t"
            f"KT({args.kt_metric},{kt_fit.own_label}/{kt_fit.other_label})="
            f"{kt_fit.ratio:.6g}"
        )


if __name__ == "__main__":
    main()
