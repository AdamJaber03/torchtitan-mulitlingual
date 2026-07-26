"""Shared configuration helpers for controlled en1/en2 experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


P_READJUST_FACTOR = 1 / 1.04

TOKEN_STATS = {
    # Mean document lengths measured with the 65k_paired tokenizer.
    "fineweb-edu-ar-en": 1109.5,
    "fineweb-edu-ar-ar": 762.4,
    "fineweb-edu-ar-ar-translated_1to1map": 893.5,
    "fineweb-edu-ar-ar-translated": 807.9,
    "gemini_seeds_en": 19.7,
    "gemini_seeds_ar": 20.2,
    "gemini_seeds_tr2en": 19.8,
    "gemini_seeds_tr2en_1to1map": 23.4,
    "from_domains_humans_ar": 23.3,
    "from_domains_humans_en": 26.1,
    "from_domains_humans_tr2en_1to1map": 23.3 * (23.4 / 20.2),
}


@dataclass(frozen=True)
class EntityCorpusSpec:
    """Deterministic description of one fictional-entity corpus."""

    name: str
    root: Path
    count: int = 2080
    rng: Literal["python", "numpy"] = "python"
    seed: int = 43
    token_stats_key: str = "gemini_seeds_en"


@dataclass(frozen=True)
class EntityRecord:
    corpus: str
    entity_id: int
    data_path: Path


@dataclass(frozen=True)
class InjectionProbabilityPlan:
    probabilities: tuple[float, ...]
    unadjusted_probabilities: tuple[float, ...]
    total_documents: float
    total_injection_probability: float
    residual_base_tokens: float


def _ordered_entity_ids(spec: EntityCorpusSpec) -> list[int]:
    if spec.count <= 0:
        raise ValueError(f"Entity count must be positive for {spec.name!r}.")

    order = list(range(spec.count))
    if spec.rng == "python":
        random.Random(spec.seed).shuffle(order)
    elif spec.rng == "numpy":
        np.random.default_rng(spec.seed).shuffle(order)
    else:
        raise ValueError(f"Unsupported RNG {spec.rng!r} for {spec.name!r}.")
    return order


def build_entity_records(
    corpus_specs: Sequence[EntityCorpusSpec],
    *,
    data_filename: str,
    rate_count: int = 4,
) -> list[EntityRecord]:
    """Build the canonical ordered entity list used by training and evaluation."""

    if not corpus_specs:
        raise ValueError("At least one entity corpus is required.")
    if rate_count <= 0:
        raise ValueError("rate_count must be positive.")

    grid_size = rate_count**2
    records: list[EntityRecord] = []
    for spec in corpus_specs:
        if spec.count % grid_size != 0:
            raise ValueError(
                f"Entity count for {spec.name!r} must be divisible by "
                f"rate_count^2={grid_size}; got {spec.count}."
            )
        records.extend(
            EntityRecord(
                corpus=spec.name,
                entity_id=entity_id,
                data_path=spec.root / str(entity_id) / data_filename,
            )
            for entity_id in _ordered_entity_ids(spec)
        )
    return records


def assign_rate_grid(
    records: Sequence[EntityRecord],
    target_counts: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Assign every ordered entity to the complete en1/en2 target-count grid."""

    if not target_counts:
        raise ValueError("target_counts must not be empty.")
    grid_size = len(target_counts) ** 2
    if len(records) % grid_size != 0:
        raise ValueError(
            f"Entity count must be divisible by target grid size {grid_size}; "
            f"got {len(records)}."
        )

    en1_targets = [
        float(target_counts[index % len(target_counts)])
        for index in range(len(records))
    ]
    en2_targets = [
        float(target_counts[(index // len(target_counts)) % len(target_counts)])
        for index in range(len(records))
    ]
    return en1_targets, en2_targets


def get_injection_probability_plan(
    target_counts: Sequence[float],
    tot_tokens: float,
    ds: str,
    inj_ds: str | Sequence[str],
    *,
    entity_counts: int | Sequence[int] = 2080,
    readjust_factor: float = P_READJUST_FACTOR,
) -> InjectionProbabilityPlan:
    """Convert desired per-entity counts into source sampling probabilities.

    ``tot_tokens`` is the token budget for one clean language source in one
    curriculum stage. Each entity corpus is assumed to contain a balanced copy
    of the complete target-count grid.
    """

    targets = tuple(float(count) for count in target_counts)
    if not targets or any(count < 0 for count in targets):
        raise ValueError("target_counts must be a non-empty nonnegative sequence.")
    if tot_tokens <= 0:
        raise ValueError(f"tot_tokens must be positive; got {tot_tokens}.")
    if not 0 < readjust_factor <= 1:
        raise ValueError(
            f"readjust_factor must be in (0, 1]; got {readjust_factor}."
        )
    if ds not in TOKEN_STATS:
        raise ValueError(
            f"Dataset {ds!r} is missing token statistics; "
            f"choose from {sorted(TOKEN_STATS)}."
        )

    injection_datasets = [inj_ds] if isinstance(inj_ds, str) else list(inj_ds)
    if not injection_datasets:
        raise ValueError("At least one injection dataset is required.")
    missing_stats = [name for name in injection_datasets if name not in TOKEN_STATS]
    if missing_stats:
        raise ValueError(
            f"Injection datasets are missing token statistics: {missing_stats}."
        )

    if isinstance(entity_counts, int):
        counts_by_corpus = [entity_counts] * len(injection_datasets)
    else:
        counts_by_corpus = list(entity_counts)
    if len(counts_by_corpus) != len(injection_datasets):
        raise ValueError(
            "entity_counts must contain one value per injection dataset."
        )
    if any(count <= 0 or count % len(targets) != 0 for count in counts_by_corpus):
        raise ValueError(
            "Each entity count must be positive and divisible by the number of "
            "target-count levels."
        )

    injection_documents_by_corpus = [
        sum(targets) * (entity_count / len(targets))
        for entity_count in counts_by_corpus
    ]
    total_injection_tokens = sum(
        documents * TOKEN_STATS[dataset_name]
        for documents, dataset_name in zip(
            injection_documents_by_corpus, injection_datasets, strict=True
        )
    )
    residual_base_tokens = tot_tokens - total_injection_tokens
    if residual_base_tokens <= 0:
        raise ValueError(
            "Injection targets consume the entire stage token budget: "
            f"target injection tokens={total_injection_tokens:g}, "
            f"available tokens={tot_tokens:g}."
        )

    total_documents = (
        residual_base_tokens / TOKEN_STATS[ds]
        + sum(injection_documents_by_corpus)
    )
    unadjusted_probabilities = tuple(count / total_documents for count in targets)
    probabilities = tuple(
        probability * readjust_factor
        for probability in unadjusted_probabilities
    )
    total_injection_probability = sum(
        (entity_count / len(targets)) * sum(probabilities)
        for entity_count in counts_by_corpus
    )
    if total_injection_probability >= 1:
        raise ValueError(
            "Total injection probability must be below one; got "
            f"{total_injection_probability:g}."
        )

    return InjectionProbabilityPlan(
        probabilities=probabilities,
        unadjusted_probabilities=unadjusted_probabilities,
        total_documents=total_documents,
        total_injection_probability=total_injection_probability,
        residual_base_tokens=residual_base_tokens,
    )


def get_injection_probabilities(
    target_counts: Sequence[float],
    tot_tokens: float,
    ds: str,
    inj_ds: str | Sequence[str],
    *,
    entity_counts: int | Sequence[int] = 2080,
    readjust_factor: float = P_READJUST_FACTOR,
) -> list[float]:
    """Compatibility wrapper returning only calibrated probabilities."""

    return list(
        get_injection_probability_plan(
            target_counts,
            tot_tokens,
            ds,
            inj_ds,
            entity_counts=entity_counts,
            readjust_factor=readjust_factor,
        ).probabilities
    )
