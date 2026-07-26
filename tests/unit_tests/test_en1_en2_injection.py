import os
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch

from torchtitan.config import ConfigManager
from torchtitan.experiments.en1_en2 import (
    EntityCorpusSpec,
    P_READJUST_FACTOR,
    assign_rate_grid,
    build_entity_records,
    get_injection_probability_plan,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataset


TARGET_COUNTS = (0.0, 20.0, 100.0, 1000.0)


class TestEn1En2InjectionPlan(unittest.TestCase):
    def test_probability_plan_reconstructs_targets_before_readjustment(self):
        plan = get_injection_probability_plan(
            TARGET_COUNTS,
            3000 * 768 * 2048 * 0.2,
            "fineweb-edu-ar-en",
            "gemini_seeds_en",
        )

        for target, raw_probability, probability in zip(
            TARGET_COUNTS,
            plan.unadjusted_probabilities,
            plan.probabilities,
            strict=True,
        ):
            self.assertAlmostEqual(raw_probability * plan.total_documents, target)
            self.assertAlmostEqual(
                probability * plan.total_documents,
                target * P_READJUST_FACTOR,
            )
        self.assertLess(plan.total_injection_probability, 1)
        self.assertGreater(plan.residual_base_tokens, 0)

    def test_entity_manifest_is_deterministic_and_grid_balanced(self):
        specs = [
            EntityCorpusSpec(
                name="gemini",
                root=Path("/gemini"),
                count=2080,
                rng="python",
                seed=43,
            )
        ]
        first = build_entity_records(specs, data_filename="en_data.jsonl")
        second = build_entity_records(specs, data_filename="en_data.jsonl")
        self.assertEqual(first, second)

        en1_targets, en2_targets = assign_rate_grid(first, TARGET_COUNTS)
        cells = Counter(zip(en1_targets, en2_targets, strict=True))
        self.assertEqual(set(cells.values()), {130})
        self.assertEqual(len(cells), 16)

    def test_combined_manifest_uses_independent_rngs(self):
        specs = [
            EntityCorpusSpec(
                name="gemini",
                root=Path("/gemini"),
                count=16,
                rng="python",
                seed=43,
            ),
            EntityCorpusSpec(
                name="human",
                root=Path("/human"),
                count=16,
                rng="numpy",
                seed=48,
            ),
        ]
        records = build_entity_records(specs, data_filename="en_data.jsonl")
        self.assertEqual(len(records), 32)
        self.assertEqual([record.corpus for record in records[:16]], ["gemini"] * 16)
        self.assertEqual([record.corpus for record in records[16:]], ["human"] * 16)
        self.assertNotEqual(
            [record.entity_id for record in records[:16]],
            [record.entity_id for record in records[16:]],
        )

        en1_targets, en2_targets = assign_rate_grid(records, TARGET_COUNTS)
        self.assertEqual(
            set(Counter(zip(en1_targets, en2_targets, strict=True)).values()),
            {2},
        )

    def test_all_sentence_configs_use_correct_stage_budgets_and_names(self):
        variants = (
            "sentence_wise_code_switching",
            "sentence_parallel_doc_order",
            "sentence_parallel_sentence_order",
        )
        fractions = (0.01, 0.10, 0.60)

        for variant in variants:
            for fraction in fractions:
                with self.subTest(variant=variant, fraction=fraction):
                    env = {
                        "EN1_EN2_MIXED_DATA_FRACTION": str(fraction),
                        "EN1_EN2_RUN_TAG": "injcountfix_gemini_s43",
                        "EN1_EN2_HF_ASSETS_PATH": "tests/assets/65k_paired",
                        "EN1_EN2_INCLUDE_HUMAN_ENTITIES": "0",
                    }
                    with patch.dict(os.environ, env, clear=False):
                        config = ConfigManager().parse_args(
                            [
                                "--module",
                                "llama3",
                                "--config",
                                f"smollm2_360m_en1_en2_{variant}",
                            ]
                        )

                    self.assertTrue(
                        config.checkpoint.folder.endswith(
                            "_injcountfix_gemini_s43"
                        )
                    )
                    stage1, clean = config.dataloader.stages
                    self.assertAlmostEqual(
                        sum(source["weight"] for source in stage1["sources"]),
                        1.0,
                    )
                    self.assertAlmostEqual(
                        sum(source["weight"] for source in clean["sources"]),
                        1.0,
                    )

                    stage1_clean_sources = [
                        source
                        for source in stage1["sources"]
                        if source.get("injection_paths")
                    ]
                    self.assertEqual(len(stage1_clean_sources), 2)
                    self.assertEqual(
                        {
                            source["injection_summary_name"]
                            for source in stage1_clean_sources
                        },
                        {"en1", "en2"},
                    )

                    stage1_plan = get_injection_probability_plan(
                        [target * 0.75 for target in TARGET_COUNTS],
                        3000 * 768 * 2048 * ((1 - fraction) / 2),
                        "fineweb-edu-ar-en",
                        "gemini_seeds_en",
                    )
                    clean_plan = get_injection_probability_plan(
                        [target * 0.25 for target in TARGET_COUNTS],
                        1000 * 768 * 2048 * 0.5,
                        "fineweb-edu-ar-en",
                        "gemini_seeds_en",
                    )
                    self.assertEqual(
                        sorted(set(stage1_clean_sources[0]["injection_probs"])),
                        sorted(set(stage1_plan.probabilities)),
                    )
                    self.assertEqual(
                        sorted(set(clean["sources"][0]["injection_probs"])),
                        sorted(set(clean_plan.probabilities)),
                    )

    def test_full_mixed_stage_is_rejected(self):
        with patch.dict(
            os.environ,
            {"EN1_EN2_MIXED_DATA_FRACTION": "1.0"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
                ConfigManager().parse_args(
                    [
                        "--module",
                        "llama3",
                        "--config",
                        "smollm2_360m_en1_en2_sentence_wise_code_switching",
                    ]
                )

    def test_upstream_reference_configs_construct(self):
        for config_name in (
            "smollm2_360m_flex_en1_en2",
            "smollm2_360m_flex_en1_en2_codeswitching",
        ):
            with self.subTest(config=config_name):
                config = ConfigManager().parse_args(
                    ["--module", "llama3", "--config", config_name]
                )
                first_stage = config.dataloader.stages[0]
                injected_sources = [
                    source
                    for source in first_stage["sources"]
                    if source.get("injection_paths")
                ]
                self.assertTrue(injected_sources)
                self.assertEqual(
                    {len(source["injection_paths"]) for source in injected_sources},
                    {4160},
                )

    def test_injection_summary_groups_sampled_counts_by_target(self):
        dataset = object.__new__(HuggingFaceTextDataset)
        dataset.injection_paths = ["a", "b", "c", "d"]
        dataset.injection_target_counts = [0.0, 20.0, 0.0, 20.0]
        dataset.injection_expected_counts = [0.0, 15.0, 0.0, 15.0]
        dataset.injection_summary_name = "en1"
        dataset.injection_counts = torch.tensor([0, 13, 1, 17])

        summary = dataset.injection_summary()

        self.assertEqual(
            [
                {
                    **{key: value for key, value in item.items() if key != "counts"},
                    "counts": item["counts"].tolist(),
                }
                for item in summary
            ],
            [
                {
                    "source": "en1",
                    "target_count": 0.0,
                    "expected_stage_count": 0.0,
                    "entity_count": 2,
                    "counts": [0, 1],
                },
                {
                    "source": "en1",
                    "target_count": 20.0,
                    "expected_stage_count": 15.0,
                    "entity_count": 2,
                    "counts": [13, 17],
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
