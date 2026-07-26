import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import ConfigManager
from torchtitan.hf_datasets.text_datasets import (
    En1En2TranslationValidationDataset,
)
from torchtitan.experiments.en1_en2_translation import (
    EXPECTED_TRANSLATION_LOSSES,
    prepare_local_step,
    read_step_metrics,
)


class _Encoding:
    def __init__(self, text: str):
        self.ids = [ord(char) for char in text]
        self.word_ids = list(range(len(text)))
        self.offsets = [(idx, idx + 1) for idx in range(len(text))]


class _BackendTokenizer:
    def encode(self, text: str) -> _Encoding:
        return _Encoding(text)


class _Tokenizer:
    bos_id = 1
    eos_id = 2
    tokenizer = _BackendTokenizer()


class _SentenceTokenizer:
    def span_tokenize(self, text: str):
        return [(0, len(text))]


class TestEn1En2TranslationValidation(unittest.TestCase):
    def test_translation_examples_shift_and_mask_only_target_tokens(self):
        vocab_size = 1000
        dataset = En1En2TranslationValidationDataset(
            dataset_name="fineweb-edu-ar-en",
            dataset_path=None,
            tokenizer=_Tokenizer(),
            seq_len=8,
            infinite=False,
            direction="both",
            vocab_size=vocab_size,
            eos_token_id=2,
            data=[{"text": "Hi."}],
            sentence_tokenizer=_SentenceTokenizer(),
        )

        examples = list(dataset)
        self.assertEqual(len(examples), 2)

        en1_to_en2_inputs, en1_to_en2_labels = examples[0]
        en2_to_en1_inputs, en2_to_en1_labels = examples[1]

        self.assertEqual(
            en1_to_en2_labels.tolist(),
            [
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                ord("H") + vocab_size,
                ord("i") + vocab_size,
                ord(".") + vocab_size,
                IGNORE_INDEX,
            ],
        )
        self.assertEqual(
            en2_to_en1_labels.tolist(),
            [
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                ord("H"),
                ord("i"),
                ord("."),
                IGNORE_INDEX,
            ],
        )
        self.assertEqual(
            en2_to_en1_inputs["input"].tolist()[1:4],
            [ord("H") + vocab_size, ord("i") + vocab_size, ord(".") + vocab_size],
        )

    def test_sentence_config_adds_translation_validation_loaders_when_enabled(self):
        env = {
            "EN1_EN2_TRANSLATION_VALIDATION_ENABLE": "1",
            "EN1_EN2_VALIDATOR_STEPS": "7",
            "EN1_EN2_TRANSLATION_VAL_STEPS": "1",
            "EN1_EN2_HF_ASSETS_PATH": "tests/assets/65k_paired",
        }
        with patch.dict(os.environ, env):
            config = ConfigManager().parse_args(
                [
                    "--module",
                    "llama3",
                    "--config",
                    "smollm2_360m_en1_en2_sentence_wise_code_switching",
                ]
            )

        self.assertEqual(
            set(config.validator.dataloader),
            {
                "en1",
                "en2",
                "translation_en1_to_en2",
                "translation_en2_to_en1",
            },
        )
        self.assertEqual(config.validator.steps, 7)
        self.assertEqual(
            config.validator.dataloader["translation_en1_to_en2"].validation_steps,
            1,
        )
        self.assertEqual(
            config.validator.dataloader["translation_en2_to_en1"].validation_steps,
            1,
        )

    def test_complete_local_step_is_retry_safe(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            csv_path = output_dir / "validation_metrics.csv"
            jsonl_path = output_dir / "validation_metrics.jsonl"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["step", "metric", "value"]
                )
                writer.writeheader()
                for metric in sorted(EXPECTED_TRANSLATION_LOSSES):
                    writer.writerow(
                        {"step": 3000, "metric": metric, "value": 0.5}
                    )
            with jsonl_path.open("w", encoding="utf-8") as f:
                for metric in sorted(EXPECTED_TRANSLATION_LOSSES):
                    f.write(
                        json.dumps(
                            {
                                "step": 3000,
                                "metrics": {metric: 0.5},
                            }
                        )
                        + "\n"
                    )

            self.assertFalse(
                prepare_local_step(output_dir, 3000, "skip-complete")
            )
            self.assertTrue(
                prepare_local_step(output_dir, 3000, "overwrite")
            )
            self.assertEqual(read_step_metrics(output_dir, 3000), {})

    def test_partial_local_step_fails_by_default(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            csv_path = output_dir / "validation_metrics.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["step", "metric", "value"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "step": 3000,
                        "metric": sorted(EXPECTED_TRANSLATION_LOSSES)[0],
                        "value": 0.5,
                    }
                )

            with self.assertRaisesRegex(RuntimeError, "partial"):
                prepare_local_step(output_dir, 3000, "skip-complete")


if __name__ == "__main__":
    unittest.main()
