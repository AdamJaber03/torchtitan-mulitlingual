"""
Measure average tokens-per-document for datasets used in get_injection_probabilities.

Run on the cluster (needs HF datasets and the 65k_paired tokenizer):

    cd /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
    python fictional_entity_data/measure_token_stats.py

Outputs the values to paste into config_registry.py token_stats dict.
Currently only measures the Russian-specific entries that are missing (TODO items).
"""

import json
import os
import glob
import sys

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer

TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tests/assets/65k_paired")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def avg_tokens(texts, label):
    if not texts:
        print(f"  {label}: no samples found", flush=True)
        return None
    lengths = [len(tokenizer.encode(t)) for t in texts]
    avg = sum(lengths) / len(lengths)
    print(f"  {label}: {avg:.1f} avg tokens/doc  (n={len(lengths)})", flush=True)
    return avg


print("=== Measuring Russian token stats ===\n")

# --- gemini_seeds/ru_data.jsonl ---
print("gemini_seeds_ru: sampling ru_data.jsonl files ...", flush=True)
ru_entity_texts = []
for i in range(2080):
    path = os.path.join(PROJECT_ROOT, f"fictional_entity_data/gemini_seeds/{i}/ru_data.jsonl")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or ""
                if text:
                    ru_entity_texts.append(text)
            except json.JSONDecodeError:
                pass
    if len(ru_entity_texts) >= 500:
        break

gemini_seeds_ru = avg_tokens(ru_entity_texts, "gemini_seeds_ru")


# --- fineweb2-hq-ru: sample from HuggingFace dataset stream ---
print("\nfineweb2-hq-ru: loading from HuggingFace datasets (streaming) ...", flush=True)
try:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="hq_ru", split="train", streaming=True)
    ru_texts = []
    for sample in ds:
        text = sample.get("text", "")
        if text:
            ru_texts.append(text)
        if len(ru_texts) >= 500:
            break
    fineweb2_hq_ru = avg_tokens(ru_texts, "fineweb2-hq-ru")
except Exception as e:
    print(f"  fineweb2-hq-ru: failed to load ({e})", flush=True)
    fineweb2_hq_ru = None


print("\n=== Paste these values into config_registry.py token_stats ===")
print(f'    "fineweb2-hq-ru": {fineweb2_hq_ru},')
print(f'    "gemini_seeds_ru": {gemini_seeds_ru},')
