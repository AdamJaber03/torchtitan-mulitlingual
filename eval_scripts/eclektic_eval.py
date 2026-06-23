#!/usr/bin/env python3
"""
ECLeKTic evaluation script for torchtitan-mulitlingual models.

Downloads ECLeKTic from Kaggle (or reads from a local path), runs generation
with a HuggingFace-compatible model, and computes word-level recall scores.

Usage:
    # Option 1: download from Kaggle first
    pip install kaggle
    kaggle datasets download -d googleai/eclektic -p /path/to/data --unzip

    # Option 2: already-downloaded data
    python eclektic_eval.py \
        --model /path/to/hf_model \
        --data /path/to/eclektic_data \
        --output /path/to/results \
        --batch_size 8 \
        --max_new_tokens 64
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import torch


# ── scoring ──────────────────────────────────────────────────────────────────

_CJK_RE = re.compile(r'[一-鿿㐀-䶿＀-￯'
                     r'　-〿가-힯]')

def _tokenize(text: str, lang: str) -> list[str]:
    if lang in ("zh", "ja"):
        tokens = []
        buf = ""
        for ch in text:
            if _CJK_RE.match(ch):
                if buf.strip():
                    tokens.append(buf.strip())
                buf = ""
                tokens.append(ch)
            else:
                buf += ch
        if buf.strip():
            tokens.append(buf.strip())
        return tokens
    return text.split()


def compute_single_recall(gold: str, pred: str, lang: str) -> float:
    gold_words = _tokenize(gold, lang)
    if not gold_words:
        return 0.0
    return sum(1 for w in gold_words if w in pred) / len(gold_words)


def compute_eclektic_scores(records: list[dict]) -> dict:
    """
    records: list of dicts with q_id, original_language, target_language,
             answer, prediction
    Returns dict with 'overall' and 'transfer' scores.
    """
    # group by q_id
    by_qid: dict[str, list[dict]] = {}
    for r in records:
        by_qid.setdefault(r["q_id"], []).append(r)

    in_lang_recall: dict[str, float] = {}
    for qid, rows in by_qid.items():
        for row in rows:
            if row["original_language"] == row["target_language"]:
                in_lang_recall[qid] = compute_single_recall(
                    row["answer"], row["prediction"], row["target_language"]
                )
                break

    results = []
    for r in records:
        cross = compute_single_recall(
            r["answer"], r["prediction"], r["target_language"]
        )
        in_lang = in_lang_recall.get(r["q_id"], 0.0)
        results.append(cross * in_lang)

    overall = float(sum(results) / len(results)) if results else 0.0

    in_lang_sum = sum(in_lang_recall.values())
    transfer_denom = in_lang_sum * 11
    transfer = (float(sum(results) / transfer_denom)
                if transfer_denom > 0 else 0.0)

    return {"overall": overall, "transfer": transfer, "n": len(results)}


# ── dataset loading ───────────────────────────────────────────────────────────

def load_eclektic(data_dir: str) -> list[dict]:
    """
    Load ECLeKTic from a local directory (Kaggle download unzipped).
    Tries CSV then JSONL; expects columns: q_id, original_language,
    target_language, question, answer.
    """
    data_path = Path(data_dir)
    records = []

    csv_files = list(data_path.glob("**/*.csv"))
    jsonl_files = list(data_path.glob("**/*.jsonl"))

    if csv_files:
        import pandas as pd
        for f in csv_files:
            df = pd.read_csv(f)
            records.extend(df.to_dict("records"))
    elif jsonl_files:
        for f in jsonl_files:
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        records.append(json.loads(line))
    else:
        raise FileNotFoundError(
            f"No CSV or JSONL files found in {data_dir}. "
            "Download the dataset: "
            "kaggle datasets download -d googleai/eclektic -p <dir> --unzip"
        )

    required = {"q_id", "original_language", "target_language", "question", "answer"}
    missing = required - set(records[0].keys())
    if missing:
        raise ValueError(f"Dataset missing fields: {missing}. Got: {list(records[0].keys())}")

    return records


# ── generation ────────────────────────────────────────────────────────────────

def run_generation(
    records: list[dict],
    model_path: str,
    batch_size: int,
    max_new_tokens: int,
    device: str,
) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # format prompt
    def make_prompt(question: str) -> str:
        return f"Question: {question}\nAnswer:"

    output_records = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        prompts = [make_prompt(r["question"]) for r in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, rec in enumerate(batch):
            input_len = inputs["input_ids"][j].shape[0]
            pred_ids = out_ids[j][input_len:]
            prediction = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            output_records.append(
                {
                    "q_id": rec["q_id"],
                    "original_language": rec["original_language"],
                    "target_language": rec["target_language"],
                    "answer": rec["answer"],
                    "prediction": prediction,
                }
            )

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(records))
            print(f"  {done}/{len(records)} examples processed")

    return output_records


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ECLeKTic evaluation")
    parser.add_argument("--model", required=True, help="Path to HF model directory")
    parser.add_argument("--data", required=True, help="Path to unzipped ECLeKTic dataset dir")
    parser.add_argument("--output", required=True, help="Directory to save predictions + scores")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--predictions_file", default=None,
                        help="If provided, skip generation and score this existing JSONL")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.predictions_file:
        with open(args.predictions_file) as f:
            output_records = [json.loads(l) for l in f if l.strip()]
        print(f"Loaded {len(output_records)} existing predictions from {args.predictions_file}")
    else:
        print("Loading ECLeKTic dataset ...")
        records = load_eclektic(args.data)
        print(f"  {len(records)} examples across languages")
        langs = sorted(set(r["target_language"] for r in records))
        print(f"  Languages: {langs}")

        output_records = run_generation(
            records, args.model, args.batch_size, args.max_new_tokens, args.device
        )

        pred_path = os.path.join(args.output, "eclektic_predictions.jsonl")
        with open(pred_path, "w") as f:
            for r in output_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Predictions saved to {pred_path}")

    scores = compute_eclektic_scores(output_records)
    scores_path = os.path.join(args.output, "eclektic_scores.json")
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)

    print("\n── ECLeKTic Results ──────────────────────")
    print(f"  Overall score : {scores['overall']:.4f}")
    print(f"  Transfer score: {scores['transfer']:.4f}")
    print(f"  Examples      : {scores['n']}")
    print(f"Results saved to {scores_path}")


if __name__ == "__main__":
    main()
