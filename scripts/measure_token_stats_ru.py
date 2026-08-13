"""Measure tokens-per-document for every source the Russian runs consume.

Produces the token_stats table in get_injection_probabilities_ru. Run on the
cluster, where OUTPUT_BASE_DIR and the gemini_seeds/from_domains_humans trees are
present:

    cd /gpfs/ess6000-1/proj/dmfexp/trAr/torchtitan-mulitlingual
    python scripts/measure_token_stats_ru.py                 # all sources, 20k docs each
    python scripts/measure_token_stats_ru.py --docs 2000      # quick pass
    python scripts/measure_token_stats_ru.py --only fineweb2-hq-ru gemini_seeds_ru

Four properties make the output comparable to what training actually sees:

* the tokenizer is tests/assets/65k_en1.0_ru1.0, the en/ru vocab the models are
  trained and released with -- not the en/ar 65k_paired vocab;
* documents are counted through the project's own encode_with_encoding, so the
  per-document EOS is included exactly as the dataloader appends it;
* the translated_1to1map rows apply the real training-time augmentation
  (WordwiseUnigramCodeSwitching, prob=1.0, GOST fallback) to the original
  Russian, rather than reading the precomputed translated_1to1map directory,
  because the augmentation is what the dataloader feeds the model;
* the fineweb rows are sampled on a systematic grid over the chunk files rather
  than off the head of the stream. Document length in these corpora is
  non-stationary across chunks (per-chunk means span 766-1485) and within them
  (the first documents of a chunk are the long ones), and the dataloader's
  20k-document shuffle buffer never leaves the head of a 1105-chunk corpus, so a
  streamed sample reads these rows low no matter how many documents it draws.
  Pass --head-sample to reproduce that biased figure for comparison.

Run --shard/--n-shards to split the fineweb rows across jobs; the shard means are
independent replicates, so their spread is the standard error of the pooled value.

Paste the printed block into get_injection_probabilities_ru.
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.augmentations import WordwiseUnigramCodeSwitching
from torchtitan.hf_datasets.text_datasets import DATASETS, encode_with_encoding

TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tests/assets/65k_en1.0_ru1.0")
RU_DICT_PATH = os.path.join(PROJECT_ROOT, "torchtitan/tests/assets/translations/top_russian_translated_fineweb_newregex_1to1.json")
N_SEED_DIRS = 2080

# Mirrors the "augmentations" block of llama3_7B_en_translated_ru.
RU_AUG_CONFIG = {
    "name": "wordwise_unigram_codeswitching",
    "prob": 1.0,
    "fallback_to_transliteration": True,
    "dict_paths": {"fineweb2-hq-ru": RU_DICT_PATH},
}


def stream_hf(dataset_name, limit, start_idx=0):
    """Yield raw document texts from a DATASETS entry, as training reads them."""
    cfg = DATASETS[dataset_name]
    stream = cfg.loader(cfg.path, start_idx)
    for i, sample in enumerate(stream):
        if i >= limit:
            return
        yield cfg.sample_processor(sample)


CHUNKED_CORPORA = {
    # stat key -> (directory of chunk_NNNN.jsonl files, number of chunks)
    "fineweb-edu-ar-en": ("data/fineweb_translated/en-original", 613),
    "fineweb2-hq-ru": ("data/fineweb2_hq/rus_Cyrl/original", 1105),
}


def stream_chunked(stat_key, chunk_step, stride, shard=0, n_shards=1):
    """Yield documents on a systematic grid: every chunk_step-th chunk, every stride-th document.

    Chunks are dealt out to shards round-robin, so each shard is itself a spread
    sample of the whole corpus rather than a contiguous slice of it.
    """
    subdir, n_chunks = CHUNKED_CORPORA[stat_key]
    root = os.path.join(PROJECT_ROOT, subdir)
    chunks = [c for i, c in enumerate(range(1, n_chunks + 1, chunk_step)) if i % n_shards == shard]
    for chunk in chunks:
        path = os.path.join(root, f"chunk_{chunk:04d}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % stride:
                    continue
                text = json.loads(line).get("text") or ""
                if text.strip():
                    yield text


def stream_jsonl_dirs(subdir, filename, limit):
    """Yield document texts from fictional_entity_data/<subdir>/<i>/<filename>."""
    yielded = 0
    for i in range(N_SEED_DIRS):
        path = os.path.join(PROJECT_ROOT, "fictional_entity_data", subdir, str(i), filename)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") or obj.get("content") or ""
                if not text:
                    continue
                yield text
                yielded += 1
                if yielded >= limit:
                    return


EOS_TOKEN_ID = 0  # matches eos_token_id in the Russian dataloader configs


def measure(label, texts, tokenizer, augment=None):
    """Average tokens per document, counting the EOS the dataloader appends."""
    total = 0
    n = 0
    for text in texts:
        if augment is not None:
            text = augment({"text": text}, dataset_name="fineweb2-hq-ru")["text"]
        tokens, _ = encode_with_encoding(tokenizer, text)
        if not tokens or tokens[-1] != EOS_TOKEN_ID:
            tokens = tokens + [EOS_TOKEN_ID]
        total += len(tokens)
        n += 1
        if n % 2000 == 0:
            print(f"  {label}: {n} docs, running mean {total / n:.1f}", flush=True)
    if n == 0:
        print(f"  {label}: NO DOCUMENTS FOUND -- check paths", flush=True)
        return None
    mean = total / n
    print(f"  {label}: {mean:.1f} tokens/doc  (n={n})", flush=True)
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=20_000, help="documents per source")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="skip this many streamed docs first; the committed table used 0")
    parser.add_argument("--only", nargs="*", default=None, help="subset of stat keys to measure")
    parser.add_argument("--head-sample", action="store_true",
                        help="draw the fineweb rows off the head of the stream instead of the "
                             "systematic grid; reproduces the biased figure, do not commit it")
    parser.add_argument("--chunk-step", type=int, default=5,
                        help="sample every Nth chunk file of the fineweb corpora")
    parser.add_argument("--stride", type=int, default=100,
                        help="sample every Nth document within each sampled chunk")
    parser.add_argument("--shard", type=int, default=0, help="this shard's index")
    parser.add_argument("--n-shards", type=int, default=1, help="total number of shards")
    args = parser.parse_args()

    def fineweb(stat_key):
        """The document stream for a fineweb row, spread over the corpus unless --head-sample."""
        if args.head_sample:
            return stream_hf(stat_key, args.docs, args.start_idx)
        return stream_chunked(stat_key, args.chunk_step, args.stride, args.shard, args.n_shards)

    tokenizer = HuggingFaceTokenizer.Config().build(tokenizer_path=TOKENIZER_PATH)
    assert tokenizer.bos_id is None, "table assumes no BOS is prepended per document"
    assert tokenizer.eos_id == EOS_TOKEN_ID, f"tokenizer eos_id {tokenizer.eos_id} != dataloader eos_token_id {EOS_TOKEN_ID}"
    ru_aug = WordwiseUnigramCodeSwitching({**RU_AUG_CONFIG, "tokenizer": tokenizer})

    sources = {
        "fineweb-edu-ar-en": lambda: measure(
            "fineweb-edu-ar-en", fineweb("fineweb-edu-ar-en"), tokenizer),
        "fineweb2-hq-ru": lambda: measure(
            "fineweb2-hq-ru", fineweb("fineweb2-hq-ru"), tokenizer),
        "fineweb2-hq-ru-translated_1to1map": lambda: measure(
            "fineweb2-hq-ru-translated_1to1map", fineweb("fineweb2-hq-ru"),
            tokenizer, augment=ru_aug),
        "gemini_seeds_en": lambda: measure(
            "gemini_seeds_en", stream_jsonl_dirs("gemini_seeds", "en_data.jsonl", args.docs), tokenizer),
        "gemini_seeds_ru": lambda: measure(
            "gemini_seeds_ru", stream_jsonl_dirs("gemini_seeds", "ru_data.jsonl", args.docs), tokenizer),
        "gemini_seeds_ru_translated_1to1map": lambda: measure(
            "gemini_seeds_ru_translated_1to1map",
            stream_jsonl_dirs("gemini_seeds", "ru_data.jsonl", args.docs), tokenizer, augment=ru_aug),
        "from_domains_humans_en": lambda: measure(
            "from_domains_humans_en",
            stream_jsonl_dirs("from_domains_humans", "en_data.jsonl", args.docs), tokenizer),
    }

    selected = args.only or list(sources)
    unknown = [key for key in selected if key not in sources]
    assert not unknown, f"unknown stat keys {unknown}; choose from {list(sources)}"

    sampling = (f"head of stream, {args.docs} docs" if args.head_sample
                else f"every {args.chunk_step}th chunk, every {args.stride}th doc, "
                     f"shard {args.shard}/{args.n_shards}")
    print(f"=== tokenizer {TOKENIZER_PATH} | fineweb rows: {sampling} ===\n")
    results = {}
    for key in selected:
        print(f"{key} ...", flush=True)
        results[key] = sources[key]()

    print("\n=== Paste into get_injection_probabilities_ru token_stats ===")
    for key, value in results.items():
        print(f'    "{key}": {value:.1f},' if value is not None else f'    # "{key}": MEASUREMENT FAILED')


if __name__ == "__main__":
    main()
