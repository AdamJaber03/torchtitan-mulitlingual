"""Build the Arabic<->English 1-to-1 shared-token map.

For every (arabic, english) entry in the translation dict, encode the
*space-prefixed* (mid-sentence) form of each side with the paired tokenizer and
keep the entries where BOTH sides are exactly one token. The resulting map is
used to remap the tokenizer vocab so the Arabic token shares the English token's
id (English-anchored), creating a shared cross-lingual anchor.

Outputs a JSON with:
  - pairs:            list of per-pair records
  - vocab_remap:      { arabic_byte_key (vocab key str) : english_id }
  - decode_side_table:{ english_id : {"en": surface, "ar": [surfaces...]} }

Usage:
  .venv/bin/python scripts/build_shared_token_map.py \
      --tokenizer tests/assets/65k_paired/tokenizer.json \
      --dict /home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex.json \
      --out  /home/adamga/leshemg/adamga/data/translations/ar_en_1to1_token_map.json
"""
import argparse
import json

from tokenizers import Tokenizer


def single_token(tok: Tokenizer, surface: str):
    """Return (vocab_key_str, token_id) if `surface` is exactly one token, else (None, None)."""
    enc = tok.encode(surface, add_special_tokens=False)
    if len(enc.ids) == 1:
        return enc.tokens[0], enc.ids[0]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tests/assets/65k_paired/tokenizer.json")
    ap.add_argument(
        "--dict",
        default="/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex.json",
    )
    ap.add_argument(
        "--out",
        default="/home/adamga/leshemg/adamga/data/translations/ar_en_1to1_token_map.json",
    )
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    with open(args.dict) as f:
        d = json.load(f)

    pairs = []
    vocab_remap = {}
    decode_side_table = {}
    seen_ar_ids = set()
    skipped_conflict = 0

    for ar, en in d.items():
        ar_key, ar_id = single_token(tok, " " + ar)
        en_key, en_id = single_token(tok, " " + en)
        if ar_key is None or en_key is None:
            continue
        # An arabic single-token surface is unique, so ar_id should appear once.
        if ar_id in seen_ar_ids:
            skipped_conflict += 1
            continue
        seen_ar_ids.add(ar_id)

        pairs.append(
            {
                "arabic": ar,
                "english": en,
                "arabic_byte_key": ar_key,
                "arabic_id": ar_id,
                "english_byte_key": en_key,
                "english_id": en_id,
                "arabic_surface": " " + ar,
                "english_surface": " " + en,
            }
        )
        vocab_remap[ar_key] = en_id
        entry = decode_side_table.setdefault(
            str(en_id), {"en": " " + en, "ar": []}
        )
        entry["ar"].append(" " + ar)

    out = {
        "meta": {
            "tokenizer": args.tokenizer,
            "source_dict": args.dict,
            "form": "space_prefixed",
            "anchor": "english",
            "num_pairs": len(pairs),
            "num_distinct_english_ids": len(decode_side_table),
            "skipped_conflicts": skipped_conflict,
        },
        "pairs": pairs,
        # {arabic_id: english_id} -- consumed by the post-tokenization remap augmentation.
        "id_remap": {str(p["arabic_id"]): p["english_id"] for p in pairs},
        # {arabic_byte_key: english_id} -- kept for reference (a vocab-level remap is NOT
        # viable: HF BPE represents merges by token id, so reassigning ids corrupts merges).
        "vocab_remap": vocab_remap,
        "decode_side_table": decode_side_table,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"total dict entries     : {len(d)}")
    print(f"1-to-1 pairs kept       : {len(pairs)}")
    print(f"distinct english anchors: {len(decode_side_table)}")
    print(f"skipped (ar_id conflict): {skipped_conflict}")
    print(f"wrote                   : {args.out}")
    print("\nsample pairs:")
    for p in pairs[:12]:
        print(f"  {p['arabic']!r}({p['arabic_id']}) -> {p['english']!r}({p['english_id']})")


if __name__ == "__main__":
    main()
