"""Alignment / positional probe across DEPTH for the contrastive (Ar<->En translation) embeddings.

For each requested layer we pool the per-word hidden state (mean+max over the word's tokens) and
compare cosine similarities of word-pairs:
    true       = cos(z_ar[d,k], z_en[d,k])           # same doc, same rank, same content
    positional = cos(z_ar[d,k], z_en[d',k]) d'!=d    # same RANK, different doc/content
    random     = cos(z_ar[d,k], z_en[d',k']) k'!=k   # different rank, different doc
plus the worst-10% of the true cosine and the retrieval rank of the true partner among all N.

CPU-only & fast: block_causal only isolates documents and RoPE only affects *relative* attention, so
a document's reps are identical whether packed (training, flex) or alone with plain causal attention.
We rebuild the model with attn_backend="sdpa"/causal, put each doc in its own single-doc row, load the
flex-trained weights, and run layers 0..max(requested) capturing the requested depths.

Run on a 0-GPU node:
    srun -p leshem.q --mem=120G -c 32 ... uv run python scripts/probe_positional_shortcut.py \
        --flavor <flavor> --ckpt <.../step-N> --layers emb,2,4,8,16,24,30
"""

import argparse
import glob
import json
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.augmentations import AUGMENTATIONS_REGISTRY
from torchtitan.hf_datasets.post_tokenization_augmentations import WordWiseContrastive
from torchtitan.hf_datasets.text_datasets import encode_with_encoding
from torchtitan.models.llama3 import model_registry
from torchtitan.models.llama3.model import Llama3Model

HF_ASSETS = "/home/adamga/torchtitan/tests/assets/65k_paired"
AR_DICT = "/home/adamga/leshemg/adamga/data/translations/top_arabic_translated_fineweb_newregex_1to1.json"
AR_GLOB = "/home/adamga/leshemg/adamga/data/fineweb_translated/original/chunk_*.jsonl"


def maybe_init_dist():
    if not dist.is_initialized():
        s = socket.socket()
        s.bind(("", 0))
        free = s.getsockname()[1]
        s.close()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(free)   # free port -> safe to run many in parallel
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def build_augs(tok):
    cfgs = [
        {"name": "text_duplication", "n": 2},
        {"name": "wordwise_unigram_codeswitching", "prob": 0.0, "dict_paths": {"fineweb-edu-ar-ar": AR_DICT}, "idx": 0},
        {"name": "wordwise_unigram_codeswitching", "prob": 1.0, "dict_paths": {"fineweb-edu-ar-ar": AR_DICT}, "idx": 1},
        {"name": "merge_seperators", "n_merge": 3, "idx": 0},
        {"name": "merge_seperators", "n_merge": 3, "idx": 1},
    ]
    augs = []
    for c in cfgs:
        c = dict(c)
        c["tokenizer"] = tok
        augs.append(AUGMENTATIONS_REGISTRY[c["name"]](c))
    return augs


def prep_doc(text, augs, wwc):
    x = {"text": text}
    for aug in augs:
        x = aug(x, dataset_name="fineweb-edu-ar-ar")
    if not (isinstance(x, list) and len(x) == 2):
        return None
    out = []
    for side in (0, 1):
        d = dict(x[side])
        d["tokens"], d["encoding"] = encode_with_encoding(wwc.tokenizer, d["text"])
        n, mask = wwc(d, 0)               # mask[t] = word id; word i -> i+1
        out.append((d["tokens"], mask, n))
    (ar_tok, ar_mask, n_ar), (en_tok, en_mask, n_en) = out
    if n_ar != n_en or n_ar < 2:
        return None
    return (ar_tok, ar_mask), (en_tok, en_mask), n_ar


def row_for(tokens, mask, K):
    cutoff = sum(1 for m in mask if 1 <= m <= K)
    toks = tokens[:cutoff]
    per_word = [[m == (k + 1) for m in mask[:cutoff]] for k in range(K)]
    return toks, per_word


@torch.no_grad()
def multi_layer_forward(model, tokens, capture_blocks, capture_emb, max_block):
    """Run embeddings + blocks 0..max_block (sdpa causal); return {layer: h} for requested depths.
    layer -1 == embeddings; layer i == output of transformer block i."""
    h = model.tok_embeddings(tokens)
    outs = {}
    if capture_emb:
        outs[-1] = h
    for i in range(max_block + 1):
        h = model.layers[str(i)](h, model.freqs_cis, None, None)
        if i in capture_blocks:
            outs[i] = h
    return outs


def pool_meanmax(h, cmask):
    hm = h.unsqueeze(1)                       # [b,1,L,dim]
    bm = cmask.unsqueeze(-1).bool()           # [b,K,L,1]
    fm = bm.float()
    mean = (hm * fm).sum(2) / fm.sum(2).clamp(min=1e-9)
    mx = hm.masked_fill(~bm, -1e9).max(2)[0]
    return torch.cat([mean, mx], dim=-1)      # [b,K,2*dim]


def analyze(cv, M, K):
    """cv: [2M, K, D] normalized; first M = Arabic rows, next M = English rows."""
    z_ar = cv[:M].reshape(M * K, -1)
    z_en = cv[M:].reshape(M * K, -1)
    N, D = z_ar.shape
    S = z_ar @ z_en.T
    idx = torch.arange(N)
    rank_id = idx % K
    doc_id = idx // K
    true_t = S.diag().clone()
    re = rank_id[:, None] == rank_id[None, :]
    de = doc_id[:, None] == doc_id[None, :]

    def moments(mask):
        cnt = mask.sum().clamp(min=1)
        m = (S * mask).sum() / cnt
        v = (S * S * mask).sum() / cnt - m * m
        return m.item(), v.clamp(min=0).sqrt().item()

    tm, ts = true_t.mean().item(), true_t.std().item()
    pm, _ = moments(re & ~de)
    rm, _ = moments(~re & ~de)
    worst10 = torch.topk(true_t, max(1, N // 10), largest=False).values.mean().item()
    ranks = (S > true_t[:, None]).sum(dim=1) + 1
    at1 = (ranks == 1).float().mean().item()
    maxr = int(ranks.max().item())
    return dict(D=D, true=tm, true_std=ts, pos=pm, rnd=rm, gap=tm - rm,
                worst10=worst10, at1=at1, maxr=maxr)


def main(args):
    maybe_init_dist()
    torch.manual_seed(0)
    K = args.k
    t0 = time.time()

    # parse --layers ("emb" -> embeddings, ints -> block indices)
    capture_emb = False
    capture_blocks = []
    for tok_ in args.layers.split(","):
        tok_ = tok_.strip()
        if tok_ in ("emb", "-1", "0e"):
            capture_emb = True
        else:
            capture_blocks.append(int(tok_))
    capture_blocks = sorted(set(capture_blocks))
    max_block = max(capture_blocks)
    layer_order = ([-1] if capture_emb else []) + capture_blocks

    tok = HuggingFaceTokenizer.Config().build(tokenizer_path=HF_ASSETS)
    eos = tok.eos_id if tok.eos_id is not None else 0

    cfg = model_registry(args.flavor).model
    cfg.layer.attention.attn_backend = "sdpa"
    cfg.layer.attention.attn_mask_type = "causal"
    target = cfg.contrastive_target_layer
    model = Llama3Model(cfg)
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device("cpu"))
    model.eval()
    model.requires_grad_(False)
    for i in range(max_block + 1, cfg.n_layers):   # only need blocks 0..max_block
        del model.layers[str(i)]
    model.output = None
    model.norm = None
    print(f"[{time.time()-t0:.0f}s] loading checkpoint (blocks 0..{max_block}): {args.ckpt}", flush=True)
    dcp.load(model.state_dict(), checkpoint_id=args.ckpt)
    print(f"[{time.time()-t0:.0f}s] checkpoint loaded.", flush=True)

    augs = build_augs(tok)
    wwc = WordWiseContrastive(tokenizer=tok)
    prepared = []
    for fp in sorted(glob.glob(AR_GLOB)):
        if len(prepared) >= args.max_docs:
            break
        with open(fp) as f:
            for line in f:
                if len(prepared) >= args.max_docs:
                    break
                words = json.loads(line).get("text", "").split()
                if len(words) < args.words_per_doc:
                    continue
                r = prep_doc(" ".join(words[: args.words_per_doc]), augs, wwc)
                if r is not None and r[2] >= K:
                    prepared.append(r)
    M = len(prepared)
    print(f"[{time.time()-t0:.0f}s] prepared M={M} docs, K={K} -> N={M*K} pairs", flush=True)

    rows, wordmasks = [], []
    for side in (0, 1):
        for p in prepared:
            t, w = row_for(p[side][0], p[side][1], K)
            rows.append(t)
            wordmasks.append(w)
    L = max(len(t) for t in rows)
    B = len(rows)
    input_ids = torch.full((B, L), eos, dtype=torch.long)
    cmask = torch.zeros(B, K, L, dtype=torch.bool)
    for b, (t, w) in enumerate(zip(rows, wordmasks)):
        input_ids[b, : len(t)] = torch.tensor(t, dtype=torch.long)
        for k in range(K):
            cmask[b, k, : len(w[k])] = torch.tensor(w[k], dtype=torch.bool)

    pooled = {Lid: torch.empty(B, K, 2 * cfg.dim) for Lid in layer_order}
    mb = args.minibatch
    with torch.no_grad():
        for s in range(0, B, mb):
            e = min(s + mb, B)
            outs = multi_layer_forward(model, input_ids[s:e], set(capture_blocks), capture_emb, max_block)
            for Lid in layer_order:
                pooled[Lid][s:e] = pool_meanmax(outs[Lid].float(), cmask[s:e])
            print(f"[{time.time()-t0:.0f}s] forward {e}/{B} rows", flush=True)

    def label(Lid):
        return "emb(0)" if Lid == -1 else f"blk{Lid}" + ("*" if Lid == target else "")

    print(f"\n========  DEPTH ALIGNMENT PROFILE  (flavor={args.flavor}, "
          f"head={getattr(model, 'contrastive_head_type', '?')}, M={M}, K={K}, N={M*K})  ========")
    print("  (* = contrastive_target_layer; vectors = pooled mean+max, pre-head)\n")
    print(f"  {'layer':8s} {'true':>7s} {'± std':>7s} {'posit':>7s} {'random':>7s} {'gap':>7s} "
          f"{'worst10':>8s} {'retr@1':>8s} {'maxrank':>8s}")
    for Lid in layer_order:
        cv = F.normalize(pooled[Lid].float(), dim=-1)
        r = analyze(cv, M, K)
        print(f"  {label(Lid):8s} {r['true']:7.3f} {r['true_std']:7.3f} {r['pos']:7.3f} {r['rnd']:7.3f} "
              f"{r['gap']:7.3f} {r['worst10']:8.3f} {r['at1']:8.2%} {r['maxr']:8d}")

    # bonus: what InfoNCE actually sees at the target layer (after the head)
    if target in pooled and model.contrastive_proj is not None:
        cvh = F.normalize(model.contrastive_proj(pooled[target]).float(), dim=-1)
        r = analyze(cvh, M, K)
        print(f"  {'blk%d+head' % target:8s} {r['true']:7.3f} {r['true_std']:7.3f} {r['pos']:7.3f} "
              f"{r['rnd']:7.3f} {r['gap']:7.3f} {r['worst10']:8.3f} {r['at1']:8.2%} {r['maxr']:8d}")

    print(f"\n[{time.time()-t0:.0f}s] done.")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--flavor", required=True)
    p.add_argument("--max-docs", type=int, default=1000)
    p.add_argument("--words-per-doc", type=int, default=50)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--layers", default="emb,2,4,8,16,24,30", help="comma list; 'emb' = token embeddings")
    main(p.parse_args())
