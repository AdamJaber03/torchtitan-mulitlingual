"""End-to-end checks for the hybrid-anchor port. CPU only; no GPU or cluster data needed.

    python scripts/verify_hybrid_anchor.py
"""

import dataclasses
import os
import sys

import torch
from torch import nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import torchtitan.models.llama3 as _llama3
from torchtitan.models.common.hybrid_anchor_embedding import (
    HybridAnchorEmbedding,
    TiedAnchorOutput,
)
from torchtitan.models.llama3 import llama3_configs, model_registry
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

assert os.path.abspath(_llama3.__file__).startswith(_REPO_ROOT + os.sep), (
    f"imported torchtitan from {_llama3.__file__}, expected a path under {_REPO_ROOT}"
)

V = 65536
FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}: {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def build_init(cfg):
    torch.manual_seed(1234)
    model = cfg.build()
    torch.manual_seed(1234)
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device("cpu"))
    return model


# --- 1. the real 7B flavor: shapes, on meta device so we never allocate 7B of floats ----------
with torch.device("meta"):
    m7 = llama3_configs["7B_flex_tagged_hybrid_anchor"].build()
emb = m7.tok_embeddings
check("7B tok_embeddings is HybridAnchorEmbedding", isinstance(emb, HybridAnchorEmbedding))
check(
    "7B anchor_table shape == (65536, 4092)",
    tuple(emb.anchor_table.weight.shape) == (65536, 4092),
    str(tuple(emb.anchor_table.weight.shape)),
)
check(
    "7B residual_table shape == (131072, 4)",
    tuple(emb.residual_table.weight.shape) == (131072, 4),
    str(tuple(emb.residual_table.weight.shape)),
)
check(
    "7B head is an untied nn.Linear (not TiedAnchorOutput)",
    isinstance(m7.output, nn.Linear) and not isinstance(m7.output, TiedAnchorOutput),
    type(m7.output).__name__,
)
n_emb = sum(p.numel() for p in emb.parameters())
check(
    "7B anchored embedding is 268.7M params (vs 536.9M dense)",
    n_emb == 65536 * 4092 + 131072 * 4,
    f"{n_emb:,}",
)

# --- 2. group ids: every tagged token shares its twin's group ---------------------------------
gid = emb._group_id_list
check("group id count == vocab_size", len(gid) == 131072, str(len(gid)))
check("num_groups == 65536", emb.anchor_table.weight.shape[0] == 65536)
check(
    "gid[V+i] == gid[i] for all i",
    all(gid[V + i] == gid[i] for i in range(0, V, 97)),
)

# --- 3. real-weights tying test on a small model ----------------------------------------------
# dim must be large enough that round(dim * 0.999) < dim, otherwise the residual subspace is
# zero-width, both "halves" below are empty slices, and the comparison passes vacuously.
# dim=1024 -> anchor 1023, residual 1 (dim=256 would round to a full 256-dim anchor).
small = dataclasses.replace(
    llama3_configs["7B_flex_tagged_hybrid_anchor"], dim=1024, n_layers=2, vocab_size=2 * 512
)
small = dataclasses.replace(
    small,
    anchor_embedding_path="identity_shift:512",
    layer=dataclasses.replace(
        small.layer,
        attention=dataclasses.replace(small.layer.attention, n_heads=8, n_kv_heads=2),
        feed_forward=dataclasses.replace(small.layer.feed_forward, hidden_dim=2048),
    ),
    rope=dataclasses.replace(small.rope, dim=128, max_seq_len=128),
)
ms = build_init(small)
d_anchor = ms.tok_embeddings.dim_anchor
check(
    "small model has a non-empty residual subspace (test is not vacuous)",
    ms.tok_embeddings.dim_residual > 0,
    f"anchor={d_anchor}, residual={ms.tok_embeddings.dim_residual}",
)
ids = torch.arange(512)
out_lo = ms.tok_embeddings(ids)
out_hi = ms.tok_embeddings(ids + 512)
check(
    "anchor halves bit-identical between twin tokens",
    torch.equal(out_lo[:, :d_anchor], out_hi[:, :d_anchor]),
)
check(
    "residual halves differ between twin tokens",
    not torch.equal(out_lo[:, d_anchor:], out_hi[:, d_anchor:]),
)

# gradient reaches the shared row from BOTH members
ms.zero_grad()
loss = ms.tok_embeddings(torch.tensor([7, 7 + 512])).sum()
loss.backward()
g = ms.tok_embeddings.anchor_table.weight.grad
check(
    "shared anchor row accumulates gradient from both twins",
    g is not None and torch.allclose(g[ms.tok_embeddings.anchor_group_id[7]], torch.tensor(2.0)),
    None if g is None else f"row sum={g[ms.tok_embeddings.anchor_group_id[7]][0].item()}",
)

# --- 4. to_hf export --------------------------------------------------------------------------
adapter = Llama3StateDictAdapter(small, None)
hf = adapter.to_hf(ms.state_dict())
ew = hf.get("model.embed_tokens.weight")
check("to_hf emits model.embed_tokens.weight", ew is not None)
check(
    "exported embedding shape == (1024, 1024)",
    ew is not None and tuple(ew.shape) == (1024, 1024),
    None if ew is None else str(tuple(ew.shape)),
)
check(
    "exported embedding matches the live module",
    ew is not None and torch.equal(ew, ms.tok_embeddings(torch.arange(1024))),
)
check(
    "no anchor_table key survives export",
    not any("anchor_table" in k for k in hf),
)

# --- 5. nparams/flops accounting ---------------------------------------------------------------
nparams, _ = small.get_nparams_and_flops(ms, 128)
check("get_nparams_and_flops runs on an anchor model", nparams > 0, f"{nparams:,}")

# --- 6. TP guard ------------------------------------------------------------------------------
class _P:
    context_parallel_degree = 1
    tensor_parallel_degree = 2


class _T:
    seq_len = 2048
    dtype = "bfloat16"


class _TC:
    training = _T()
    parallelism = _P()


try:
    dataclasses.replace(llama3_configs["7B_flex_tagged_hybrid_anchor"]).update_from_config(
        trainer_config=_TC()
    )
    check("TP is rejected for anchor models", False, "no exception raised")
except NotImplementedError:
    check("TP is rejected for anchor models", True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("all checks passed")
