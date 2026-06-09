# Vendored from https://github.com/goombalab/hnet (MIT License,
# Copyright (c) 2025 brwa-cartesia). Flattened into a single package and
# import paths rewritten to be relative for torchtitan integration.
# Functional model code is kept verbatim; see torchtitan/models/hnet/README.md.

from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class AttnConfig:

    num_heads: List = field(default_factory=list)
    rotary_emb_dim: List = field(default_factory=list)
    window_size: List = field(default_factory=list)
    # torchtitan addition: per-stage number of KV heads for grouped-query
    # attention (GQA). Defaults (per stage) to num_heads => standard MHA.
    num_kv_heads: List = field(default_factory=list)


@dataclass
class SSMConfig:

    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256


@dataclass
class HNetConfig:
    arch_layout: List[Union[str, List]] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    # intermediate dimension for the FFNs (0 indicates no FFN)
    d_intermediate: List[int] = field(default_factory=list)
    vocab_size: int = 256
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    tie_embeddings: bool = False
