# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any

import torch

logger = logging.getLogger()

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import Llama3Model


class Llama3StateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_config: Llama3Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)

        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    # HuggingFace permutation function (exact copy from their conversion script)
    def _permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
            .clone()
        )

    def _reverse_permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        n_heads = self.model_config.layer.attention.n_heads
        n_kv_heads = (
            self.model_config.layer.attention.n_kv_heads
            # pyrefly: ignore [missing-attribute]
            if self.model_config.layer.attention.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_config.dim
        head_dim = dim // n_heads
        hf_state_dict = {}

        # --- HybridAnchorEmbedding: materialize the dense (vocab_size, dim) matrix that
        # to_hf_map's "tok_embeddings.weight" entry would otherwise expect. The live module has
        # no such key -- its params live under tok_embeddings.anchor_table.weight /
        # tok_embeddings.residual_table.weight (plus the anchor_group_id buffer) -- so the normal
        # per-key loop below would KeyError on `to_hf_map[key]`. Pop them out and emit the
        # equivalent dense matrix under the regular HF key instead; the rest of the pipeline
        # (including downstream vocab-slicing) then works unchanged on a normal dense matrix.
        state_dict = dict(state_dict)  # shallow copy -- don't mutate the caller's dict
        anchor_w = state_dict.pop("tok_embeddings.anchor_table.weight", None)
        residual_w = state_dict.pop("tok_embeddings.residual_table.weight", None)
        anchor_group_id = state_dict.pop("tok_embeddings.anchor_group_id", None)
        if anchor_w is not None:
            full_emb = torch.cat([anchor_w[anchor_group_id], residual_w], dim=-1)
            hf_state_dict["model.embed_tokens.weight"] = full_emb
            # Tied case: TiedAnchorOutput holds tok_embeddings as a submodule, so the SAME
            # params are also reachable (and present in this state_dict) under
            # output.tok_embeddings.*. Discard that duplicate path and instead emit lm_head.weight
            # as an independent clone of the materialized matrix -- mirroring how the standard
            # (non-anchor) tied case already always emits a separate "lm_head.weight" entry
            # regardless of tying, later `.clone()`-d to break shared-storage aliasing.
            state_dict.pop("output.tok_embeddings.anchor_table.weight", None)
            state_dict.pop("output.tok_embeddings.residual_table.weight", None)
            state_dict.pop("output.tok_embeddings.anchor_group_id", None)
            if self.model_config.enable_weight_tying:
                hf_state_dict["lm_head.weight"] = full_emb.clone()

        for key, value in state_dict.items():
            if "contrastive_proj" in key:
                print(f"Skipping {key} since since projector layers are for training only and not present in HuggingFace checkpoint.")
                continue
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if abstract_key == "layers.{}.attention.wq.weight":
                    value = self._permute(value, n_heads)
                if abstract_key == "layers.{}.attention.wk.weight":
                    # pyrefly: ignore [unsupported-operation]
                    key_value_dim = head_dim * n_kv_heads
                    value = self._permute(value, n_kv_heads, key_value_dim, dim)

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        n_heads = self.model_config.layer.attention.n_heads
        n_kv_heads = (
            self.model_config.layer.attention.n_kv_heads
            # pyrefly: ignore [missing-attribute]
            if self.model_config.layer.attention.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_config.dim
        head_dim = dim // n_heads
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]

                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                    value = self._reverse_permute(value, n_heads)
                if abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                    # pyrefly: ignore [unsupported-operation]
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(value, n_kv_heads, key_value_dim, dim)

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map[key]

            # pyrefly: ignore [unsupported-operation]
            state_dict[new_key] = value
        # pyrefly: ignore [bad-return]
        return state_dict
