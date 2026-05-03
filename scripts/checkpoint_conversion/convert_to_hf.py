# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP


@torch.inference_mode()
def convert_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
    extract_vocab, # <-- Added argument
):
    # load model and model args so that we can get the state dict shape
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model

    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    # pyrefly: ignore[bad-instantiation, not-callable]
    sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."

    # allocate state dict memory with empty weights to load checkpoint
    state_dict = model._get_state_dict()
    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
    )

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    # --- NEW: Vocab Slicing Logic ---
    if extract_vocab in ["base", "tagged"]:
        V = model_config.vocab_size // 2
        print(f"Slicing vocabulary from {model_config.vocab_size} to {V} for '{extract_vocab}' model.")
        
        # Standard HF keys
        emb_key = "model.embed_tokens.weight"
        out_key = "lm_head.weight"
        
        if extract_vocab == "base":
            hf_state_dict[emb_key] = hf_state_dict[emb_key][:V, :].clone()
            if out_key in hf_state_dict:
                hf_state_dict[out_key] = hf_state_dict[out_key][:V, :].clone()
                
        elif extract_vocab == "tagged":
            hf_state_dict[emb_key] = hf_state_dict[emb_key][V:, :].clone()
            if out_key in hf_state_dict:
                hf_state_dict[out_key] = hf_state_dict[out_key][V:, :].clone()
    # --------------------------------

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # map and apply export dtype if needed
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    # --- ADD THIS FIX TO BREAK SHARED MEMORY ---
    if "lm_head.weight" in hf_state_dict:
        hf_state_dict["lm_head.weight"] = hf_state_dict["lm_head.weight"].clone()
    # -------------------------------------------

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )
    
    # # --- NEW: Update config.json vocab_size if slicing ---
    # if extract_vocab in ["base", "tagged"]:
    #     config_path = output_dir / "config.json"
    #     # Wait a moment for HuggingFaceStorageWriter to finish dumping metadata
    #     if config_path.exists():
    #         with open(config_path, "r") as f:
    #             config_data = json.load(f)
            
    #         config_data["vocab_size"] = V
            
    #         with open(config_path, "w") as f:
    #             json.dump(config_data, f, indent=2)
    #         print(f"Updated config.json vocab_size to {V}")
    #     else:
    #         print("Warning: config.json not found in output directory to update vocab_size.")
    # # -----------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with DCP weights."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory. This is used to get the model.safetensors.index.json mapping",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for HF checkpoint (default: float32)",
    )
    
    # New argument for vocabulary slicing
    parser.add_argument("--extract_vocab", type=str, choices=["none", "base", "tagged"], default="none", 
                        help="Slice the model into 'base' (0-V) or 'tagged' (V-2V) vocabularies.")
    
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
        args.extract_vocab,
    )