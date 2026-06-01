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
    num_vocabs,    # <-- Added argument
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
    if num_vocabs > 1:
        print(f"Extracting vocab slice {extract_vocab} out of {num_vocabs} total slices.")
        V = model_config.vocab_size // num_vocabs
        start_idx = extract_vocab * V
        end_idx = (extract_vocab + 1) * V
        print(f"Slicing vocabulary from {start_idx} to {end_idx} for slice {extract_vocab}.")
        
        # Standard HF keys
        emb_key = "model.embed_tokens.weight"
        out_key = "lm_head.weight"
        
        hf_state_dict[emb_key] = hf_state_dict[emb_key][start_idx:end_idx, :].clone()
        if out_key in hf_state_dict:
            hf_state_dict[out_key] = hf_state_dict[out_key][start_idx:end_idx, :].clone()
    elif num_vocabs == 1:
        print("No vocab slicing applied, using the full vocabulary.")

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # map and apply export dtype if needed
    # target_dtype = TORCH_DTYPE_MAP[export_dtype]
    # if target_dtype != torch.float32:
    #     hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}
    # map and apply export dtype unconditionally -- gemini fix
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    cleaned_dict = {}
    
    for k, v in list(hf_state_dict.items()):
        if not isinstance(v, torch.Tensor):
            continue
            
        try:
            # 1. Force the tensor to CPU
            v_cpu = v.cpu()
            
            # 2. If it's wrapped in a custom TorchAO or DTensor class, 
            #    this attempts to get the raw underlying tensor.
            if hasattr(v_cpu, "_local_tensor"):
                v_cpu = v_cpu._local_tensor
            elif hasattr(v_cpu, "dequantize"):
                v_cpu = v_cpu.dequantize()

            # 3. Standardize dtype
            dt = target_dtype if v_cpu.is_floating_point() else v_cpu.dtype
            
            # 4. The ultimate wash: cast to the target type, detach from graph, 
            #    and force a brand new memory allocation.
            clean_tensor = v_cpu.to(dt).detach().clone()
            
            cleaned_dict[k] = clean_tensor
            
        except Exception as e:
            print(f"Failed to clean tensor {k}. Error: {e}")
            raise

    hf_state_dict = cleaned_dict    # ----------------------------    # --- ADD THIS FIX TO BREAK SHARED MEMORY ---
    if "lm_head.weight" in hf_state_dict:
        hf_state_dict["lm_head.weight"] = hf_state_dict["lm_head.weight"].clone()
    # -------------------------------------------
    print("--- DEBUG: Inspecting hf_state_dict before save ---")
    bad_tensors = 0
    for k, v in hf_state_dict.items():
        # Check if anything made it through that isn't a standard tensor
        if not isinstance(v, torch.Tensor):
            print(f"❌ WARNING: {k} is not a tensor! Type: {type(v)}")
            bad_tensors += 1
            continue
            
        # Check for unmapped or weird dtypes
        safetensor_valid_dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool]
        if v.dtype not in safetensor_valid_dtypes:
            print(f"❌ WARNING: {k} has a potentially unsupported dtype: {v.dtype}")
            bad_tensors += 1
    
    print(f"--- DEBUG: Found {bad_tensors} suspicious tensors ---")
    # dcp.save(
    #     hf_state_dict,
    #     storage_writer=storage_writer,
    # )
    # --- BYPASS DCP WRITER ---
    from safetensors.torch import save_file
    
    output_file = output_dir / "model.safetensors"
    
    # Create the target directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Bypassing DCP writer. Saving pure safetensors to {output_file}...")
    
    # save_file calculates perfect byte offsets based strictly on the current dict
    save_file(hf_state_dict, output_file, metadata={"format": "pt"})
    print("Save complete!")    
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
    parser.add_argument("--extract_vocab", type=int, default=0, 
                        help="Slice the model into 'base' (0-V) or 'tagged' (V-2V) vocabularies.")    
    parser.add_argument("--num_vocabs", type=int, default=1, 
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
        args.num_vocabs,
    )