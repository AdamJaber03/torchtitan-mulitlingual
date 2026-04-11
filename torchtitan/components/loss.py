# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Union

from torchtitan.config import CompileConfig, LossConfig
from torchtitan.tools.logging import logger
from collections.abc import Callable
from typing import TypeAlias

import torch.compiler

IGNORE_INDEX = -100
LossFunction: TypeAlias = Callable[..., torch.Tensor]
# ==========================================
# 1. Loss Registry
# ==========================================
LOSS_REGISTRY = {}

def register_loss(name: str):
    def wrapper(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return wrapper

# ==========================================
# 2. Specific Loss Definitions
# ==========================================

@register_loss("cross_entropy")
class CrossEntropyLoss(nn.Module):
    def __init__(
        self, 
        key: str = "output", 
        label_key: str = "labels", 
        ignore_index: int = IGNORE_INDEX,
        **kwargs,
    ):
        """
        Args:
            key: The key to extract logits from the input_dict.
            label_key: The key to extract targets if labels is a dict.
            ignore_index: Standard CE ignore index.
        """
        super().__init__()
        self.key = key
        self.label_key = label_key
        self.ignore_index = ignore_index

    def forward(self, input_dict: Dict[str, Any], labels: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        # Extract the specific tensor this loss cares about
        logits = input_dict[self.key]
        
        # Handle labels (allows flexibility if labels is a single tensor or a dict of multiple targets)
        targets = labels[self.label_key] if isinstance(labels, dict) else labels

        # Flatten for standard causal LM Cross Entropy
        # CRITICAL: Use reduction="sum" to match torchtitan's token-based normalization
        return F.cross_entropy(
            logits.flatten(0, 1).float(),
            targets.flatten(0, 1),
            reduction="sum",
            ignore_index=self.ignore_index,
        )


@register_loss("contrastive")
class BidirectionalInfoNCELoss(nn.Module):
    def __init__(
        self, 
        key: str = "contrastive_vectors", 
        temperature: float = 0.05,
        **kwargs,
    ):
        """
        Args:
            key: The key to extract contrastive embeddings from input_dict. 
                 Expected to point to interleaved [2N, D] tensor of (EN, AR) vectors.
        """
        super().__init__()
        self.key = key
        self.register_buffer('temperature', torch.tensor(temperature))

    @torch.compiler.disable
    def forward(self, input_dict: Dict[str, Any], labels: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        if self.key not in input_dict or input_dict[self.key] is None:
            return torch.tensor(0.0, device=labels.device if not isinstance(labels, dict) else labels['labels'].device)
            
        raw_vectors = input_dict[self.key]        # [Batch, MaxSeqs, Proj_Dim]
        valid_mask = input_dict["valid_seq_mask"] # [Batch, MaxSeqs]
        
        # --- DYNAMIC SLICING ---
        # Flattens the batch and extracts ONLY the real sequences.
        # Shape becomes: [Total_Valid_Seqs, Proj_Dim]
        contrastive_vectors = raw_vectors[valid_mask] 
        
        num_vectors = contrastive_vectors.size(0)
        
        # --- THE FSDP AUTOGRAD TRICK ---
        # If empty or odd, return a 0 derived from the graph
        if num_vectors == 0 or num_vectors % 2 != 0:
             return (raw_vectors * 0.0).sum()
            
        N = num_vectors // 2
        
        # The rest of your InfoNCE math stays exactly the same
        pairs = contrastive_vectors.view(N, 2, -1).float()
        z_en = pairs[:, 0, :]
        z_ar = pairs[:, 1, :]
        
        z_en = F.normalize(z_en, p=2, dim=1)
        z_ar = F.normalize(z_ar, p=2, dim=1)
        
        sim_matrix = torch.matmul(z_en, z_ar.T) / self.temperature
        labels_contrastive = torch.arange(N, device=sim_matrix.device)
        
        loss_en_ar = F.cross_entropy(sim_matrix, labels_contrastive)
        loss_ar_en = F.cross_entropy(sim_matrix.T, labels_contrastive)
        info_nce_loss_mean = (loss_en_ar + loss_ar_en) / 2.0
        
        targets = labels["labels"] if isinstance(labels, dict) else labels
        valid_tokens = (targets != -100).sum()
        
        return info_nce_loss_mean * valid_tokens

# ==========================================
# 3. The Generic Mixed Loss Module
# ==========================================
class MixedLoss(nn.Module):
    def __init__(self, loss_config: Dict[str, Dict[str, Any]], **global_params):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}

        for loss_item in loss_config.losses:
            loss_name = loss_item.get("name", None)
            if loss_name not in LOSS_REGISTRY:
                raise ValueError(f"Loss '{loss_name}' is not registered. Available: {list(LOSS_REGISTRY.keys())}")

            weight = loss_item.get("weight", 1.0)
            local_params = loss_item.get("params", {})
            merged_params = {**global_params, **local_params}

            self.losses[loss_name] = LOSS_REGISTRY[loss_name](**merged_params)
            self.weights[loss_name] = weight

    def forward(self, input_dict: Dict[str, Any], labels: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Strict signature: takes exactly the model's output dict and the ground truth labels.
        """
        total_loss = 0.0
        loss_breakdown = {}

        for loss_name, loss_fn in self.losses.items():
            # Pass only the dict and the labels. The loss implementation routes the keys.
            l_val = loss_fn(input_dict, labels)
            
            weighted_loss = l_val * self.weights[loss_name]
            total_loss += weighted_loss
            loss_breakdown[f"{loss_name}_loss"] = l_val

        return total_loss, loss_breakdown

# ==========================================
# 4. Builder with Torch Compile Integration
# ==========================================
def build_loss(loss_config: LossConfig, compile_config: CompileConfig, **global_params) -> Union[MixedLoss, Any]:
    """
    Instantiates the mixed loss framework and applies torch.compile if enabled in the config.
    """
    loss_fn = MixedLoss(loss_config=loss_config, **global_params)
    
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the mixed loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
        
    return loss_fn