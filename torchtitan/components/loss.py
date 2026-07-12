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


def _extract_contrastive_pairs(
    raw_vectors: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor | None:
    """Shared pair-extraction helper for paired-vector losses (InfoNCE, L2 alignment, ...).

    Args:
        raw_vectors: [Batch, MaxSeqs, D] pooled contrastive vectors for one layer.
        valid_mask: [Batch, MaxSeqs] boolean mask of valid (non-padding) sequence slots.
    Returns:
        [N, 2, D] float tensor of (member0, member1) pairs, or None if there are zero or an
        odd number of valid vectors (caller should fall back to an FSDP-safe zero in that case).
    """
    contrastive_vectors = raw_vectors[valid_mask]  # [Total_Valid_Seqs, D]
    num_vectors = contrastive_vectors.size(0)
    if num_vectors == 0 or num_vectors % 2 != 0:
        return None
    N = num_vectors // 2
    return contrastive_vectors.view(N, 2, -1).float()


@register_loss("contrastive")
class BidirectionalInfoNCELoss(nn.Module):
    def __init__(
        self,
        key: str = "contrastive_vectors",
        temperature: float = 0.05,
        layer_reduction: str = "mean",
        **kwargs,
    ):
        """
        Args:
            key: The key to extract contrastive embeddings from input_dict. Points to either a single
                 [Batch, MaxSeqs, D] tensor (single-layer) or a dict {layer: tensor} (multi-depth).
            temperature: InfoNCE temperature.
            layer_reduction: how to combine the per-layer InfoNCE losses in the multi-depth case,
                 "mean" (default; keeps magnitude ~ single-layer) or "sum".
        """
        super().__init__()
        self.key = key
        self.layer_reduction = layer_reduction
        self.register_buffer('temperature', torch.tensor(temperature))

    def _info_nce(self, raw_vectors: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Bidirectional InfoNCE for one layer's vectors [Batch, MaxSeqs, D].

        Returns a scalar; if there are no/odd valid pairs, returns a graph-derived zero (the FSDP
        autograd trick) so the term still participates in the backward graph.
        """
        pairs = _extract_contrastive_pairs(raw_vectors, valid_mask)
        if pairs is None:
            return (raw_vectors * 0.0).sum()
        z_en = F.normalize(pairs[:, 0, :], p=2, dim=1)
        z_ar = F.normalize(pairs[:, 1, :], p=2, dim=1)
        N = pairs.size(0)
        sim_matrix = torch.matmul(z_en, z_ar.T) / self.temperature
        labels_contrastive = torch.arange(N, device=sim_matrix.device)
        loss_en_ar = F.cross_entropy(sim_matrix, labels_contrastive)
        loss_ar_en = F.cross_entropy(sim_matrix.T, labels_contrastive)
        return (loss_en_ar + loss_ar_en) / 2.0

    @torch.compiler.disable
    def forward(self, input_dict: Dict[str, Any], labels: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        if self.key not in input_dict or input_dict[self.key] is None:
            logger.info("skipping contrastive loss because key missing or None")
            return torch.tensor(0.0, device=labels.device if not isinstance(labels, dict) else labels['labels'].device)

        vecs = input_dict[self.key]
        valid_mask = input_dict["valid_seq_mask"]     # [Batch, MaxSeqs] (shared across layers)

        if isinstance(vecs, dict):
            # Multi-depth: one [Batch, MaxSeqs, D] tensor per contrastive layer.
            per_layer = [self._info_nce(v, valid_mask) for v in vecs.values()]
            if len(per_layer) == 0:
                return torch.tensor(0.0, device=valid_mask.device)
            combined = sum(per_layer)
            if self.layer_reduction != "sum":
                combined = combined / len(per_layer)   # mean
            logger.info(f"contrastive (multi-depth, {len(per_layer)} layers, {self.layer_reduction}): {combined}")
        else:
            combined = self._info_nce(vecs, valid_mask)
            logger.info(f"contrastive (single layer): {combined}")

        targets = labels["labels"] if isinstance(labels, dict) else labels
        valid_tokens = (targets != -100).sum()
        return combined * valid_tokens


@register_loss("l2_alignment")
class BidirectionalL2Loss(nn.Module):
    """Simple L2 (MSE) alignment loss between paired Ar/En pooled vectors.

    Structurally mirrors BidirectionalInfoNCELoss (same key/multi-depth/layer_reduction/
    valid_tokens-scaling machinery) but replaces the normalize+cosine-similarity+cross-entropy
    InfoNCE math with a direct MSE between each matched pair's two vectors -- no temperature,
    no negatives, just pull each pair together.

    normalize=True (default) L2-normalizes both vectors to unit norm before MSE, matching the
    normalization InfoNCE already applies before its cosine-similarity step. This keeps the loss
    bounded in [0, 4] and comparable across the different contrastive_target_layers (raw
    residual-stream activation norm grows with depth in this pre-norm architecture, so
    un-normalized MSE would be implicitly dominated by the deepest layers). Set normalize=False
    for literal raw-vector MSE instead.
    """

    def __init__(
        self,
        key: str = "contrastive_vectors",
        layer_reduction: str = "mean",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.key = key
        self.layer_reduction = layer_reduction
        self.normalize = normalize

    def _l2(self, raw_vectors: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Bidirectional L2/MSE alignment for one layer's vectors [Batch, MaxSeqs, D].

        Returns a scalar; if there are no/odd valid pairs, returns a graph-derived zero (the FSDP
        autograd trick) so the term still participates in the backward graph.
        """
        pairs = _extract_contrastive_pairs(raw_vectors, valid_mask)
        if pairs is None:
            return (raw_vectors * 0.0).sum()
        a = pairs[:, 0, :]
        b = pairs[:, 1, :]
        if self.normalize:
            a = F.normalize(a, p=2, dim=1)
            b = F.normalize(b, p=2, dim=1)
        # Sum over the feature dim (squared Euclidean distance per pair), then mean over pairs --
        # NOT F.mse_loss's default (mean over every element including D), which would shrink
        # toward 0 as D grows and break the normalized [0,4] bound this loss relies on.
        return (a - b).pow(2).sum(dim=-1).mean()

    @torch.compiler.disable
    def forward(self, input_dict: Dict[str, Any], labels: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        if self.key not in input_dict or input_dict[self.key] is None:
            logger.info("skipping l2_alignment loss because key missing or None")
            return torch.tensor(0.0, device=labels.device if not isinstance(labels, dict) else labels['labels'].device)

        vecs = input_dict[self.key]
        valid_mask = input_dict["valid_seq_mask"]     # [Batch, MaxSeqs] (shared across layers)

        if isinstance(vecs, dict):
            # Multi-depth: one [Batch, MaxSeqs, D] tensor per contrastive layer.
            per_layer = [self._l2(v, valid_mask) for v in vecs.values()]
            if len(per_layer) == 0:
                return torch.tensor(0.0, device=valid_mask.device)
            combined = sum(per_layer)
            if self.layer_reduction != "sum":
                combined = combined / len(per_layer)   # mean
            logger.info(f"l2_alignment (multi-depth, {len(per_layer)} layers, {self.layer_reduction}): {combined}")
        else:
            combined = self._l2(vecs, valid_mask)
            logger.info(f"l2_alignment (single layer): {combined}")

        targets = labels["labels"] if isinstance(labels, dict) else labels
        valid_tokens = (targets != -100).sum()
        return combined * valid_tokens


@register_loss("hnet_ratio")
class HNetRatioLoss(nn.Module):
    """H-Net dynamic-chunking load-balancing ("ratio") loss.

    The model (torchtitan/models/hnet/model.py) precomputes the summed
    per-stage ratio loss and puts it in ``input_dict["ratio_loss"]``. This loss
    is a small O(1) scalar (a mean over the batch), but the trainer divides the
    *total* loss by the global valid-token count. To keep this term at its
    configured weight after that division, we scale it by the local valid-token
    count (same trick as the contrastive loss above).
    """

    def __init__(self, key: str = "ratio_loss", ignore_index: int = IGNORE_INDEX, **kwargs):
        super().__init__()
        self.key = key
        self.ignore_index = ignore_index

    def forward(self, input_dict: Dict[str, Any], labels: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        targets = labels["labels"] if isinstance(labels, dict) else labels
        ratio_loss = input_dict.get(self.key, None)
        if ratio_loss is None:
            return torch.tensor(0.0, device=targets.device)
        valid_tokens = (targets != self.ignore_index).sum()
        return ratio_loss * valid_tokens


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