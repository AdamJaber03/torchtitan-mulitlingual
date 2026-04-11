# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import TypeAlias

import torch

from torchtitan.config import CompileConfig
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

# --- NEW: Contrastive Loss Globals ---
CONTRASTIVE_LAMBDA = 0.02  # Weight of the alignment loss relative to LM loss
CONTRASTIVE_TEMP = 0.05    # Temperature for InfoNCE scaling

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None], labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss with sum reduction for token-based normalization, plus optional InfoNCE."""
    
    # --- NEW: Unpack the tuple if contrastive alignment is enabled ---
    if isinstance(pred, tuple):
        logits, contrastive_vectors = pred
    else:
        logits = pred
        contrastive_vectors = None

    # 1. Standard Language Modeling Loss (Sum Reduction)
    ce_loss_sum = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )

    # 2. Contrastive Alignment Loss (InfoNCE)
    if contrastive_vectors is not None and contrastive_vectors.size(0) > 0:
        num_vectors = contrastive_vectors.size(0)
        
        # Safety check: ensure we have an even number of interleaved vectors
        if num_vectors % 2 == 0:
            N = num_vectors // 2
            
            # Reshape from [2N, D] -> [N, 2, D] and split
            pairs = contrastive_vectors.view(N, 2, -1).float()
            z_en = pairs[:, 0, :]
            z_ar = pairs[:, 1, :]
            
            # L2 Normalization (Crucial for cosine similarity)
            z_en = torch.nn.functional.normalize(z_en, p=2, dim=1)
            z_ar = torch.nn.functional.normalize(z_ar, p=2, dim=1)
            
            # Compute Cosine Similarity Matrix [N, N] scaled by temperature
            sim_matrix = torch.matmul(z_en, z_ar.T) / CONTRASTIVE_TEMP
            
            # The correct pairs are on the diagonal (0 matches 0, 1 matches 1...)
            labels_contrastive = torch.arange(N, device=sim_matrix.device)
            
            # Symmetric Cross Entropy (English -> Arabic AND Arabic -> English)
            loss_en_ar = torch.nn.functional.cross_entropy(sim_matrix, labels_contrastive)
            loss_ar_en = torch.nn.functional.cross_entropy(sim_matrix.T, labels_contrastive)
            info_nce_loss_mean = (loss_en_ar + loss_ar_en) / 2.0
            
            # --- The Torchtitan Scaling Trick ---
            # Torchtitan scales the final gradient by doing: loss_sum / global_valid_tokens.
            # To ensure our lambda coefficient remains accurate after that global division,
            # we must multiply our mean InfoNCE loss by the local valid tokens here.
            valid_tokens = (labels != IGNORE_INDEX).sum()
            contrastive_loss_sum = info_nce_loss_mean * valid_tokens * CONTRASTIVE_LAMBDA
            
            return ce_loss_sum + contrastive_loss_sum

    return ce_loss_sum


def build_cross_entropy_loss(compile_config: CompileConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = cross_entropy_loss
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn


def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common MSE loss function with sum reduction for Transformer models training."""
    return torch.nn.functional.mse_loss(
        pred.float(), labels.float().detach(), reduction="sum"
    )


def build_mse_loss(compile_config: CompileConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = mse_loss
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn
