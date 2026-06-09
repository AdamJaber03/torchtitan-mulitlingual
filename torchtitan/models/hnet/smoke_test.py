# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Standalone (non-distributed) forward/backward smoke test for the byte-level
# H-Net debug model. Run on a GPU node *after* installing the kernels:
#
#   source .venv/bin/activate
#   python -m torchtitan.models.hnet.smoke_test
#
# It isolates the model from the training harness: builds the debugmodel on GPU,
# runs a forward on random byte tokens, checks the logits shape and the ratio
# loss, then runs a backward and checks that gradients are finite.

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.hnet import hnet_configs


def main() -> None:
    assert torch.cuda.is_available(), "H-Net smoke test requires a CUDA GPU."
    device = torch.device("cuda")
    dtype = torch.bfloat16

    cfg = hnet_configs["debugmodel"]
    with torch.device(device), torch.autocast("cuda", dtype=dtype):
        model = cfg.build().to(device)
    model.init_weights(buffer_device=device)
    model.train()

    B, L, V = 2, 256, cfg.vocab_size
    tokens = torch.randint(0, V, (B, L), device=device)
    labels = torch.randint(0, V, (B, L), device=device)
    labels[:, : L // 8] = IGNORE_INDEX  # exercise the ignore path

    with torch.autocast("cuda", dtype=dtype):
        out = model(tokens)
    logits, ratio_loss = out["output"], out["ratio_loss"]
    assert logits.shape == (B, L, V), logits.shape
    assert ratio_loss.ndim == 0 and torch.isfinite(ratio_loss), ratio_loss
    print(f"forward OK: logits {tuple(logits.shape)}, ratio_loss {ratio_loss.item():.4f}")

    ce = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(), labels.flatten(0, 1), ignore_index=IGNORE_INDEX
    )
    loss = ce + 0.03 * ratio_loss
    loss.backward()

    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    all_finite = all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert all_finite, "non-finite gradients detected"
    print(
        f"backward OK: loss {loss.item():.4f} (ce {ce.item():.4f}), "
        f"{n_grad} params have finite grads"
    )
    print("H-Net smoke test PASSED")


if __name__ == "__main__":
    main()
