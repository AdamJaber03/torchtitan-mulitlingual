# Vendored from https://github.com/goombalab/hnet (MIT License,
# Copyright (c) 2025 brwa-cartesia). Flattened into a single package and
# import paths rewritten to be relative for torchtitan integration.
# Functional model code is kept verbatim; see torchtitan/models/hnet/README.md.

# Base code imported from
# https://github.com/state-spaces/mamba
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.ops.activations import swiglu


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model,
        d_intermediate=None,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        d_intermediate = (
            d_intermediate if d_intermediate is not None else int(8 * d_model / 3)
        )
        d_intermediate = (d_intermediate + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(d_model, 2 * d_intermediate, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(d_intermediate, d_model, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = swiglu(gate, y)
        y = self.fc2(y)
        return y
