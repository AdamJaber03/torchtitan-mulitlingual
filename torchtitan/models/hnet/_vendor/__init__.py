# Vendored H-Net reference implementation.
#
# Source: https://github.com/goombalab/hnet (MIT License, (c) 2025 brwa-cartesia)
# "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
# Hwang, Wang, Gu — arXiv:2507.07955.
#
# The original `hnet/` package (split across `models/` and `modules/`) has been
# flattened into this single package and its intra-package imports rewritten to
# be relative. The functional model code is otherwise kept verbatim so it tracks
# upstream. See torchtitan/models/hnet/README.md for details and the kernel
# dependencies (mamba_ssm, causal_conv1d, flash_attn) that this code requires.
#
# NOTE: This package is intentionally import-light at the top level. The modules
# that pull in CUDA kernels (mixer_seq, hnet, isotropic, block, dc, mha, mlp,
# rotary) are imported lazily by torchtitan/models/hnet/model.py so that the
# rest of torchtitan can be imported on machines without those kernels.
