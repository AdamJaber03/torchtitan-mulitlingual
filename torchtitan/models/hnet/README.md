# H-Net (Hierarchical Network with Dynamic Chunking)

Byte-level, tokenizer-free language model that learns content-dependent
**dynamic chunking** end-to-end, replacing the tokenize → LM → detokenize
pipeline with a single model. U-Net-style hierarchy:
`encoder → routing → chunk → main network → dechunk → decoder`.

- Paper: *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*,
  Hwang, Wang, Gu — [arXiv:2507.07955](https://arxiv.org/abs/2507.07955).
- Upstream code (vendored, MIT): https://github.com/goombalab/hnet

The model code under [`_vendor/`](./_vendor) is the official implementation,
flattened into one package with intra-package imports rewritten to be relative.
The torchtitan glue lives in `model.py`, `parallelize.py`, `config_registry.py`,
and `__init__.py`.

## Dependencies (required, GPU only)

H-Net uses Mamba-2 SSD and FlashAttention CUDA kernels. These must be built on a
**GPU node with a CUDA toolkit** (`nvcc`); there are no prebuilt wheels for the
torch version in this repo, and they cannot be installed on a CPU-only box.

```bash
source .venv/bin/activate
# Build against the installed torch; needs nvcc + matching CUDA toolkit.
# Keep MAX_JOBS modest for flash-attn or its compile will OOM (it is memory-hungry).
export TORCH_CUDA_ARCH_LIST="12.0"   # set to your GPU's arch (12.0 = Blackwell sm_120)
MAX_JOBS=8 uv pip install --no-build-isolation causal-conv1d
MAX_JOBS=8 uv pip install --no-build-isolation mamba-ssm
MAX_JOBS=8 uv pip install --no-build-isolation flash-attn
# small pure-python dep used by the vendored Isotropic module
uv pip install optree
# sanity check
python -c "import mamba_ssm, causal_conv1d, flash_attn, optree; print('kernels OK')"
```

If your cluster lacks a CUDA toolkit new enough for your GPU (e.g. Blackwell
sm_120 needs nvcc >= 12.8 and the module system only ships older CUDA), you can
assemble a toolchain from NVIDIA redistributables without root, e.g.:

```bash
# download + extract cuda_nvcc and cuda_cudart redist tarballs into ~/cuda129,
# then point the build at it:
export CUDA_HOME=$HOME/cuda129
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
# (cub/thrust headers come from the pip nvidia-cuda-cccl-cu12 wheel, merged into
#  $CUDA_HOME/include). See the project notes for the exact redist URLs.
```

`model.py` imports these lazily and raises a clear error if they are missing, so
the rest of torchtitan still imports on machines without the kernels.

## Run

```bash
# 1-process forward/backward smoke test on the debug model (10 steps)
MODULE=hnet CONFIG=hnet_debugmodel NGPU=1 ./run_train.sh

# Multi-GPU FSDP
MODULE=hnet CONFIG=hnet_debugmodel NGPU=8 ./run_train.sh

# Small real run on C4
MODULE=hnet CONFIG=hnet_1stage_small NGPU=8 ./run_train.sh
```

The dataloader feeds raw text; the `ByteTokenizer`
(`torchtitan/components/tokenizer.py`) encodes it to UTF-8 bytes
(`vocab_size=256`, `eos_id=255`). No tokenizer files are needed —
`hf_assets_path` is ignored by the byte tokenizer.

## Configuration

`arch_layout` encodes the hierarchy. For a 1-stage net:

```
[ "<encoder>", [ "<main>" ], "<decoder>" ]
```

Each `"<...>"` is a sequence of `"<letter><count>"` groups:

| letter | mixer                | FFN (SwiGLU) |
|--------|----------------------|--------------|
| `m`    | Mamba-2              | no           |
| `M`    | Mamba-2              | yes          |
| `t`    | causal attention     | no           |
| `T`    | causal attention     | yes          |

e.g. `"m2T2"` = 2 Mamba layers then 2 (attention + FFN) layers. Per-stage list
fields (`d_model`, `d_intermediate`, `attn_num_heads`, `attn_rotary_emb_dim`,
`attn_window_size`) are indexed by stage: index 0 = outer (encoder/decoder),
index 1 = inner (main network). All per-stage lists must have one entry per
stage even if a stage uses no attention.

`target_compression` is the target downsampling factor `N` used by the
dynamic-chunking ratio loss (one per non-innermost stage).

## Loss

The model returns `{"output": logits, "ratio_loss": <scalar>}`. Training uses a
`MixedLoss` of:

- `cross_entropy` (next-byte prediction), and
- `hnet_ratio` (the chunking load-balancing loss; weight ~0.03).

Both are registered in `torchtitan/components/loss.py`.

## Limitations (v1)

- **FSDP / HSDP / DDP only.** Dynamic chunking yields variable-length inner
  sequences, which are incompatible with Tensor / Pipeline / Context Parallel —
  these raise a clear `NotImplementedError`.
- **`torch.compile` off by default** (data-dependent dynamic shapes).
- **1-stage** flavors provided (`debugmodel`, `1stage_small`). The vendored code
  supports deeper hierarchies; add a 2-stage `arch_layout` + extra per-stage
  list entries to extend.
- Inference/generation (`step()` paths in the vendored code) is not wired into
  torchtitan; only training is.

## Weight init note

torchtitan builds on `meta`, then `to_empty()` + `init_weights()`.
`HNetModel.init_weights` therefore fully re-initializes every parameter/buffer —
Mamba-2 `A_log`/`dt_bias`/`D`, conv1d, RMSNorm, rotary `inv_freq`, the routing
module's identity projections, and the zero-init residual projections — in
addition to the upstream Linear/embedding init. Validate numerics with the
debug-model smoke run on a GPU node.
