# Installing JAX for ROCm on Debian 13 Trixie with an AMD Radeon RX 6700 XT

This document describes the working installation setup for using JAX and Pallas on Debian 13 Trixie with an AMD Radeon RX 6700 XT using ROCm/HIP, `uv` project-based Python dependency management, and AMD's ROCm wheel repository.

The target GPU, from `rocminfo`, is:

```text
Name:                    gfx1030
Marketing Name:          AMD Radeon RX 6700 XT
Compute Unit:            40
Wavefront Size:          32
Workgroup Max Size:      1024
Max Waves Per CU:        32
VRAM Pool Size:          ~12 GB
L2 Cache:                3072 KB
L3 Cache:                98304 KB
```

## Motivation

AMD ROCm support for JAX is not as turnkey as CUDA/TPU, especially on Debian and consumer Radeon GPUs. This setup uses `uv` instead of manually created virtual environments and `pip` because it gives:

- reproducible project dependencies,
- a lockfile,
- fast dependency resolution,
- isolated project environments without manually activating venv,
- easy script execution via `uv run`.

The AMD ROCm wheel index provides:

- ROCm user-space runtime packages,
- device-specific ROCm libraries,
- the JAX ROCm plugin packages:
  - `jax-rocm7-plugin`
  - `jax-rocm7-pjrt`

The normal PyPI `rocm` package is **not** the AMD ROCm stack and should not be selected. Therefore the `pyproject.toml` below explicitly maps the AMD packages to AMD's wheel index and leaves normal `jax`/`jaxlib` resolution to PyPI.

This particular setup was validated with the following resolved dependency tree:

```text
jax v0.10.0
jaxlib v0.10.0
jax-rocm7-pjrt v0.10.0+rocm7.14.0
jax-rocm7-plugin v0.10.0+rocm7.14.0
rocm v7.14.0
rocm-sdk-core v7.14.0
rocm-sdk-libraries v7.14.0
rocm-sdk-device-gfx1030 v7.14.0
```

## Prerequisites

### 1.1 ROCm driver/runtime visibility

Before installing Python packages, confirm that ROCm can see the GPU:

```bash
rocminfo
rocm-smi
```

You should see the GPU listed as:

```text
gfx1030
AMD Radeon RX 6700 XT
```

Useful filtered check:

```bash
rocminfo | grep -E 'gfx1030|Marketing Name|Compute Unit|Wavefront Size|Workgroup Max Size'
```

### 1.2 User permissions

Make sure your user has access to the GPU device nodes. Typically this means being in the `video` and `render` groups:

```bash
groups
```

If needed:

```bash
sudo usermod -aG video,render $USER
```

Then log out and log back in.

### 1.3 Install uv

Install `uv` using the standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload your shell or source your profile:

```bash
source ~/.bashrc
```

Verify:

```bash
uv --version
```

## Create the uv project

Create and enter the project directory:

```bash
uv init amd_jax
cd amd_jax
```

Install and pin Python 3.11. Python 3.11 is a safe choice for ROCm/JAX wheel availability:

```bash
uv python install 3.11
uv python pin 3.11
```

Verify the project-local Python version:

```bash
uv run python --version
```

## Working pyproject.toml

Replace the contents of `pyproject.toml` with the following:

```toml
[project]
name = "amd_jax"
version = "0.1.0"
description = "JAX and Pallas on AMD ROCm"
requires-python = ">=3.11,<3.14"
dependencies = [
    # AMD ROCm meta package
    "rocm>=7.13",

    # Explicitly pull in the AMD SDK pieces that rocm depends on.
    # Without these as direct dependencies, uv may try to find them on PyPI.
    "rocm-sdk-core>=7.13",
    "rocm-sdk-libraries>=7.13",
    "rocm-sdk-device-gfx1030>=7.13",

    # AMD JAX plugin packages
    "jax-rocm7-plugin",
    "jax-rocm7-pjrt",

    # Standard JAX from PyPI
    "jax==0.10.0",
    "jaxlib==0.10.0",
]

[[tool.uv.index]]
name = "amd-rocm"
url = "https://repo.amd.com/rocm/whl-multi-arch/"
explicit = true

[tool.uv.sources]
rocm = { index = "amd-rocm" }
rocm-sdk-core = { index = "amd-rocm" }
rocm-sdk-libraries = { index = "amd-rocm" }
rocm-sdk-device-gfx1030 = { index = "amd-rocm" }
jax-rocm7-plugin = { index = "amd-rocm" }
jax-rocm7-pjrt = { index = "amd-rocm" }
```

### Why this form is necessary

The AMD wheel index is configured with:

```toml
explicit = true
```

This prevents `uv` from querying the AMD index for ordinary packages such as `jax`, which can result in a `403 Forbidden` error when the package does not exist there.

The `[tool.uv.sources]` section maps the AMD-specific packages to the AMD index. The `rocm-sdk-*` packages are listed as direct dependencies because the `rocm` meta-package depends on them transitively, and explicit source mapping is more reliable when those packages are direct project dependencies.

## Install dependencies

Remove any old lockfile and sync:

```bash
rm -rf uv.lock
uv sync
```

Expected behavior:

- `uv` resolves the AMD ROCm packages from `https://repo.amd.com/rocm/whl-multi-arch/`.
- `uv` resolves normal `jax` and `jaxlib` from PyPI.
- The project environment is created/managed automatically by `uv`.

## Verify resolved packages

Check the dependency tree:

```bash
uv tree | grep -E 'rocm|jax'
```

You should see output similar to:

```text
amd-jax v0.1.0
├── jax v0.10.0
│   ├── jaxlib v0.10.0
├── jax-rocm7-pjrt v0.10.0+rocm7.14.0
├── jax-rocm7-plugin v0.10.0+rocm7.14.0
│   └── jax-rocm7-pjrt v0.10.0+rocm7.14.0
├── jaxlib v0.10.0 (*)
├── rocm v7.14.0
│   └── rocm-sdk-core v7.14.0
├── rocm-sdk-core v7.14.0
├── rocm-sdk-device-gfx1030 v7.14.0
│   └── rocm-sdk-libraries v7.14.0
└── rocm-sdk-libraries v7.14.0
```

The important thing is that you do **not** see:

```text
rocm v0.1.0
```

`rocm v0.1.0` is the wrong PyPI placeholder package.

## Verify that JAX sees the ROCm GPU

Run:

```bash
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

## Pallas smoke test

Create `pallas_smoke.py` with the following content:

```python
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

BLOCK_SIZE = 256


def vector_add_kernel(x_ref, y_ref, z_ref):
    z_ref[...] = x_ref[...] + y_ref[...]


def vector_add(x: jax.Array, y: jax.Array) -> jax.Array:
    n = x.shape[0]

    if n % BLOCK_SIZE != 0:
        raise ValueError("n must be divisible by BLOCK_SIZE")

    grid = (n // BLOCK_SIZE,)

    block_spec = pl.BlockSpec(
        lambda i: (i,),
        (BLOCK_SIZE,),
    )

    return pl.pallas_call(
        vector_add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=[block_spec, block_spec],
        out_specs=block_spec,
        name="vector_add",
    )(x, y)


def main() -> None:
    print("JAX devices:", jax.devices())

    x = jnp.ones(8192, dtype=jnp.float32)
    y = jnp.arange(8192, dtype=jnp.float32)

    z = jax.jit(vector_add)(x, y)
    z.block_until_ready()

    print(z[:8])
    print("Pallas smoke test completed")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python pallas_smoke.py
```

Expected output ends with:

```text
Pallas smoke test completed
```

## RX 6700 XT / gfx1030 compatibility notes

The RX 6700 XT is RDNA2:

```text
gfx1030
```

AMD's official AI-stack support is strongest on Instinct GPUs and newer Radeon generations. Your card may work natively, but if JAX or Pallas fails to detect or compile for the device, try spoofing a supported RDNA3 target:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run python verify_jax.py
```

or:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run python pallas_smoke.py
```

If the override causes crashes, remove it and test native again:

```bash
unset HSA_OVERRIDE_GFX_VERSION
```

For Pallas kernel tuning on this specific GPU, use the `rocminfo` limits:

- wavefront size: 32
- max workgroup size: 1024
- max waves per CU: 32
- compute units: 40

Good starting block sizes:

```python
BLOCK_SIZE = 256
BLOCK_SIZE = 512
BLOCK_SIZE = 1024
```

For 2D kernels, reasonable starting tiles include:

```python
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
```

## Debugging environment variables

If something fails, use these environment variables to get more synchronized and verbose behavior:

```bash
HIP_LAUNCH_BLOCKING=1 \
AMD_LOG_LEVEL=3 \
JAX_TRACEBACK_FILTERING=off \
uv run python verify_jax.py
```

For XLA/IR dumps:

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" \
uv run python pallas_smoke.py
```

Then inspect:

```bash
ls -lh /tmp/xla_dump
```

Clear JAX/Triton caches if compilation state becomes corrupted:

```bash
rm -rf ~/.cache/jax
rm -rf ~/.triton
```

## Host ROCm version caveat

This Python environment installs ROCm user-space packages at:

```text
rocm v7.14.0
rocm-sdk-core v7.14.0
jax-rocm7-plugin v0.10.0+rocm7.14.0
jax-rocm7-pjrt v0.10.0+rocm7.14.0
```

Your host kernel driver/runtime must be compatible with these user-space libraries. If you see HSA, ROCr, HIP runtime, or GPU initialization errors, you have two good options:

- Use a ROCm container image matching the Python ROCm stack.
- Pin the Python ROCm packages to versions matching your host ROCm installation, if AMD publishes those exact wheels.

For containerized use, the AMD Container Toolkit can simplify GPU passthrough using CDI:

```bash
docker run --device amd.com/gpu=all --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  rocm/jax:rocm7.14-jax0.10.0-py3.11 \
  bash
```

## Commit the working configuration

After successful installation and verification, commit the project files:

```bash
git add pyproject.toml uv.lock
git commit -m "Add working JAX ROCm uv project for Debian 13 / RX 6700 XT"
```

On another machine, the environment can then be restored with:

```bash
uv sync
```

## Summary of the working workflow

```bash
# create project
uv init amd_jax
cd amd_jax
uv python install 3.11
uv python pin 3.11

# replace pyproject.toml with the working version above

# install dependencies
rm -rf uv.lock
uv sync

# inspect dependency tree
uv tree | grep -E 'rocm|jax'

# verify JAX devices
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"

# run verification script
uv run python verify_jax.py

# optional Pallas test
uv run python pallas_smoke.py
```
