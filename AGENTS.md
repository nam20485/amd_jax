# AGENTS.md

This repo is a working, documented `uv`-managed configuration for running **JAX and Pallas on AMD ROCm** (target GPU: Radeon RX 6700 XT / `gfx1030`) on Debian 13 Trixie. It is a config + docs project, not an application — there is no application code, only smoke-test scripts and setup documentation.

The authoritative setup guide is `docs/install-guides/jax/guide.md`. Read it before changing dependency config.

## Project layout

- `src/` — Python smoke-test scripts (`verify_jax.py`, `pallas_smoke.py`, `verify_torch.py`).
- `scripts/validate.sh` — runs all smoke tests and reports PASS/FAIL; exits 0 only if all pass.
- `pyproject.toml` / `uv.lock` — pinned ROCm/JAX/Triton dependencies.
- `docs/rocminfo.txt` — captured host `rocminfo` output (GPU details).
- `docs/amdrx6700xt_gpu.md` — AMD Radeon RX 6700 XT GPU (i.e. `gfx1030`) specification sheet. (installed system AMD GPU)
- `docs/install-guides/` — step-by-step setup guides (JAX, AMD Container Toolkit) plus their companion install scripts.

## Commands

All Python operations go through `uv` — never raw `pip` or `python`:

- `uv sync` — install/sync dependencies into the project venv
- `./scripts/validate.sh` — run all smoke tests (`src/verify_jax.py`, `src/pallas_smoke.py`, `src/verify_torch.py`); exits 0 only if all pass
- `uv run python src/<file>` — run a script in the project env
- `uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"` — verify JAX sees the ROCm GPU (should report a ROCm/GPU backend, not CPU)
- `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` — verify PyTorch sees the ROCm GPU (should report `2.12.0+rocm7.14.0 True`)
- `uv tree | grep -E 'rocm|jax|torch'` — verify the resolved dependency tree

There are **no tests, lint, typecheck, or CI** configured. `scripts/validate.sh` (and the `import jax` check above) is the de-facto verification step.

## Dependency config is fragile — do not "clean up"

`pyproject.toml` has non-obvious constraints that must be preserved:

- The AMD wheel index (`[[tool.uv.index]] name = "amd-rocm"`) is set to `explicit = true`. This is required: without it, `uv` queries AMD's index for ordinary packages and gets `403 Forbidden`. Do not remove `explicit = true`.
- The `rocm-sdk-*` packages look redundant with the `rocm` meta-package but are listed as direct deps **deliberately**, each mapped to the AMD index in `[tool.uv.sources]`. Explicit source mapping is more reliable than relying on transitive resolution. Do not remove them.
- `triton` is mapped to the AMD index because Pallas on ROCm requires AMD's ROCm Triton build (PyPI's `triton` is CUDA-only). Do not move it to PyPI.
- `rocm` must resolve to `rocm v7.x` from the AMD index, **not** the PyPI placeholder `rocm v0.1.0`. After any dependency change, check with `uv tree | grep -E 'rocm|jax|torch'`.
- `jax==0.10.0` / `jaxlib==0.10.0` are pinned exactly; `jax-rocm7-plugin` / `jax-rocm7-pjrt` are intentionally unpinned to track the plugin's matching ROCm build.
- `torch` / `torchvision` / `torchaudio` are mapped to the AMD index because PyPI's `torch` is a CUDA/CPU build. They are pinned to `2.12.0` / `0.27.0` / `2.11.0` (the RELEASES.md compatibility-matrix row with Linux wheels for all of Python 3.11–3.13; the newer `2.13.x` / `0.28.x` sets are cp312-only). `amd-torch-device-gfx1030` / `amd-torchvision-device-gfx1030` are listed as direct deps for the same reason as `rocm-sdk-*` — they are AMD-index-only and would not resolve transitively under `explicit = true`. `torch` constrains `triton` to `3.7.1` (down from the standalone `3.8.0`); both are the AMD ROCm build and Pallas still passes.

When changing dependencies, regenerate the lockfile: `rm -rf uv.lock && uv sync`. `uv.lock` is committed.

## Device-specific package

`rocm-sdk-device-gfx1030` matches what the host ROCm runtime reports for the RX 6700 XT (`gfx1030`). JAX/XLA/Triton compile for whatever the host HSA runtime reports. For a different GPU, replace it with the matching device package (e.g. `rocm-sdk-device-gfx1100`). GPU details are in `docs/rocminfo.txt`.

If JAX/Pallas fails to detect or compile for the device, the guide documents spoofing RDNA3:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 uv run python src/<file>
```

## Pallas on ROCm

Pallas kernels require the **Triton** lowering backend. JAX defaults to the Mosaic GPU backend (`JAX_PALLAS_USE_MOSAIC_GPU=1`), which does not support ROCm. Pass `compiler_params=jax.experimental.pallas.triton.CompilerParams()` per `pallas_call` (see `src/pallas_smoke.py`), or set the env var `JAX_PALLAS_USE_MOSAIC_GPU=0`. Without this, `pallas_call` fails with `Cannot lower pallas_call on platform: gpu`.

## Python version

`.python-version` pins 3.12; `pyproject.toml` allows `>=3.11,<3.14`. ROCm/JAX wheel availability is most reliable on 3.11–3.12.
