# amd_jax — JAX & Pallas on AMD ROCm (Radeon RX 6700 XT)

A working, documented `uv`-managed setup for running **JAX and Pallas** on **AMD ROCm** on Debian 13 "Trixie", targeting the **AMD Radeon RX 6700 XT** (`gfx1030`, as reported by the host ROCm runtime).

This is a **configuration + documentation** project, not an application. It pins a known-good ROCm/JAX/Triton stack and ships smoke tests that verify the GPU works end to end.

## Project layout

```text
src/                          # Python smoke-test scripts
  verify_jax.py               # JAX backend detection + GPU matmul
  pallas_smoke.py             # tiled Pallas vector-add kernel
scripts/
  validate.sh                 # runs both smoke tests, reports PASS/FAIL
docs/
  install-guides/             # step-by-step setup guides
    jax/                      # JAX-on-ROCm install guide + example pyproject
    amd-container-toolkit/    # AMD Container Toolkit guide + install script
  rocminfo.txt                # captured host rocminfo output
  amdrx6700xt_gpu.md          # GPU specification sheet
pyproject.toml                # pinned ROCm/JAX/Triton dependencies (uv)
uv.lock                       # resolved dependency lockfile
```

## Quick start

```bash
uv sync                       # install/sync dependencies
./scripts/validate.sh         # verify JAX + Pallas on the GPU
```

`validate.sh` exits `0` only if both `src/verify_jax.py` and `src/pallas_smoke.py` pass.

## Documentation

- **JAX on ROCm setup:** [`docs/install-guides/jax/guide.md`](docs/install-guides/jax/guide.md)
- **AMD Container Toolkit:** [`docs/install-guides/amd-container-toolkit/guide.md`](docs/install-guides/amd-container-toolkit/guide.md)

## Notes

- All Python operations go through `uv` — never raw `pip`/`python`.
- The host ROCm runtime reports the GPU as `gfx1030`; a container with a newer ROCm stack may report `gfx1031` (the native silicon). See the container-toolkit guide for details.
- Pallas on ROCm requires AMD's Triton build (`triton` from the ROCm wheel index) and the Triton lowering backend (`compiler_params=jax.experimental.pallas.triton.CompilerParams()`).
