#!/usr/bin/env bash
#
# validate.sh — verify the JAX-on-ROCm setup end to end.
#
# Runs the smoke-test scripts through `uv run` and reports PASS/FAIL for
# each. Exits 0 only if ALL succeed:
#   - src/verify_jax.py   : JAX detects the ROCm GPU and a matmul computes on it.
#   - src/pallas_smoke.py : a tiled Pallas vector-add kernel compiles and runs.
#   - src/verify_torch.py : PyTorch detects the ROCm GPU and a matmul computes on it.
#
# Usage (from anywhere):
#   ./scripts/validate.sh

set -uo pipefail

# Resolve the repo root from this script's location, so it works from any CWD.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SCRIPTS=("src/verify_jax.py" "src/pallas_smoke.py" "src/verify_torch.py")

PASS=0
FAIL=0

# Make sure the scripts exist before we start.
for f in "${SCRIPTS[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found in ${REPO_ROOT}" >&2
        exit 2
    fi
done

run_check() {
    local label="$1"
    local script="$2"

    echo
    echo "================================================================"
    echo "RUNNING: ${label}"
    echo "  \$ uv run python ${script}"
    echo "----------------------------------------------------------------"

    if uv run python "${script}"; then
        echo "----------------------------------------------------------------"
        echo "RESULT:  PASS  (${label})"
        PASS=$((PASS + 1))
    else
        local rc=$?
        echo "----------------------------------------------------------------"
        echo "RESULT:  FAIL  (${label}, exit code ${rc})"
        FAIL=$((FAIL + 1))
    fi
}

run_check "JAX backend + GPU compute" "src/verify_jax.py"
run_check "Pallas tiled vector-add kernel" "src/pallas_smoke.py"
run_check "PyTorch backend + GPU compute" "src/verify_torch.py"

echo
echo "================================================================"
echo "SUMMARY: ${PASS} passed, ${FAIL} failed"
echo "================================================================"

if [ "${FAIL}" -ne 0 ]; then
    exit 1
fi
