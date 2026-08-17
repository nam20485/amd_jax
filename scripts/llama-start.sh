#!/usr/bin/env bash
set -euo pipefail

LLAMA_DIR="$HOME/llama-b10448"
MODELS="$HOME/models"
PROFILE="${1:-8b}"
PORT="${2:-9999}"   # default avoids common dev-service collisions (8080 et al.)
LLOCAL_LLAMA_CPP_API_KEY="${LLOCAL_LLAMA_CPP_API_KEY:-sk-local}"   # API key clients must send; edit here to change

# Pass the key via a 0600 temp file, not argv, so it never shows in ps output
KEY_FILE="$(mktemp)"
trap 'rm -f "$KEY_FILE"' EXIT
printf '%s\n' "$LLOCAL_LLAMA_CPP_API_KEY" > "$KEY_FILE"

# Kill any previous instance (only one fits in 12 GB VRAM)
if pgrep -f "$LLAMA_DIR/llama-server" >/dev/null; then
  echo "Existing llama-server found — killing..."
  pkill -f "$LLAMA_DIR/llama-server" 2>/dev/null || true
  for _ in $(seq 1 30); do
    pgrep -f "$LLAMA_DIR/llama-server" >/dev/null || break
    sleep 1
  done
  if pgrep -f "$LLAMA_DIR/llama-server" >/dev/null; then
    echo "ERROR: existing server still running after 30s; aborting (VRAM not freed)" >&2
    exit 1
  fi
  echo "Gone."
fi

SPEC_ARGS=(--spec-type ngram-simple)   # default; 8b-s overrides with a draft model

case "$PROFILE" in
  8b)
    MODEL="$MODELS/Qwen3-8B-Q4_K_M.gguf"
    ALIAS="qwen3-8b"
    CTX=32768          # big-context workspace ingestion profile
    ;;
  8b-r)
    MODEL="$MODELS/Qwen3-8B-Q4_K_M.gguf"
    ALIAS="qwen3-8b-r"
    CTX=32768          # same as 8b but reasoning/think-tokens enabled
    REASONING=1        # non-empty → omit --reasoning-format none
    ;;
  8b-s)
    MODEL="$MODELS/Qwen3-8B-Q4_K_M.gguf"
    DRAFT="$MODELS/Qwen3-0.6B-Q8_0.gguf"
    ALIAS="qwen3-8b-s"
    CTX=32768          # 8B with 0.6B draft-model speculative decoding
    SPEC_ARGS=(-md "$DRAFT" -ngld 99 --spec-draft-n-max 8)
    ;;
  14b)
    MODEL="$MODELS/Qwen3-14B-Q4_K_M.gguf"
    ALIAS="qwen3-14b"
    CTX=16384          # architect profile, keep ctx lower for VRAM
    ;;
  *)
    echo "Usage: llama-start [8b|8b-r|8b-s|14b] [port]"; exit 1 ;;
esac

# Fail fast on missing prerequisites
[ -x "$LLAMA_DIR/llama-server" ] || { echo "ERROR: llama-server not found at $LLAMA_DIR" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
[ -z "${DRAFT:-}" ] || [ -f "$DRAFT" ] || { echo "ERROR: draft model not found: $DRAFT" >&2; exit 1; }
command -v curl >/dev/null || { echo "ERROR: curl is required for the health check" >&2; exit 1; }

nohup "$LLAMA_DIR/llama-server" \
  -m "$MODEL" \
  --alias "$ALIAS" \
  --api-key-file "$KEY_FILE" \
  -ngl 99 \
  -fit off \
  -c "$CTX" \
  -b 2048 -ub 1024 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fa on \
  "${SPEC_ARGS[@]}" \
  --cache-reuse 256 \
  $([ -n "${REASONING:-}" ] || echo --reasoning-format none) \
  --jinja \
  --host 127.0.0.1 --port "$PORT" \
  > "$HOME/.llama-server.log" 2>&1 &

echo "Starting $ALIAS (ctx=$CTX) ..."
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && exit_ok=1 && break
  sleep 1
done
if [ "${exit_ok:-0}" -ne 1 ]; then
  echo "FAILED to start $ALIAS; last log lines:" >&2
  tail -n 20 "$HOME/.llama-server.log" >&2
  exit 1
fi
echo "READY → http://127.0.0.1:$PORT/v1  (model id: $ALIAS)"
echo "Logs: tail -f ~/.llama-server.log"
