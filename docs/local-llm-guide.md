# Local LLM on RX 6700 XT — llama.cpp (Vulkan) Ops Guide

Hardware (from rocminfo):

- GPU: AMD Radeon RX 6700 XT (gfx1030, RDNA2), 40 CUs, 96 MB L3 Infinity Cache
- VRAM Pool: 12,566,528 KB (~11.98 GiB usable)
- Memory bandwidth: ~384 GB/s (192-bit bus) ← the real bottleneck for generation
- CPU: Xeon W-2155 (20 threads), ~125 GB RAM (used only for loading, never offload)

Verified benchmarks (Vulkan prebuilt b10448, -ngl 99, 100% GPU resident):

| Setup                                      | Prompt t/s | Gen t/s |
|--------------------------------------------|-----------:|--------:|
| Qwen3-8B  Q4_K_M + FA + N-gram             |    ~239    |  ~62.5  |
| Qwen3-8B  + 0.6B draft spec-decode         |    ~243    |  ~61.8  |
| Qwen3-14B Q4_K_M + FA + N-gram             |    ~130    |  ~19.5  |

Conclusion: FA + N-gram ≥ draft-model spec-decode on this card
(96 MB Infinity Cache stays hot with one model; no cache thrashing).

## Golden Rules

1. NEVER let total VRAM exceed ~11.5 GB or layers spill to PCIe and tok/s dies.
2. Only ONE server profile at a time (8B ≈ 10 GB, 14B ≈ 10.8 GB — can't coexist).
3. Generation speed is bandwidth-bound: 384 GB/s ÷ model-GB ≈ theoretical t/s cap.

=====================================================================
PART 1 — Save the flag sets as switchable profiles
=====================================================================

## 1.1 The switcher script: ~/.local/bin/llama-start

The versioned source is `scripts/llama-start.sh` in this repo. Deploy:

    mkdir -p ~/.local/bin
    ln -sf /home/nam20485/src/github/nam20485/amd_jax/scripts/llama-start.sh \
      ~/.local/bin/llama-start

(Use the absolute path to the main checkout — not `$(pwd)` — so you don't
accidentally link to a transient git worktree or second clone.)

(`~/.local/bin` is on PATH by default on Debian.) The symlink always runs
the repo version — edit `scripts/llama-start.sh` and the change is live on
the next launch; no re-deploy step.

What it does: kills any running instance and waits for VRAM to free,
fail-fast checks the llama-server binary / model file / curl, launches the
profile, then health-checks up to 120 s and exits 1 with the last 20 log
lines if the server never came up.
`--tail` follows ~/.llama-server.log in the foreground after READY (Ctrl-C
stops only the tail, not the server).

Why these flags:

- --alias          → stable model id clients can reference (qwen3-8b / qwen3-14b)
- --api-key-file   → requires clients to authenticate; key is written to a
                     0600 temp file (deleted at exit) so it never appears in
                     ps output; value is the LLOCAL_LLAMA_CPP_API_KEY var at
                     the top of llama-start.sh — edit the script to change it
- -ngl 99 + -fit off → full GPU offload; -fit off silences the "failed to
                        fit params" warning (auto-fit aborts anyway since
                        -ngl is explicitly set)
- --cache-reuse 256 → reuses KV across agent tool-loop turns (big speedup)
- --reasoning-format none → suppresses `<think>` tokens; REQUIRED for
                            agent tool-calling (Kilo/Cline/OpenCode) to
                            parse. OMITTED in the 8b-r profile, which keeps
                            reasoning output (see Part 2 for 8b-r caveats)
- --jinja          → correct Qwen3 chat/tool-call template rendering
- --tools all       → enables llama-server's built-in agent tools (read_file,
                      grep_search, exec_shell_command, write_file, ...). This
                      is server-side code execution: fine on this API-key'd
                      127.0.0.1-only server, but note it also forces CORS to
                      localhost (agent WebUI browser calls still work)
- -fa on + --spec-type ngram-simple → your verified fastest combo (8b-s
                        swaps in a 0.6B draft model instead: -md + -ngld 99 +
                        --spec-draft-n-max 8)

## 1.2 Shell aliases (marker-guarded block in ~/.bashrc)

One-time setup — the markers make it replaceable without hand-editing:

    cat >> ~/.bashrc <<'EOF'
    # >>> llama aliases >>>
    alias ai8='llama-start 8b'      # daily driver, 32K ctx, ~62 t/s
    alias ai8r='llama-start 8b-r'   # reasoning-enabled 8B, 32K ctx, ~62 t/s
    alias ai8s='llama-start 8b-s'   # 8B + 0.6B draft spec-decode, 32K ctx
    alias ai14='llama-start 14b'    # architect, 16K ctx, ~20 t/s
    alias aistop='pkill -f llama-server'
    alias ailog='tail -f ~/.llama-server.log'
    alias aivram='watch -n1 "echo $(( $(cat /sys/class/drm/card0/device/mem_info_vram_used)/1048576 )) MB"'
    # <<< llama aliases <<<
    EOF
    source ~/.bashrc

The `# >>>` / `# <<<` comment lines just delimit the block so it's easy
to spot (and select-and-delete) if you ever edit it by hand.

Usage:
    ai8      # start 8B profile (no reasoning — the agent-safe default)
    ai8r     # start 8B reasoning profile (same model, `<think>` kept)
    ai8s     # start 8B + 0.6B draft-model spec-decode profile
    ai14     # switch to 14B (auto-kills any running profile first)
    aistop   # stop

## 1.3 Verify the API

    curl -H "Authorization: Bearer sk-local" http://127.0.0.1:9999/v1/models
    curl http://127.0.0.1:9999/v1/chat/completions \
      -H "Authorization: Bearer sk-local" \
      -H "Content-Type: application/json" \
      -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"ping"}],"max_tokens":16}'

## 1.4 Optional: fixed-profile systemd unit (auto-start 8B at boot)

How --api-key-file works under systemd: llama-server reads the file's
(trimmed) content ONCE at startup and requires it as the Bearer token — it
does not watch the file, so rotating the key needs a service restart. The
llama-start mktemp-and-delete-on-exit trick doesn't apply here because
systemd execs llama-server directly (no wrapper to create/clean the temp
file), so you need a persistent key file readable only by the service user:

    mkdir -p ~/.config/llama.cpp
    printf 'sk-local\n' > ~/.config/llama.cpp/api-key
    chmod 600 ~/.config/llama.cpp/api-key

    sudo tee /etc/systemd/system/llama-cpp.service <<'EOF'
    [Unit]
    Description=llama.cpp Vulkan server (Qwen3-8B)
    After=network.target

    [Service]
    Type=simple
    User=YOUR_USERNAME
    WorkingDirectory=/home/YOUR_USERNAME/llama-b10448
    ExecStart=/home/YOUR_USERNAME/llama-b10448/llama-server \
      -m /home/YOUR_USERNAME/models/Qwen3-8B-Q4_K_M.gguf \
      --alias qwen3-8b -ngl 99 -c 32768 -b 2048 -ub 1024 \
      --cache-type-k q8_0 --cache-type-v q8_0 -fa on \
      --spec-type ngram-simple --cache-reuse 256 \
      --api-key-file %h/.config/llama.cpp/api-key \
      --reasoning-format none --jinja --host 127.0.0.1 --port 9999
    Restart=on-failure

    [Install]
    WantedBy=multi-user.target
    EOF
    sudo systemctl daemon-reload && sudo systemctl enable --now llama-cpp

(%h expands to the home dir of User=. For an auto-start 8b-r variant: drop
--reasoning-format none and set --alias qwen3-8b-r — but remember agents
must then NOT point at it.)

Hardened alternative — systemd credentials (key never lives in $HOME or
backups): `sudo install -m600 /dev/null /etc/llama-cpp.key` and write the
key into it (root-owned), then in [Service] add
`LoadCredential=api-key:/etc/llama-cpp.key` and use
`--api-key-file %d/api-key` in ExecStart. %d resolves to a per-service
tmpfs dir (/run/credentials/llama-cpp.service/), auto-cleaned on stop and
readable only by the service user.

(Notes: don't run systemd unit AND llama-start at once; pick one workflow.
Key rotation = edit the key file, then `systemctl restart llama-cpp`.)

=====================================================================
PART 2 — Client integrations
=====================================================================

All clients point at the SAME endpoint:  <http://127.0.0.1:9999/v1>
API key: `sk-local` — the server enforces it (--api-key-file; the key is
passed via a 0600 temp file, not argv, so it doesn't leak in ps). Change it by
editing LLOCAL_LLAMA_CPP_API_KEY at the top of llama-start.sh, then update
every client to match. Model id: qwen3-8b, qwen3-8b-r, qwen3-8b-s, or
qwen3-14b.
Switching profiles = just run `ai8` / `ai8r` / `ai8s` / `ai14`; clients keep
the same URL, but the configured model id must match the ACTIVE profile — with ai8r
running, a client still set to qwen3-8b gets a model-not-found error.
Also: 8b-r emits reasoning (`<think>`) output, which breaks agent
tool-call parsing — keep Kilo/Cline/OpenCode on ai8 and use 8b-r only for
interactive chats where you want the reasoning trace.
(Set client context window BELOW the active profile's ctx — 24576 for 8b
and 8b-r, 12288 for 14b. The server counts prompt + output against -c and rejects any
single prompt > -c with `exceed_context_size_error`, so the client must
compact/truncate before that; the 8K gap leaves output headroom.)

## 2.1 Kilo Code (VS Code VSIX)

1. Open Kilo Code panel → provider settings → choose "OpenAI Compatible".
2. Fill:
   - Base URL:   <http://127.0.0.1:9999/v1>
   - API Key:    sk-local
   - Model ID:   qwen3-8b
3. Advanced: set context window 24576 (NOT 32768 — see the note above),
   max output tokens 8192. If Kilo autodetects a larger window from the
   GGUF metadata (Qwen3 advertises 32K+/YaRN), override it manually.
4. Disable any "vision/image" option (text-only model).

## 2.2 Cline (VS Code VSIX)

1. Provider: "OpenAI Compatible"
2. Base URL: <http://127.0.0.1:9999/v1>
3. API Key: sk-local
4. Model ID: qwen3-8b
5. Manually set Context Window Size: 24576 (below server ctx; Cline doesn't
   autodetect)
6. Uncheck "Use VLM" / image support.

## 2.3 OpenCode CLI (your main terminal agent)

~/.config/opencode/opencode.json:

    {
      "$schema": "https://opencode.ai/config.json",
      "provider": {
        "local": {
          "npm": "@ai-sdk/openai-compatible",
          "name": "llama.cpp",
          "options": {
            "baseURL": "http://127.0.0.1:9999/v1",
            "apiKey": "sk-local"
          },
          "models": {
            "qwen3-8b": {
              "name": "Qwen3-8B  (local, 32K)",
              "limit": { "context": 24576, "output": 8192 }
            },
            "qwen3-8b-r": {
              "name": "Qwen3-8B reasoning (local, 32K)",
              "limit": { "context": 24576, "output": 8192 }
            },
            "qwen3-8b-s": {
              "name": "Qwen3-8B draft spec-decode (local, 32K)",
              "limit": { "context": 24576, "output": 8192 }
            },
            "qwen3-14b": {
              "name": "Qwen3-14B (local, 16K)",
              "limit": { "context": 12288, "output": 8192 }
            }
          }
        }
      }
    }

Run:
    export OPENAI_API_KEY=sk-local
    opencode --model local/qwen3-8b
    opencode --model local/qwen3-8b-r   # only while ai8r is the running profile
    opencode --model local/qwen3-14b

Tip: agent loops burn tokens fast → default to local/qwen3-8b (62 t/s);
local/qwen3-8b-r adds <think> traces (pick it only for planning sessions);
reserve local/qwen3-14b for deep architectural/debugging sessions.

## 2.4 Kilo Code CLI

Same provider model as the VSIX. Add to its config (or `kilocode provider` flow):

- Provider type: openai-compatible
- baseURL: <http://127.0.0.1:9999/v1>
- apiKey: sk-local
- model:   qwen3-8b

## 2.5 Qwen Code CLI

Supports OpenAI-compatible endpoints. Either env vars:

    export OPENAI_API_KEY=sk-local
    export OPENAI_BASE_URL=http://127.0.0.1:9999/v1
    qwen --model openai/qwen3-8b

or ~/.qwen/settings.json:

    {
      "modelProviders": {
        "openai": [
          {
            "id": "qwen3-8b",
            "name": "Local Qwen3-8B",
            "baseUrl": "http://127.0.0.1:9999/v1",
            "apiKey": "sk-local"
          }
        ]
      }
    }

(Exact key names vary slightly by version; the env-var route always works.)

=====================================================================
PART 3 — Daily workflow cheat-sheet
=====================================================================

    ai8                       # coding/refactor/boilerplate: 32K ctx, ~62 t/s
    ai8r                      # interactive reasoning chats: same 8B, <think> kept
    ai8s                      # 8B + 0.6B draft spec-decode: 32K ctx (experiment;
                              #   n-gram ai8 stayed faster in the b10448 bench)
    ai14                      # architecture/debugging:      16K ctx, ~20 t/s
    ailog                     # watch server logs
    aivram                    # live VRAM usage (keep < ~11500 MB)
    aistop                    # free the GPU

Profile VRAM budget (must stay under ~11.5 GB):
    8B : weights 5.0 + KV@32K q8_0 4.0 + overhead 1.0 ≈ 10.0 GB ✔ (8b-r same)
    8B-s: 8B budget + 0.65 draft weights + draft KV ≈ 10.7 GB ✔
    14B: weights 8.5 + KV@16K q8_0 1.3 + overhead 1.0 ≈ 10.8 GB ✔
    14B @ 32K would be ~12.1 GB → spills to PCIe → unusable. Don't.

Workspace-ingestion tip: don't cat whole repos into the prompt. Extract
signatures/headers/skeletons (tree-sitter or a grep script) so the project
context fits in a few thousand tokens and leaves room for the answer.

Troubleshooting:

- "request (N tokens) exceeds the available
  context size (32768)"            → the client sent a prompt bigger than -c.
                                       Fix client-side: set the client context
                                       window below server ctx (24576 for 8b),
                                       then compact (/compact, or a fresh
                                       session) to clear the oversized one.
                                       Don't raise -c on 8b: 40K ≈ 11.0 GB is
                                       the VRAM ceiling; 48K only fits with
                                       --cache-type-k/v q4_0 (~9 GB, quality
                                       unverified on this card).
- Slow generation suddenly?            → check aivram; you're over VRAM, spill to RAM
- Tool calls failing in Kilo/Cline?    → confirm --reasoning-format none + --jinja
- Model id mismatch errors?            → use exactly the ACTIVE alias:
                                         qwen3-8b / qwen3-8b-r / qwen3-8b-s /
                                         qwen3-14b (ai8r serves qwen3-8b-r, etc.)
- `<think>` blocks breaking an agent?  → the 8b-r profile is active; run ai8
                                         (--reasoning-format none is required
                                         for tool-calling agents)
- Draft-model spec decode              → already a profile: ai8s (8b-s) runs
                                         -md Qwen3-0.6B-Q8_0.gguf -ngld 99
                                         --spec-draft-n-max 8 instead of
                                         --spec-type ngram-simple
