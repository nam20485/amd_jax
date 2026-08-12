# Installing the AMD Container Toolkit on Debian 13 Trixie (Radeon RX 6700 XT)

This guide installs the **AMD Container Toolkit** (`amd-container-toolkit`) on Debian 13 "Trixie" and configures Docker so containers can access the AMD GPU through **CDI** (the Container Device Interface). It is the companion to the JAX-on-ROCm guide: once the toolkit is in place you can run the official `rocm/jax` and `rocm/dev-ubuntu-*` images with full GPU access, instead of maintaining the ROCm user-space stack directly on the host.

## System tested on

- **OS:** Debian 13 "Trixie" (kernel 6.12.x)
- **GPU:** AMD Radeon RX 6700 XT — native silicon is **`gfx1031`** (Navi 22). Note: the host may report `gfx1030`; see [Why the container reports `gfx1031` but the host reports `gfx1030`](#why-the-container-reports-gfx1031-but-the-host-reports-gfx1030) below.
- **Docker:** 29.7.2 (from the official `download.docker.com` Debian repository)
- **Toolkit:** `amd-container-toolkit 1.3.0~24.04`

> **Debian 13 caveat (important).** AMD's `amd-container-toolkit` APT repository does **not** publish a `trixie` suite. This guide uses the Ubuntu 24.04 **`noble`** suite instead, which is binary-compatible. It is the only suite the repository publishes — verified directly: `noble` returns HTTP 200, while `ubuntu`, `debian`, and `trixie` all return 404.

## Motivation

AMD ROCm support for JAX is easiest to consume from inside a container that ships a known-good, matching ROCm + JAX stack. Running ROCm in containers requires a way to pass the GPU into the container. Historically this meant manually passing `--device /dev/kfd --device /dev/dri/renderD128` and a pile of `--group-add` / security options. The AMD Container Toolkit replaces all of that with **CDI**, which lets you request the GPU with a single stable identifier:

```bash
--device amd.com/gpu=all
```

Benefits:

- one declarative device handle instead of raw device nodes,
- works the same way across Docker (and other CDI-aware runtimes),
- keeps the heavy ROCm user-space libraries inside the image, leaving the host lean,
- makes the ROCm/JAX environment reproducible and portable (the exact workflow the JAX guide's container section relies on).

## Prerequisites

1. **A working AMD GPU + ROCm kernel driver on the host.** The toolkit only passes the GPU through; the kernel-side `amdgpu`/KFD driver and firmware must already work:

   ```bash
   rocminfo | grep -E 'gfx103|Marketing Name'
   rocm-smi
   ```

   You should see the Radeon RX 6700 XT listed.

2. **Your user in the `video` and `render` groups** so it can access the GPU device nodes:

   ```bash
   groups
   ```

   If missing:

   ```bash
   sudo usermod -aG video,render "$USER"
   ```

   Then log out and back in for it to take effect.

3. **Docker 25 or newer**, because CDI device injection became generally available in Docker 25+. Install it from the official Docker repository (Debian 13 "trixie" suites are published there). Verify:

   ```bash
   docker version --format 'Client: {{.Client.Version}} / Server: {{.Server.Version}}'
   ```

   Both client and server should report `25.x` or higher (this guide was validated on 29.7.2).

4. **`sudo` rights**, and a few helper packages (`wget`, `gnupg`, `ca-certificates`, `jq`).

## Step-by-step installation

### Step 1 — Install prerequisite packages

```bash
sudo apt-get update
sudo apt-get install -y wget gnupg ca-certificates jq
```

`wget` fetches the AMD GPG key, `gnupg` dearmors it into the keyring format APT expects, and `jq` is handy for inspecting the generated CDI spec later.

### Step 2 — Create the APT keyrings directory

Modern APT stores third-party signing keys in `/etc/apt/keyrings/`. Ensure it exists with correct permissions:

```bash
sudo install -d -m 0755 /etc/apt/keyrings
```

### Step 3 — Add AMD's GPG key

Fetch the ROCm/Radeon GPG key, dearmor it, and install it into the keyrings directory:

```bash
wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor \
    | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
```

Verify it landed:

```bash
ls -l /etc/apt/keyrings/rocm.gpg
```

### Step 4 — Add the AMD Container Toolkit APT repository

Create the sources list entry, signed by the key from Step 3. Note the **`noble`** suite (the Debian 13 workaround):

```bash
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amd-container-toolkit/apt/ noble main" \
    | sudo tee /etc/apt/sources.list.d/amd-container-toolkit.list
```

Refresh the package index:

```bash
sudo apt-get update
```

You should see the new repo fetched:

```text
Get:.. https://repo.radeon.com/amd-container-toolkit/apt noble InRelease
Get:.. https://repo.radeon.com/amd-container-toolkit/apt noble/main amd64 Packages
```

### Step 5 — Install the toolkit

```bash
sudo apt-get install -y amd-container-toolkit
```

Confirm the version:

```bash
amd-ctk version
```

This guide was validated with `1.3.0~24.04`.

### Step 6 — Configure the Docker runtime and enable CDI

Tell the toolkit to configure Docker:

```bash
sudo amd-ctk runtime configure --runtime docker
```

This rewrites `/etc/docker/daemon.json` to:

- register the `amd-container-runtime`, and
- enable Docker's CDI feature (`"features": { "cdi": true }`).

Inspect the result:

```bash
cat /etc/docker/daemon.json
```

Expected:

```json
{
    "features": {
        "cdi": true
    },
    "runtimes": {
        "amd": {
            "args": [],
            "path": "amd-container-runtime"
        }
    }
}
```

### Step 7 — Generate the CDI spec (critical, easy to miss)

This is the step that is missing from many older walkthroughs and is the single most common reason GPU passthrough fails after install. Enabling CDI only tells Docker *how* to consume CDI devices; you must also generate the **spec** that actually defines `amd.com/gpu=all` for your hardware.

First, confirm the toolkit sees the GPU:

```bash
amd-ctk cdi list
```

Expected (one GPU):

```text
Found 1 AMD GPU device
amd.com/gpu=all
amd.com/gpu=0
  /dev/dri/renderD128
```

Then generate the spec into Docker's default CDI directory (`/etc/cdi`):

```bash
sudo install -d -m 0755 /etc/cdi
sudo amd-ctk cdi generate
```

Output:

```text
Generated CDI spec: /etc/cdi/amd.json
```

Verify it exists and describes your device:

```bash
ls -l /etc/cdi/amd.json
jq '.cdiVersion, .devices[].name' /etc/cdi/amd.json
```

> **Without this step**, a container started with `--device amd.com/gpu=all` fails with:
> ```text
> docker: Error response from daemon: CDI device injection failed: unresolvable CDI devices amd.com/gpu=all
> ```
> because Docker has nothing to resolve that device name against. See [Troubleshooting](#troubleshooting).

### Step 8 — Restart Docker

So it picks up the new `daemon.json` and the CDI spec:

```bash
sudo systemctl restart docker
```

### Step 9 — Verify GPU passthrough from inside a container

Run a stock ROCm dev image and probe the GPU with `rocminfo`:

```bash
docker run --device amd.com/gpu=all --rm -it rocm/dev-ubuntu-22.04:latest rocminfo
```

Success looks like an `HSA Agents` section listing both your CPU and the GPU, e.g.:

```text
*******
Agent 2
*******
  Name:                    gfx1031
  Marketing Name:          AMD Radeon RX 6700 XT
  Vendor Name:             AMD
  Device Type:             GPU
  Compute Unit:            40
  Wavefront Size:          32(0x20)
  Workgroup Max Size:      1024(0x400)
  ...
```

> **Which image to use?** The container image's Ubuntu version is **decoupled** from your host OS and from the `noble` APT repo used above — it only needs to be compatible with your host kernel driver. `rocm/dev-ubuntu-22.04:latest` (80+ tags) and `rocm/dev-ubuntu-24.04:latest` (47+ tags) both work; pick whichever ships the ROCm/JAX version you want inside the container. For JAX specifically, prefer the matching `rocm/jax:rocm<ver>-jax<ver>-py<ver>` image referenced in the JAX guide.

## Running ROCm/JAX containers day-to-day

With the toolkit configured, GPU access is a single flag. To get a shell in a ROCm container with your project mounted:

```bash
docker run --device amd.com/gpu=all --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  rocm/jax:rocm7.14-jax0.10.0-py3.11 \
  bash
```

Inside, verify JAX sees the GPU:

```bash
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Device targeting options:

- `--device amd.com/gpu=all` — all GPUs.
- `--device amd.com/gpu=0` — a specific GPU (useful on multi-GPU hosts).

## Troubleshooting

### `unresolvable CDI devices amd.com/gpu=all`

CDI is enabled but the spec is missing or stale. Regenerate and restart:

```bash
sudo amd-ctk cdi generate
sudo systemctl restart docker
```

Then confirm the spec defines the device:

```bash
amd-ctk cdi list
jq '.devices[].name' /etc/cdi/amd.json
```

If you added/removed a GPU or updated the driver, regenerate — the spec is a snapshot of the hardware at generation time.

### Docker ignores the CDI config

Confirm `daemon.json` has `"features": { "cdi": true }`, that your Docker is ≥ 25, and that you restarted the daemon after editing. Re-run `sudo amd-ctk runtime configure --runtime docker` if `daemon.json` was clobbered.

### Permission denied on `/dev/kfd` or `/dev/dri/renderD128`

Your user is not in the `video`/`render` groups (see [Prerequisites](#prerequisites)), or you did not re-login after adding them.

### `apt-get update` 404s on the AMD repo

You are using the wrong suite. The repo only publishes `noble`. Make sure your sources line uses `noble main`, not `trixie`, `debian`, or `ubuntu`.

## Why the container reports `gfx1031` but the host reports `gfx1030`

This is expected, not a bug, and it confuses almost everyone on first contact.

- **Hardware truth:** the RX 6700 XT is Navi 22, whose GCN ISA target is **`gfx1031`** (`gfx1030` is Navi 21 — the RX 6800 / 6800 XT / 6900 XT family). The container confirms this directly.
- **The host reports `gfx1030` regardless of the Python device package.** Verified on this machine: both `/usr/bin/rocminfo` (system ROCm 7.1) and `.venv/bin/rocminfo` (ROCm 7.14) report `gfx1030`. Swapping the `rocm-sdk-device-gfx1030` package for `gfx1031` did **not** change the host's report — so the device package is *not* what drives it.
- **The container reports `gfx1031`** because the `rocm/dev-ubuntu-*` image ships its own ROCm build whose device-name mapping resolves this GPU to `gfx1031`.

The same physical GPU reads as `gfx1030` under the host's ROCm and `gfx1031` under the container's ROCm. The difference lives in each ROCm userspace build's device mapping — not in the shared kernel driver and not in the Python device package.

**Practical guidance:** JAX/XLA/Triton compile for whatever the host HSA runtime reports — on this host that is `gfx1030` — so this repo uses the matching `rocm-sdk-device-gfx1030` package and the committed `rocminfo.txt` shows `gfx1030`. The container's `gfx1031` identity is internal to that image and does not affect host-side JAX.

## Uninstall / cleanup

Remove the toolkit, its repository, and the CDI spec:

```bash
sudo apt-get remove -y amd-container-toolkit
sudo rm -f /etc/apt/sources.list.d/amd-container-toolkit.list
sudo rm -f /etc/apt/keyrings/rocm.gpg /etc/cdi/amd.json
sudo apt-get update
```

Restore `/etc/docker/daemon.json` manually if you want to disable the `amd` runtime and CDI feature, then `sudo systemctl restart docker`.

## Reference: automated install script

All of the steps above are automated in:

```text
docs/install-guides/amd-container-toolkit/install-amd-container-toolkit.sh
```

It performs the full sequence — prerequisites, keyring, GPG key, `noble` repo, install, Docker runtime/CDI configuration, **CDI spec generation**, and Docker restart — and is the recommended way to reproduce this setup. Run it with:

```bash
chmod +x docs/install-guides/amd-container-toolkit/install-amd-container-toolkit.sh
./docs/install-guides/amd-container-toolkit/install-amd-container-toolkit.sh
```
