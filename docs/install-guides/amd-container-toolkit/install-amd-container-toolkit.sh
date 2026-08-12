#!/usr/bin/env bash
set -euo pipefail

# Debian 13 Trixie workaround:
# AMD's amd-container-toolkit APT repo does not currently publish "trixie".
# Use Ubuntu 24.04 "noble" instead.

export DEBIAN_FRONTEND=noninteractive

echo "Installing prerequisites..."
sudo apt-get update
sudo apt-get install -y wget gnupg ca-certificates jq

# Docker is required: the AMD Container Toolkit configures a CDI-compatible runtime.
# Install it yourself if missing (Docker 25+ or another CDI-compatible runtime).
if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required but not installed. Install Docker 25+ and re-run." >&2
    exit 1
fi

echo "Ensuring APT keyrings directory exists..."
sudo install -d -m 0755 /etc/apt/keyrings

echo "Fetching AMD ROCm GPG key..."
wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor \
    | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "Adding AMD Container Toolkit repository..."
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amd-container-toolkit/apt/ noble main" \
    | sudo tee /etc/apt/sources.list.d/amd-container-toolkit.list

echo "Updating APT..."
sudo apt-get update

echo "Installing amd-container-toolkit..."
sudo apt-get install -y amd-container-toolkit

echo "Configuring Docker for AMD GPU CDI access..."
sudo amd-ctk runtime configure --runtime docker

echo "Restarting Docker..."
sudo systemctl restart docker

echo "Done."
echo
echo "Test with:"
echo "  docker run --device amd.com/gpu=all --rm -it rocm/dev-ubuntu-22.04:latest rocminfo"
