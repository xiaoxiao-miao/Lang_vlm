#!/usr/bin/env bash
# One-time host setup for: local agent_client + Docker VLM (offline runtime).
# Run:  sudo bash vlm_server/0_setup_docker_prereqs.sh
# If curl fails under sudo, proxy env was stripped — this script auto-detects local proxy.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run with sudo: sudo bash $0" >&2
  exit 1
fi

TARGET_USER="${SUDO_USER:-${USER}}"

# sudo often drops proxy; restore common local proxy if present.
_setup_proxy() {
  if [[ -n "${https_proxy:-}${HTTPS_PROXY:-}${http_proxy:-}${HTTP_PROXY:-}" ]]; then
    return 0
  fi
  # Clash / mihomo default HTTP port on this machine
  if ss -ltn 2>/dev/null | grep -q ':7897 '; then
    export http_proxy="http://127.0.0.1:7897/"
    export https_proxy="http://127.0.0.1:7897/"
    export HTTP_PROXY="${http_proxy}"
    export HTTPS_PROXY="${https_proxy}"
    echo "[proxy] using ${http_proxy}"
  fi
  unset ALL_PROXY all_proxy || true
}
_setup_proxy

echo "[1/3] Add ${TARGET_USER} to docker group"
groupadd -f docker
usermod -aG docker "${TARGET_USER}"

echo "[2/3] Install NVIDIA Container Toolkit (for --gpus)"
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  TMP_KEY="$(mktemp)"
  TMP_LIST="$(mktemp)"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o "${TMP_KEY}"
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg "${TMP_KEY}"
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list -o "${TMP_LIST}"
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    "${TMP_LIST}" > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  rm -f "${TMP_KEY}" "${TMP_LIST}"

  # apt may also need proxy for nvidia repo
  if [[ -n "${http_proxy:-}" ]]; then
    mkdir -p /etc/apt/apt.conf.d
    cat > /etc/apt/apt.conf.d/99proxy-temp-nvidia <<EOF
Acquire::http::Proxy "${http_proxy}";
Acquire::https::Proxy "${https_proxy}";
EOF
  fi

  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker

  rm -f /etc/apt/apt.conf.d/99proxy-temp-nvidia
else
  echo "nvidia-ctk already present"
fi

echo "[3/3] Verify GPU inside Docker"
# Pull may need daemon proxy (sudo strips shell proxy). Soft-fail with hint.
if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi; then
  echo
  echo "WARN: GPU verify failed — usually Docker Hub pull timeout (no daemon proxy)."
  echo "nvidia-container-toolkit is installed. Next fix pull, then re-verify:"
  echo "  sudo bash vlm_server/0_setup_docker_proxy.sh"
  echo "  newgrp docker"
  echo "  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"
  exit 0
fi

echo
echo "Done. Activate docker group in this shell:"
echo "  newgrp docker"
echo "Then:"
echo "  pkill -f 'vllm.entrypoints.openai.api_server' || true"
echo "  cd ~/Documents/Miao_space/Lang_vlm/vlm_server && ./docker_2_run.sh"
