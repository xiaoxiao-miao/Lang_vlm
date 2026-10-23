#!/usr/bin/env bash
# Path B / step 1 — 本机 Docker：构建 vlm-server:v2.3（5090 / 新驱动栈）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

DOCKER="${DOCKER:-docker}"
BUILD_UID="${BUILD_UID:-1000}"
BUILD_GID="${BUILD_GID:-100}"
IMAGE_REPO="${IMAGE_REPO:-vlm-server}"
IMAGE_TAG="${IMAGE_TAG:-v2.3}"
LOCAL_IMAGE="${LOCAL_IMAGE:-${IMAGE_REPO}:${IMAGE_TAG}}"
BASE_IMAGE="${BASE_IMAGE:-swr.cn-north-9.myhuaweicloud.com/sai-guon/lerobot-gpu:v6-tmux}"

STORAGE_ROOT="${STORAGE_ROOT:-/mnt/dev/vlm}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${STORAGE_ROOT}/model_cache}"
HF_HOME="${HF_HOME:-${MODEL_CACHE_DIR}/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
TORCH_HOME="${TORCH_HOME:-${MODEL_CACHE_DIR}/torch}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/ma-user/.cache/triton}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if ! ${DOCKER} info >/dev/null 2>&1; then
  echo "error: docker not usable. One-time:" >&2
  echo "  sudo bash ${ROOT}/0_setup_docker_prereqs.sh" >&2
  echo "  sudo bash ${ROOT}/0_setup_docker_proxy.sh" >&2
  echo "  newgrp docker" >&2
  exit 1
fi

echo "[docker_1] pull base ${BASE_IMAGE}"
# shellcheck disable=SC2086
${DOCKER} pull "${BASE_IMAGE}"

export DOCKER_BUILDKIT=1
# shellcheck disable=SC2086
${DOCKER} build \
  -f Dockerfile \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
  --build-arg "TORCH_INDEX=${TORCH_INDEX}" \
  --build-arg "UID=${BUILD_UID}" \
  --build-arg "GID=${BUILD_GID}" \
  --build-arg "HOME=/home/ma-user" \
  --build-arg "STORAGE_ROOT=${STORAGE_ROOT}" \
  --build-arg "MODEL_CACHE_DIR=${MODEL_CACHE_DIR}" \
  --build-arg "HF_HOME=${HF_HOME}" \
  --build-arg "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
  --build-arg "TORCH_HOME=${TORCH_HOME}" \
  --build-arg "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}" \
  --build-arg "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}" \
  --build-arg "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}" \
  -t "${LOCAL_IMAGE}" \
  .

${DOCKER} image inspect "${LOCAL_IMAGE}" >/dev/null
echo "[docker_1] built ${LOCAL_IMAGE} → next: ./docker_2_run.sh"
