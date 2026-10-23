#!/usr/bin/env bash
# Path C — 构建并推送 vlm-server:v2.3 到 Huawei SWR（Docker V2，非 OCI）
#
# Usage:
#   docker login -u <user>@<domain> swr.cn-north-9.myhuaweicloud.com
#   ./cloud_2_push_swr.sh
#
# Optional overrides:
#   LOCAL_IMAGE=vlm-server:v2.3
#   SWR_IMAGE=swr.cn-north-9.myhuaweicloud.com/sai-guon/vlm-server:v2.3
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

DOCKER="${DOCKER:-docker}"

BUILD_UID="${BUILD_UID:-1000}"
BUILD_GID="${BUILD_GID:-100}"
STORAGE_ROOT="${STORAGE_ROOT:-/mnt/dev/vlm}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${STORAGE_ROOT}/model_cache}"
HF_HOME="${HF_HOME:-${MODEL_CACHE_DIR}/huggingface}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/ma-user/.cache/triton}"

LOCAL_IMAGE="${LOCAL_IMAGE:-vlm-server:v2.3}"
SWR_IMAGE="${SWR_IMAGE:-swr.cn-north-9.myhuaweicloud.com/sai-guon/vlm-server:v2.3}"
BASE_IMAGE="${BASE_IMAGE:-swr.cn-north-9.myhuaweicloud.com/sai-guon/lerobot-gpu:v6-tmux}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

echo "[cloud_2] SWR_IMAGE=${SWR_IMAGE}"
echo "[cloud_2] ensure: docker login ... swr.cn-north-9.myhuaweicloud.com"
echo "[cloud_2] base=${BASE_IMAGE}"
# shellcheck disable=SC2086
${DOCKER} pull "${BASE_IMAGE}"

echo "[cloud_2] rebuild ${LOCAL_IMAGE} -> ${SWR_IMAGE} (Docker V2)"
# shellcheck disable=SC2086
${DOCKER} buildx build \
  --builder default \
  --provenance=false \
  --sbom=false \
  --platform linux/amd64 \
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
  --build-arg "TRANSFORMERS_CACHE=${HF_HOME}" \
  --build-arg "TORCH_HOME=${MODEL_CACHE_DIR}/torch" \
  --build-arg "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}" \
  --build-arg "HF_HUB_OFFLINE=1" \
  --build-arg "TRANSFORMERS_OFFLINE=1" \
  -t "${LOCAL_IMAGE}" \
  -t "${SWR_IMAGE}" \
  --output "type=docker,oci-mediatypes=false" \
  .

echo "[cloud_2] docker push ${SWR_IMAGE}"
# shellcheck disable=SC2086
${DOCKER} push "${SWR_IMAGE}"
echo "[cloud_2] done. VLM_IMAGE=${SWR_IMAGE}"
