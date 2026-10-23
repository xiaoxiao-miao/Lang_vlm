#!/usr/bin/env bash
# Path C / step 3 — 云端 Docker：在 ModelArts / 云主机上拉镜像并启动
# Run on the cloud GPU host (not the laptop build machine).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

ENV_FILE="${ENV_FILE:-${ROOT}/.env.cloud}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "error: missing ${ENV_FILE}" >&2
  echo "  cp .env.cloud.example .env.cloud   # set VLM_IMAGE to SWR tag" >&2
  exit 1
fi

# shellcheck disable=SC1090
set -a
# shellcheck source=/dev/null
source "${ENV_FILE}"
set +a

: "${VLM_IMAGE:?VLM_IMAGE required in ${ENV_FILE}}"
HF_HOME_IN_CONTAINER="${HF_HOME:-/mnt/dev/vlm/model_cache/huggingface}"
HF_CACHE="${HF_CACHE:-${HF_HOME_IN_CONTAINER}}"
MODEL_DIR="${HF_CACHE}/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "warn: expected cache dir not found: ${MODEL_DIR}" >&2
  echo "  Upload with ./0_upload_hf_model_to_obs.sh then SFS import uid=1000 gid=100" >&2
fi

if ss -ltn 2>/dev/null | grep -q ":${HOST_PORT:-8000} "; then
  echo "error: port ${HOST_PORT:-8000} in use" >&2
  exit 1
fi

echo "[cloud_3] VLM_IMAGE=${VLM_IMAGE}"
echo "[cloud_3] HF_HOME=${HF_HOME_IN_CONTAINER}"
docker compose --env-file "${ENV_FILE}" -f docker-compose.cloud.yaml pull || true
docker compose --env-file "${ENV_FILE}" -f docker-compose.cloud.yaml up -d

echo "[cloud_3] waiting for /v1/models ..."
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${HOST_PORT:-8000}/v1/models" >/dev/null 2>&1; then
    curl -sS "http://127.0.0.1:${HOST_PORT:-8000}/v1/models"
    echo
    echo "[cloud_3] ready. Laptop Agent:"
    echo "  uv run vlm-agent --config configs/config.cloud.yaml --image <img> --text '...'"
    exit 0
  fi
  sleep 5
done

echo "error: not ready. Logs:" >&2
docker compose --env-file "${ENV_FILE}" -f docker-compose.cloud.yaml logs --tail 100
exit 1
