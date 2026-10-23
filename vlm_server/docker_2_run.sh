#!/usr/bin/env bash
# Path B / step 2 — 本机 Docker：离线启动容器（挂载本机 HF 缓存）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

export HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
MODEL_DIR="${HF_CACHE}/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "error: model cache missing: ${MODEL_DIR}" >&2
  echo "Run ./local_2_run.sh once online to download, then retry." >&2
  exit 1
fi

if ss -ltn 2>/dev/null | grep -q ':8000 '; then
  echo "error: port 8000 in use. Stop host vLLM first:" >&2
  echo "  pkill -f 'vllm.entrypoints.openai.api_server' || true" >&2
  exit 1
fi

echo "[docker_2] HF_CACHE=${HF_CACHE} (read-only mount, offline)"
docker compose up -d --build

echo "[docker_2] waiting for /v1/models ..."
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then
    curl -sS "http://127.0.0.1:8000/v1/models"
    echo
    echo "[docker_2] ready → next: ./docker_3_agent.sh"
    exit 0
  fi
  sleep 5
done

echo "error: not ready. Logs:" >&2
docker compose logs --tail 80
exit 1
