#!/usr/bin/env bash
# Path A / step 1 — 本机 local：检查环境（GPU、venv、端口提示）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
cd "${ROOT}"

echo "[local_1] repo=${REPO}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found" >&2
  exit 1
fi
nvidia-smi -L || true

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "error: missing ${ROOT}/.venv" >&2
  echo "Create a venv with CUDA torch + vllm matching your GPU, e.g.:" >&2
  echo "  cd ${ROOT} && python3 -m venv .venv && .venv/bin/pip install vllm qwen-vl-utils" >&2
  exit 1
fi

MODEL_DIR="${HF_CACHE:-${HOME}/.cache/huggingface}/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "warn: model cache not found: ${MODEL_DIR}"
  echo "  First online run of local_2_run.sh will download it (needs network)."
else
  echo "[local_1] model cache ok: ${MODEL_DIR}"
fi

if ss -ltn 2>/dev/null | grep -q ':8000 '; then
  echo "warn: :8000 in use — stop before local_2_run.sh:"
  echo "  pkill -f 'vllm.entrypoints.openai.api_server' || true"
fi

echo "[local_1] ok → next: ./local_2_run.sh"
