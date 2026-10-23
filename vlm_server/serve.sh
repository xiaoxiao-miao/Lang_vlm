#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ID:?Set MODEL_ID to a Hugging Face model ID, for example google/gemma-4-26B-A4B-it}"
: "${MODEL_CACHE_DIR:?MODEL_CACHE_DIR must be set}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":4}}"

if [[ ! -d "$MODEL_CACHE_DIR" ]]; then
  echo "[serve] ERROR: MODEL_CACHE_DIR not found: $MODEL_CACHE_DIR" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[serve] ERROR: nvidia-smi not found" >&2
  exit 1
fi

VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
VISIBLE_GPU_COUNT="${VISIBLE_GPU_COUNT//[[:space:]]/}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${VISIBLE_GPU_COUNT}}"

if (( VISIBLE_GPU_COUNT < 1 )); then
  echo "[serve] ERROR: no visible NVIDIA GPU" >&2
  exit 1
fi

if (( TENSOR_PARALLEL_SIZE < 1 || TENSOR_PARALLEL_SIZE > VISIBLE_GPU_COUNT )); then
  echo "[serve] ERROR: TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}, visible GPUs=${VISIBLE_GPU_COUNT}" >&2
  exit 1
fi

args=(
  serve "$MODEL_ID"
  --served-model-name "$SERVED_MODEL_NAME"
  --download-dir "$MODEL_CACHE_DIR"
  --host "$VLLM_HOST"
  --port "$VLLM_PORT"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --dtype bfloat16
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT"
)

if [[ -n "${VLLM_API_KEY:-}" ]]; then
  args+=(--api-key "$VLLM_API_KEY")
fi

echo "[serve] model=${MODEL_ID}"
echo "[serve] served_model_name=${SERVED_MODEL_NAME}"
echo "[serve] model_cache_dir=${MODEL_CACHE_DIR}"
echo "[serve] offline=${HF_HUB_OFFLINE:-<unset>}"
echo "[serve] listen=${VLLM_HOST}:${VLLM_PORT}"
echo "[serve] visible_gpus=${VISIBLE_GPU_COUNT} tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
echo "[serve] max_model_len=${MAX_MODEL_LEN} gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo "[serve] limit_mm_per_prompt=${LIMIT_MM_PER_PROMPT}"

exec vllm "${args[@]}"
