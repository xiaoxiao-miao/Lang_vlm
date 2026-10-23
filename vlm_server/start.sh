#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# --- LD_LIBRARY_PATH (system CUDA + pip nvidia libs) --------------------------
if [[ -f /app/docker_smoke_import.py ]]; then
  _nvidia_lib_path="$(python /app/docker_smoke_import.py --print-libpath)"
else
  _nvidia_lib_path="/usr/local/cuda/lib64"
fi
if [[ -n "${_nvidia_lib_path}" ]]; then
  export LD_LIBRARY_PATH="${_nvidia_lib_path}${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH:-}"
fi

# --- Runtime defaults ---------------------------------------------------------
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
# vLLM 0.26+ uses --attention-backend CLI; avoid unknown-env warning.
unset VLLM_ATTENTION_BACKEND 2>/dev/null || true
MM_ENCODER_ATTN_BACKEND="${MM_ENCODER_ATTN_BACKEND:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
# vLLM >=0.26 needs JSON. Do NOT use ${VAR:-{...}} — the closing } ends the expansion early.
# Also ignore broken image ENV leftovers like image=8,video=0 or image=8,video=0}
if [[ -z "${LIMIT_MM_PER_PROMPT:-}" || "${LIMIT_MM_PER_PROMPT}" != \{* ]]; then
  LIMIT_MM_PER_PROMPT='{"image":8,"video":0}'
fi
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-0}"
# LangChain / robot-agent tool-calling needs these (vLLM rejects tool_choice=auto otherwise).
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
# Qwen2.5(-VL) uses Hermes-style <tool_call> tags.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"

# Triton/Inductor must be on local disk — SFS/NFS breaks os.replace.
_triton_default="${HOME:-/tmp}/.cache/triton"
_torchinductor_default="${HOME:-/tmp}/.cache/torchinductor"
case "${TRITON_CACHE_DIR:-}" in
  /mnt/*|"") export TRITON_CACHE_DIR="${TRITON_CACHE_DIR_LOCAL:-${_triton_default}}" ;;
esac
case "${TORCHINDUCTOR_CACHE_DIR:-}" in
  /mnt/*|"") export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR_LOCAL:-${_torchinductor_default}}" ;;
esac
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

_VLLM_HELP="$(python -m vllm.entrypoints.openai.api_server --help 2>&1 || true)"
_has_flag() { grep -qE -- "$1" <<<"${_VLLM_HELP}"; }

EXTRA_ARGS=()
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  EXTRA_ARGS+=(--trust-remote-code)
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if _has_flag '--attention-backend'; then
  EXTRA_ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
fi
if [[ -n "${MM_ENCODER_ATTN_BACKEND}" ]] && _has_flag '--mm-encoder-attn-backend'; then
  EXTRA_ARGS+=(--mm-encoder-attn-backend "${MM_ENCODER_ATTN_BACKEND}")
fi
if [[ -n "${LIMIT_MM_PER_PROMPT}" ]] && _has_flag '--limit-mm-per-prompt'; then
  EXTRA_ARGS+=(--limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}")
fi
if [[ "${SKIP_MM_PROFILING}" == "1" ]] && _has_flag '--skip-mm-profiling'; then
  EXTRA_ARGS+=(--skip-mm-profiling)
fi
if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]] && _has_flag '--enable-auto-tool-choice'; then
  EXTRA_ARGS+=(--enable-auto-tool-choice)
  if [[ -n "${TOOL_CALL_PARSER}" ]] && _has_flag '--tool-call-parser'; then
    EXTRA_ARGS+=(--tool-call-parser "${TOOL_CALL_PARSER}")
  fi
fi

echo "[vlm_server] user=$(id -un):$(id -gn) uid=$(id -u) gid=$(id -g)"
echo "[vlm_server] note: ModelArts may set gid=0; UID 1000 must match SFS ownership"
echo "[vlm_server] HF_HOME=${HF_HOME:-unset}"
echo "[vlm_server] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "[vlm_server] TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
echo "[vlm_server] MODEL_NAME=${MODEL_NAME}"
echo "[vlm_server] listening on ${HOST}:${PORT}"
echo "[vlm_server] VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER}"
echo "[vlm_server] attn=${ATTENTION_BACKEND} mm_encoder_attn=${MM_ENCODER_ATTN_BACKEND:-unset} ENFORCE_EAGER=${ENFORCE_EAGER}"
echo "[vlm_server] LIMIT_MM_PER_PROMPT=${LIMIT_MM_PER_PROMPT} SKIP_MM_PROFILING=${SKIP_MM_PROFILING}"
echo "[vlm_server] tool_choice=${ENABLE_AUTO_TOOL_CHOICE} parser=${TOOL_CALL_PARSER}"
echo "[vlm_server] LD_LIBRARY_PATH (ordered): ${_nvidia_lib_path:-none}"

python - <<'PY'
import torch, torchaudio
try:
    import flash_attn
    fa = flash_attn.__version__
except Exception as e:
    fa = f"unavailable ({e})"
try:
    import vllm
    vv = getattr(vllm, "__version__", "?")
except Exception as e:
    vv = f"unavailable ({e})"
print(f"[vlm_server] torch={torch.__version__} cuda={torch.version.cuda} torchaudio={torchaudio.__version__} flash_attn={fa} vllm={vv}")
try:
    print(f"[vlm_server] gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
except Exception as e:
    print(f"[vlm_server] gpu probe failed: {e}")
PY

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
