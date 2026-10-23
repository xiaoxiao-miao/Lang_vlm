# syntax=docker/dockerfile:1
# VLM OpenAI-compatible server (vLLM + Qwen2.5-VL)
# Tag: v2.3 — FROM lerobot-gpu:v6-tmux (already has CUDA/torch/flash-attn)
#
# Why this base:
#   Compiling flash-attn locally OOMs / kills the host. Reuse the prebuilt stack from
#   DockerSpaces (lerobot-gpu-base → lerobot-gpu:v6-tmux on SWR).
#
# Base already provides (see docker/lerobot-gpu-base.Dockerfile + .devcontainer/lerobot/Dockerfile):
#   CUDA 12.8.1 + Ubuntu 24.04 + Python 3.12 + /opt/venv
#   torch 2.11 (cu128) + flash-attn 2.8.3 + ma-user (UID 1000)
#
# This layer only adds inference deps (NO flash-attn rebuild):
#   vllm + qwen-vl-utils + huggingface_hub + torchaudio
#   then re-pin torch/vision/audio to cu128 (same versions) so flash-attn ABI stays valid
#
ARG BASE_IMAGE=swr.cn-north-9.myhuaweicloud.com/sai-guon/lerobot-gpu:v6-tmux
FROM ${BASE_IMAGE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.11.0
ARG VLLM_VERSION=
ARG UID=1000
ARG GID=100
ARG HOME=/home/ma-user
ARG STORAGE_ROOT=/mnt/dev/vlm
ARG MODEL_CACHE_DIR=/mnt/dev/vlm/model_cache
ARG HF_HOME=/mnt/dev/vlm/model_cache/huggingface
ARG TRANSFORMERS_CACHE=/mnt/dev/vlm/model_cache/huggingface
ARG TORCH_HOME=/mnt/dev/vlm/model_cache/torch
ARG TRITON_CACHE_DIR=/home/ma-user/.cache/triton
ARG HF_HUB_OFFLINE=1
ARG TRANSFORMERS_OFFLINE=1
ARG VENV_PATH=/opt/venv

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=${VENV_PATH} \
    PATH=${VENV_PATH}/bin:/usr/local/cuda/bin:/usr/local/bin:${PATH} \
    HOME=${HOME} \
    STORAGE_ROOT=${STORAGE_ROOT} \
    MODEL_CACHE_DIR=${MODEL_CACHE_DIR} \
    HF_HOME=${HF_HOME} \
    TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} \
    TORCH_HOME=${TORCH_HOME} \
    TRITON_CACHE_DIR=${TRITON_CACHE_DIR} \
    TORCHINDUCTOR_CACHE_DIR=/home/ma-user/.cache/torchinductor \
    MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct \
    HOST=0.0.0.0 \
    PORT=8000 \
    DTYPE=bfloat16 \
    GPU_MEMORY_UTILIZATION=0.85 \
    MAX_MODEL_LEN=8192 \
    MAX_NUM_SEQS=8 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    MM_ENCODER_ATTN_BACKEND=TRITON_ATTN \
    ENFORCE_EAGER=1 \
    SKIP_MM_PROFILING=1 \
    ENABLE_AUTO_TOOL_CHOICE=1 \
    TOOL_CALL_PARSER=hermes \
    HF_HUB_OFFLINE=${HF_HUB_OFFLINE} \
    TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE} \
    HF_HUB_DISABLE_TELEMETRY=1

USER root
# Lightweight extras only (base already has toolchain / tmux / ffmpeg / etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      findutils \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p "$HOME" "$HOME/.cache/triton" "$HOME/.cache/torchinductor" \
      "$STORAGE_ROOT" "$MODEL_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" /app \
    && chown -R "${UID}:${GID}" "$HOME" "$STORAGE_ROOT" /app \
    && chown -R "${UID}:${GID}" "${VENV_PATH}" || true

USER ma-user
WORKDIR /app
COPY --chown=${UID}:${GID} start.sh docker_smoke_import.py /app/
RUN chmod +x /app/start.sh

# Reuse base /opt/venv — never rebuild flash-attn.
# uv is present in lerobot-gpu images; fall back to pip if missing.
RUN python -c "import torch, flash_attn; print(f'base ok: torch={torch.__version__} flash_attn={flash_attn.__version__}')" \
    && uv pip install --no-cache-dir \
         torchaudio \
         --index-url "${TORCH_INDEX}" \
    && if [ -n "${VLLM_VERSION}" ]; then \
         uv pip install --no-cache-dir "vllm==${VLLM_VERSION}"; \
       else \
         uv pip install --no-cache-dir "vllm>=0.8.0"; \
       fi \
    && uv pip install --no-cache-dir \
         "qwen-vl-utils>=0.0.8" \
         "huggingface_hub" \
    && uv pip install --no-cache-dir --reinstall \
         "torch==${TORCH_VERSION}" \
         torchvision \
         torchaudio \
         --index-url "${TORCH_INDEX}" \
    && (uv pip install --no-cache-dir "nvidia-cuda-runtime-cu13" || true) \
    && (uv pip install --no-cache-dir "nvidia-cuda-runtime-cu12" || true) \
    && LD_LIBRARY_PATH="$(python /app/docker_smoke_import.py --print-libpath)" \
         python /app/docker_smoke_import.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=10 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/v1/models" || exit 1

CMD ["/app/start.sh"]
