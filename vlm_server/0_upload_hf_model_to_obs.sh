#!/usr/bin/env bash
# Upload local Hugging Face cache (model) to OBS — patterned after
# DockerSpaces/steps_to_go/4_upload_datasets.sh
#
# Why flatten (--copy-links):
#   HF hub uses snapshots/* -> ../../blobs/<hash> symlinks. OBS→SFS import often
#   turns those into 0-byte mode-000 files (Permission denied / bad JSON).
#   Flattening uploads real files so SFS import is usable offline.
#
# Prerequisites:
#   - obsutil configured (same as DockerSpaces)
#   - local model already downloaded under ~/.cache/huggingface/hub/...
#
# Usage:
#   # Dry-run Qwen2.5-VL-7B (default)
#   DRY_RUN=1 ./0_upload_hf_model_to_obs.sh
#
#   # Real upload
#   DRY_RUN=0 ./0_upload_hf_model_to_obs.sh
#
#   # Custom model + OBS root
#   MODEL_DIR_NAME=models--Qwen--Qwen2.5-VL-7B-Instruct \
#   OBS_DATA_ROOT=obs://obs-a100-sai15/gn/vlm \
#   DRY_RUN=0 ./0_upload_hf_model_to_obs.sh
#
# After OBS upload: import to SFS with 用户ID=1000 用户组ID=100, mount as
#   /mnt/dev/vlm/model_cache/huggingface
# so that hub/<MODEL_DIR_NAME>/ exists under HF_HOME.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"

# Prefer DockerSpaces obsutil if present.
_DEFAULT_OBSUTIL_CANDIDATES=(
  "${OBSUTIL:-}"
  "${REPO_ROOT}/tools/obsutil_linux_amd64_5.8.3/obsutil"
  "${HOME}/Documents/Git_space/DockerSpaces/tools/obsutil_linux_amd64_5.8.3/obsutil"
)

OBSUTIL=""
for c in "${_DEFAULT_OBSUTIL_CANDIDATES[@]}"; do
  [[ -z "$c" ]] && continue
  if [[ -x "$c" ]]; then
    OBSUTIL="$c"
    break
  fi
done

OBS_DATA_ROOT="${OBS_DATA_ROOT:-obs://obs-a100-sai15/gn/vlm}"
LOCAL_HF_HOME="${LOCAL_HF_HOME:-${HOME}/.cache/huggingface}"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-models--Qwen--Qwen2.5-VL-7B-Instruct}"

OBSUTIL_OUT_DIR="${OBSUTIL_OUT_DIR:-${ROOT}/.obsutil/out}"
OBSUTIL_CPD_DIR="${OBSUTIL_CPD_DIR:-${ROOT}/.obsutil/cpd}"

DRY_RUN="${DRY_RUN:-1}"
METHOD="${METHOD:-sync}"
JOBS="${JOBS:-5}"
PARALLELS="${PARALLELS:-5}"
DISABLE_PROXY="${DISABLE_PROXY:-1}"
FLATTEN="${FLATTEN:-1}"

LOCAL_MODEL_SRC="${LOCAL_HF_HOME}/hub/${MODEL_DIR_NAME}"
REMOTE_HUB_PARENT="${OBS_DATA_ROOT%/}/model_cache/huggingface/hub"
REMOTE_SYNC_DIR="${REMOTE_HUB_PARENT}/${MODEL_DIR_NAME}"

mkdir -p "$OBSUTIL_OUT_DIR" "$OBSUTIL_CPD_DIR"

if [[ -z "$OBSUTIL" || ! -x "$OBSUTIL" ]]; then
  echo "[upload-hf] ERROR: obsutil not found/executable." >&2
  echo "[upload-hf] Set OBSUTIL=/path/to/obsutil (e.g. DockerSpaces/tools/.../obsutil)" >&2
  exit 1
fi

if [[ ! -d "$LOCAL_MODEL_SRC" ]]; then
  echo "[upload-hf] ERROR: local model not found: $LOCAL_MODEL_SRC" >&2
  echo "[upload-hf] Download first, or set LOCAL_HF_HOME / MODEL_DIR_NAME" >&2
  exit 1
fi

STAGE_DIR=""
cleanup() {
  if [[ -n "${STAGE_DIR}" && -d "${STAGE_DIR}" ]]; then
    rm -rf "${STAGE_DIR}"
  fi
}
trap cleanup EXIT

LOCAL_UPLOAD_DIR="$LOCAL_MODEL_SRC"
if [[ "$FLATTEN" == "1" ]]; then
  STAGE_DIR="$(mktemp -d /tmp/vlm-hf-flat.XXXXXX)"
  echo "[upload-hf] flattening symlinks → ${STAGE_DIR}/${MODEL_DIR_NAME}"
  mkdir -p "${STAGE_DIR}/${MODEL_DIR_NAME}"
  # Prefer rsync --copy-links; fall back to cp -aL
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --copy-links "${LOCAL_MODEL_SRC}/" "${STAGE_DIR}/${MODEL_DIR_NAME}/"
  else
    cp -aL "${LOCAL_MODEL_SRC}/." "${STAGE_DIR}/${MODEL_DIR_NAME}/"
  fi
  LOCAL_UPLOAD_DIR="${STAGE_DIR}/${MODEL_DIR_NAME}"
  # Quick integrity check on flattened config.json
  if [[ -f "${LOCAL_UPLOAD_DIR}/snapshots" ]] || [[ -d "${LOCAL_UPLOAD_DIR}/snapshots" ]]; then
    cfg="$(find "${LOCAL_UPLOAD_DIR}/snapshots" -name config.json -type f 2>/dev/null | head -1 || true)"
    if [[ -n "$cfg" ]]; then
      python3 -c "import json; json.load(open('${cfg}')); print('[upload-hf] flattened config.json OK:', '${cfg}')"
    fi
  fi
fi

case "$METHOD" in
  sync)
    args=(
      sync "$LOCAL_UPLOAD_DIR" "$REMOTE_SYNC_DIR"
      "-j=${JOBS}"
      "-p=${PARALLELS}"
      "-o=${OBSUTIL_OUT_DIR}"
      "-cpd=${OBSUTIL_CPD_DIR}"
    )
    ;;
  cp)
    args=(
      cp "$LOCAL_UPLOAD_DIR" "$REMOTE_HUB_PARENT"
      -r
      -f
      -u
      "-j=${JOBS}"
      "-p=${PARALLELS}"
      "-o=${OBSUTIL_OUT_DIR}"
      "-cpd=${OBSUTIL_CPD_DIR}"
    )
    ;;
  *)
    echo "[upload-hf] ERROR: METHOD must be sync or cp, got: $METHOD" >&2
    exit 1
    ;;
esac

if [[ "$DRY_RUN" == "1" ]]; then
  args+=(-dryRun)
fi

echo "[upload-hf] obsutil=${OBSUTIL}"
echo "[upload-hf] local_hf_home=${LOCAL_HF_HOME}"
echo "[upload-hf] model_dir_name=${MODEL_DIR_NAME}"
echo "[upload-hf] local_upload_dir=${LOCAL_UPLOAD_DIR}"
echo "[upload-hf] remote_sync_dir=${REMOTE_SYNC_DIR}"
echo "[upload-hf] flatten=${FLATTEN} method=${METHOD} dry_run=${DRY_RUN}"
echo "[upload-hf] jobs=${JOBS} parallels=${PARALLELS} disable_proxy=${DISABLE_PROXY}"
echo "[upload-hf] command=${OBSUTIL} ${args[*]}"

if [[ "$DISABLE_PROXY" == "1" ]]; then
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    "$OBSUTIL" "${args[@]}"
else
  "$OBSUTIL" "${args[@]}"
fi

echo "[upload-hf] done."
echo "[upload-hf] next: SFS import from OBS with UID=1000 GID=100"
echo "[upload-hf] mount → /mnt/dev/vlm/model_cache/huggingface (HF_HOME)"
echo "[upload-hf] verify: ls -la \$HF_HOME/hub/${MODEL_DIR_NAME}/snapshots/*/config.json"
