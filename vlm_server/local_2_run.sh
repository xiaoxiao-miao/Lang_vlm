#!/usr/bin/env bash
# Path A / step 2 — 本机 local：启动宿主机 vLLM（无 Docker）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

# httpx rejects scheme "socks://"; keep http(s)_proxy.
unset ALL_PROXY all_proxy || true
if [[ "${HTTP_PROXY:-}" == socks://* ]]; then
  export HTTP_PROXY="socks5://${HTTP_PROXY#socks://}"
fi
if [[ "${HTTPS_PROXY:-}" == socks://* ]]; then
  export HTTPS_PROXY="socks5://${HTTPS_PROXY#socks://}"
fi

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "error: run ./local_1_prepare.sh first (missing .venv)" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
exec "${ROOT}/start.sh" "$@"
