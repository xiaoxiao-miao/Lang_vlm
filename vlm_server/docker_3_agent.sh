#!/usr/bin/env bash
# Path B / step 3 — 本机 Docker：Agent 联调（逻辑在 agent_client.main）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
cd "${REPO}"
exec uv run vlm-agent "$@"
