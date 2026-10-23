#!/usr/bin/env bash
# Configure Docker daemon to pull images via local HTTP proxy (Clash :7897).
# Prefer proxy → Docker Hub. Avoid flaky public mirrors that return 403.
# Run: sudo bash vlm_server/0_setup_docker_proxy.sh
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run with sudo: sudo bash $0" >&2
  exit 1
fi

PROXY_URL="${DOCKER_HTTP_PROXY:-http://127.0.0.1:7897}"
# Domestic registries must bypass proxy (SWR push fails via Clash: closed connection).
NO_PROXY_LIST="${DOCKER_NO_PROXY:-localhost,127.0.0.1,::1,.myhuaweicloud.com,.huaweicloud.com,swr.cn-north-9.myhuaweicloud.com}"

mkdir -p /etc/systemd/system/docker.service.d
cat > /etc/systemd/system/docker.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=${PROXY_URL}"
Environment="HTTPS_PROXY=${PROXY_URL}"
Environment="NO_PROXY=${NO_PROXY_LIST}"
EOF

python3 - <<'PY'
import json
from pathlib import Path
p = Path("/etc/docker/daemon.json")
data = {}
if p.exists():
    data = json.loads(p.read_text() or "{}")
# Keep nvidia runtime; drop broken mirrors that 403.
data.pop("registry-mirrors", None)
data["ip6tables"] = False
# Ensure nvidia runtime block remains if present
p.write_text(json.dumps(data, indent=2) + "\n")
print("wrote", p)
print(p.read_text())
PY

systemctl daemon-reload
systemctl restart docker
sleep 2
echo "Docker proxy set to ${PROXY_URL}"
echo "NO_PROXY=${NO_PROXY_LIST}"
echo "Hub pulls use proxy; Huawei SWR push/pull bypass proxy."
echo "Test Hub:  docker pull nvidia/cuda:12.4.1-base-ubuntu22.04"
echo "Test SWR:  docker push swr.<region>.myhuaweicloud.com/<org>/vlm-server:v1"
