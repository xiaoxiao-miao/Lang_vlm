# vlm_server 脚本索引

命名约定：

| 前缀 | 含义 |
|------|------|
| `0_*` | 公用 / 一次性（三路径复用） |
| `local_*` | 本机 local：`local_1_` / `local_2_` / `local_3_` |
| `docker_*` | 本机 Docker：`docker_1_` / `docker_2_` / `docker_3_` |
| `cloud_*` | 云端 Docker：`cloud_2_` / `cloud_3_` |
| 无前缀工作流名 | 容器运行时（`start.sh` 等），不要当工作流入口 |

---

## 路径 A — 本机 local（宿主机 Python）

```bash
./local_1_prepare.sh
./local_2_run.sh          # 前台跑 vLLM
# 另开终端
./local_3_agent.sh
```

## 路径 B — 本机 Docker

一次性：

```bash
sudo bash ./0_setup_docker_prereqs.sh
sudo bash ./0_setup_docker_proxy.sh
newgrp docker
```

每次：

```bash
./docker_1_build.sh
./docker_2_run.sh
./docker_3_agent.sh
```

## 路径 C — 云端 Docker（推 v2.3 到 SWR）

本机构建推送：

```bash
docker login -u <user>@<domain> swr.cn-north-9.myhuaweicloud.com
./cloud_2_push_swr.sh
# 模型（一次）：DRY_RUN=0 ./0_upload_hf_model_to_obs.sh → SFS 导入 uid=1000 gid=100
```

云主机：

```bash
cp .env.cloud.example .env.cloud   # 填 VLM_IMAGE
./cloud_3_run.sh
```

---

## 公用 `0_*`

| 脚本 | 用途 |
|------|------|
| `0_setup_docker_prereqs.sh` | docker 组 + nvidia-container-toolkit |
| `0_setup_docker_proxy.sh` | Docker daemon 代理（构建/拉镜像） |
| `0_upload_hf_model_to_obs.sh` | HF 模型 flatten 上传 OBS |

---

## 机器人 Agent（仓库根目录）

```bash
# 在仓库根目录
uv run robot-agent --scripted "抓取桌面红色物体"   # 无 LLM
uv run robot-agent "抓取桌面红色物体"              # 需 ./local_2_run.sh
```
