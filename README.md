# physical-vlm-agent

本地 LangChain Agent + VLM OpenAI 兼容推理服务。

- 本地 GPU：RTX 5090（32GB）
- 云端 GPU：华为云 NVIDIA A100
- 默认模型：`Qwen/Qwen2.5-VL-7B-Instruct`

脚本约定见 [`vlm_server/SCRIPTS.md`](vlm_server/SCRIPTS.md)：`0_*` 公用；`local_*` / `docker_*` / `cloud_*` 分路径。

---

## 路径 A：本机 local（宿主机 vLLM）

```bash
cd vlm_server
./local_1_prepare.sh
./local_2_run.sh
# 另开终端
./local_3_agent.sh
```

---

## 路径 B：本机 Docker（离线容器）

```
宿主机 HF 缓存 ──volume──► Docker vlm-server:v2.3 ──:8000──► Agent
```

一次性：

```bash
sudo bash vlm_server/0_setup_docker_prereqs.sh
sudo bash vlm_server/0_setup_docker_proxy.sh
newgrp docker
```

每次：

```bash
cd vlm_server
pkill -f 'vllm.entrypoints.openai.api_server' || true
./docker_1_build.sh
./docker_2_run.sh
./docker_3_agent.sh
```

运维：`docker compose logs -f` / `down` / `up -d`

| 现象 | 处理 |
|------|------|
| docker.sock 权限 | `newgrp docker` |
| :8000 占用 | `pkill -f vllm.entrypoints` |
| pull/build 超时 | `0_setup_docker_proxy.sh` |

---

## 路径 C：云端 Docker（v2.3 → SWR）

镜像：`vlm-server:v2.3`（与本机 Docker 同栈）

本机：

```bash
cd vlm_server
docker login -u <user>@<domain> swr.cn-north-9.myhuaweicloud.com
./cloud_2_push_swr.sh

# 模型（一次）：flatten 上传 OBS → SFS 导入 uid=1000 gid=100
DRY_RUN=0 ./0_upload_hf_model_to_obs.sh
```

云主机：

```bash
cp .env.cloud.example .env.cloud   # VLM_IMAGE=.../vlm-server:v2.3
./cloud_3_run.sh
```

`cloud_2_push_swr.sh` 强制 Docker V2 构建并推送。

---

## RTX 5090

- `VLLM_USE_FLASHINFER_SAMPLER=0`
- 勿用 `ALL_PROXY=socks://...`（`local_2_run.sh` 会清理）

更多：`VLM_Docker_LangChain_Workflow.md`、`vlm_server/SCRIPTS.md`。

---

## 机器人智能体（Phase 1：Dummy + Mock）

包：`packages/robot_agent`。不接真机；用内存场景 + MockPolicy 演示「抓取桌面红色物体」。

```bash
# 离线验收（不需要 VLM）
uv run robot-agent --scripted "抓取桌面红色物体"

# 需本机 vLLM 在 :8000（tool-calling）
uv run robot-agent "抓取桌面红色物体"
```

配置：`configs/config.robot.local.yaml`。

---

## 场端 Environment Agent（LangGraph + World Model）

包：`packages/field_robot_agent`。显式闭环（Observe → UpdateWorld → Plan → Dispatch → Verify），结构化 World Model，第一阶段 Mock Robot。

```bash
# 离线验收（不需要 VLM）
uv run field-agent --mock-llm --task "把苹果放进抽屉"

# 交互
uv run field-agent --mock-llm

# 真实 vLLM（OpenAI-compatible :8000）+ Mock Robot
uv run field-agent
```

说明见 [`packages/field_robot_agent/README.md`](packages/field_robot_agent/README.md)。
