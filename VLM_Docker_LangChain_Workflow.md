# VLM Docker Inference Server + Local LangChain Agent Workflow

## 1. 项目目标

构建一个基于多模态大模型（Vision Language Model, VLM）的具身智能基础服务架构：

- 本地 Agent 使用 LangChain 进行任务编排；
- 使用 **uv**（推荐）或 **pixi** 管理 Python 依赖与可复现环境；
- Docker 部署 VLM 推理服务（本地 RTX 5090 / 华为云 A100 同一镜像）；
- 支持 Image + Text 输入；
- VLM 输出场景理解、任务规划、机器人操作建议；
- **先在本地 RTX 5090 完成端到端验证**；
- 再迁移至 **华为云 NVIDIA A100**（CUDA / vLLM，与本地同架构）。

核心流程：

```
Camera Image
      |
      v
Local LangChain Agent（uv / pixi 管理）
      |
      v
HTTP OpenAI Compatible API
      |
      v
Docker VLM Server（本地 5090 或云端 A100）
      |
      v
Qwen2.5-VL（默认 7B）
      |
      v
Response
```

------------------------------------------------------------------------

## 2. 硬件与部署路径（重要）

| 阶段 | 环境 | GPU | 推理栈 | 说明 |
|------|------|-----|--------|------|
| Phase 1–3 | 本地 | **RTX 5090（32GB）** | CUDA + vLLM | 开发与端到端测试 |
| Phase 4 | 华为云 | **NVIDIA A100（40/80GB）** | **同一** CUDA + vLLM | 仅改 `base_url` / 镜像仓库 |
| 可选远期 | Ascend | 910B | MindIE + CANN | API 仍保持 OpenAI 兼容（非本阶段目标） |

设计原则：

- 本地与华为云均为 **NVIDIA GPU + CUDA**，Docker 镜像与启动参数高度复用；
- A100 显存更大，可按需升级到更大 VLM（如 32B），**不必改 Agent 代码**；
- Ascend 作为独立扩展路径，不阻塞当前 A100 云端落地。

------------------------------------------------------------------------

## 3. 系统架构

```
Local Host / Robot PC

ROS2 Camera
      |
      v
Image Capture
      |
      v
LangChain Agent（agent_client，uv/pixi）
- Prompt Manager
- Memory
- Task Planner
- Tool Calling
      |
 OpenAI Compatible API
      |
      v
Docker VLM Server（vlm_server）
- vLLM OpenAI API Server
- Qwen2.5-VL
      |
      v
GPU: RTX 5090（本地） / A100（华为云）
```

------------------------------------------------------------------------

## 4. 包管理：uv（推荐）或 pixi

工程用现代 Python 包管理工具统一本地 Agent 与 Server 构建依赖，避免散落的 `pip install`。

### 4.1 选型建议

| 工具 | 适用 | 特点 |
|------|------|------|
| **uv**（推荐默认） | Agent + Docker 构建脚本 | 极快、`pyproject.toml` 标准、易写 `uv sync` / `uv run` |
| **pixi** | 需要 conda 级系统依赖时 | 基于 conda-forge，适合 CUDA/系统库一并锁定 |

本仓库默认以 **uv** 为主；若团队更熟悉 conda 生态，可用 pixi 等价替换下方命令。

### 4.2 仓库结构（uv workspace）

```
physical_vlm_agent/
├── pyproject.toml              # workspace 根：成员 + 共享工具
├── uv.lock                     # 可复现锁定（提交入库）
├── pixi.toml                   # 可选：pixi 等价环境
├── .python-version             # 如 3.11
│
├── packages/
│   └── agent_client/           # 本地 LangChain 智能体
│       ├── pyproject.toml
│       ├── src/agent_client/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── chains.py       # LangChain 链条
│       │   ├── vlm_client.py
│       │   ├── image_encoder.py
│       │   └── prompts.py
│       └── tests/
│
├── vlm_server/                 # Docker 推理服务
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   ├── docker-compose.cloud.yaml
│   ├── start.sh
│   ├── pyproject.toml          # 镜像内依赖（vLLM 等）
│   └── requirements.lock.txt   # 可选：由 uv export 生成供 Docker 安装
│
├── configs/
│   ├── config.local.yaml       # 5090 / localhost:8000
│   └── config.cloud.yaml       # 华为云 A100 / cloud_ip:8000
│
└── README.md
```

### 4.3 根 `pyproject.toml` 示意（uv workspace）

```toml
[project]
name = "physical-vlm-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[tool.uv.workspace]
members = ["packages/agent_client"]

[dependency-groups]
dev = ["pytest", "ruff", "httpx"]
```

### 4.4 Agent 包依赖示意

```toml
# packages/agent_client/pyproject.toml
[project]
name = "agent-client"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "langchain>=0.3",
  "langchain-openai>=0.2",
  "langchain-core>=0.3",
  "openai>=1.40",
  "pillow>=10",
  "httpx",
  "pyyaml",
  "python-dotenv",
]

[project.scripts]
vlm-agent = "agent_client.main:main"
```

### 4.5 常用命令

**uv：**

```bash
# 安装 uv（一次性）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync

# 运行本地 Agent
uv run --package agent-client vlm-agent

# 跑测试
uv run pytest packages/agent_client/tests

# 导出 Docker 用冻结依赖（可选）
uv export --package agent-client -o packages/agent_client/requirements.txt
```

**pixi（可选等价）：**

```bash
pixi install
pixi run vlm-agent
pixi run pytest
```

------------------------------------------------------------------------

## 5. 本地 RTX 5090：开源 VLM 选型

目标硬件：**RTX 5090，约 32GB 显存**。优先保证 BF16/FP16 下稳定跑通多模态 API，再考虑更大模型。

### 5.1 推荐默认

| 优先级 | 模型 | 约显存（BF16 + KV，中等上下文） | 建议 |
|--------|------|----------------------------------|------|
| **默认** | **Qwen/Qwen2.5-VL-7B-Instruct** | ~14–18GB | 5090 充足余量；OpenAI 兼容；场景理解强 |
| 备选轻量 | Qwen2.5-VL-3B-Instruct | ~8–12GB | 快速迭代 / 调试 |
| 备选增强 | InternVL2.5-8B / InternVL3-8B | ~16–20GB | 对比评测 |
| 云端可升级 | Qwen2.5-VL-32B-Instruct | 单卡 5090 偏紧；**A100 80GB 舒适** | Phase 4 在华为云升级 |

**本地 Phase 1–3 固定使用：`Qwen/Qwen2.5-VL-7B-Instruct`。**

理由：

- 官方 Instruct 多模态，适合 Image + Text；
- vLLM 支持完善，OpenAI `/v1/chat/completions` 开箱即用；
- 5090 上 BF16 仍有余量给图像 token、KV cache、并发；
- 与华为云 A100 使用**同一模型镜像**即可对照，云端再按需改 `MODEL_NAME` 升到 32B。

### 5.2 环境变量

```bash
MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
# 云端可选：
# MODEL_NAME=Qwen/Qwen2.5-VL-32B-Instruct
```

------------------------------------------------------------------------

# Part 1 — VLM Docker Server（本地 / 云端同一镜像）

## 6. Docker 职责

容器负责：

- 加载多模态模型；
- GPU 推理（5090 或 A100）；
- 提供 OpenAI 兼容 API。

接口：

```
POST /v1/chat/completions
GET  /v1/models
```

Agent **不**进容器；只通过 HTTP 调用。

------------------------------------------------------------------------

## 7. 基础环境（本地 5090）

- Ubuntu
- NVIDIA Driver（支持 Blackwell / 5090）
- CUDA（与镜像匹配，推荐 12.4+）
- Docker + NVIDIA Container Toolkit

验证：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

------------------------------------------------------------------------

## 8. Docker Image

推荐基础镜像：

```
nvidia/cuda:12.4.1-runtime-ubuntu22.04
```

镜像内包含：

- Python 3.11+
- PyTorch（CUDA）
- vLLM（含 OpenAI API Server）
- qwen-vl-utils（按模型需要）
- 由 `vlm_server/pyproject.toml` 或 `uv export` 锁定依赖

`Dockerfile` 建议用多阶段或直接 `uv pip install` / `pip install -r requirements.lock.txt`，保证本地 build 与华为云 SWR 推送产物一致。

------------------------------------------------------------------------

## 9. 启动要求

- CUDA runtime + GPU 透传（`--gpus all`）
- 模型加载（HF cache 挂载）
- 端口 **8000** 暴露
- `MODEL_NAME` 可配置

vLLM 启动示例：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
```

5090 本地可酌情加：

```bash
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

A100（华为云）可提高并发与上下文，例如：

```bash
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --max-num-seqs 64
```

------------------------------------------------------------------------

## 10. 本地 5090 构建与测试

```bash
cd vlm_server
docker build -t vlm-server:v1 .

docker run -d \
  --name vlm-server \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct \
  vlm-server:v1
```

健康检查：

```bash
curl localhost:8000/v1/models
```

`docker-compose.yaml`（本地）与 `docker-compose.cloud.yaml`（华为云 A100）共用同一 image tag，仅差异化：

- GPU 资源声明 / 实例规格；
- 镜像仓库地址（本地 Docker vs 华为云 SWR）；
- 可选更大的 `MODEL_NAME`。

------------------------------------------------------------------------

# Part 2 — Local LangChain Agent（uv / pixi）

## 11. Client 职责

| 位置 | 职责 |
|------|------|
| 本地 `agent_client` | 摄像头采集、图片编码、Prompt、LangChain 链条、Memory、Tool Calling |
| Docker `vlm_server` | 仅 VLM 推理 |

配置通过 `configs/config.local.yaml`：

```yaml
vlm:
  base_url: "http://localhost:8000/v1"
  model: "Qwen/Qwen2.5-VL-7B-Instruct"
  api_key: "EMPTY"
```

云端仅改：

```yaml
vlm:
  base_url: "http://<huawei-cloud-a100-ip>:8000/v1"
  model: "Qwen/Qwen2.5-VL-7B-Instruct"  # 或 32B
```

------------------------------------------------------------------------

## 12. LangChain 智能体链条（工程重点）

建议链条（可逐步加厚）：

```
[Image Encoder] → [Vision Prompt] → [ChatOpenAI(base_url=VLM)] → [Output Parser]
                         ↑
                   [Memory / Session]
                         ↑
              [可选 Tools：ROS 查询、技能库]
```

实现要点：

1. 使用 `langchain-openai` 的 `ChatOpenAI`，`base_url` 指向本地/云端 vLLM；
2. 消息内容为 OpenAI Vision 格式（`image_url` + `text`）；
3. 用 `Runnable` / LCEL 组装：`prompt | llm | parser`；
4. 后续扩展：`create_tool_calling_agent` + 场景图 / Affordance 工具。

最小调用示意：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="Qwen/Qwen2.5-VL-7B-Instruct",
)

msg = HumanMessage(content=[
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    {"type": "text", "text": "分析场景并给出可执行操作建议"},
])
print(llm.invoke([msg]).content)
```

------------------------------------------------------------------------

## 13. Image 处理流程

```
Camera → JPEG → Binary → Base64 → JSON → HTTP → VLM
```

Base64：把二进制图变成可放进 HTTP JSON 的文本，符合 OpenAI Vision 协议。

请求示例：

```json
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": { "url": "data:image/jpeg;base64,xxxx" }
        },
        { "type": "text", "text": "分析图片中的场景" }
      ]
    }
  ]
}
```

------------------------------------------------------------------------

# Part 3 — Embodied Agent Extension（后续）

```
Human Voice → Whisper → Task Agent → LangChain
      → Environment Agent → VLM → Scene Graph
      → Affordance → Skill Planner → Robot Controller
```

------------------------------------------------------------------------

# Part 4 — ROS2 Integration（后续）

```
ROS2 Camera (sensor_msgs/Image)
  → Image Bridge → JPEG → Base64 → VLM API
```

------------------------------------------------------------------------

# Part 5 — 多相机策略

不要：8 路 × 30 FPS 持续上传。

采用：Frame Selector → Key Frame → VLM，典型 **1–5 Hz** 查询。

------------------------------------------------------------------------

# Part 6 — 华为云部署（NVIDIA A100）

本地与云端**同架构**（NVIDIA + CUDA + vLLM + 同一 Docker 镜像）：

```
本地:  RTX 5090 + Docker + Qwen2.5-VL-7B
  ↓ 推送镜像到华为云 SWR
云端:  A100   + Docker + 同镜像（可改 MODEL_NAME 升 32B）
```

客户端只改配置中的：

```
base_url: localhost:8000  →  http://<cloud_ip>:8000
```

注意：

- 安全组放行 8000（或经网关 / VPN）；
- HuggingFace 缓存可挂云盘，避免每次冷启动拉模型；
- A100 40GB / 80GB 选择影响是否上 32B。

------------------------------------------------------------------------

# Part 7 — Ascend 扩展（非当前路径）

当前华为云目标是 **A100**，不是 Ascend。

远期若迁 910B：

```
NVIDIA: vLLM + CUDA
  → 替换为 MindIE + CANN + Ascend Runtime
API 保持: OpenAI Compatible
```

Agent 与 `base_url` 契约不变。

------------------------------------------------------------------------

# 开发 Milestone

## Phase 1 — 本地 5090 + Docker VLM

- [ ] `uv sync` 初始化 workspace
- [ ] 构建 `vlm-server:v1`
- [ ] 启动 `Qwen2.5-VL-7B-Instruct`
- [ ] `curl /v1/models` 与简单 vision 请求通过

## Phase 2 — LangChain Agent 链条

- [ ] `agent_client`：Image + Text → VLM
- [ ] LCEL 链条 + 输出解析
- [ ] `configs/config.local.yaml` 跑通

## Phase 3 — 机器人侧

- [ ] ROS2 Camera → Base64 → Agent
- [ ] Scene Query / Affordance 推理

## Phase 4 — 华为云 A100

- [ ] 同一镜像推送 SWR
- [ ] A100 上启动；Agent 改 `base_url`
- [ ] （可选）`MODEL_NAME` 升级 32B

------------------------------------------------------------------------

# Coding Agent 执行顺序

```
Step 1: 用 uv 初始化 monorepo（pyproject + agent_client + vlm_server）
Step 2: 编写 vlm_server Dockerfile / start.sh / compose
Step 3: 本地 5090 启动 Qwen2.5-VL-7B，curl 验证 OpenAI API
Step 4: 实现 LangChain Vision 链条（ChatOpenAI + base_url）
Step 5: 本地端到端：图片 + 文本 → 场景理解响应
Step 6: （可选）ROS2 camera 接入
Step 7: 同一镜像部署到华为云 A100，只改 base_url
```
