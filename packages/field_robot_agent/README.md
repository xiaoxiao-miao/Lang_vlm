# Field Robot Agent

场端 Environment Agent 原型：LangGraph 显式闭环 + 结构化 World Model + 可替换 Robot Interface。第一阶段使用 Mock Robot，不接 ROS2。

## Architecture

```text
CLI
 ↓
LangGraph
 ↓
Environment Agent
 ├─ Planner
 ├─ World Model
 ├─ Memory
 ├─ VLM Perception
 └─ Executive
       ↓
 Robot Interface
       ↓
 Mock Robot
```

## Architecture Principles

- World Model != Chat History
- Planner != Robot Controller
- Environment Agent produces semantic subgoals
- Robot Agent owns low-level execution
- State updates are change-centric
- LangGraph owns deterministic workflow

Environment Agent 只决定 **WHAT**（`open_drawer` / `grasp` / `place`）。Robot 侧决定 **HOW**（运动、VLA、控制器）。第一阶段 HOW 由 Mock Robot 的规则仿真承担。

## Quick Start

在仓库根目录：

```bash
uv sync
uv run pytest
uv run field-agent --mock-llm
```

然后输入：

```text
Task > 把苹果放进抽屉
Task > 苹果在哪里
```

同一交互进程里，World Model 和 Mock Robot 会跨 `Task >` 保留。第二句是查询，从当前世界回答，不会把场景重置回「苹果在桌上」。一次性 `--task` 仍从 demo 厨房开局。

或一次性：

```bash
uv run field-agent --mock-llm --task "把苹果放进抽屉"
```

不调用远程 vLLM。Planner / World update / Verifier / Goal checker 全部走规则实现，用于离线验收和 CI。

## Example

```text
Field-Robot Environment Agent
-----------------------------

Task > 把苹果放进抽屉

[Task]
把苹果放进抽屉

[World Revision 0]
apple_1 on table_1
drawer_1 part_of cabinet_1
drawer_1 closed
robot holding nothing

[Plan]
Drawer must be opened before placing the apple.
1. open_drawer drawer_1
2. grasp apple_1
3. place apple_1 drawer_1

[Execute]
open_drawer(drawer_1)

[Robot]
success

[World Update]
drawer_1.state: closed -> open

...

[Goal]
SUCCESS
```

最终 WorldState：

```text
drawer_1.state = open
apple_1 inside drawer_1
robot_1.holding_object = None
```

Trace 写在 `logs/task_<id>/`（每次模型调用的 request/response JSON，以及 `episodic.jsonl`）。

## vLLM Setup

本机 vLLM 或华为云隧道都走同一地址：

```text
base_url: http://127.0.0.1:8000/v1
```

离线（不连模型）：

```bash
uv run field-agent --mock-llm
```

本机 vLLM 已在 `:8000`：

```bash
uv run field-agent
```

华为云：先停掉本机占用 `:8000` 的进程，再开 SSH 隧道（见 [`docs/verify_huaweicloud_vllm.md`](../../docs/verify_huaweicloud_vllm.md)），然后同样打 `:8000`：

```bash
ssh -N -L 8000:127.0.0.1:8000 -p <作业端口> ma-user@<host>
export NO_PROXY=127.0.0.1,localhost
uv run field-agent --model Qwen/Qwen3-VL-30B-A3B-Instruct
```

`--model` 必须与云端 `SERVED_MODEL_NAME` 一致。也可用环境变量：

```text
FIELD_AGENT_BASE_URL
FIELD_AGENT_MODEL
FIELD_AGENT_API_KEY
```

配置文件：[`config/model.yaml`](config/model.yaml)、[`config/agent.yaml`](config/agent.yaml)。本包不绑定具体模型名（Qwen3-VL / Gemma 4 均可，只要 OpenAI-compatible multimodal API）。

## 后续接 ROS2 时替换哪一层

只替换 Robot backend，Environment Agent / LangGraph / World Model 不改：

```text
MockRobot
    ↓ replace
ROS2RobotInterface(RobotInterface)
```

未来链路：

```text
Environment Agent
       ↓
ROS2 Action: execute_skill
       ↓
Robot Agent
```

Request：`skill` / `target` / `parameters`  
Result：`success` / `message` / `error_code` / observation metadata

当前 **不** 引入 `rclpy`。

## 当前限制

- Mock Robot 无真实运动学；技能是语义级规则仿真
- `--mock-llm` 的规则规划器针对「苹果放进抽屉」场景；「在哪里 / 吗」类问题从 World Model 回答，不规划 inspect、不重新操作
- 真实 VLM 模式依赖远程服务的 structured output 质量
- 第一阶段不包含 Web UI、真实机械臂、VLA、向量库
