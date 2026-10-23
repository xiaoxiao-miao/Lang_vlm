你是一名高级机器人系统工程师、LLM Agent 工程师和 Python 架构工程师。请基于下面的需求，在当前本地 workspace 中直接实现一个**可运行、可扩展、结构清晰的原型系统**。

目标不是写 demo 伪代码，而是完成一个真实可运行的 Python Project，使我能够在本地通过 Shell 输入任务，Environment Agent 基于 LangGraph 维护任务状态、调用远程 vLLM 上部署的 VLM/LLM、维护结构化 World Model、进行长序列任务规划，并通过一个可替换的 Robot Interface 执行子任务。

第一阶段暂时不接真实 ROS2 机器人，而是实现一个 Mock Robot。架构上必须为后续接 ROS2 留出清晰接口。

---

# 1. 项目目标

我要实现一个“场端融合”的机器人智能体系统。

系统中有两个逻辑主体：

- Environment Agent：运行在环境侧服务器/PC
- Robot Agent：运行在机器人本体侧

当前阶段只真正实现 Environment Agent 和 Mock Robot Agent。

整体逻辑为：

```text
Shell / CLI
    ↓
LangGraph
    ↓
Environment Agent
    ├── VLM Perception
    ├── World Model
    ├── Planner
    ├── Memory
    └── Task Executive
            ↓
        Robot Interface
            ↓
        Mock Robot
```

未来需要可以替换为：

```text
Environment Agent
        ↓
      ROS2
        ↓
Robot Agent
        ↓
Skill / VLA / Motion / Controller
```

因此，不要把 Robot 具体实现和 Environment Agent 强耦合。

---

# 2. 核心设计思想

系统不能简单实现成：

```text
user prompt
   ↓
LLM agent
   ↓
tool calling
   ↓
response
```

我要的是一个显式的、可控制的闭环：

```text
Task
 ↓
Observe Environment
 ↓
Update World Model
 ↓
Plan
 ↓
Select Subgoal
 ↓
Execute Robot Skill
 ↓
Receive Execution Result
 ↓
Observe Again
 ↓
Update World Model
 ↓
Verify
 ↓
Continue / Replan
 ↓
Task Complete
```

即：

```text
Observe
  ↓
UpdateWorld
  ↓
CheckGoal
  ↓
Plan
  ↓
Dispatch
  ↓
Execute
  ↓
Observe
  ↓
...
```

LangGraph 负责组织这个 workflow，而不是让一个自由 Agent 自己决定整个控制流。

---

# 3. 一个重要原则：LLM 不等于 World Model

World Model 必须是独立的结构化状态。

不要把当前环境状态主要存储在聊天 history 或 prompt 中。

定义：

```text
W_t = system belief about current world state at time t
```

Robot 执行动作：

```text
a_t
```

得到新 observation：

```text
o_{t+1}
```

系统更新：

```text
W_t + a_t + o_{t+1}
    ↓
  ΔW
    ↓
W_{t+1}
```

World Model 第一版不需要实现复杂神经世界模型。

先实现：

```text
Dynamic Scene Graph
+
Robot State
+
Task State
+
Affordance State
+
Uncertainty
```

---

# 4. 技术栈

请优先使用：

```text
Python >= 3.11
uv
LangGraph
LangChain
Pydantic v2
httpx
Typer 或 argparse
PyYAML
Rich
pytest
```

模型统一通过远程 vLLM 提供。

vLLM 提供 OpenAI-compatible API：

```text
http://<server>:8000/v1
```

模型可能是：

```text
Qwen3-VL
Gemma 4
```

因此模型客户端不要和具体模型强绑定。

第一版可以基于：

```python
langchain_openai.ChatOpenAI
```

连接：

```text
base_url = http://server:8000/v1
```

但是 VLM 部分需要支持 multimodal message，即：

```text
text + image
```

未来最好能支持：

```text
text + multiple images
text + short video
```

第一阶段必须支持：

```text
before image
after image
```

---

# 5. 项目目录

请创建类似以下结构：

```text
field_robot_agent/
├── pyproject.toml
├── README.md
├── .env.example
├── config/
│   ├── model.yaml
│   └── agent.yaml
│
├── src/
│   └── field_robot_agent/
│       ├── __init__.py
│       ├── main.py
│       │
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── graph.py
│       │   ├── state.py
│       │   ├── planner.py
│       │   ├── executive.py
│       │   └── verifier.py
│       │
│       ├── world_model/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   ├── manager.py
│       │   └── events.py
│       │
│       ├── perception/
│       │   ├── __init__.py
│       │   ├── vlm_client.py
│       │   └── scene_updater.py
│       │
│       ├── robot/
│       │   ├── __init__.py
│       │   ├── interface.py
│       │   ├── mock_robot.py
│       │   └── skills.py
│       │
│       ├── memory/
│       │   ├── __init__.py
│       │   └── episodic.py
│       │
│       └── utils/
│           ├── logging.py
│           └── image.py
│
└── tests/
    ├── test_world_model.py
    ├── test_planner.py
    ├── test_mock_robot.py
    └── test_graph.py
```

目录不必一字不差，但职责划分必须保持。

---

# 6. LangGraph State

定义一个显式的 AgentState。

例如：

```python
class AgentState(TypedDict):
    task_id: str

    goal: str

    world_state: WorldState

    plan: list[Subgoal]

    current_step: int

    current_subgoal: Subgoal | None

    robot_result: RobotExecutionResult | None

    latest_observation: Observation | None

    recent_events: list[WorldEvent]

    task_status: Literal[
        "initialized",
        "running",
        "waiting_robot",
        "replanning",
        "success",
        "failed",
    ]

    iteration: int

    error: str | None
```

不要在 state 里随便塞大量无结构 dict。

核心对象都使用 Pydantic model。

---

# 7. World Model Schema

请认真设计 `world_model/schema.py`。

至少需要：

## 7.1 ObjectState

例如：

```python
class ObjectState(BaseModel):
    id: str
    type: str

    attributes: dict[str, Any] = {}

    position: list[float] | None = None
    frame: str | None = None

    visible: bool = True

    confidence: float = 1.0

    affordances: list[str] = []
```

---

## 7.2 Relation

例如：

```python
class Relation(BaseModel):
    subject: str
    predicate: str
    object: str

    confidence: float = 1.0
```

例如：

```text
apple_1 on table_1
drawer_1 part_of cabinet_1
cup_1 inside drawer_1
box_1 blocks drawer_handle_1
```

---

## 7.3 RobotState

例如：

```python
class RobotState(BaseModel):
    robot_id: str = "robot_1"

    holding_object: str | None = None

    available_skills: list[str] = []

    current_skill: str | None = None

    status: str = "idle"
```

---

## 7.4 TaskState

例如：

```python
class TaskState(BaseModel):
    goal: str

    completed_subgoals: list[str] = []

    current_subgoal: str | None = None

    failed_subgoals: list[str] = []
```

---

## 7.5 WorldState

例如：

```python
class WorldState(BaseModel):
    objects: dict[str, ObjectState]

    relations: list[Relation]

    robot: RobotState

    task: TaskState

    revision: int = 0
```

这里要注意：

WorldState 表示“当前世界是什么样”。

它不是历史数据库。

---

# 8. Event / Change Schema

机器人操作之后，不要让 VLM 每次重新生成完整 WorldState。

优先采用 change-centric update。

定义：

```python
class AttributeChange(BaseModel):
    entity_id: str
    attribute: str
    old_value: Any | None
    new_value: Any
    confidence: float
```

定义：

```python
class RelationChange(BaseModel):
    action: Literal["add", "remove"]

    subject: str
    predicate: str
    object: str

    confidence: float
```

定义：

```python
class WorldEvent(BaseModel):
    event_type: str

    description: str

    attribute_changes: list[AttributeChange] = []

    relation_changes: list[RelationChange] = []

    new_objects: list[ObjectState] = []

    confidence: float = 1.0
```

WorldModelManager 负责：

```python
apply_event(world_state, event) -> WorldState
```

要求：

- revision 自动增加
- 更新 object attributes
- 更新 relations
- 加入新对象
- 保持 schema validation
- 不允许无条件覆盖高可信状态
- 简单处理 confidence

---

# 9. Memory 和 World Model 必须分离

创建一个简单 `EpisodicMemory`。

它保存：

```text
发生过什么
```

例如：

```text
14:31 robot tried grasp cup_1
14:31 grasp failed
14:32 camera observed cup handle
14:33 retry succeeded
```

第一版可以只使用：

```text
JSONL
```

或者：

```text
SQLite
```

如果考虑简洁，优先 JSONL。

每个 Event 至少记录：

```text
timestamp
task_id
event_type
description
subgoal
robot_result
world_revision
```

不要第一版上 vector database。

---

# 10. Robot Skill Schema

Environment Agent 不直接生成 joint/cartesian trajectory。

Environment Agent 输出 semantic skill。

例如：

```python
class SkillCall(BaseModel):
    skill: str
    target: str | None = None
    parameters: dict[str, Any] = {}
```

例如：

```json
{
  "skill": "open_drawer",
  "target": "drawer_1",
  "parameters": {}
}
```

或者：

```json
{
  "skill": "grasp",
  "target": "apple_1",
  "parameters": {
    "grasp_type": "top"
  }
}
```

Robot Agent 以后负责：

```text
semantic skill
 ↓
affordance grounding
 ↓
VLA / motion planner
 ↓
controller
```

---

# 11. 第一版 Skills

至少实现以下抽象 skill：

```text
inspect
open_drawer
close_drawer
grasp
place
move_object
navigate
```

第一版 Mock Robot 不需要真实运动学。

---

# 12. Robot Interface

定义抽象接口，例如：

```python
class RobotInterface(ABC):

    @abstractmethod
    def execute_skill(
        self,
        skill: SkillCall
    ) -> RobotExecutionResult:
        ...
```

未来 ROS2 应该直接实现同样接口：

```text
MockRobot
    ↓ replace
ROS2RobotInterface
```

Environment Agent 其他代码不能因为替换 Robot backend 而修改。

---

# 13. Mock Robot

实现一个可以真正推进 World State 的 Mock Robot。

不要只返回固定 success。

Mock Robot 内部可以维护一个简单 simulated world。

例如：

初始化：

```text
apple_1 on table_1

drawer_1:
    state = closed

robot:
    holding = None
```

如果执行：

```text
open_drawer(drawer_1)
```

返回：

```python
RobotExecutionResult(
    success=True,
    message="drawer opened",
    simulated_changes=...
)
```

执行：

```text
grasp(apple_1)
```

要求：

```text
apple visible
robot currently holding nothing
```

否则失败。

执行成功后：

```text
robot.holding_object = apple_1
apple no longer on table
```

执行：

```text
place(apple_1, drawer_1)
```

要求：

```text
robot holding apple_1
drawer open
```

成功后：

```text
apple inside drawer
robot holding = None
```

这样能够真实验证 planning loop。

---

# 14. Observation Schema

定义：

```python
class Observation(BaseModel):
    timestamp: datetime

    text_description: str | None = None

    before_image: str | None = None

    after_image: str | None = None

    metadata: dict[str, Any] = {}
```

当前 Mock Robot 模式下可以不依赖真实图片。

但架构上必须允许：

```text
before.jpg
after.jpg
```

进入 VLM。

---

# 15. VLM Client

创建独立 `VLMClient`。

职责：

```text
远程调用 vLLM
```

配置例如：

```yaml
provider: openai_compatible

base_url: http://127.0.0.1:8000/v1

api_key: EMPTY

model: Qwen/Qwen3-VL-30B-A3B-Instruct

temperature: 0.1
```

环境变量允许覆盖。

需要支持：

```python
analyze_scene(...)
analyze_change(...)
plan_task(...)
verify_subgoal(...)
```

可以内部都走同一个 model，但 prompt 必须分离。

不要设计成一个巨大 `ask_llm()` 到处乱用。

---

# 16. VLM Change Detection Prompt

重点实现：

```python
analyze_change(
    previous_world: WorldState,
    robot_action: SkillCall,
    robot_result: RobotExecutionResult,
    before_image: str | None,
    after_image: str | None,
) -> WorldEvent
```

Prompt 思路：

```text
You are the world-state update module of a robot environment agent.

Current structured world model:
...

Robot executed:
...

Execution result:
...

You may also receive:
- image before execution
- image after execution

Do NOT regenerate the entire world model.

Determine only the state changes caused or observed after this action.

Return structured JSON containing:
- attribute changes
- relation additions/removals
- newly visible objects
- unexpected events
- confidence
```

输出通过 Pydantic structured output validation。

如果无法验证，允许 retry 一次。

---

# 17. Planner

实现一个 task-level planner。

输入：

```text
goal
+
current WorldState
+
available skills
+
recent episodic memory
```

输出：

```python
class Subgoal(BaseModel):
    id: str

    description: str

    skill: SkillCall

    preconditions: list[str] = []

    expected_effects: list[str] = []
```

Planner 返回：

```python
class TaskPlan(BaseModel):
    reasoning_summary: str

    subgoals: list[Subgoal]
```

不要返回隐藏 chain-of-thought。

`reasoning_summary` 只写简短可观察规划依据，例如：

```text
Drawer must be opened before placing the apple.
```

不要要求模型输出冗长推理过程。

---

# 18. 示例规划

输入：

```text
Goal:
Put apple_1 into drawer_1
```

WorldState：

```text
apple_1 on table_1
drawer_1 closed
robot holding nothing
```

期待：

```text
1. open_drawer(drawer_1)
2. grasp(apple_1)
3. place(apple_1, drawer_1)
```

如果 drawer 已经 open：

```text
1. grasp
2. place
```

如果 robot 已经 holding apple：

```text
1. open_drawer
2. place
```

---

# 19. Verifier

不要单纯相信：

```text
robot_result.success == True
```

需要实现 SubgoalVerifier。

第一版可以综合：

```text
robot result
+
current WorldState
+
expected effects
```

例如：

```text
subgoal:
open_drawer(drawer_1)
```

如果：

```text
world_state.drawer_1.state == open
```

才算 verified。

---

# 20. Goal Checker

需要独立：

```python
check_goal(
    goal,
    world_state
) -> GoalCheckResult
```

可以第一版通过 VLM / LLM 判断，但最好支持 rule based。

对 mock scenario：

```text
goal:
put apple_1 into drawer_1
```

可以检查 relation：

```text
apple_1 inside drawer_1
```

返回：

```python
class GoalCheckResult(BaseModel):
    satisfied: bool
    explanation: str
```

---

# 21. LangGraph Nodes

至少实现以下 nodes：

```text
initialize_task

observe

update_world

check_goal

plan

select_subgoal

dispatch_skill

verify_subgoal

record_memory

replan

finish
```

建议 graph：

```text
START
  ↓
initialize_task
  ↓
observe
  ↓
update_world
  ↓
check_goal
  │
  ├── satisfied → finish
  │
  └── not satisfied
          ↓
         plan
          ↓
    select_subgoal
          ↓
    dispatch_skill
          ↓
       observe
          ↓
     update_world
          ↓
   verify_subgoal
      │       │
 success    failure
      │       │
      ↓       ↓
 check_goal  replan
      │       │
      └───────┘
```

一定要设置：

```text
max_iterations
```

防止无限循环。

例如：

```yaml
max_iterations: 20
```

---

# 22. 初始 World State

为了第一版能够直接运行，请提供一个 demo world：

```text
table_1

cabinet_1

drawer_1:
    state: closed
    part_of cabinet_1

apple_1:
    on table_1
    visible
    graspable

robot_1:
    holding nothing
```

CLI 启动以后显示：

```text
Current world:
- apple_1 is on table_1
- drawer_1 is closed
- robot hand is empty
```

---

# 23. CLI

启动：

```bash
uv run field-agent
```

或者：

```bash
uv run python -m field_robot_agent.main
```

终端显示：

```text
Field-Robot Environment Agent
-----------------------------

Task >
```

输入：

```text
Put the apple into the drawer
```

或者中文：

```text
把苹果放进抽屉
```

系统应该打印每一轮高层状态，例如：

```text
[Task]
把苹果放进抽屉

[World Revision 0]
apple_1 on table_1
drawer_1 closed
robot holding nothing

[Plan]
1. open_drawer drawer_1
2. grasp apple_1
3. place apple_1 into drawer_1

[Execute]
open_drawer(drawer_1)

[Robot]
success

[World Update]
drawer_1.state: closed -> open

[Execute]
grasp(apple_1)

...

[Goal]
SUCCESS
```

使用 Rich 做可读输出即可。

---

# 24. Debug / Trace

这一点很重要。

每一次 LLM/VLM 调用都必须可观测。

实现 log，例如：

```text
logs/
  task_xxx/
    001_plan_request.json
    001_plan_response.json

    002_world_update_request.json
    002_world_update_response.json

    003_verify_request.json
    003_verify_response.json
```

每次模型调用至少保存：

```text
timestamp

model

messages

structured schema name

raw model response

parsed response

latency

token usage if available
```

注意：

不要记录 API key。

这个 trace 功能是必须实现的，不是 optional。

因为后续我要分析：

```text
每一轮到底给模型输入了什么 context
模型返回了什么
world state 如何被更新
```

---

# 25. Prompt 文件化

不要把所有 prompt 硬编码在 Python 中。

创建：

```text
src/field_robot_agent/prompts/
```

例如：

```text
planner.txt

world_update.txt

goal_check.txt

subgoal_verify.txt

initial_scene.txt
```

Python 中加载。

后续我要方便调 prompt。

---

# 26. 配置文件

`config/model.yaml`

例如：

```yaml
model:
  base_url: "http://127.0.0.1:8000/v1"
  api_key: "EMPTY"
  model_name: "Qwen/Qwen3-VL-30B-A3B-Instruct"

  temperature: 0.1
  timeout: 60
```

`config/agent.yaml`

```yaml
agent:
  max_iterations: 20

  trace_enabled: true

  memory_enabled: true

robot:
  backend: mock
```

---

# 27. 无模型 Mock Mode

这个功能一定要有。

增加：

```text
--mock-llm
```

例如：

```bash
uv run field-agent --mock-llm
```

在这个模式下：

- 不调用远程 vLLM
- Planner 使用简单 rule-based implementation
- World update 使用 Mock Robot 返回的 changes
- Goal Checker 使用 rule-based
- Verifier 使用 rule-based

这样即使没有 GPU/vLLM server，我也能够验证整个 LangGraph 系统。

这个模式是单元测试和 CI 的基础。

---

# 28. 实际 VLM Mode

正常启动：

```bash
uv run field-agent
```

则：

```text
Planner
World updater
Verifier
```

使用远程模型。

但是对于 Mock Robot，可以继续使用模拟环境。

即：

```text
real VLM
+
mock robot
```

也应该可以工作。

---

# 29. Image Support

实现一个 image utility：

```python
encode_image(path) -> data URL
```

但是优先兼容 OpenAI-compatible multimodal format：

```python
{
    "type": "image_url",
    "image_url": {
        "url": ...
    }
}
```

支持：

```text
local image file
HTTP URL
```

对于 local file，可使用 base64 data URL。

---

# 30. 暂时不需要做的事情

当前版本不要实现：

```text
Web UI
FastAPI
OpenClaw
Docker
真实 ROS2
真实机械臂控制
VLA
motion planning
vector database
3D reconstruction
VGGT
3DGS
SLAM
video streaming
多智能体 swarm
复杂权限系统
```

这些都不是第一阶段目标。

不要过度工程化。

---

# 31. ROS2 扩展接口

虽然第一版不实现 ROS2，但请在 README 中说明未来如何替换：

```text
MockRobot
```

为：

```text
ROS2RobotInterface
```

未来接口大致是：

```text
Environment Agent
       ↓
ROS2 Action:
execute_skill
       ↓
Robot Agent
```

Request：

```text
skill
target
parameters
```

Result：

```text
success
message
error_code
observation metadata
```

当前不要真的引入 rclpy dependency。

---

# 32. 测试

至少完成：

## test_world_model.py

验证：

```text
closed -> open

relation add/remove

confidence

revision increment
```

## test_mock_robot.py

验证：

```text
cannot place object if not holding

cannot place into closed drawer

can open drawer

can grasp apple

can place apple
```

## test_planner.py

Mock planner 输入：

```text
apple on table
drawer closed
```

输出：

```text
open → grasp → place
```

## test_graph.py

运行整个：

```text
put apple into drawer
```

最终：

```text
goal == success

apple inside drawer

robot holding none

drawer open
```

测试必须能够：

```bash
uv run pytest
```

通过。

---

# 33. README

README 必须包含：

## Architecture

画 ASCII 图：

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

## Quick Start

```bash
uv sync

uv run field-agent --mock-llm
```

## vLLM Setup

示例：

```text
base_url:
http://localhost:8000/v1
```

说明只要求远程服务满足 OpenAI-compatible multimodal API。

## Example

给出完整：

```text
Task > 把苹果放进抽屉
```

运行示例。

## Architecture Principles

明确说明：

```text
World Model != Chat History

Planner != Robot Controller

Environment Agent produces semantic subgoals

Robot Agent owns low-level execution

State updates are change-centric

LangGraph owns deterministic workflow
```

---

# 34. Coding Style

要求：

- 完整 type hints
- Pydantic v2
- dataclass / TypedDict 适当使用
- 不写巨大单文件
- 避免超过约 300 行的核心模块
- docstring 简洁
- logging 清晰
- 所有外部 I/O 尽量隔离
- model client 与 business logic 分离
- prompt 与 code 分离
- 不允许在各个模块中直接创建 ChatOpenAI
- 统一 Model/VLM client
- Robot backend 使用 dependency injection
- 不要大量使用全局变量

---

# 35. 最重要的系统边界

请严格遵守下面的架构原则：

## Environment Agent 决定：

```text
WHAT should be done next
```

例如：

```text
open drawer
grasp apple
place apple
inspect shelf
```

## Robot Agent 决定：

```text
HOW to physically execute it
```

例如未来：

```text
target pose

affordance point

VLA action

motion planning

trajectory

controller command
```

因此 Environment Agent 不输出：

```text
joint positions

Cartesian trajectories

velocity commands
```

---

# 36. World Model 的定义

本项目里的 World Model 定义为：

> A persistent, structured and dynamically updated belief of the current environment used by the Environment Agent for planning, verification and task execution.

也就是说：

```text
World Model
=
Dynamic Scene Graph
+
Object State
+
Relations
+
Affordances
+
Robot State
+
Task State
+
Uncertainty
```

注意：

```text
World Model
!=
episodic memory
```

Memory 回答：

```text
What happened before?
```

World Model 回答：

```text
What is the world like now?
```

---

# 37. 第一阶段成功标准

完成项目后，我应该能够执行：

```bash
uv sync

uv run pytest

uv run field-agent --mock-llm
```

然后：

```text
Task > 把苹果放进抽屉
```

系统自动完成：

```text
recognize current state
 ↓
plan:
 open drawer
 grasp apple
 place apple
 ↓
execute mock skills
 ↓
update world state
 ↓
verify each subgoal
 ↓
check final goal
 ↓
SUCCESS
```

最终 WorldState：

```text
drawer_1.state = open

apple_1 inside drawer_1

robot_1.holding_object = None
```

同时生成完整 trace。

---

# 38. 工作方式

请直接在当前 workspace 开始实现。

执行顺序建议：

```text
1. 初始化 uv project
2. 建立 schema
3. 实现 WorldModelManager
4. 实现 RobotInterface
5. 实现 MockRobot
6. 实现 rule-based Mock Planner
7. 实现 AgentState
8. 实现 LangGraph
9. 实现 CLI
10. 跑通 mock end-to-end
11. 写测试
12. 加 vLLM/VLM client
13. 加 structured output
14. 加 prompt files
15. 加 trace
16. 完善 README
```

优先保证：

```text
uv run pytest
```

和：

```text
uv run field-agent --mock-llm
```

真正能运行。

不要只创建文件框架。

每完成一个主要模块，请自行执行相关测试和静态检查并修正错误。

如果当前环境缺少某个依赖，请通过 `pyproject.toml` 正确添加，不要依赖全局 Python 环境。

如果某些 LangGraph / LangChain API 与当前安装版本不同，请以当前稳定版本的实际 API 为准调整实现，但保持上述架构不变。

最终请给我：

1. 实现了哪些模块
2. 项目目录
3. 如何安装
4. 如何启动 mock demo
5. 如何连接真实 vLLM
6. 测试结果
7. 后续接 ROS2 时具体应替换哪一层
8. 当前存在的限制