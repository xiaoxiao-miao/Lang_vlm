"""LangChain tools bound to a shared DummyRobot + MockPolicy runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from robot_agent.policies.mock import MockPolicy
from robot_agent.robot.dummy import DEFAULT_JOINTS, DummyRobot
from robot_agent.runner import run_grasp
from robot_agent.scene import SceneState


@dataclass
class RobotRuntime:
    """Shared state for tools within one agent session."""

    scene: SceneState = field(default_factory=SceneState.default)
    joint_names: list[str] = field(default_factory=lambda: list(DEFAULT_JOINTS))
    horizon: int = 8
    vlm_cfg: dict[str, Any] = field(default_factory=dict)
    image_path: Path | None = None
    robot: DummyRobot = field(init=False)
    policy: MockPolicy = field(init=False)

    def __post_init__(self) -> None:
        self.robot = DummyRobot(scene=self.scene, joint_names=list(self.joint_names))
        self.policy = MockPolicy(horizon=self.horizon)


def build_tools(runtime: RobotRuntime) -> list[BaseTool]:
    def list_objects() -> str:
        """列出当前桌面场景中的物体（名称、颜色、位置）。"""
        return json.dumps(runtime.scene.list_objects(), ensure_ascii=False, indent=2)

    def describe_scene() -> str:
        """描述当前场景。若配置了图像且 VLM 可用则调用视觉模型，否则返回内存场景描述。"""
        fallback = runtime.scene.describe()
        if runtime.image_path is None or not runtime.image_path.is_file():
            return fallback
        vlm = runtime.vlm_cfg
        if not vlm.get("base_url"):
            return fallback
        try:
            from agent_client.chains import analyze_image
            from agent_client.vlm_client import ensure_vlm_ready

            ensure_vlm_ready(str(vlm["base_url"]), timeout=3.0)
            return analyze_image(
                image_path=str(runtime.image_path),
                user_text="简要描述桌面物体，尤其是颜色与可否抓取。",
                base_url=str(vlm["base_url"]),
                model=str(vlm.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")),
                api_key=str(vlm.get("api_key", "EMPTY")),
                temperature=float(vlm.get("temperature", 0.2)),
            )
        except Exception as exc:  # noqa: BLE001 — tool must not crash the agent
            return f"{fallback}\n（VLM 不可用，已回退内存描述：{exc}）"

    def grasp_object(instruction: str) -> str:
        """对桌面物体执行抓取。传入完整自然语言指令，例如「抓取桌面红色物体」。"""
        result = run_grasp(
            robot=runtime.robot,
            policy=runtime.policy,
            instruction=instruction,
        )
        return result.as_text()

    return [
        StructuredTool.from_function(
            func=list_objects,
            name="list_objects",
            description="列出当前桌面场景中的物体（名称、颜色、位置）。",
        ),
        StructuredTool.from_function(
            func=describe_scene,
            name="describe_scene",
            description="描述当前场景；有图且 VLM 可用时用视觉模型，否则用内存场景。",
        ),
        StructuredTool.from_function(
            func=grasp_object,
            name="grasp_object",
            description="执行抓取。参数 instruction 为完整指令，如「抓取桌面红色物体」。",
        ),
    ]
