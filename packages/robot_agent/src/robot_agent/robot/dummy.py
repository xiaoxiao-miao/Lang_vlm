"""Dummy robot: in-memory joints + gripper over a SceneState."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robot_agent.scene import SceneObject, SceneState

DEFAULT_JOINTS = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "gripper",
]


@dataclass
class DummyRobot:
    scene: SceneState
    joint_names: list[str] = field(default_factory=lambda: list(DEFAULT_JOINTS))
    joint_state: dict[str, float] = field(default_factory=dict)
    holding: SceneObject | None = None
    _grasp_armed: bool = False
    _pending_instruction: str | None = None

    def __post_init__(self) -> None:
        if not self.joint_state:
            self.joint_state = {name: 0.0 for name in self.joint_names}

    def get_observation(self) -> dict[str, Any]:
        state = [self.joint_state[n] for n in self.joint_names]
        return {
            "observation.state": state,
            "observation.joint_names": list(self.joint_names),
            "observation.holding": None if self.holding is None else self.holding.name,
            "observation.scene": self.scene.list_objects(),
        }

    def begin_episode(self, instruction: str) -> None:
        self._pending_instruction = instruction
        self._grasp_armed = False

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply one action tick. Values must be float (or convertible)."""
        for key, value in action.items():
            if key == "grasp":
                self._grasp_armed = bool(float(value) > 0.5)
                continue
            if key in self.joint_state:
                self.joint_state[key] = float(value)

        gripper = self.joint_state.get("gripper", 0.0)
        if self._grasp_armed and gripper >= 0.7 and self.holding is None:
            target = self.scene.find_from_instruction(self._pending_instruction or "红色")
            if target is not None and not target.grasped:
                target.grasped = True
                target.location = "gripper"
                self.holding = target
                self._grasp_armed = False

        return {
            "joint_state": dict(self.joint_state),
            "holding": None if self.holding is None else self.holding.name,
            "ok": True,
        }

    def summary(self) -> str:
        if self.holding is not None:
            return f"夹爪持有：{self.holding.name}（{self.holding.color}）"
        return "夹爪空载"
