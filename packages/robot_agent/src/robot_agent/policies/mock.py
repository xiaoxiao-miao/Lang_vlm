"""Mock policy: scripted approach + grasp trajectory (no images)."""

from __future__ import annotations

import math
from typing import Any

from robot_agent.policies.base import Policy


class MockPolicy(Policy):
    """Generate a short open-loop chunk that ends with gripper close + grasp flag."""

    def __init__(self, horizon: int = 8, **kwargs: Any) -> None:
        self.horizon = max(4, int(horizon))
        self.robot_state_keys: list[str] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = list(robot_state_keys)

    def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        keys = list(self.robot_state_keys)
        if not keys:
            names = observation_dict.get("observation.joint_names")
            if isinstance(names, list) and names:
                keys = [str(n) for n in names]
            else:
                state = observation_dict.get("observation.state") or []
                keys = [f"joint_{i}" for i in range(len(state))] or [
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "gripper",
                ]

        actions: list[dict[str, Any]] = []
        n = self.horizon
        for t in range(n):
            phase = t / max(n - 1, 1)
            tick: dict[str, Any] = {}
            for i, key in enumerate(keys):
                if key == "gripper":
                    # open → close in the last third
                    tick[key] = 0.1 if phase < 0.66 else 0.95
                else:
                    # smooth sinusoid approach
                    tick[key] = 0.3 * math.sin(2 * math.pi * (phase + i * 0.07))
            # arm grasp intent near the end
            tick["grasp"] = 1.0 if phase >= 0.66 else 0.0
            actions.append(tick)
        return actions
