"""Sync policy runner: obs → get_actions → robot.step → summary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robot_agent.policies.base import Policy
from robot_agent.robot.dummy import DummyRobot


@dataclass
class RunResult:
    ok: bool
    instruction: str
    steps: int
    holding: str | None
    message: str
    trajectory: list[dict[str, Any]]

    def as_text(self) -> str:
        status = "成功" if self.ok else "失败"
        return (
            f"[{status}] instruction={self.instruction!r} steps={self.steps} "
            f"holding={self.holding!r} — {self.message}"
        )


def run_grasp(
    *,
    robot: DummyRobot,
    policy: Policy,
    instruction: str,
) -> RunResult:
    """Execute one mock grasp episode against the dummy robot."""
    robot.begin_episode(instruction)
    if hasattr(policy, "set_robot_state_keys"):
        policy.set_robot_state_keys(list(robot.joint_names))  # type: ignore[attr-defined]

    obs = robot.get_observation()
    actions = policy.get_actions(obs, instruction)
    trajectory: list[dict[str, Any]] = []
    for action in actions:
        info = robot.step(action)
        trajectory.append({"action": action, "info": info})

    holding = None if robot.holding is None else robot.holding.name
    ok = robot.holding is not None
    if ok:
        message = f"已抓取 {robot.holding.name}（{robot.holding.color}）"
    else:
        target = robot.scene.find_from_instruction(instruction)
        hint = f"目标候选={target.name}" if target else "未匹配到场景物体"
        message = f"未抓取成功（{hint}）"

    return RunResult(
        ok=ok,
        instruction=instruction,
        steps=len(trajectory),
        holding=holding,
        message=message,
        trajectory=trajectory,
    )
