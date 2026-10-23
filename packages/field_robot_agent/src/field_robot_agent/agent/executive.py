"""Dispatch a semantic skill through RobotInterface."""

from __future__ import annotations

from field_robot_agent.robot.interface import RobotInterface
from field_robot_agent.robot.skills import RobotExecutionResult, SkillCall


class TaskExecutive:
    def __init__(self, robot: RobotInterface) -> None:
        self.robot = robot

    def dispatch(self, skill: SkillCall) -> RobotExecutionResult:
        return self.robot.execute_skill(skill)
