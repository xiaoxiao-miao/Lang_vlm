"""Replaceable robot backend. MockRobot and future ROS2RobotInterface share this."""

from __future__ import annotations

from abc import ABC, abstractmethod

from field_robot_agent.robot.skills import Observation, RobotExecutionResult, SkillCall


class RobotInterface(ABC):
    @abstractmethod
    def execute_skill(self, skill: SkillCall) -> RobotExecutionResult:
        """Run one semantic skill and return an execution result."""

    @abstractmethod
    def get_observation(self) -> Observation:
        """Return the latest environment observation (text and optional images)."""
