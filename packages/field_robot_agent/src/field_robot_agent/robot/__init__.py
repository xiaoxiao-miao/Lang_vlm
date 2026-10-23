"""Robot backends. Swap MockRobot for a future ROS2RobotInterface."""

from field_robot_agent.robot.interface import RobotInterface
from field_robot_agent.robot.mock_robot import MockRobot
from field_robot_agent.robot.skills import Observation, RobotExecutionResult, SkillCall

__all__ = [
    "MockRobot",
    "Observation",
    "RobotExecutionResult",
    "RobotInterface",
    "SkillCall",
]
