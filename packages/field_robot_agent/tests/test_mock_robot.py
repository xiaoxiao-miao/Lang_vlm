"""MockRobot skill preconditions and world advancement."""

from field_robot_agent.robot.mock_robot import MockRobot
from field_robot_agent.robot.skills import SkillCall
from field_robot_agent.world_model.schema import has_relation


def test_cannot_place_if_not_holding():
    robot = MockRobot()
    result = robot.execute_skill(
        SkillCall(skill="place", target="apple_1", parameters={"destination": "drawer_1"})
    )
    assert result.success is False
    assert result.error_code == "not_holding"


def test_cannot_place_into_closed_drawer():
    robot = MockRobot()
    grasped = robot.execute_skill(SkillCall(skill="grasp", target="apple_1"))
    assert grasped.success
    result = robot.execute_skill(
        SkillCall(skill="place", target="apple_1", parameters={"destination": "drawer_1"})
    )
    assert result.success is False
    assert result.error_code == "drawer_closed"


def test_open_grasp_place_succeeds():
    robot = MockRobot()
    opened = robot.execute_skill(SkillCall(skill="open_drawer", target="drawer_1"))
    assert opened.success
    assert robot.sim.objects["drawer_1"].attributes["state"] == "open"

    grasped = robot.execute_skill(SkillCall(skill="grasp", target="apple_1"))
    assert grasped.success
    assert robot.sim.robot.holding_object == "apple_1"
    assert not has_relation(robot.sim, "apple_1", "on", "table_1")

    placed = robot.execute_skill(
        SkillCall(skill="place", target="apple_1", parameters={"destination": "drawer_1"})
    )
    assert placed.success
    assert robot.sim.robot.holding_object is None
    assert has_relation(robot.sim, "apple_1", "inside", "drawer_1")


def test_open_drawer_fails_if_already_open():
    robot = MockRobot()
    assert robot.execute_skill(SkillCall(skill="open_drawer", target="drawer_1")).success
    again = robot.execute_skill(SkillCall(skill="open_drawer", target="drawer_1"))
    assert again.success is False
    assert again.error_code == "already_open"


def test_grasp_fails_when_hands_full():
    robot = MockRobot()
    assert robot.execute_skill(SkillCall(skill="grasp", target="apple_1")).success
    again = robot.execute_skill(SkillCall(skill="grasp", target="apple_1"))
    assert again.success is False
    assert again.error_code == "hands_full"
