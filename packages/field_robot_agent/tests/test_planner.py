"""Rule-based planner for the apple-into-drawer demo."""

from field_robot_agent.agent.planner import RuleBasedPlanner
from field_robot_agent.world_model.schema import build_demo_world


def test_planner_open_grasp_place_when_drawer_closed():
    world = build_demo_world("把苹果放进抽屉")
    plan = RuleBasedPlanner().plan("把苹果放进抽屉", world)
    skills = [sg.skill.skill for sg in plan.subgoals]
    assert skills == ["open_drawer", "grasp", "place"]
    assert plan.subgoals[0].skill.target == "drawer_1"
    assert plan.subgoals[1].skill.target == "apple_1"
    assert plan.subgoals[2].skill.parameters.get("destination") == "drawer_1"
    assert "Drawer must be opened" in plan.reasoning_summary


def test_planner_skips_open_when_drawer_already_open():
    world = build_demo_world("put apple into drawer")
    world.objects["drawer_1"].attributes["state"] = "open"
    plan = RuleBasedPlanner().plan("put apple into drawer", world)
    assert [sg.skill.skill for sg in plan.subgoals] == ["grasp", "place"]


def test_planner_when_already_holding_apple():
    world = build_demo_world("put apple")
    world.robot.holding_object = "apple_1"
    plan = RuleBasedPlanner().plan("put apple into drawer", world)
    assert [sg.skill.skill for sg in plan.subgoals] == ["open_drawer", "place"]


def test_planner_does_not_manipulate_on_location_query():
    world = build_demo_world("苹果在哪里")
    plan = RuleBasedPlanner().plan("苹果在哪里", world)
    assert plan.subgoals == []
    assert "query" in plan.reasoning_summary.lower()


def test_planner_does_not_manipulate_on_yes_no_query():
    world = build_demo_world("苹果在桌子上吗")
    plan = RuleBasedPlanner().plan("苹果在桌子上吗", world)
    assert plan.subgoals == []
