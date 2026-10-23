"""End-to-end LangGraph loop with --mock-llm semantics."""

from pathlib import Path

from field_robot_agent.agent.runtime import build_runtime, run_task
from field_robot_agent.robot.mock_robot import MockRobot
from field_robot_agent.world_model.schema import build_demo_world, has_relation


def _settings() -> dict:
    return {
        "agent": {"max_iterations": 20, "trace_enabled": True, "memory_enabled": True},
        "robot": {"backend": "mock"},
        "model": {},
    }


def test_graph_put_apple_into_drawer(tmp_path: Path):
    runtime = build_runtime(
        mock_llm=True, settings=_settings(), logs_root=tmp_path, task_id="pytest"
    )
    result = run_task("把苹果放进抽屉", runtime)
    assert result["task_status"] == "success"
    world = result["world_state"]
    assert world.objects["drawer_1"].attributes["state"] == "open"
    assert world.robot.holding_object is None
    assert has_relation(world, "apple_1", "inside", "drawer_1")
    assert (tmp_path / "task_pytest" / "episodic.jsonl").is_file()


def test_graph_english_goal(tmp_path: Path):
    runtime = build_runtime(mock_llm=True, settings=_settings(), logs_root=tmp_path)
    result = run_task("Put the apple into the drawer", runtime)
    assert result["task_status"] == "success"
    world = result["world_state"]
    assert has_relation(world, "apple_1", "inside", "drawer_1")


def test_world_persists_and_location_query_does_not_replay(tmp_path: Path):
    robot = MockRobot(build_demo_world(""))
    first = build_runtime(
        mock_llm=True, settings=_settings(), logs_root=tmp_path, robot=robot, task_id="put"
    )
    put = run_task("把苹果放进抽屉", first)
    assert put["task_status"] == "success"
    assert has_relation(put["world_state"], "apple_1", "inside", "drawer_1")
    revision_after_put = put["world_state"].revision

    second = build_runtime(
        mock_llm=True, settings=_settings(), logs_root=tmp_path, robot=robot, task_id="ask"
    )
    ask = run_task("苹果在哪里", second, world_state=put["world_state"])
    assert ask["task_status"] == "success"
    world = ask["world_state"]
    assert has_relation(world, "apple_1", "inside", "drawer_1")
    assert not has_relation(world, "apple_1", "on", "table_1")
    assert world.objects["drawer_1"].attributes["state"] == "open"
    assert world.revision == revision_after_put
    assert "apple_1 inside drawer_1" in ask["goal_check"].explanation


def test_yes_no_query_does_not_inspect_loop(tmp_path: Path):
    robot = MockRobot(build_demo_world(""))
    first = build_runtime(
        mock_llm=True, settings=_settings(), logs_root=tmp_path, robot=robot, task_id="put2"
    )
    put = run_task("把苹果放进抽屉", first)
    second = build_runtime(
        mock_llm=True, settings=_settings(), logs_root=tmp_path, robot=robot, task_id="ask2"
    )
    ask = run_task("苹果在桌子上吗", second, world_state=put["world_state"])
    assert ask["task_status"] == "success"
    assert ask["goal_check"].explanation.startswith("No.")
    assert has_relation(ask["world_state"], "apple_1", "inside", "drawer_1")
    assert ask["world_state"].revision == put["world_state"].revision
    assert not (ask.get("plan") or [])
