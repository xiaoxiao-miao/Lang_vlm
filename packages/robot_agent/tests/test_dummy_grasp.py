"""Dummy grasp loop tests (no LLM)."""

from pathlib import Path

from robot_agent.policies.mock import MockPolicy
from robot_agent.robot.dummy import DummyRobot
from robot_agent.runner import run_grasp
from robot_agent.scene import SceneState
from robot_agent.tools import RobotRuntime, build_tools


def _fixture_scene_jpg() -> Path:
    # packages/robot_agent/tests -> packages/agent_client/tests/fixtures
    return (
        Path(__file__).resolve().parents[2]
        / "agent_client"
        / "tests"
        / "fixtures"
        / "scene_sample.jpg"
    )


def test_run_grasp_red_object_succeeds():
    scene = SceneState.default()
    robot = DummyRobot(scene=scene)
    policy = MockPolicy(horizon=8)
    result = run_grasp(robot=robot, policy=policy, instruction="抓取桌面红色物体")
    assert result.ok
    assert result.holding == "红色方块"
    assert robot.holding is not None
    assert robot.holding.color == "红色"


def test_list_and_grasp_tools():
    runtime = RobotRuntime()
    tools = {t.name: t for t in build_tools(runtime)}
    listed = tools["list_objects"].invoke({})
    assert "红色方块" in listed
    out = tools["grasp_object"].invoke({"instruction": "抓取桌面红色物体"})
    assert "成功" in out
    assert runtime.robot.holding is not None


def test_scripted_path_text():
    from robot_agent.agent import run_scripted

    runtime = RobotRuntime()
    text = run_scripted(runtime, "抓取桌面红色物体")
    assert "list_objects" in text
    assert "成功" in text
    assert runtime.robot.holding.name == "红色方块"


def test_invoke_tool_calls_helper():
    from langchain_core.messages import AIMessage

    from robot_agent.agent import _invoke_tool_calls

    runtime = RobotRuntime()
    tools = {t.name: t for t in build_tools(runtime)}
    ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "list_objects",
                "args": {},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    msgs = _invoke_tool_calls(1, ai, tools)
    assert len(msgs) == 1
    assert "红色方块" in msgs[0].content


def test_build_user_message_with_image():
    from robot_agent.agent import build_user_message

    image = _fixture_scene_jpg()
    assert image.is_file(), image
    msg = build_user_message("抓取图片中红色物体", image)
    assert isinstance(msg.content, list)
    assert msg.content[0]["type"] == "image_url"
    assert "抓取图片中红色物体" in msg.content[1]["text"]


def test_format_messages_shortens_data_url():
    from langchain_core.messages import HumanMessage

    from robot_agent.agent import _format_messages_for_log

    long_b64 = "A" * 200
    msg = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{long_b64}"},
            },
            {"type": "text", "text": "抓取图片中红色物体"},
        ]
    )
    text = _format_messages_for_log([msg])
    assert "抓取图片中红色物体" in text
    assert "200 chars" in text
    assert long_b64 not in text
