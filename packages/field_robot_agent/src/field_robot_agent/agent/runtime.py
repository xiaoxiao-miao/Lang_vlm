"""Assemble GraphRuntime and run one task."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from field_robot_agent.agent.executive import TaskExecutive
from field_robot_agent.agent.goal_check import RuleBasedGoalChecker, VLMGoalChecker
from field_robot_agent.agent.graph import GraphRuntime, build_graph
from field_robot_agent.agent.planner import RuleBasedPlanner, VLMPlanner
from field_robot_agent.agent.state import AgentState
from field_robot_agent.agent.verifier import RuleBasedVerifier, VLMVerifier
from field_robot_agent.memory.episodic import EpisodicMemory
from field_robot_agent.perception.scene_updater import SceneUpdater
from field_robot_agent.perception.vlm_client import VLMClient
from field_robot_agent.robot.interface import RobotInterface
from field_robot_agent.robot.mock_robot import MockRobot
from field_robot_agent.utils.console import RunPrinter
from field_robot_agent.utils.logging import Tracer
from field_robot_agent.world_model.manager import WorldModelManager
from field_robot_agent.world_model.schema import WorldState, build_demo_world


def build_runtime(
    *,
    mock_llm: bool,
    settings: dict[str, Any],
    logs_root: Path,
    printer: RunPrinter | None = None,
    robot: RobotInterface | None = None,
    task_id: str | None = None,
) -> GraphRuntime:
    tid = task_id or uuid4().hex[:10]
    task_dir = logs_root / f"task_{tid}"
    agent_cfg = settings.get("agent") or {}
    tracer = Tracer(task_dir, enabled=bool(agent_cfg.get("trace_enabled", True)))
    memory = EpisodicMemory(
        task_dir / "episodic.jsonl",
        enabled=bool(agent_cfg.get("memory_enabled", True)),
    )
    vlm: VLMClient | None = None
    if not mock_llm:
        vlm = VLMClient.from_config(settings.get("model") or {}, tracer=tracer)

    backend = ((settings.get("robot") or {}).get("backend") or "mock").lower()
    if robot is None:
        if backend != "mock":
            raise ValueError(f"unsupported robot backend {backend!r}; first stage is mock only")
        robot = MockRobot(build_demo_world(""))

    planner = RuleBasedPlanner() if mock_llm else VLMPlanner(vlm)
    verifier = RuleBasedVerifier() if mock_llm else VLMVerifier(vlm)
    goal_checker = RuleBasedGoalChecker() if mock_llm else VLMGoalChecker(vlm)
    return GraphRuntime(
        robot=robot,
        world_manager=WorldModelManager(),
        planner=planner,
        verifier=verifier,
        goal_checker=goal_checker,
        scene_updater=SceneUpdater(mock_llm=mock_llm, vlm=vlm),
        executive=TaskExecutive(robot),
        memory=memory,
        tracer=tracer,
        printer=printer or RunPrinter(),
        max_iterations=int(agent_cfg.get("max_iterations", 20)),
        mock_llm=mock_llm,
        task_id=tid,
    )


def run_task(
    goal: str,
    runtime: GraphRuntime,
    task_id: str | None = None,
    world_state: WorldState | None = None,
) -> AgentState:
    graph = build_graph(runtime)
    initial: AgentState = {
        "goal": goal,
        "task_id": task_id or runtime.task_id,
    }
    if world_state is not None:
        initial["world_state"] = world_state
    result = graph.invoke(initial, config={"recursion_limit": 256})
    return result  # type: ignore[return-value]
