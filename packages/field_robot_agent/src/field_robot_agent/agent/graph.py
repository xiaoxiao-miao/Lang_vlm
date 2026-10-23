"""LangGraph owns the deterministic Observe → Plan → Dispatch → Verify loop."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from field_robot_agent.agent.executive import TaskExecutive
from field_robot_agent.agent.goal_check import GoalChecker, answer_world_query, is_world_query
from field_robot_agent.agent.planner import Planner
from field_robot_agent.agent.state import AgentState, GoalCheckResult
from field_robot_agent.agent.verifier import Verifier
from field_robot_agent.memory.episodic import EpisodicMemory
from field_robot_agent.perception.scene_updater import SceneUpdater
from field_robot_agent.robot.interface import RobotInterface
from field_robot_agent.utils.console import RunPrinter
from field_robot_agent.utils.logging import Tracer
from field_robot_agent.world_model.manager import WorldModelManager
from field_robot_agent.world_model.schema import TaskState, WorldState, build_demo_world


@dataclass
class GraphRuntime:
    robot: RobotInterface
    world_manager: WorldModelManager
    planner: Planner
    verifier: Verifier
    goal_checker: GoalChecker
    scene_updater: SceneUpdater
    executive: TaskExecutive
    memory: EpisodicMemory
    tracer: Tracer
    printer: RunPrinter
    max_iterations: int
    mock_llm: bool
    task_id: str


def build_graph(rt: GraphRuntime) -> Any:
    def initialize_task(state: AgentState) -> dict[str, Any]:
        task_id = state.get("task_id") or rt.task_id or uuid.uuid4().hex[:10]
        goal = state["goal"]
        world = _continue_world(state.get("world_state"), rt.robot, goal)
        rt.printer.task(goal)
        rt.printer.world(world)
        rt.memory.record(
            task_id=task_id,
            event_type="task_start",
            description=f"task started: {goal}",
            world_revision=world.revision,
        )
        rt.tracer.log_event("initialize_task", {"goal": goal, "world": world.model_dump()})
        return {
            "task_id": task_id,
            "world_state": world,
            "plan": [],
            "current_step": 0,
            "current_subgoal": None,
            "robot_result": None,
            "latest_observation": None,
            "recent_events": [],
            "task_status": "initialized",
            "iteration": 0,
            "error": None,
            "phase": "boot",
            "verify_result": None,
            "goal_check": None,
        }

    def observe(state: AgentState) -> dict[str, Any]:
        obs = rt.robot.get_observation()
        return {"latest_observation": obs, "task_status": "running"}

    def update_world(state: AgentState) -> dict[str, Any]:
        subgoal = state.get("current_subgoal")
        event = rt.scene_updater.update(
            world=state["world_state"],
            observation=state.get("latest_observation"),
            skill=subgoal.skill if subgoal else None,
            result=state.get("robot_result"),
        )
        world = state["world_state"]
        if event.has_changes():
            world = rt.world_manager.apply_event(world, event)
            rt.printer.world_update(event)
        events = list(state.get("recent_events") or [])
        events.append(event)
        rt.tracer.log_event(
            "update_world",
            {"event": event.model_dump(), "revision": world.revision, "mock_llm": rt.mock_llm},
        )
        return {"world_state": world, "recent_events": events[-20:]}

    def check_goal(state: AgentState) -> dict[str, Any]:
        if int(state.get("iteration") or 0) >= rt.max_iterations:
            msg = f"max_iterations={rt.max_iterations} exceeded"
            return {
                "task_status": "failed",
                "error": msg,
                "goal_check": GoalCheckResult(satisfied=False, explanation=msg),
            }
        goal = state["goal"]
        world = state["world_state"]
        result = rt.goal_checker.check(goal, world)
        if result.satisfied:
            return {"goal_check": result, "task_status": "success"}
        if _noop_inspect_stall(state):
            if is_world_query(goal):
                answered = GoalCheckResult(
                    satisfied=True,
                    explanation=answer_world_query(goal, world),
                )
                return {"goal_check": answered, "task_status": "success"}
            msg = "inspect did not change the world; stopping to avoid an inspect loop"
            return {
                "goal_check": GoalCheckResult(satisfied=False, explanation=msg),
                "task_status": "failed",
                "error": msg,
            }
        return {"goal_check": result, "task_status": "running"}

    def plan(state: AgentState) -> dict[str, Any]:
        task_plan = rt.planner.plan(state["goal"], state["world_state"], rt.memory.recent_text())
        rt.printer.plan(task_plan)
        rt.tracer.log_event("plan", {"plan": task_plan.model_dump(), "mock_llm": rt.mock_llm})
        return {"plan": task_plan.subgoals, "current_step": 0, "task_status": "running"}

    def select_subgoal(state: AgentState) -> dict[str, Any]:
        plan_items = state.get("plan") or []
        step = int(state.get("current_step") or 0)
        if not plan_items or step >= len(plan_items):
            return {
                "current_subgoal": None,
                "task_status": "failed",
                "error": "planner returned no remaining subgoals",
            }
        sg = plan_items[step]
        world = state["world_state"]
        task = world.task.model_copy(update={"current_subgoal": sg.id})
        return {"current_subgoal": sg, "world_state": world.model_copy(update={"task": task})}

    def dispatch_skill(state: AgentState) -> dict[str, Any]:
        sg = state["current_subgoal"]
        if sg is None:
            return {"task_status": "failed", "error": "no subgoal to dispatch"}
        rt.printer.execute(sg.skill)
        result = rt.executive.dispatch(sg.skill)
        rt.printer.robot(result)
        return {
            "robot_result": result,
            "phase": "post_execute",
            "iteration": int(state.get("iteration") or 0) + 1,
            "task_status": "waiting_robot",
        }

    def verify_subgoal(state: AgentState) -> dict[str, Any]:
        sg = state["current_subgoal"]
        if sg is None:
            return {
                "verify_result": None,
                "task_status": "failed",
                "error": "no subgoal to verify",
            }
        result = rt.verifier.verify(sg, state["world_state"], state.get("robot_result"))
        world = state["world_state"]
        task = world.task
        if result.success:
            task = task.model_copy(
                update={
                    "completed_subgoals": [*task.completed_subgoals, sg.id],
                    "current_subgoal": None,
                }
            )
        else:
            task = task.model_copy(
                update={
                    "failed_subgoals": [*task.failed_subgoals, sg.id],
                    "current_subgoal": None,
                }
            )
        return {"verify_result": result, "world_state": world.model_copy(update={"task": task})}

    def record_memory(state: AgentState) -> dict[str, Any]:
        sg = state.get("current_subgoal")
        result = state.get("robot_result")
        vr = state.get("verify_result")
        skill_name = sg.skill.skill if sg else "?"
        ok = bool(vr and vr.success)
        rt.memory.record(
            task_id=state["task_id"],
            event_type="verify",
            description=f"robot tried {skill_name}; {'succeeded' if ok else 'failed'}",
            subgoal=sg.id if sg else None,
            robot_result=result.message if result else None,
            world_revision=state["world_state"].revision,
        )
        return {}

    def replan(state: AgentState) -> dict[str, Any]:
        task_plan = rt.planner.plan(state["goal"], state["world_state"], rt.memory.recent_text())
        rt.printer.plan(task_plan)
        rt.tracer.log_event("replan", {"plan": task_plan.model_dump()})
        return {
            "plan": task_plan.subgoals,
            "current_step": 0,
            "task_status": "replanning",
        }

    def finish(state: AgentState) -> dict[str, Any]:
        gc = state.get("goal_check")
        if gc is not None and gc.satisfied:
            if is_world_query(state.get("goal") or ""):
                rt.printer.answer(gc.explanation)
            else:
                rt.printer.goal(gc)
            return {"task_status": "success"}
        msg = state.get("error") or (gc.explanation if gc else "task failed")
        rt.printer.fail(msg or "task failed")
        return {"task_status": "failed"}

    def route_after_update(state: AgentState) -> Literal["verify_subgoal", "check_goal"]:
        if state.get("phase") == "post_execute":
            return "verify_subgoal"
        return "check_goal"

    def route_after_goal(state: AgentState) -> Literal["finish", "plan"]:
        if state.get("task_status") in {"success", "failed"}:
            return "finish"
        return "plan"

    def route_after_select(state: AgentState) -> Literal["finish", "dispatch_skill"]:
        if state.get("task_status") == "failed" or state.get("current_subgoal") is None:
            return "finish"
        return "dispatch_skill"

    def route_after_memory(state: AgentState) -> Literal["check_goal", "replan"]:
        vr = state.get("verify_result")
        if vr is not None and vr.success:
            return "check_goal"
        return "replan"

    graph = StateGraph(AgentState)
    graph.add_node("initialize_task", initialize_task)
    graph.add_node("observe", observe)
    graph.add_node("update_world", update_world)
    graph.add_node("check_goal", check_goal)
    graph.add_node("plan", plan)
    graph.add_node("select_subgoal", select_subgoal)
    graph.add_node("dispatch_skill", dispatch_skill)
    graph.add_node("verify_subgoal", verify_subgoal)
    graph.add_node("record_memory", record_memory)
    graph.add_node("replan", replan)
    graph.add_node("finish", finish)

    graph.add_edge(START, "initialize_task")
    graph.add_edge("initialize_task", "observe")
    graph.add_edge("observe", "update_world")
    graph.add_conditional_edges(
        "update_world",
        route_after_update,
        {"verify_subgoal": "verify_subgoal", "check_goal": "check_goal"},
    )
    graph.add_conditional_edges(
        "check_goal",
        route_after_goal,
        {"finish": "finish", "plan": "plan"},
    )
    graph.add_edge("plan", "select_subgoal")
    graph.add_conditional_edges(
        "select_subgoal",
        route_after_select,
        {"finish": "finish", "dispatch_skill": "dispatch_skill"},
    )
    graph.add_edge("dispatch_skill", "observe")
    graph.add_edge("verify_subgoal", "record_memory")
    graph.add_conditional_edges(
        "record_memory",
        route_after_memory,
        {"check_goal": "check_goal", "replan": "replan"},
    )
    graph.add_edge("replan", "select_subgoal")
    graph.add_edge("finish", END)
    return graph.compile()


def _continue_world(
    previous: WorldState | None,
    robot: RobotInterface,
    goal: str,
) -> WorldState:
    """Keep objects/relations across turns; only reset the current task fields."""
    if previous is not None:
        world = previous.model_copy(deep=True)
    else:
        sim = getattr(robot, "sim", None)
        if isinstance(sim, WorldState):
            world = sim.model_copy(deep=True)
        else:
            world = build_demo_world(goal)
    world.task = TaskState(goal=goal)
    return world


def _noop_inspect_stall(state: AgentState) -> bool:
    """Inspect that did not patch the world will never satisfy a remaining goal."""
    sg = state.get("current_subgoal")
    if sg is None or sg.skill.skill != "inspect":
        return False
    events = state.get("recent_events") or []
    return bool(events) and not events[-1].has_changes()
