"""Task-level planner: rule-based (mock-llm) or VLM-backed."""

from __future__ import annotations

from typing import Protocol

from field_robot_agent.agent.goal_check import is_world_query, looks_like_apple_in_drawer
from field_robot_agent.agent.state import Subgoal, TaskPlan
from field_robot_agent.robot.skills import SkillCall
from field_robot_agent.world_model.schema import WorldState, has_relation


class Planner(Protocol):
    def plan(self, goal: str, world: WorldState, memory_text: str = "") -> TaskPlan: ...


def _drawer_open(world: WorldState, drawer_id: str = "drawer_1") -> bool:
    drawer = world.objects.get(drawer_id)
    return bool(drawer and drawer.attributes.get("state") == "open")


def _apple_in_drawer(world: WorldState) -> bool:
    return has_relation(world, "apple_1", "inside", "drawer_1")


class RuleBasedPlanner:
    """Deterministic planner for the apple-into-drawer demo (and close paraphrases)."""

    def plan(self, goal: str, world: WorldState, memory_text: str = "") -> TaskPlan:
        del memory_text
        if is_world_query(goal):
            return TaskPlan(
                reasoning_summary="World-state query; no manipulation skills required.",
                subgoals=[],
            )
        if not looks_like_apple_in_drawer(goal) and goal.strip():
            return TaskPlan(
                reasoning_summary="No remaining semantic skills for this goal.",
                subgoals=[],
            )
        if _apple_in_drawer(world):
            return TaskPlan(reasoning_summary="Apple is already inside the drawer.", subgoals=[])

        subgoals: list[Subgoal] = []
        step = 1
        if not _drawer_open(world):
            subgoals.append(
                Subgoal(
                    id=f"sg_{step}",
                    description="Open the drawer so it can receive the apple.",
                    skill=SkillCall(skill="open_drawer", target="drawer_1"),
                    preconditions=["drawer_1 exists"],
                    expected_effects=["drawer_1.state == open"],
                )
            )
            step += 1
        if world.robot.holding_object != "apple_1":
            if world.robot.holding_object is not None:
                subgoals.append(
                    Subgoal(
                        id=f"sg_{step}",
                        description="Place the currently held object before grasping the apple.",
                        skill=SkillCall(
                            skill="place",
                            target=world.robot.holding_object,
                            parameters={"destination": "table_1"},
                        ),
                        expected_effects=["robot holding nothing"],
                    )
                )
                step += 1
            subgoals.append(
                Subgoal(
                    id=f"sg_{step}",
                    description="Grasp the apple.",
                    skill=SkillCall(skill="grasp", target="apple_1"),
                    preconditions=["apple_1 visible", "robot holding nothing"],
                    expected_effects=["robot holding apple_1"],
                )
            )
            step += 1
        if world.robot.holding_object == "apple_1" or any(
            s.skill.skill == "grasp" for s in subgoals
        ):
            if not any(s.skill.skill == "open_drawer" for s in subgoals) and not _drawer_open(
                world
            ):
                subgoals.insert(
                    0,
                    Subgoal(
                        id="sg_open",
                        description="Open the drawer before placing.",
                        skill=SkillCall(skill="open_drawer", target="drawer_1"),
                        expected_effects=["drawer_1.state == open"],
                    ),
                )
            subgoals.append(
                Subgoal(
                    id=f"sg_{step}",
                    description="Place the apple into the drawer.",
                    skill=SkillCall(
                        skill="place",
                        target="apple_1",
                        parameters={"destination": "drawer_1"},
                    ),
                    preconditions=["robot holding apple_1", "drawer_1.state == open"],
                    expected_effects=["apple_1 inside drawer_1", "robot holding nothing"],
                )
            )

        if not subgoals:
            return TaskPlan(
                reasoning_summary="No remaining semantic skills for this goal.",
                subgoals=[],
            )

        if not _drawer_open(world):
            summary = "Drawer must be opened before placing the apple."
        elif world.robot.holding_object == "apple_1":
            summary = "Apple is already in hand; place it into the open drawer."
        else:
            summary = "Grasp the apple then place it into the open drawer."
        return TaskPlan(reasoning_summary=summary, subgoals=subgoals)


class VLMPlanner:
    def __init__(self, vlm: object) -> None:
        self._vlm = vlm

    def plan(self, goal: str, world: WorldState, memory_text: str = "") -> TaskPlan:
        if is_world_query(goal):
            return TaskPlan(
                reasoning_summary="World-state query; no manipulation skills required.",
                subgoals=[],
            )
        return self._vlm.plan_task(goal=goal, world=world, memory_text=memory_text)
