"""Structured world-model schema. This is the agent's belief, not chat history."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

AVAILABLE_SKILLS: list[str] = [
    "inspect",
    "open_drawer",
    "close_drawer",
    "grasp",
    "place",
    "move_object",
    "navigate",
]


class ObjectState(BaseModel):
    id: str
    type: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    position: list[float] | None = None
    frame: str | None = None
    visible: bool = True
    confidence: float = 1.0
    affordances: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0


class RobotState(BaseModel):
    robot_id: str = "robot_1"
    holding_object: str | None = None
    available_skills: list[str] = Field(default_factory=lambda: list(AVAILABLE_SKILLS))
    current_skill: str | None = None
    status: str = "idle"


class TaskState(BaseModel):
    goal: str
    completed_subgoals: list[str] = Field(default_factory=list)
    current_subgoal: str | None = None
    failed_subgoals: list[str] = Field(default_factory=list)


class WorldState(BaseModel):
    objects: dict[str, ObjectState]
    relations: list[Relation]
    robot: RobotState
    task: TaskState
    revision: int = 0


def build_demo_world(goal: str = "") -> WorldState:
    """Kitchen-table demo: apple on table, closed drawer, empty gripper."""
    return WorldState(
        objects={
            "table_1": ObjectState(id="table_1", type="table", affordances=["support"]),
            "cabinet_1": ObjectState(id="cabinet_1", type="cabinet"),
            "drawer_1": ObjectState(
                id="drawer_1",
                type="drawer",
                attributes={"state": "closed"},
                affordances=["open", "close"],
            ),
            "apple_1": ObjectState(
                id="apple_1",
                type="apple",
                visible=True,
                affordances=["grasp"],
            ),
        },
        relations=[
            Relation(subject="drawer_1", predicate="part_of", object="cabinet_1"),
            Relation(subject="apple_1", predicate="on", object="table_1"),
        ],
        robot=RobotState(),
        task=TaskState(goal=goal),
    )


def format_world(world: WorldState) -> str:
    lines: list[str] = []
    for rel in world.relations:
        lines.append(f"{rel.subject} {rel.predicate} {rel.object}")
    for obj in world.objects.values():
        state = obj.attributes.get("state")
        if state is not None:
            lines.append(f"{obj.id} {state}")
    holding = world.robot.holding_object
    lines.append("robot holding nothing" if not holding else f"robot holding {holding}")
    return "\n".join(lines)


def has_relation(world: WorldState, subject: str, predicate: str, obj: str) -> bool:
    return any(
        r.subject == subject and r.predicate == predicate and r.object == obj
        for r in world.relations
    )
