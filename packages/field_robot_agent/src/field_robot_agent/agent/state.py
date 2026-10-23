"""Agent graph state. Core objects are Pydantic models, not loose dicts."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from field_robot_agent.robot.skills import Observation, RobotExecutionResult, SkillCall
from field_robot_agent.world_model.events import WorldEvent
from field_robot_agent.world_model.schema import WorldState

TaskStatus = Literal[
    "initialized",
    "running",
    "waiting_robot",
    "replanning",
    "success",
    "failed",
]

Phase = Literal["boot", "post_execute"]


class Subgoal(BaseModel):
    id: str
    description: str
    skill: SkillCall
    preconditions: list[str] = Field(default_factory=list)
    expected_effects: list[str] = Field(default_factory=list)


class TaskPlan(BaseModel):
    reasoning_summary: str
    subgoals: list[Subgoal]


class GoalCheckResult(BaseModel):
    satisfied: bool
    explanation: str


class SubgoalVerifyResult(BaseModel):
    success: bool
    explanation: str


class AgentState(TypedDict, total=False):
    task_id: str
    goal: str
    world_state: WorldState
    plan: list[Subgoal]
    current_step: int
    current_subgoal: Subgoal | None
    robot_result: RobotExecutionResult | None
    latest_observation: Observation | None
    recent_events: list[WorldEvent]
    task_status: TaskStatus
    iteration: int
    error: str | None
    phase: Phase
    verify_result: SubgoalVerifyResult | None
    goal_check: GoalCheckResult | None
