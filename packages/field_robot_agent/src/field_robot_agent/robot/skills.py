"""Semantic skill schema. Environment Agent never emits joint/cartesian commands."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from field_robot_agent.world_model.events import WorldEvent


class SkillCall(BaseModel):
    skill: str
    target: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class RobotExecutionResult(BaseModel):
    success: bool
    message: str
    error_code: str | None = None
    simulated_changes: WorldEvent | None = None
    observation_metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    timestamp: datetime
    text_description: str | None = None
    before_image: str | None = None
    after_image: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
