"""Change-centric world events. VLM should emit these, not a full WorldState."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from field_robot_agent.world_model.schema import ObjectState


class AttributeChange(BaseModel):
    entity_id: str
    attribute: str
    old_value: Any | None = None
    new_value: Any = None
    confidence: float = 1.0


class RelationChange(BaseModel):
    action: Literal["add", "remove"]
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0


class WorldEvent(BaseModel):
    event_type: str
    description: str
    attribute_changes: list[AttributeChange] = Field(default_factory=list)
    relation_changes: list[RelationChange] = Field(default_factory=list)
    new_objects: list[ObjectState] = Field(default_factory=list)
    confidence: float = 1.0

    def has_changes(self) -> bool:
        return bool(self.attribute_changes or self.relation_changes or self.new_objects)
