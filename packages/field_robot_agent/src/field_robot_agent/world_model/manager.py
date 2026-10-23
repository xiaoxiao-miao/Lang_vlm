"""Apply WorldEvent patches onto WorldState with confidence gating."""

from __future__ import annotations

from field_robot_agent.world_model.events import AttributeChange, WorldEvent
from field_robot_agent.world_model.schema import Relation, WorldState

OBJECT_FIELDS = {"visible", "confidence", "position", "frame", "type"}
ROBOT_FIELDS = {"holding_object", "status", "current_skill"}
PROTECT_THRESHOLD = 0.8


class WorldModelManager:
    def apply_event(self, world_state: WorldState, event: WorldEvent) -> WorldState:
        world = world_state.model_copy(deep=True)
        world.revision += 1

        for obj in event.new_objects:
            existing = world.objects.get(obj.id)
            if existing is None or obj.confidence >= existing.confidence:
                world.objects[obj.id] = obj

        for change in event.attribute_changes:
            self._apply_attribute(world, change)

        for rel_change in event.relation_changes:
            if not self._confidence_ok(rel_change.confidence, event.confidence):
                continue
            match = Relation(
                subject=rel_change.subject,
                predicate=rel_change.predicate,
                object=rel_change.object,
                confidence=rel_change.confidence,
            )
            if rel_change.action == "remove":
                world.relations = [
                    r
                    for r in world.relations
                    if not (
                        r.subject == match.subject
                        and r.predicate == match.predicate
                        and r.object == match.object
                    )
                ]
            else:
                if not any(
                    r.subject == match.subject
                    and r.predicate == match.predicate
                    and r.object == match.object
                    for r in world.relations
                ):
                    world.relations.append(match)
                else:
                    world.relations = [
                        match
                        if (
                            r.subject == match.subject
                            and r.predicate == match.predicate
                            and r.object == match.object
                            and match.confidence >= r.confidence
                        )
                        else r
                        for r in world.relations
                    ]

        return WorldState.model_validate(world.model_dump())

    def _apply_attribute(self, world: WorldState, change: AttributeChange) -> None:
        if (
            world.robot.robot_id == change.entity_id or change.entity_id == "robot"
        ) and change.attribute in ROBOT_FIELDS:
            current_conf = 1.0
            if not self._may_overwrite(current_conf, change.confidence):
                return
            setattr(world.robot, change.attribute, change.new_value)
            return

        obj = world.objects.get(change.entity_id)
        if obj is None:
            return
        if not self._may_overwrite(obj.confidence, change.confidence):
            return

        if change.attribute in OBJECT_FIELDS:
            setattr(obj, change.attribute, change.new_value)
            if change.attribute != "confidence":
                obj.confidence = max(obj.confidence, change.confidence)
            return

        obj.attributes[change.attribute] = change.new_value
        obj.confidence = max(obj.confidence, change.confidence)

    @staticmethod
    def _may_overwrite(existing: float, incoming: float) -> bool:
        if incoming >= existing:
            return True
        if existing >= PROTECT_THRESHOLD and incoming < existing:
            return False
        return incoming >= existing

    @staticmethod
    def _confidence_ok(change_conf: float, event_conf: float) -> bool:
        return min(change_conf, event_conf) > 0.0
