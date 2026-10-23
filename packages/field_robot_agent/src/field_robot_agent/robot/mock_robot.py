"""Mock robot with an internal simulated world that actually advances."""

from __future__ import annotations

from datetime import UTC, datetime

from field_robot_agent.robot.interface import RobotInterface
from field_robot_agent.robot.skills import Observation, RobotExecutionResult, SkillCall
from field_robot_agent.world_model.events import AttributeChange, RelationChange, WorldEvent
from field_robot_agent.world_model.manager import WorldModelManager
from field_robot_agent.world_model.schema import WorldState, build_demo_world, format_world


def _now() -> datetime:
    return datetime.now(UTC)


class MockRobot(RobotInterface):
    """In-memory skill simulator. Independent from the agent's WorldState belief."""

    def __init__(self, world: WorldState | None = None) -> None:
        self.sim: WorldState = (world or build_demo_world("")).model_copy(deep=True)
        self._manager = WorldModelManager()

    def get_observation(self) -> Observation:
        return Observation(
            timestamp=_now(),
            text_description=format_world(self.sim),
            metadata={"revision": self.sim.revision, "backend": "mock"},
        )

    def execute_skill(self, skill: SkillCall) -> RobotExecutionResult:
        handlers = {
            "inspect": self._inspect,
            "open_drawer": self._open_drawer,
            "close_drawer": self._close_drawer,
            "grasp": self._grasp,
            "place": self._place,
            "move_object": self._move_object,
            "navigate": self._navigate,
        }
        handler = handlers.get(skill.skill)
        if handler is None:
            return RobotExecutionResult(
                success=False,
                message=f"unknown skill {skill.skill!r}",
                error_code="unknown_skill",
            )
        result = handler(skill)
        if result.success and result.simulated_changes and result.simulated_changes.has_changes():
            self.sim = self._manager.apply_event(self.sim, result.simulated_changes)
        return result

    def _inspect(self, skill: SkillCall) -> RobotExecutionResult:
        target = skill.target or "scene"
        return RobotExecutionResult(
            success=True,
            message=f"inspected {target}: {format_world(self.sim)}",
            simulated_changes=WorldEvent(
                event_type="inspect",
                description=f"inspected {target}",
            ),
        )

    def _open_drawer(self, skill: SkillCall) -> RobotExecutionResult:
        drawer_id = skill.target or "drawer_1"
        drawer = self.sim.objects.get(drawer_id)
        if drawer is None:
            return RobotExecutionResult(
                success=False, message=f"{drawer_id} not found", error_code="not_found"
            )
        if drawer.attributes.get("state") == "open":
            return RobotExecutionResult(
                success=False, message=f"{drawer_id} already open", error_code="already_open"
            )
        return RobotExecutionResult(
            success=True,
            message="drawer opened",
            simulated_changes=WorldEvent(
                event_type="drawer_opened",
                description=f"{drawer_id} opened",
                attribute_changes=[
                    AttributeChange(
                        entity_id=drawer_id,
                        attribute="state",
                        old_value="closed",
                        new_value="open",
                        confidence=1.0,
                    )
                ],
            ),
        )

    def _close_drawer(self, skill: SkillCall) -> RobotExecutionResult:
        drawer_id = skill.target or "drawer_1"
        drawer = self.sim.objects.get(drawer_id)
        if drawer is None:
            return RobotExecutionResult(
                success=False, message=f"{drawer_id} not found", error_code="not_found"
            )
        if drawer.attributes.get("state") == "closed":
            return RobotExecutionResult(
                success=False, message=f"{drawer_id} already closed", error_code="already_closed"
            )
        return RobotExecutionResult(
            success=True,
            message="drawer closed",
            simulated_changes=WorldEvent(
                event_type="drawer_closed",
                description=f"{drawer_id} closed",
                attribute_changes=[
                    AttributeChange(
                        entity_id=drawer_id,
                        attribute="state",
                        old_value="open",
                        new_value="closed",
                        confidence=1.0,
                    )
                ],
            ),
        )

    def _grasp(self, skill: SkillCall) -> RobotExecutionResult:
        target = skill.target
        if not target:
            return RobotExecutionResult(
                success=False, message="grasp requires target", error_code="missing_target"
            )
        obj = self.sim.objects.get(target)
        if obj is None:
            return RobotExecutionResult(
                success=False, message=f"{target} not found", error_code="not_found"
            )
        if not obj.visible:
            return RobotExecutionResult(
                success=False, message=f"{target} not visible", error_code="not_visible"
            )
        if self.sim.robot.holding_object is not None:
            return RobotExecutionResult(
                success=False,
                message=f"already holding {self.sim.robot.holding_object}",
                error_code="hands_full",
            )
        support = next(
            (r.object for r in self.sim.relations if r.subject == target and r.predicate == "on"),
            None,
        )
        changes = [
            AttributeChange(
                entity_id=self.sim.robot.robot_id,
                attribute="holding_object",
                old_value=None,
                new_value=target,
                confidence=1.0,
            )
        ]
        rels: list[RelationChange] = []
        if support is not None:
            rels.append(
                RelationChange(
                    action="remove",
                    subject=target,
                    predicate="on",
                    object=support,
                    confidence=1.0,
                )
            )
        return RobotExecutionResult(
            success=True,
            message=f"grasped {target}",
            simulated_changes=WorldEvent(
                event_type="grasped",
                description=f"robot grasped {target}",
                attribute_changes=changes,
                relation_changes=rels,
            ),
        )

    def _place(self, skill: SkillCall) -> RobotExecutionResult:
        held = self.sim.robot.holding_object
        if held is None:
            return RobotExecutionResult(
                success=False, message="not holding any object", error_code="not_holding"
            )
        obj_id, dest = self._place_args(skill, held)
        if obj_id != held:
            return RobotExecutionResult(
                success=False,
                message=f"holding {held}, cannot place {obj_id}",
                error_code="wrong_object",
            )
        dest_obj = self.sim.objects.get(dest) if dest else None
        if dest_obj is None:
            return RobotExecutionResult(
                success=False,
                message="place requires destination",
                error_code="missing_destination",
            )
        if dest_obj.type == "drawer" and dest_obj.attributes.get("state") != "open":
            return RobotExecutionResult(
                success=False, message=f"{dest} is closed", error_code="drawer_closed"
            )
        predicate = "inside" if dest_obj.type == "drawer" else "on"
        return RobotExecutionResult(
            success=True,
            message=f"placed {obj_id} {predicate} {dest}",
            simulated_changes=WorldEvent(
                event_type="placed",
                description=f"placed {obj_id} {predicate} {dest}",
                attribute_changes=[
                    AttributeChange(
                        entity_id=self.sim.robot.robot_id,
                        attribute="holding_object",
                        old_value=obj_id,
                        new_value=None,
                        confidence=1.0,
                    )
                ],
                relation_changes=[
                    RelationChange(
                        action="add",
                        subject=obj_id,
                        predicate=predicate,
                        object=dest,
                        confidence=1.0,
                    )
                ],
            ),
        )

    def _move_object(self, skill: SkillCall) -> RobotExecutionResult:
        if self.sim.robot.holding_object is None:
            grasp = self._grasp(skill)
            if not grasp.success:
                return grasp
            if grasp.simulated_changes and grasp.simulated_changes.has_changes():
                self.sim = self._manager.apply_event(self.sim, grasp.simulated_changes)
        return self._place(skill)

    def _navigate(self, skill: SkillCall) -> RobotExecutionResult:
        dest = skill.target or skill.parameters.get("location") or "scene"
        return RobotExecutionResult(
            success=True,
            message=f"navigated to {dest}",
            simulated_changes=WorldEvent(
                event_type="navigated",
                description=f"robot navigated to {dest}",
                attribute_changes=[
                    AttributeChange(
                        entity_id=self.sim.robot.robot_id,
                        attribute="status",
                        old_value=self.sim.robot.status,
                        new_value="idle",
                        confidence=1.0,
                    )
                ],
            ),
        )

    @staticmethod
    def _place_args(skill: SkillCall, held: str) -> tuple[str, str | None]:
        dest = (
            skill.parameters.get("destination")
            or skill.parameters.get("into")
            or skill.parameters.get("location")
        )
        if dest:
            return skill.target or held, str(dest)
        if skill.target and skill.target != held:
            return held, skill.target
        return skill.target or held, None
