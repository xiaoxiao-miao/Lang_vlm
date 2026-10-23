"""Subgoal verification: robot_result + world + expected effects. Do not trust success alone."""

from __future__ import annotations

from typing import Protocol

from field_robot_agent.agent.state import Subgoal, SubgoalVerifyResult
from field_robot_agent.robot.skills import RobotExecutionResult
from field_robot_agent.world_model.schema import WorldState, has_relation


class Verifier(Protocol):
    def verify(
        self,
        subgoal: Subgoal,
        world: WorldState,
        robot_result: RobotExecutionResult | None,
    ) -> SubgoalVerifyResult: ...


def _effect_holds(effect: str, world: WorldState) -> bool:
    text = effect.strip().lower().replace(" ", "")
    drawer = world.objects.get("drawer_1")
    drawer_state = drawer.attributes.get("state") if drawer else None
    holding = world.robot.holding_object

    checks: dict[str, bool] = {
        "drawer_1.state==open": drawer_state == "open",
        "drawer_1.open": drawer_state == "open",
        "drawer_1.state==closed": drawer_state == "closed",
        "robotholdingapple_1": holding == "apple_1",
        "robotholdingnothing": holding is None,
        "apple_1insidedrawer_1": has_relation(world, "apple_1", "inside", "drawer_1"),
    }
    compact = effect.strip().lower()
    for key, ok in checks.items():
        if key in text or key.replace("==", " ") in compact.replace(" ", ""):
            return ok
    if "inside" in compact and "apple" in compact and "drawer" in compact:
        return has_relation(world, "apple_1", "inside", "drawer_1")
    if "holding nothing" in compact or "hand is empty" in compact:
        return holding is None
    if "holding apple" in compact:
        return holding == "apple_1"
    if "drawer" in compact and "open" in compact:
        return drawer_state == "open"
    return True


class RuleBasedVerifier:
    def verify(
        self,
        subgoal: Subgoal,
        world: WorldState,
        robot_result: RobotExecutionResult | None,
    ) -> SubgoalVerifyResult:
        if robot_result is not None and not robot_result.success:
            return SubgoalVerifyResult(
                success=False,
                explanation=robot_result.message,
            )
        missing = [e for e in subgoal.expected_effects if not _effect_holds(e, world)]
        if missing:
            return SubgoalVerifyResult(
                success=False,
                explanation=f"expected effects not met: {missing}",
            )
        return SubgoalVerifyResult(
            success=True,
            explanation=f"subgoal {subgoal.id} verified against world revision {world.revision}",
        )


class VLMVerifier:
    def __init__(self, vlm: object) -> None:
        self._vlm = vlm

    def verify(
        self,
        subgoal: Subgoal,
        world: WorldState,
        robot_result: RobotExecutionResult | None,
    ) -> SubgoalVerifyResult:
        return self._vlm.verify_subgoal(subgoal=subgoal, world=world, robot_result=robot_result)
