"""Turn observations / robot results into WorldEvent patches."""

from __future__ import annotations

from field_robot_agent.perception.vlm_client import VLMClient
from field_robot_agent.robot.skills import Observation, RobotExecutionResult, SkillCall
from field_robot_agent.world_model.events import WorldEvent
from field_robot_agent.world_model.schema import WorldState


class SceneUpdater:
    def __init__(self, *, mock_llm: bool, vlm: VLMClient | None = None) -> None:
        self.mock_llm = mock_llm
        self.vlm = vlm

    def update(
        self,
        world: WorldState,
        observation: Observation | None,
        skill: SkillCall | None,
        result: RobotExecutionResult | None,
    ) -> WorldEvent:
        if self.mock_llm:
            if result is not None and result.simulated_changes is not None:
                return result.simulated_changes
            return WorldEvent(
                event_type="observe",
                description=(observation.text_description if observation else "observe"),
            )
        if self.vlm is None:
            raise RuntimeError("VLMClient required when mock_llm is false")
        if result is not None:
            return self.vlm.analyze_change(
                previous_world=world,
                robot_action=skill,
                robot_result=result,
                before_image=observation.before_image if observation else None,
                after_image=observation.after_image if observation else None,
            )
        if observation is None:
            return WorldEvent(event_type="observe", description="no observation")
        return self.vlm.analyze_scene(world, observation)
