"""Rich console printer for the Environment Agent loop."""

from __future__ import annotations

from rich.console import Console

from field_robot_agent.agent.state import GoalCheckResult, TaskPlan
from field_robot_agent.robot.skills import RobotExecutionResult, SkillCall
from field_robot_agent.world_model.events import WorldEvent
from field_robot_agent.world_model.schema import WorldState, format_world


class RunPrinter:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def banner(self) -> None:
        self.console.print("Field-Robot Environment Agent")
        self.console.print("-----------------------------")

    def task(self, goal: str) -> None:
        self.console.print(f"\n[Task]\n{goal}")

    def world(self, world: WorldState) -> None:
        self.console.print(f"\n[World Revision {world.revision}]")
        self.console.print(format_world(world))

    def plan(self, plan: TaskPlan) -> None:
        self.console.print("\n[Plan]")
        if plan.reasoning_summary:
            self.console.print(plan.reasoning_summary)
        for i, sg in enumerate(plan.subgoals, start=1):
            dest = sg.skill.parameters.get("destination")
            extra = f" {dest}" if dest else ""
            target = sg.skill.target or ""
            self.console.print(f"{i}. {sg.skill.skill} {target}{extra}".rstrip())

    def execute(self, skill: SkillCall) -> None:
        dest = skill.parameters.get("destination")
        args = skill.target or ""
        if dest:
            args = f"{skill.target}, {dest}"
        self.console.print(f"\n[Execute]\n{skill.skill}({args})")

    def robot(self, result: RobotExecutionResult) -> None:
        status = "success" if result.success else "failed"
        self.console.print(f"\n[Robot]\n{status}")
        if result.message:
            self.console.print(result.message)

    def world_update(self, event: WorldEvent) -> None:
        self.console.print("\n[World Update]")
        if not event.has_changes():
            self.console.print(event.description or "(no changes)")
            return
        for ch in event.attribute_changes:
            self.console.print(f"{ch.entity_id}.{ch.attribute}: {ch.old_value} -> {ch.new_value}")
        for rel in event.relation_changes:
            self.console.print(f"{rel.action} {rel.subject} {rel.predicate} {rel.object}")
        for obj in event.new_objects:
            self.console.print(f"new object {obj.id} ({obj.type})")

    def goal(self, result: GoalCheckResult) -> None:
        label = "SUCCESS" if result.satisfied else "not satisfied"
        self.console.print(f"\n[Goal]\n{label}")
        if result.explanation:
            self.console.print(result.explanation)

    def answer(self, explanation: str) -> None:
        self.console.print(f"\n[Answer]\n{explanation}")

    def fail(self, message: str) -> None:
        self.console.print(f"\n[Goal]\nFAILED\n{message}")
