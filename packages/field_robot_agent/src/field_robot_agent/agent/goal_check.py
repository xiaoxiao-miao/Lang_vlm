"""Independent goal checker. Mock scenario: apple_1 inside drawer_1."""

from __future__ import annotations

from typing import Protocol

from field_robot_agent.agent.state import GoalCheckResult
from field_robot_agent.world_model.schema import WorldState, format_world, has_relation

_QUERY_MARKERS_ZH = ("在哪里", "在哪儿", "在哪", "在什么地方", "当前世界", "现在的世界", "世界状态")
_QUERY_MARKERS_EN = ("where is", "where are", "where's", "what is the world")
_YES_NO_MARKERS_ZH = ("是不是", "有没有", "对不对", "是吗")
_YES_NO_SUFFIX = ("吗", "嘛", "呢", "？", "?")
_YES_NO_PREFIX_EN = ("is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ")
_OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "apple_1": ("apple", "苹果"),
    "drawer_1": ("drawer", "抽屉"),
    "table_1": ("table", "桌子", "桌"),
    "cabinet_1": ("cabinet", "柜子", "柜"),
}
_STATIC_TYPES = {"table", "cabinet", "drawer"}


class GoalChecker(Protocol):
    def check(self, goal: str, world: WorldState) -> GoalCheckResult: ...


def looks_like_apple_in_drawer(goal: str) -> bool:
    g = goal.lower()
    has_apple = "apple" in g or "苹果" in goal
    has_drawer = "drawer" in g or "抽屉" in goal
    return has_apple and has_drawer


def is_world_query(goal: str) -> bool:
    """True for location / yes-no questions, not manipulation commands."""
    text = goal.strip()
    if not text:
        return False
    if any(marker in text for marker in _QUERY_MARKERS_ZH):
        return True
    lowered = text.lower()
    if any(marker in lowered for marker in _QUERY_MARKERS_EN):
        return True
    return _is_yes_no_question(text)


def _is_yes_no_question(goal: str) -> bool:
    text = goal.strip()
    if not text:
        return False
    if text.endswith(_YES_NO_SUFFIX):
        return True
    if any(marker in text for marker in _YES_NO_MARKERS_ZH):
        return True
    lowered = text.lower()
    return lowered.startswith(_YES_NO_PREFIX_EN)


def describe_mentioned_objects(goal: str, world: WorldState) -> str:
    """Facts about objects named in the utterance."""
    lowered = goal.lower()
    mentioned = [
        obj_id
        for obj_id, aliases in _OBJECT_ALIASES.items()
        if any(alias in goal or alias in lowered for alias in aliases)
    ]
    if not mentioned:
        return format_world(world)

    lines: list[str] = []
    for obj_id in mentioned:
        if world.robot.holding_object == obj_id:
            lines.append(f"{obj_id} held by robot")
        rels = [rel for rel in world.relations if rel.subject == obj_id]
        if rels:
            lines.extend(f"{rel.subject} {rel.predicate} {rel.object}" for rel in rels)
        elif world.robot.holding_object != obj_id:
            obj = world.objects.get(obj_id)
            if obj is None or obj.type in _STATIC_TYPES:
                state = obj.attributes.get("state") if obj else None
                if state is not None:
                    lines.append(f"{obj_id} {state}")
                continue
            lines.append(f"{obj_id} location unknown")
        obj = world.objects.get(obj_id)
        state = obj.attributes.get("state") if obj else None
        if state is not None and f"{obj_id} {state}" not in lines:
            lines.append(f"{obj_id} {state}")
    return "\n".join(lines) if lines else format_world(world)


def _yes_no_claim(goal: str, world: WorldState) -> bool | None:
    lowered = goal.lower()
    apple = "苹果" in goal or "apple" in lowered
    if apple and ("桌" in goal or "table" in lowered):
        return has_relation(world, "apple_1", "on", "table_1")
    if apple and ("抽屉" in goal or "drawer" in lowered):
        return has_relation(world, "apple_1", "inside", "drawer_1")
    if "抽屉" in goal or "drawer" in lowered:
        drawer = world.objects.get("drawer_1")
        open_ = bool(drawer and drawer.attributes.get("state") == "open")
        if "关" in goal or "close" in lowered:
            return not open_
        if "开" in goal or "open" in lowered:
            return open_
    return None


def answer_world_query(goal: str, world: WorldState) -> str:
    """Answer a question from the World Model. Yes/no uses world relations, not VLM."""
    facts = describe_mentioned_objects(goal, world)
    if not _is_yes_no_question(goal):
        return facts
    claim = _yes_no_claim(goal, world)
    if claim is None:
        return facts
    return f"{'Yes' if claim else 'No'}. {facts}"


class RuleBasedGoalChecker:
    def check(self, goal: str, world: WorldState) -> GoalCheckResult:
        if is_world_query(goal):
            return GoalCheckResult(satisfied=True, explanation=answer_world_query(goal, world))
        if looks_like_apple_in_drawer(goal) or not goal.strip():
            ok = has_relation(world, "apple_1", "inside", "drawer_1")
            return GoalCheckResult(
                satisfied=ok,
                explanation=("apple_1 inside drawer_1" if ok else "apple_1 is not inside drawer_1"),
            )
        return GoalCheckResult(
            satisfied=False,
            explanation="no rule-based matcher for this goal; use VLM mode",
        )


class VLMGoalChecker:
    def __init__(self, vlm: object) -> None:
        self._vlm = vlm

    def check(self, goal: str, world: WorldState) -> GoalCheckResult:
        if is_world_query(goal):
            return GoalCheckResult(satisfied=True, explanation=answer_world_query(goal, world))
        return self._vlm.check_goal(goal=goal, world=world)
