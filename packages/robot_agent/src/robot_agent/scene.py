"""In-memory tabletop scene for phase-1 dummy robot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SceneObject:
    name: str
    color: str
    location: str = "table"
    pose: tuple[float, float, float] = (0.3, 0.0, 0.05)
    grasped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "color": self.color,
            "location": self.location,
            "pose": list(self.pose),
            "grasped": self.grasped,
        }


@dataclass
class SceneState:
    """Default scene: a red cube on the table (plus optional distractors)."""

    objects: list[SceneObject] = field(default_factory=list)

    @classmethod
    def default(cls) -> SceneState:
        return cls(
            objects=[
                SceneObject(name="红色方块", color="红色", pose=(0.35, 0.05, 0.05)),
                SceneObject(name="蓝色方块", color="蓝色", pose=(0.25, -0.10, 0.05)),
            ]
        )

    def list_objects(self) -> list[dict[str, Any]]:
        return [obj.to_dict() for obj in self.objects]

    def describe(self) -> str:
        if not self.objects:
            return "桌面为空。"
        parts = []
        for obj in self.objects:
            state = "已被夹持" if obj.grasped else f"在{obj.location}上"
            label = obj.name if obj.color in obj.name else f"{obj.color}{obj.name}"
            parts.append(f"{label}（{state}，位姿≈{obj.pose}）")
        return "桌面上可见：" + "；".join(parts) + "。"

    def find_by_color(self, color: str) -> SceneObject | None:
        needle = color.strip().lower()
        for obj in self.objects:
            if needle in obj.color.lower() or needle in obj.name.lower():
                return obj
        # Chinese aliases
        aliases = {"red": "红", "蓝色": "蓝", "blue": "蓝", "绿色": "绿", "green": "绿"}
        for key, zh in aliases.items():
            if key in needle or zh in needle:
                for obj in self.objects:
                    if zh in obj.color or zh in obj.name:
                        return obj
        return None

    def find_from_instruction(self, instruction: str) -> SceneObject | None:
        text = instruction.strip().lower()
        for color in ("红色", "蓝色", "绿色", "黄色", "red", "blue", "green"):
            if color in text:
                found = self.find_by_color(color)
                if found is not None:
                    return found
        # fallback: first ungrasped object on table
        for obj in self.objects:
            if not obj.grasped and obj.location == "table":
                return obj
        return None
