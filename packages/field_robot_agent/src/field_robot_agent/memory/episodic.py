"""JSONL episodic memory: what happened, not what the world is now."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    timestamp: datetime
    task_id: str
    event_type: str
    description: str
    subgoal: str | None = None
    robot_result: str | None = None
    world_revision: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EpisodicMemory:
    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._records: list[MemoryRecord] = []
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: MemoryRecord) -> None:
        self._records.append(record)
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def recent_text(self, limit: int = 8) -> str:
        lines: list[str] = []
        for rec in self._records[-limit:]:
            ts = rec.timestamp.strftime("%H:%M")
            lines.append(f"{ts} {rec.description}")
        return "\n".join(lines)

    def record(
        self,
        *,
        task_id: str,
        event_type: str,
        description: str,
        subgoal: str | None = None,
        robot_result: str | None = None,
        world_revision: int | None = None,
    ) -> None:
        self.append(
            MemoryRecord(
                timestamp=datetime.now(UTC),
                task_id=task_id,
                event_type=event_type,
                description=description,
                subgoal=subgoal,
                robot_result=robot_result,
                world_revision=world_revision,
            )
        )
