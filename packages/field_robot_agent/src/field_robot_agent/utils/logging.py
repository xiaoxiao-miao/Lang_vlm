"""Per-task JSON traces for every model call. Never log API keys."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Tracer:
    def __init__(self, task_dir: Path, enabled: bool = True) -> None:
        self.task_dir = task_dir
        self.enabled = enabled
        self._seq = 0
        if enabled:
            self.task_dir.mkdir(parents=True, exist_ok=True)

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def log_pair(
        self,
        op: str,
        request: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        seq = self.next_seq()
        prefix = f"{seq:03d}_{op}"
        self._write(f"{prefix}_request.json", request)
        self._write(f"{prefix}_response.json", response)

    def log_event(self, name: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        seq = self.next_seq()
        self._write(f"{seq:03d}_{name}.json", payload)

    def _write(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.task_dir / filename
        safe = {k: v for k, v in payload.items() if "api_key" not in k.lower()}
        path.write_text(
            json.dumps(safe, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )


def model_trace_payload(
    *,
    model: str,
    messages: list[Any],
    schema_name: str | None,
    raw_response: Any,
    parsed: Any,
    latency_s: float,
    usage: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = {
        "timestamp": _now(),
        "model": model,
        "messages": messages,
        "structured_schema_name": schema_name,
    }
    response = {
        "timestamp": _now(),
        "model": model,
        "raw_model_response": raw_response,
        "parsed_response": parsed,
        "latency_s": latency_s,
        "token_usage": usage,
    }
    return request, response
