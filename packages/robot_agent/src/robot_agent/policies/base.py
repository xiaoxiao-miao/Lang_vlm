"""Minimal Policy ABC (concept-aligned with strands-robots, sync-first)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    """Produce action chunks from observation + language instruction."""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    def requires_images(self) -> bool:
        return False

    @abstractmethod
    def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return a list of action dicts. Values are float (never ndarray)."""
