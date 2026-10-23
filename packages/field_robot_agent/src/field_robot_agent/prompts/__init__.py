"""Load prompt templates shipped with the package."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def _prompts_dir() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    path = _prompts_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"prompt not found: {path}")
    return path.read_text(encoding="utf-8")
