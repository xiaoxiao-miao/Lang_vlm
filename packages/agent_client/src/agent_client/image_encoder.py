"""Image encoding helpers for OpenAI Vision / vLLM multimodal requests."""

from __future__ import annotations

import base64
from pathlib import Path


def encode_image_file(path: str | Path, mime: str | None = None) -> str:
    """Return a data-URL (data:image/...;base64,...) for an image file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"image not found: {p}")

    suffix = p.suffix.lower()
    if mime is None:
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "image/jpeg")

    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def encode_image_bytes(data: bytes, mime: str = "image/jpeg") -> str:
    """Return a data-URL from raw image bytes."""
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"
