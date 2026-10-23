"""Image helpers for OpenAI-compatible multimodal messages."""

from __future__ import annotations

from pathlib import Path

from agent_client.image_encoder import encode_image_file


def encode_image(path_or_url: str | Path) -> str:
    """Return a data URL for a local file, or pass through HTTP(S) URLs."""
    text = str(path_or_url)
    if text.startswith(("http://", "https://", "data:")):
        return text
    return encode_image_file(text)


def image_message_part(path_or_url: str | Path) -> dict[str, object]:
    return {"type": "image_url", "image_url": {"url": encode_image(path_or_url)}}
