"""Unit tests that do not require a running VLM server."""

from agent_client.image_encoder import encode_image_bytes
from agent_client.prompts import build_scene_prompt


def test_encode_image_bytes_prefix():
    # Minimal 1x1 JPEG is not required; encoder only base64-wraps bytes.
    url = encode_image_bytes(b"\xff\xd8\xff", mime="image/jpeg")
    assert url.startswith("data:image/jpeg;base64,")


def test_build_scene_prompt_with_user_text():
    text = build_scene_prompt("找到桌上的杯子")
    assert "杯子" in text
    assert "场景" in text or "场景" in text
