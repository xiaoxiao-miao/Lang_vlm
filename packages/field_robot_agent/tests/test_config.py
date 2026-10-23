"""CLI helpers for VLM connection errors."""

from field_robot_agent.utils.config import apply_model_overrides, vlm_unavailable_message


def test_vlm_unavailable_message_mentions_mock_and_tunnel():
    text = vlm_unavailable_message("http://127.0.0.1:8000/v1", RuntimeError("refused"))
    assert "http://127.0.0.1:8000/v1/models" in text
    assert "--mock-llm" in text
    assert "-L 8000:127.0.0.1:8000" in text


def test_apply_model_overrides_sets_base_url():
    settings = {"model": {"base_url": "http://127.0.0.1:8000/v1", "model_name": "x"}}
    apply_model_overrides(settings, base_url="http://127.0.0.1:8000/v1", model_name="qwen")
    assert settings["model"]["base_url"] == "http://127.0.0.1:8000/v1"
    assert settings["model"]["model_name"] == "qwen"
