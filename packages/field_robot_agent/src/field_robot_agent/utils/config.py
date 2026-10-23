"""Load YAML config with environment-variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def package_root() -> Path:
    # .../src/field_robot_agent/utils/config.py -> packages/field_robot_agent
    return Path(__file__).resolve().parents[3]


def default_config_dir() -> Path:
    return package_root() / "config"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping in {path}")
    return data


def apply_model_overrides(
    settings: dict[str, Any],
    *,
    base_url: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    model_cfg = dict(settings.get("model") or {})
    if base_url:
        model_cfg["base_url"] = base_url
    if model_name:
        model_cfg["model_name"] = model_name
    settings["model"] = model_cfg
    return settings


def vlm_unavailable_message(base_url: str, cause: BaseException) -> str:
    endpoint = f"{str(base_url).rstrip('/')}/models"
    return (
        f"VLM 连不上 {endpoint}（{cause}）\n"
        "  离线演示:  uv run field-agent --mock-llm\n"
        "  本机 vLLM: 把服务起在 :8000\n"
        "  华为云: 停掉本机 :8000 后 SSH -L 8000:127.0.0.1:8000，再 uv run field-agent"
    )


def load_settings(config_dir: Path | None = None) -> dict[str, Any]:
    cfg_dir = config_dir or default_config_dir()
    model = load_yaml(cfg_dir / "model.yaml")
    agent = load_yaml(cfg_dir / "agent.yaml")
    merged: dict[str, Any] = {**agent, **model}

    model_cfg = dict(merged.get("model") or {})
    model_cfg["base_url"] = os.environ.get("FIELD_AGENT_BASE_URL", model_cfg.get("base_url"))
    model_cfg["model_name"] = os.environ.get("FIELD_AGENT_MODEL", model_cfg.get("model_name"))
    model_cfg["api_key"] = os.environ.get("FIELD_AGENT_API_KEY", model_cfg.get("api_key", "EMPTY"))
    if "FIELD_AGENT_TEMPERATURE" in os.environ:
        model_cfg["temperature"] = float(os.environ["FIELD_AGENT_TEMPERATURE"])
    if "FIELD_AGENT_TIMEOUT" in os.environ:
        model_cfg["timeout"] = float(os.environ["FIELD_AGENT_TIMEOUT"])
    merged["model"] = model_cfg
    return merged
