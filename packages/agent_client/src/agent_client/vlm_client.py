"""Thin helpers around ChatOpenAI pointed at local/cloud vLLM."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_openai import ChatOpenAI


def create_vlm_llm(
    *,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    temperature: float = 0.2,
    timeout: float = 120.0,
    max_tokens: int = 1024,
) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
    )


def models_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def fetch_models(base_url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    """GET {base_url}/models — raises httpx.HTTPError if VLM is down."""
    response = httpx.get(models_endpoint(base_url), timeout=timeout)
    response.raise_for_status()
    return response.json()


def ensure_vlm_ready(base_url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    """Fail fast with a clear message when the OpenAI-compatible server is not up."""
    try:
        return fetch_models(base_url, timeout=timeout)
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"VLM not ready at {models_endpoint(base_url)} — start local_2_run.sh "
            f"(or docker_2_run.sh) first ({exc})"
        ) from exc


def health_check_payload() -> dict[str, Any]:
    return {"ok": True}
