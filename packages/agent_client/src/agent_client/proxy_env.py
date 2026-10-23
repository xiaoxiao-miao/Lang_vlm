"""Sanitize proxy env vars that break httpx (langchain-openai / huggingface)."""

from __future__ import annotations

import os


def sanitize_proxy_env() -> None:
    """Drop ALL_PROXY socks:// schemes; httpx rejects scheme 'socks://'."""
    for key in ("ALL_PROXY", "all_proxy"):
        val = os.environ.get(key)
        if not val:
            continue
        if val.startswith("socks://"):
            # Prefer socks5 if a socks proxy is required; otherwise unset.
            os.environ.pop(key, None)

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        val = os.environ.get(key)
        if val and val.startswith("socks://"):
            os.environ[key] = "socks5://" + val[len("socks://") :]
