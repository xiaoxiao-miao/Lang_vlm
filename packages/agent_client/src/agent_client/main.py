"""CLI entry: vlm-agent [--image path] [--text ...] [--config ...]

Defaults mirror former vlm_server/local_3_agent.sh (local smoke against :8000).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from agent_client.chains import analyze_image
from agent_client.proxy_env import sanitize_proxy_env
from agent_client.vlm_client import ensure_vlm_ready


def _repo_root() -> Path:
    # .../packages/agent_client/src/agent_client/main.py -> repo root
    return Path(__file__).resolve().parents[4]


def _package_root() -> Path:
    # .../packages/agent_client
    return Path(__file__).resolve().parents[2]


def default_image() -> Path:
    return _package_root() / "tests" / "fixtures" / "1.jpg"


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        path = _repo_root() / "configs" / "config.local.yaml"
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "vlm" not in data:
        raise ValueError(f"config missing 'vlm' section: {path}")
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local LangChain VLM agent")
    p.add_argument(
        "--image",
        default=None,
        help=f"Path to input image (default: {default_image()})",
    )
    p.add_argument(
        "--text",
        default="描述场景",
        help="Extra instruction (default: 描述场景)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (default: configs/config.local.yaml)",
    )
    p.add_argument("--base-url", default=None, help="Override vlm.base_url")
    p.add_argument("--model", default=None, help="Override vlm.model")
    p.add_argument(
        "--skip-ready-check",
        action="store_true",
        help="Do not GET /v1/models before inference",
    )
    p.add_argument(
        "positional",
        nargs="*",
        help="Optional: IMAGE [TEXT] (same as former local_3_agent.sh)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    sanitize_proxy_env()
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    vlm = cfg["vlm"]

    image_arg = args.image
    text_arg = args.text
    if args.positional:
        image_arg = args.positional[0]
        if len(args.positional) > 1:
            text_arg = args.positional[1]
        if len(args.positional) > 2:
            print("error: too many positional args (want IMAGE [TEXT])", file=sys.stderr)
            sys.exit(2)

    base_url = args.base_url or vlm["base_url"]
    model = args.model or vlm["model"]
    api_key = vlm.get("api_key", "EMPTY")
    temperature = float(vlm.get("temperature", 0.2))

    image_path = Path(image_arg) if image_arg else default_image()
    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_ready_check:
        try:
            models = ensure_vlm_ready(base_url)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        print("[vlm-agent] GET /v1/models", file=sys.stderr)
        print(json.dumps(models, ensure_ascii=False, indent=2))

    print(
        f"[vlm-agent] base_url={base_url} model={model} image={image_path}",
        file=sys.stderr,
    )
    result = analyze_image(
        image_path=str(image_path),
        user_text=text_arg,
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
    )
    print(result)


if __name__ == "__main__":
    main()
