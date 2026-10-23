"""CLI: robot-agent [--image ...] [--scripted] "抓取图片中红色物体"."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from agent_client.proxy_env import sanitize_proxy_env

from robot_agent.agent import run_agent, run_scripted
from robot_agent.robot.dummy import DEFAULT_JOINTS
from robot_agent.scene import SceneState
from robot_agent.tools import RobotRuntime


def _repo_root() -> Path:
    # .../packages/robot_agent/src/robot_agent/main.py -> repo root
    return Path(__file__).resolve().parents[4]


def _default_image() -> Path:
    return (
        _repo_root()
        / "packages"
        / "agent_client"
        / "tests"
        / "fixtures"
        / "scene_sample.jpg"
    )


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        path = _repo_root() / "configs" / "config.robot.local.yaml"
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "vlm" not in data:
        raise ValueError(f"config missing 'vlm' section: {path}")
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LangChain robot agent (dummy + mock policy)")
    p.add_argument(
        "instruction",
        nargs="?",
        default="抓取图片中红色物体",
        help="Natural-language robot instruction",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (default: configs/config.robot.local.yaml)",
    )
    p.add_argument(
        "--scripted",
        action="store_true",
        help="Run fixed tool sequence without LLM (offline demo)",
    )
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help=f"Image sent to the VLM (default: {_default_image()})",
    )
    p.add_argument(
        "--no-image",
        action="store_true",
        help="Do not attach an image to the LLM prompt",
    )
    return p


def resolve_image(args: argparse.Namespace) -> Path | None:
    if args.no_image:
        return None
    path = args.image if args.image is not None else _default_image()
    if not path.is_file():
        print(f"error: image not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path


def build_runtime(cfg: dict[str, Any], image: Path | None) -> RobotRuntime:
    robot_cfg = cfg.get("robot") or {}
    joints = robot_cfg.get("joint_names") or list(DEFAULT_JOINTS)
    return RobotRuntime(
        scene=SceneState.default(),
        joint_names=list(joints),
        horizon=int(robot_cfg.get("horizon", 8)),
        vlm_cfg=dict(cfg["vlm"]),
        image_path=image,
    )


def main(argv: list[str] | None = None) -> None:
    sanitize_proxy_env()
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    image = resolve_image(args)
    runtime = build_runtime(cfg, image)

    print(
        f"[robot-agent] instruction={args.instruction!r} "
        f"scripted={args.scripted} image={image}",
        flush=True,
    )

    if args.scripted:
        print(run_scripted(runtime, args.instruction))
        if runtime.robot.holding is None:
            sys.exit(1)
        return

    agent_cfg = cfg.get("agent") or {}
    system_prompt = agent_cfg.get("system_prompt")
    try:
        run_agent(runtime, args.instruction, system_prompt=system_prompt)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
