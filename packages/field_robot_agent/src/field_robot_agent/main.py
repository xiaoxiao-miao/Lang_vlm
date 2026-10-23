"""CLI: uv run field-agent [--mock-llm] [--task '...']."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from agent_client.proxy_env import sanitize_proxy_env
from agent_client.vlm_client import ensure_vlm_ready
from dotenv import load_dotenv
from rich.console import Console

from field_robot_agent.agent.runtime import build_runtime, run_task
from field_robot_agent.robot.mock_robot import MockRobot
from field_robot_agent.utils.config import (
    apply_model_overrides,
    default_config_dir,
    load_settings,
    vlm_unavailable_message,
)
from field_robot_agent.utils.console import RunPrinter
from field_robot_agent.world_model.schema import WorldState, build_demo_world

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _ensure_localhost_noproxy() -> None:
    extra = "127.0.0.1,localhost"
    for key in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(key, "")
        if "127.0.0.1" in cur:
            continue
        os.environ[key] = f"{cur},{extra}" if cur else extra


def _require_vlm(settings: dict, printer: RunPrinter) -> None:
    model_cfg = settings.get("model") or {}
    base_url = str(model_cfg.get("base_url") or "http://127.0.0.1:8000/v1")
    try:
        ensure_vlm_ready(base_url)
    except RuntimeError as exc:
        cause = exc.__cause__ or exc
        printer.console.print(vlm_unavailable_message(base_url, cause))
        raise typer.Exit(code=1) from exc


def _run(
    *,
    mock_llm: bool,
    task: str | None,
    config_dir: Path | None,
    logs_dir: Path | None,
    base_url: str | None,
    model: str | None,
) -> None:
    sanitize_proxy_env()
    _ensure_localhost_noproxy()
    load_dotenv()
    settings = apply_model_overrides(
        load_settings(config_dir),
        base_url=base_url,
        model_name=model,
    )
    printer = RunPrinter()
    if not mock_llm:
        _require_vlm(settings, printer)
        model_cfg = settings.get("model") or {}
        printer.console.print(
            f"[field-agent] VLM {model_cfg.get('base_url')}  model={model_cfg.get('model_name')}"
        )
    printer.banner()
    logs_root = logs_dir or Path.cwd() / "logs"
    robot = MockRobot(build_demo_world(""))
    session_world: WorldState | None = None

    def execute(goal: str) -> None:
        nonlocal session_world
        runtime = build_runtime(
            mock_llm=mock_llm,
            settings=settings,
            logs_root=logs_root,
            printer=printer,
            robot=robot,
        )
        try:
            result = run_task(goal, runtime, world_state=session_world)
        except Exception as exc:
            model_cfg = settings.get("model") or {}
            printer.fail(
                vlm_unavailable_message(str(model_cfg.get("base_url") or ""), exc)
                if not mock_llm
                else str(exc)
            )
            raise typer.Exit(code=1) from exc
        session_world = result.get("world_state") or session_world
        status = result.get("task_status")
        if status != "success":
            raise typer.Exit(code=1)

    if task:
        execute(task)
        return

    console = Console()
    while True:
        try:
            goal = console.input("\nTask > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not goal or goal.lower() in {"q", "quit", "exit"}:
            return
        try:
            execute(goal)
        except typer.Exit:
            continue


@app.callback(invoke_without_command=True)
def cli(
    ctx: typer.Context,
    mock_llm: Annotated[
        bool, typer.Option("--mock-llm", help="Rule-based loop, no remote vLLM")
    ] = False,
    task: Annotated[str | None, typer.Option("--task", help="Run one task and exit")] = None,
    config_dir: Annotated[
        Path | None, typer.Option("--config-dir", help="Directory with model.yaml / agent.yaml")
    ] = None,
    logs_dir: Annotated[
        Path | None, typer.Option("--logs-dir", help="Trace output directory")
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="Override model.base_url (default http://127.0.0.1:8000/v1)"),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", help="Override model.model_name")] = None,
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    _run(
        mock_llm=mock_llm,
        task=task,
        config_dir=config_dir or default_config_dir(),
        logs_dir=logs_dir,
        base_url=base_url,
        model=model,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
