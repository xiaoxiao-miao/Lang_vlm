"""Unified VLM/LLM client. No other module constructs ChatOpenAI."""

from __future__ import annotations

import time
from typing import Any

from agent_client.vlm_client import create_vlm_llm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from field_robot_agent.agent.state import GoalCheckResult, Subgoal, SubgoalVerifyResult, TaskPlan
from field_robot_agent.prompts import load_prompt
from field_robot_agent.robot.skills import Observation, RobotExecutionResult, SkillCall
from field_robot_agent.utils.image import image_message_part
from field_robot_agent.utils.logging import Tracer, model_trace_payload
from field_robot_agent.world_model.events import WorldEvent
from field_robot_agent.world_model.schema import AVAILABLE_SKILLS, WorldState


def _shorten_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                image = block.get("image_url") or {}
                url = str(image.get("url", "") if isinstance(image, dict) else image)
                if url.startswith("data:") and ";base64," in url:
                    head, _, b64 = url.partition(";base64,")
                    url = f"{head};base64,{b64[:48]}…({len(b64)} chars)"
                out.append({"type": "image_url", "image_url": {"url": url}})
            else:
                out.append(block)
        return out
    return content


def _dump_messages(messages: list[Any]) -> list[dict[str, Any]]:
    dumped: list[dict[str, Any]] = []
    for msg in messages:
        dumped.append(
            {
                "type": getattr(msg, "type", msg.__class__.__name__),
                "content": _shorten_content(getattr(msg, "content", str(msg))),
            }
        )
    return dumped


def _usage_from_message(raw: Any) -> dict[str, Any] | None:
    meta = getattr(raw, "usage_metadata", None)
    if isinstance(meta, dict):
        return meta
    return None


class VLMClient:
    def __init__(
        self,
        *,
        llm: BaseChatModel,
        model_name: str,
        tracer: Tracer | None = None,
    ) -> None:
        self.llm = llm
        self.model_name = model_name
        self.tracer = tracer

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any], tracer: Tracer | None = None) -> VLMClient:
        llm = create_vlm_llm(
            base_url=str(model_cfg["base_url"]),
            model=str(model_cfg.get("model_name") or model_cfg.get("model")),
            api_key=str(model_cfg.get("api_key", "EMPTY")),
            temperature=float(model_cfg.get("temperature", 0.1)),
            timeout=float(model_cfg.get("timeout", 60)),
            max_tokens=int(model_cfg.get("max_tokens", 2048)),
        )
        return cls(
            llm=llm,
            model_name=str(model_cfg.get("model_name") or model_cfg.get("model")),
            tracer=tracer,
        )

    def analyze_scene(
        self,
        world: WorldState,
        observation: Observation,
        image: str | None = None,
    ) -> WorldEvent:
        prompt = load_prompt("initial_scene.txt").format(
            world_json=world.model_dump_json(indent=2),
            observation=observation.text_description or "",
        )
        return self._structured(
            "world_update",
            WorldEvent,
            prompt,
            images=[image or observation.after_image or observation.before_image],
        )

    def analyze_change(
        self,
        previous_world: WorldState,
        robot_action: SkillCall | None,
        robot_result: RobotExecutionResult | None,
        before_image: str | None = None,
        after_image: str | None = None,
    ) -> WorldEvent:
        prompt = load_prompt("world_update.txt").format(
            world_json=previous_world.model_dump_json(indent=2),
            skill_json=(robot_action.model_dump_json() if robot_action else "{}"),
            result_json=(robot_result.model_dump_json() if robot_result else "{}"),
        )
        return self._structured(
            "world_update",
            WorldEvent,
            prompt,
            images=[before_image, after_image],
        )

    def plan_task(self, goal: str, world: WorldState, memory_text: str = "") -> TaskPlan:
        prompt = load_prompt("planner.txt").format(
            goal=goal,
            world_json=world.model_dump_json(indent=2),
            skills=", ".join(AVAILABLE_SKILLS),
            memory=memory_text or "(empty)",
        )
        return self._structured("plan", TaskPlan, prompt)

    def verify_subgoal(
        self,
        subgoal: Subgoal,
        world: WorldState,
        robot_result: RobotExecutionResult | None,
    ) -> SubgoalVerifyResult:
        prompt = load_prompt("subgoal_verify.txt").format(
            subgoal_json=subgoal.model_dump_json(indent=2),
            expected_effects="\n".join(subgoal.expected_effects) or "(none)",
            result_json=(robot_result.model_dump_json() if robot_result else "{}"),
            world_json=world.model_dump_json(indent=2),
        )
        return self._structured("verify", SubgoalVerifyResult, prompt)

    def check_goal(self, goal: str, world: WorldState) -> GoalCheckResult:
        prompt = load_prompt("goal_check.txt").format(
            goal=goal,
            world_json=world.model_dump_json(indent=2),
        )
        return self._structured("goal_check", GoalCheckResult, prompt)

    def _structured(
        self,
        op: str,
        schema: type[BaseModel],
        prompt: str,
        images: list[str | None] | None = None,
    ) -> Any:
        messages = self._messages(prompt, images)
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                return self._invoke_structured(op, schema, messages)
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = exc
        raise RuntimeError(f"structured output failed for {op}: {last_error}") from last_error

    def _invoke_structured(
        self,
        op: str,
        schema: type[BaseModel],
        messages: list[Any],
    ) -> Any:
        started = time.perf_counter()
        runnable = self.llm.with_structured_output(schema, include_raw=True)
        result = runnable.invoke(messages)
        latency = time.perf_counter() - started
        parsed: Any
        raw: Any = None
        if isinstance(result, dict):
            parsed = result.get("parsed")
            raw = result.get("raw")
            error = result.get("parsing_error")
            if parsed is None:
                raise ValueError(error or "structured output parsed=None")
        else:
            parsed = result
        if self.tracer is not None:
            req, resp = model_trace_payload(
                model=self.model_name,
                messages=_dump_messages(messages),
                schema_name=schema.__name__,
                raw_response=getattr(raw, "content", raw),
                parsed=parsed.model_dump() if isinstance(parsed, BaseModel) else parsed,
                latency_s=latency,
                usage=_usage_from_message(raw),
            )
            self.tracer.log_pair(op, req, resp)
        return parsed

    @staticmethod
    def _messages(prompt: str, images: list[str | None] | None) -> list[Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images or []:
            if image:
                content.append(image_message_part(image))
        return [
            SystemMessage(
                content="Return valid structured JSON that matches the requested schema."
            ),
            HumanMessage(content=content if len(content) > 1 else prompt),
        ]
