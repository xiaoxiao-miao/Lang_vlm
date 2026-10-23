"""Tool-calling robot agent loop with step-by-step LLM prompt/response logging."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from robot_agent.tools import RobotRuntime, build_tools

DEFAULT_SYSTEM = """你是桌面操作机器人控制器，不是聊天助手。
用户会给出图片与抓取指令。请结合图片理解目标，并用工具执行。

必须严格按顺序调用工具：
1. describe_scene 或 list_objects ——确认要抓的物体；
2. grasp_object ——参数 instruction 使用用户原话（如「抓取图片中红色物体」）。

规则：
- 禁止向用户提问或索取额外信息；
- 未调用工具前禁止声称已完成；
- 工具返回成功后，用一两句中文总结即可。"""

MAX_STEPS = 6
_HR = "─" * 72


def create_llm(vlm_cfg: dict[str, Any]) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=str(vlm_cfg["base_url"]).rstrip("/"),
        api_key=str(vlm_cfg.get("api_key", "EMPTY")),
        model=str(vlm_cfg.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")),
        temperature=float(vlm_cfg.get("temperature", 0.2)),
        timeout=float(vlm_cfg.get("timeout", 120)),
        max_tokens=int(vlm_cfg.get("max_tokens", 1024)),
    )


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content)


def _shorten_data_url(url: str, keep: int = 48) -> str:
    if url.startswith("data:") and ";base64," in url:
        head, _, b64 = url.partition(";base64,")
        return f"{head};base64,{b64[:keep]}…({len(b64)} chars)"
    return url if len(url) <= 120 else url[:117] + "..."


def _format_content_for_log(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        lines: list[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    lines.append(str(block.get("text", "")))
                elif btype == "image_url":
                    image = block.get("image_url") or {}
                    url = image.get("url", "") if isinstance(image, dict) else str(image)
                    lines.append(f"[image] {_shorten_data_url(str(url))}")
                else:
                    lines.append(json.dumps(block, ensure_ascii=False)[:500])
            else:
                lines.append(str(block))
        return "\n".join(lines)
    return str(content)


def _format_messages_for_log(messages: list[BaseMessage]) -> str:
    chunks: list[str] = []
    for i, msg in enumerate(messages):
        role = msg.__class__.__name__.replace("Message", "").upper()
        body = _format_content_for_log(msg.content)
        extra = ""
        if isinstance(msg, AIMessage) and msg.tool_calls:
            extra = "\n  tool_calls: " + json.dumps(msg.tool_calls, ensure_ascii=False, indent=2)
        if isinstance(msg, ToolMessage):
            extra = f"\n  tool_call_id={msg.tool_call_id}"
        chunks.append(f"[{i}] {role}:\n{body}{extra}")
    return "\n\n".join(chunks)


def _format_ai_for_log(ai_msg: AIMessage) -> str:
    parts = [_format_content_for_log(ai_msg.content) or "(empty content)"]
    if ai_msg.tool_calls:
        parts.append("tool_calls:")
        parts.append(json.dumps(ai_msg.tool_calls, ensure_ascii=False, indent=2))
    return "\n".join(parts)


def _print_step_header(step: int, title: str) -> None:
    print(f"\n{_HR}", flush=True)
    print(f"步骤 {step}: {title}", flush=True)
    print(_HR, flush=True)


def _print_llm_exchange(
    step: int,
    *,
    tool_choice: Any,
    messages: list[BaseMessage],
    response: AIMessage,
) -> None:
    _print_step_header(step, f"调用大模型 (tool_choice={tool_choice!r})")
    print(">>> 发给 LLM 的 prompt（messages）:", flush=True)
    print(_format_messages_for_log(messages), flush=True)
    print("\n<<< LLM 返回:", flush=True)
    print(_format_ai_for_log(response), flush=True)


def build_user_message(instruction: str, image_path: Path | None) -> HumanMessage:
    """Build multimodal HumanMessage: optional image + user instruction."""
    text = (
        f"用户指令：{instruction}\n"
        "请根据图片（若有）理解目标，立刻调用工具执行，禁止向用户提问。"
    )
    if image_path is None:
        return HumanMessage(content=text)
    from agent_client.image_encoder import encode_image_file

    image_url = encode_image_file(image_path)
    return HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": text},
        ]
    )


def _invoke_tool_calls(
    step: int,
    ai_msg: AIMessage,
    tool_map: dict[str, BaseTool],
) -> list[ToolMessage]:
    out: list[ToolMessage] = []
    for call in ai_msg.tool_calls or []:
        name = call["name"]
        args = call.get("args") or {}
        call_id = call.get("id") or name
        _print_step_header(step, f"执行工具 {name}")
        print(f"args = {json.dumps(args, ensure_ascii=False)}", flush=True)
        tool = tool_map.get(name)
        if tool is None:
            result = f"error: unknown tool {name!r}"
        else:
            try:
                result = str(tool.invoke(args))
            except Exception as exc:  # noqa: BLE001
                result = f"error: {exc}"
        print(f"result =\n{result}", flush=True)
        out.append(ToolMessage(content=result, tool_call_id=call_id))
    return out


def _call_model(
    llm: BaseChatModel,
    tools: list[BaseTool],
    messages: list[BaseMessage],
    *,
    force_tool: bool,
    step: int,
) -> AIMessage:
    """Call LLM; force_tool=True uses tool_choice=required (vLLM auto-tool-choice)."""
    choice: Any = "required" if force_tool else "auto"
    bound = llm.bind_tools(tools, tool_choice=choice)
    try:
        response = bound.invoke(messages)
        if not isinstance(response, AIMessage):
            response = AIMessage(content=_message_text(getattr(response, "content", response)))
        _print_llm_exchange(step, tool_choice=choice, messages=messages, response=response)
        return response
    except Exception:
        if not force_tool:
            raise
        print(
            f"\n[robot-agent] tool_choice=required 失败，改为 auto + nudge（步骤 {step}）",
            flush=True,
        )
        nudge = HumanMessage(
            content=(
                "不要向用户提问。立刻调用 describe_scene 或 list_objects，"
                "然后调用 grasp_object。禁止纯文本回复。"
            )
        )
        retry_messages = [*messages, nudge]
        bound = llm.bind_tools(tools, tool_choice="auto")
        response = bound.invoke(retry_messages)
        if not isinstance(response, AIMessage):
            response = AIMessage(content=_message_text(getattr(response, "content", response)))
        _print_llm_exchange(step, tool_choice="auto", messages=retry_messages, response=response)
        return response


def run_agent(
    runtime: RobotRuntime,
    instruction: str,
    *,
    llm: BaseChatModel | None = None,
    system_prompt: str | None = None,
    max_steps: int = MAX_STEPS,
) -> str:
    """ReAct-style loop: multimodal user turn + forced tools + step logging."""
    tools = build_tools(runtime)
    tool_map = {t.name: t for t in tools}
    model = llm or create_llm(runtime.vlm_cfg)
    prompt = (system_prompt or DEFAULT_SYSTEM).strip()

    if runtime.image_path is not None:
        print(f"[robot-agent] image={runtime.image_path}", flush=True)

    messages: list[BaseMessage] = [
        SystemMessage(content=prompt),
        build_user_message(instruction, runtime.image_path),
    ]

    saw_tool = False
    final_text = ""
    llm_step = 0
    flow_step = 0

    for step in range(max_steps):
        grasp_done = runtime.robot.holding is not None
        force = not saw_tool or (not grasp_done and step < max_steps - 1)
        llm_step += 1
        flow_step += 1
        ai_msg = _call_model(
            model,
            tools,
            messages,
            force_tool=force and not grasp_done,
            step=flow_step,
        )
        messages.append(ai_msg)

        if ai_msg.tool_calls:
            saw_tool = True
            flow_step += 1
            messages.extend(_invoke_tool_calls(flow_step, ai_msg, tool_map))
            continue

        final_text = _message_text(ai_msg.content).strip()
        if not saw_tool:
            print("\n[robot-agent] 本轮无 tool_calls，追加 nudge 后重试", flush=True)
            messages.append(
                HumanMessage(
                    content=(
                        "你没有调用任何工具。立即调用 describe_scene 或 list_objects，"
                        f"再 grasp_object(instruction={instruction!r})。"
                    )
                )
            )
            continue

        break

    if not saw_tool:
        raise RuntimeError(
            "模型未调用任何工具。请确认 vLLM 已用 --enable-auto-tool-choice "
            "--tool-call-parser hermes 启动，或改用 --scripted。"
        )

    if not final_text:
        flow_step += 1
        summary_messages = [
            *messages,
            HumanMessage(content="根据工具结果与图片，用一两句中文总结执行结果。"),
        ]
        summary_llm = model.bind_tools(tools, tool_choice="none")
        summary = summary_llm.invoke(summary_messages)
        if not isinstance(summary, AIMessage):
            summary = AIMessage(content=_message_text(getattr(summary, "content", summary)))
        _print_llm_exchange(
            flow_step,
            tool_choice="none",
            messages=summary_messages,
            response=summary,
        )
        final_text = _message_text(summary.content).strip()

    if not final_text and runtime.robot.holding is not None:
        final_text = f"已完成：夹持 {runtime.robot.holding.name}。"
    elif not final_text:
        final_text = "工具已执行，但模型未给出文字总结。"

    _print_step_header(flow_step + 1, "最终输出")
    print(final_text, flush=True)
    return final_text


def build_agent(
    runtime: RobotRuntime,
    *,
    llm: BaseChatModel | None = None,
    system_prompt: str | None = None,
):
    """Backward-compatible helper: returns a callable."""

    def _invoke(instruction: str) -> str:
        return run_agent(runtime, instruction, llm=llm, system_prompt=system_prompt)

    return _invoke


def run_scripted(runtime: RobotRuntime, instruction: str) -> str:
    """Deterministic demo without LLM: list → describe → grasp."""
    tools = {t.name: t for t in build_tools(runtime)}
    lines = [
        "[scripted] list_objects →",
        str(tools["list_objects"].invoke({})),
        "[scripted] describe_scene →",
        str(tools["describe_scene"].invoke({})),
        f"[scripted] grasp_object({instruction!r}) →",
        str(tools["grasp_object"].invoke({"instruction": instruction})),
    ]
    return "\n".join(lines)
