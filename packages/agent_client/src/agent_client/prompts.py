"""Prompt templates for embodied VLM scene understanding."""

from __future__ import annotations

DEFAULT_SCENE_PROMPT = (
    "你是一个具身智能场景理解助手。"
    "请根据输入图片完成：\n"
    "1. 简要描述可见场景与关键物体；\n"
    "2. 列出可交互物体及可能的操作（affordance）；\n"
    "3. 给出 1–3 条对机器人安全、可执行的下一步建议。\n"
    "用简洁中文回答，分点列出。"
)


def build_scene_prompt(user_text: str | None = None) -> str:
    if user_text and user_text.strip():
        return f"{DEFAULT_SCENE_PROMPT}\n\n用户补充指令：{user_text.strip()}"
    return DEFAULT_SCENE_PROMPT
