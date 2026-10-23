"""LangChain LCEL chains for Image + Text → VLM."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from agent_client.image_encoder import encode_image_file
from agent_client.prompts import build_scene_prompt
from agent_client.vlm_client import create_vlm_llm


def _to_vision_message(payload: dict[str, Any]) -> list[HumanMessage]:
    image_url = payload["image_url"]
    text = payload.get("text") or build_scene_prompt()
    return [
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text},
            ]
        )
    ]


def build_vision_chain(llm: ChatOpenAI):
    """Return a runnable: {image_url, text?} -> str response."""
    return RunnableLambda(_to_vision_message) | llm | RunnableLambda(lambda m: m.content)


def analyze_image(
    *,
    image_path: str,
    user_text: str | None = None,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    temperature: float = 0.2,
) -> str:
    llm = create_vlm_llm(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
    )
    chain = build_vision_chain(llm)
    image_url = encode_image_file(image_path)
    prompt = build_scene_prompt(user_text)
    return chain.invoke({"image_url": image_url, "text": prompt})
