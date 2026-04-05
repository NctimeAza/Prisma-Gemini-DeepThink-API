"""Provider-level LLM client dispatcher.

Business layers import this module and stay provider-agnostic.
Provider is now resolved per-call based on the ``provider`` argument,
which maps to a registered ProviderConfig (gemini / openai / custom).
"""

import logging
from typing import Any, AsyncGenerator, Optional

from config import LLM_PROVIDER, get_provider_config

logger = logging.getLogger(__name__)

# 延迟 import，避免循环依赖
import clients.gemini_client as _gemini
import clients.openai_client as _openai


def _resolve_type(provider: str) -> str:
    """将 provider 名称解析为底层 API 类型 ("gemini" 或 "openai")."""
    cfg = get_provider_config(provider)
    return cfg.type


async def generate_json(
    model: str,
    contents: str | list[Any],
    system_instruction: Optional[str],
    response_schema: dict[str, Any],
    thinking_budget: int,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    image_parts: list[dict] | None = None,
    debug_info: dict[str, Any] | None = None,
    *,
    provider: str = "",
    json_via_prompt: bool = False,
) -> dict[str, Any]:
    """调用 LLM 生成结构化 JSON 响应（自动分发到对应 provider）.

    Args:
        model: 模型标识符.
        contents: 用户消息内容.
        system_instruction: 系统指令.
        response_schema: JSON schema 约束.
        thinking_budget: thinking token 预算.
        temperature: 温度参数.
        image_parts: 图片列表.
        provider: provider 标识符，空则使用全局默认.

    Returns:
        解析后的 JSON dict.
    """
    p = provider or LLM_PROVIDER
    api_type = _resolve_type(p)
    if api_type == "gemini":
        return await _gemini.generate_json(
            model, contents, system_instruction,
            response_schema, thinking_budget,
            temperature=temperature,
            top_p=top_p,
            image_parts=image_parts,
            debug_info=debug_info,
            provider=p,
            json_via_prompt=json_via_prompt,
        )
    return await _openai.generate_json(
        model, contents, system_instruction,
        response_schema, thinking_budget,
        temperature=temperature,
        top_p=top_p,
        image_parts=image_parts,
        debug_info=debug_info,
        provider=p,
        json_via_prompt=json_via_prompt,
    )


async def generate_content(
    model: str,
    contents: str | list[Any],
    system_instruction: Optional[str] = None,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    thinking_budget: int = 0,
    image_parts: list[dict] | None = None,
    *,
    provider: str = "",
) -> tuple[str, str, list[dict]]:
    """非流式调用 LLM，返回 (text, thoughts, grounding_chunks) 元组.

    Args:
        model: 模型标识符.
        contents: 用户消息内容.
        system_instruction: 系统指令.
        temperature: 温度参数.
        thinking_budget: thinking token 预算.
        image_parts: 图片列表.
        provider: provider 标识符，空则使用全局默认.

    Returns:
        (text, thoughts, grounding_chunks) 元组.
    """
    p = provider or LLM_PROVIDER
    api_type = _resolve_type(p)
    if api_type == "gemini":
        return await _gemini.generate_content(
            model, contents,
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            thinking_budget=thinking_budget,
            image_parts=image_parts,
            provider=p,
        )
    return await _openai.generate_content(
        model, contents,
        system_instruction=system_instruction,
        temperature=temperature,
        top_p=top_p,
        thinking_budget=thinking_budget,
        image_parts=image_parts,
        provider=p,
    )


async def generate_content_stream(
    model: str,
    contents: str | list[Any],
    system_instruction: Optional[str] = None,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    thinking_budget: int = 0,
    image_parts: list[dict] | None = None,
    *,
    provider: str = "",
) -> AsyncGenerator[tuple[str, str, list[dict]], None]:
    """流式调用 LLM，yield (text_chunk, thought_chunk, grounding_chunks) 元组.

    Args:
        model: 模型标识符.
        contents: 用户消息内容.
        system_instruction: 系统指令.
        temperature: 温度参数.
        thinking_budget: thinking token 预算.
        image_parts: 图片列表.
        provider: provider 标识符，空则使用全局默认.

    Yields:
        (text_chunk, thought_chunk, grounding_chunks) 元组.
    """
    p = provider or LLM_PROVIDER
    api_type = _resolve_type(p)
    if api_type == "gemini":
        async for chunk in _gemini.generate_content_stream(
            model, contents,
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            thinking_budget=thinking_budget,
            image_parts=image_parts,
            provider=p,
        ):
            yield chunk
    else:
        async for chunk in _openai.generate_content_stream(
            model, contents,
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            thinking_budget=thinking_budget,
            image_parts=image_parts,
            provider=p,
        ):
            yield chunk


logger.info("[LLM] Default provider: %s", LLM_PROVIDER)

__all__ = [
    "generate_json",
    "generate_content",
    "generate_content_stream",
]
