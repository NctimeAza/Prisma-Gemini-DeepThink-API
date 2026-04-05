"""Synthesis 综合模块.

迁移自原项目 synthesis.ts。
汇总所有 Expert 结果，流式生成最终综合回复。
"""

import logging
from typing import AsyncGenerator

from clients.llm_client import generate_content_stream
from models import ExpertResult, ReviewResult
from prompts import build_prefill_contents, get_synthesis_prompt

logger = logging.getLogger(__name__)


async def stream_synthesis(
    model: str,
    query: str,
    history_context: str,
    expert_results: list[ExpertResult],
    review_results: list[ReviewResult],
    budget: int,
    temperature: float | None = None,
    top_p: float | None = None,
    user_system_prompt: str = "",
    image_parts: list[dict] | None = None,
    provider: str = "",
) -> AsyncGenerator[tuple[str, str, list[dict]], None]:
    """流式执行综合阶段.

    Args:
        model: 模型标识符.
        query: 用户原始问题.
        history_context: 最近对话上下文.
        expert_results: 所有 Expert 的执行结果.
        budget: thinking token 预算.
        temperature: 温度参数（可选）.
        user_system_prompt: 下游客户端的 system prompt.
        image_parts: Gemini inlineData 格式的图片列表.

    Yields:
        (text_chunk, thought_chunk, grounding_chunks) 元组.
    """
    prompt = get_synthesis_prompt(
        history_context, query, expert_results, review_results,
        user_system_prompt=user_system_prompt,
    )
    contents = build_prefill_contents(prompt, image_parts=image_parts)
    logger.info("[Synthesis] Starting synthesis (%d experts)", len(expert_results))

    kwargs: dict = {
        "model": model,
        "contents": contents,
        "thinking_budget": budget,
        "image_parts": None,
        "top_p": top_p,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    async for text_chunk, thought_chunk, grounding_chunks in generate_content_stream(
        **kwargs, provider=provider,
    ):
        yield text_chunk, thought_chunk, grounding_chunks

    logger.info("[Synthesis] Synthesis completed")
