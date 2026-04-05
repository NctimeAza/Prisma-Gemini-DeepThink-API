"""Gemini AI 客户端封装.

封装 google-genai SDK，提供同步/流式调用接口。
支持多 provider 实例（通过 provider 参数指定不同的 api_key / base_url）。
"""

import asyncio
import json
import logging
import random
from typing import Any, AsyncGenerator, Optional

from google import genai
from google.genai import types

from config import (
    DEFAULT_TOP_P,
    LLM_PROVIDER,
    LLM_REQUEST_DELAY_MIN,
    LLM_REQUEST_DELAY_MAX,
    LLM_REQUEST_TIMEOUT,
    LLM_TIMEOUT_RETRIES,
    LLM_NETWORK_RETRIES,
    STREAM_CHUNK_TIMEOUT,
    get_provider_config,
)
from utils.retry import with_retry

logger = logging.getLogger(__name__)

# 每个 provider 对应一个 client 实例
_clients: dict[str, genai.Client] = {}
_request_lock: Optional[asyncio.Lock] = None


def get_client(provider: str = "") -> genai.Client:
    """获取或创建指定 provider 的 Gemini 客户端.

    Args:
        provider: provider 名称，空则使用全局默认.

    Returns:
        genai.Client 实例.
    """
    p = provider or LLM_PROVIDER
    if p not in _clients:
        cfg = get_provider_config(p)
        client_kwargs: dict[str, Any] = {"api_key": cfg.api_key}
        if cfg.base_url:
            client_kwargs["http_options"] = types.HttpOptions(
                base_url=cfg.base_url
            )
            logger.info("[Gemini] Provider %s using relay URL: %s", p, cfg.base_url)
        _clients[p] = genai.Client(**client_kwargs)
    return _clients[p]


def _build_contents(
    text: str,
    image_parts: list[dict] | None = None,
) -> Any:
    """将文本和图片组合为 Gemini contents 结构.

    当存在图片时，构建 {"role": "user", "parts": [...]} 结构;
    否则返回纯文本字符串。

    Args:
        text: 文本内容.
        image_parts: Gemini inlineData 格式的图片列表.

    Returns:
        纯文本字符串或 Gemini Content 字典.
    """
    if not image_parts:
        return text
    parts: list[dict] = [{"text": text}]
    parts.extend(image_parts)
    return {"role": "user", "parts": parts}


async def _random_delay() -> None:
    """请求 LLM 前如果配置了随机延迟，则排队进行随机休眠（防风控）."""
    global _request_lock
    if _request_lock is None:
        _request_lock = asyncio.Lock()

    if LLM_REQUEST_DELAY_MAX > 0:
        async with _request_lock:
            delay = random.uniform(LLM_REQUEST_DELAY_MIN, LLM_REQUEST_DELAY_MAX)
            if delay > 0:
                logger.debug(
                    "[Gemini] Acquired request lock, queued delay %.3fs (throttling guard)",
                    delay,
                )
                await asyncio.sleep(delay)


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
    """调用 Gemini 生成结构化 JSON 响应.

    Args:
        model: 模型标识符.
        contents: 用户消息内容.
        system_instruction: 系统指令.
        response_schema: JSON schema 约束.
        thinking_budget: thinking token 预算.
        temperature: 温度参数（可选）.
        image_parts: Gemini inlineData 格式的图片列表（可选）.

    Returns:
        解析后的 JSON dict.
    """
    await _random_delay()
    client = get_client(provider)

    # 将图片注入 contents
    if isinstance(contents, str):
        contents = _build_contents(contents, image_parts)

    config_dict: dict[str, Any] = {
        "response_mime_type": "application/json",
        "response_schema": response_schema,
        "tools": [types.Tool(google_search=types.GoogleSearch())],
    }
    if system_instruction:
        config_dict["system_instruction"] = system_instruction
    if temperature is not None:
        config_dict["temperature"] = temperature
    config_dict["top_p"] = DEFAULT_TOP_P if top_p is None else top_p
    if thinking_budget > 0:
        config_dict["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=True,
        )

    async def _call():
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_dict),
        )
        return response

    response = await with_retry(
        _call,
        timeout=LLM_REQUEST_TIMEOUT,
        timeout_retries=LLM_TIMEOUT_RETRIES,
        network_retries=LLM_NETWORK_RETRIES,
    )
    raw_text = response.text or "{}"
    logger.debug(
        "[Gemini] generate_json raw response (model=%s):\n%s", model, raw_text
    )

    # 尝试清洗 JSON
    cleaned = _clean_json_string(raw_text)
    if cleaned != raw_text:
        logger.debug("[Gemini] cleaned JSON:\n%s", cleaned)
    if debug_info is not None:
        debug_info["client"] = "gemini"
        debug_info["model"] = model
        debug_info["provider"] = provider or LLM_PROVIDER
        debug_info["raw_text"] = raw_text or ""
        debug_info["cleaned_text"] = cleaned or ""
    return json.loads(cleaned)


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
    """非流式调用 Gemini，返回完整的 (text, thoughts, grounding_chunks) 元组.

    与流式调用相比，with_retry 完整覆盖整个请求生命周期，
    不存在流迭代阶段的超时/断连问题，适用于 Expert 等不需要实时输出的场景。

    Args:
        model: 模型标识符.
        contents: 用户消息内容.
        system_instruction: 系统指令（可选）.
        temperature: 温度参数.
        thinking_budget: thinking token 预算.
        image_parts: Gemini inlineData 格式的图片列表（可选）.

    Returns:
        (text, thoughts, grounding_chunks) 元组.
        grounding_chunks 为 [{"title": ..., "uri": ...}, ...] 列表.
    """
    await _random_delay()
    client = get_client(provider)

    # 将图片注入 contents
    if isinstance(contents, str):
        contents = _build_contents(contents, image_parts)

    config_dict: dict[str, Any] = {
        "temperature": temperature,
        "top_p": DEFAULT_TOP_P if top_p is None else top_p,
        "tools": [types.Tool(google_search=types.GoogleSearch())],
    }
    if system_instruction:
        config_dict["system_instruction"] = system_instruction
    if thinking_budget > 0:
        config_dict["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=True,
        )

    async def _call():
        return await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_dict),
        )

    response = await with_retry(
        _call,
        timeout=LLM_REQUEST_TIMEOUT,
        timeout_retries=LLM_TIMEOUT_RETRIES,
        network_retries=LLM_NETWORK_RETRIES,
    )

    text = ""
    thoughts = ""
    if response.candidates and response.candidates[0].content:
        for part in (response.candidates[0].content.parts or []):
            if part.thought:
                thoughts += part.text or ""
            elif part.text:
                text += part.text

    grounding = _extract_grounding_chunks(response.candidates[0] if response.candidates else None)

    logger.debug(
        "[Gemini] generate_content complete (model=%s): %d chars text, %d chars thoughts, %d grounding",
        model, len(text), len(thoughts), len(grounding),
    )
    return text, thoughts, grounding


def _extract_grounding_chunks(candidate: Any) -> list[dict]:
    """从 Gemini Candidate 中提取 grounding_chunks.

    Returns:
        [{"title": ..., "uri": ...}, ...] 列表，无数据则为空.
    """
    if candidate is None:
        return []
    meta = getattr(candidate, "grounding_metadata", None)
    if meta is None:
        return []
    raw_chunks = getattr(meta, "grounding_chunks", None) or []
    results: list[dict] = []
    for gc in raw_chunks:
        web = getattr(gc, "web", None)
        if web is None:
            continue
        entry: dict[str, str] = {}
        uri = getattr(web, "uri", None)
        title = getattr(web, "title", None)
        if uri:
            entry["uri"] = uri
        if title:
            entry["title"] = title
        if entry:
            results.append(entry)
    return results


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
    """流式调用 Gemini，yield (text, thought, grounding_chunks) 元组.

    Args:
        model: 模型标识符.
        contents: 用户消息内容.
        system_instruction: 系统指令（可选）.
        temperature: 温度参数.
        thinking_budget: thinking token 预算.
        image_parts: Gemini inlineData 格式的图片列表（可选）.

    Yields:
        (text_chunk, thought_chunk, grounding_chunks) 元组.
        grounding_chunks 仅在流末尾的 chunk 中可能非空.
    """
    await _random_delay()
    client = get_client(provider)

    # 将图片注入 contents
    if isinstance(contents, str):
        contents = _build_contents(contents, image_parts)

    config_dict: dict[str, Any] = {
        "temperature": temperature,
        "top_p": DEFAULT_TOP_P if top_p is None else top_p,
        "tools": [types.Tool(google_search=types.GoogleSearch())],
    }
    if system_instruction:
        config_dict["system_instruction"] = system_instruction
    if thinking_budget > 0:
        config_dict["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=True,
        )

    async def _call():
        return await client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_dict),
        )

    stream = await with_retry(
        _call,
        timeout=LLM_REQUEST_TIMEOUT,
        timeout_retries=LLM_TIMEOUT_RETRIES,
        network_retries=LLM_NETWORK_RETRIES,
    )

    # 手动迭代 + 单 chunk 超时保护，防止上游流 hang 住导致连接死结
    chunk_timeout = STREAM_CHUNK_TIMEOUT if STREAM_CHUNK_TIMEOUT > 0 else None
    aiter = stream.__aiter__()
    while True:
        try:
            if chunk_timeout:
                chunk = await asyncio.wait_for(
                    aiter.__anext__(), timeout=chunk_timeout
                )
            else:
                chunk = await aiter.__anext__()
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            logger.error(
                "[Gemini] stream single-chunk timeout (%.0fs), upstream may be disconnected",
                chunk_timeout,
            )
            raise

        text_chunk = ""
        thought_chunk = ""

        if chunk.candidates and chunk.candidates[0].content:
            for part in (chunk.candidates[0].content.parts or []):
                if part.thought:
                    thought_chunk += part.text or ""
                elif part.text:
                    text_chunk += part.text

        grounding = _extract_grounding_chunks(
            chunk.candidates[0] if chunk.candidates else None
        )

        if text_chunk or thought_chunk or grounding:
            yield text_chunk, thought_chunk, grounding


def _clean_json_string(s: str) -> str:
    """清洗可能被 Markdown 包裹的 JSON 字符串.

    迁移自原项目 utils.ts/cleanJsonString。

    Args:
        s: 原始字符串.

    Returns:
        清洗后的 JSON 字符串.
    """
    if not s:
        return "{}"

    import re

    # 尝试提取 markdown code block 中的 JSON
    md_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s)
    if md_match:
        return md_match.group(1).strip()

    # 提取第一个 { 到最后一个 }
    first_open = s.find("{")
    last_close = s.rfind("}")
    if first_open != -1 and last_close != -1 and last_close > first_open:
        return s[first_open : last_close + 1]

    return s.strip() if s.strip().startswith("{") else "{}"
