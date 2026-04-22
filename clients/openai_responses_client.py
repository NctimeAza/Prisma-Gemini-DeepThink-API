"""OpenAI Responses API client wrapper.

默认全阶段开启 web_search，适配 GPT-5 联网搜索。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, AsyncGenerator, Optional

from openai import AsyncOpenAI

from config import (
    DEFAULT_TOP_P,
    LLM_NETWORK_RETRIES,
    LLM_PROVIDER,
    LLM_REQUEST_DELAY_MAX,
    LLM_REQUEST_DELAY_MIN,
    LLM_REQUEST_TIMEOUT,
    LLM_TIMEOUT_RETRIES,
    OPENAI_RESPONSES_USE_STREAM_FOR_NON_STREAM,
    STREAM_CHUNK_TIMEOUT,
    get_provider_config,
)
from utils.retry import extract_status, is_retryable_error, with_retry

import clients.openai_client as _chat_client

logger = logging.getLogger(__name__)

_clients: dict[str, AsyncOpenAI] = {}
_request_lock: Optional[asyncio.Lock] = None
_WEB_SEARCH_TOOL = [{"type": "web_search"}]


def get_client(provider: str = "") -> AsyncOpenAI:
    """获取或创建指定 provider 的 Responses 客户端。"""
    p = provider or LLM_PROVIDER
    if p not in _clients:
        cfg = get_provider_config(p)
        kwargs: dict[str, Any] = {"api_key": cfg.api_key}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
            logger.info(
                "[OpenAI Responses] Provider %s using base URL: %s",
                p,
                cfg.base_url,
            )
        _clients[p] = AsyncOpenAI(**kwargs)
    return _clients[p]


async def _random_delay() -> None:
    """请求前执行可选随机延迟。"""
    global _request_lock
    if _request_lock is None:
        _request_lock = asyncio.Lock()

    if LLM_REQUEST_DELAY_MAX > 0:
        async with _request_lock:
            delay = random.uniform(LLM_REQUEST_DELAY_MIN, LLM_REQUEST_DELAY_MAX)
            if delay > 0:
                logger.debug(
                    "[OpenAI Responses] Acquired request lock, queued delay %.3fs",
                    delay,
                )
                await asyncio.sleep(delay)


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_coerce_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalize_to_response_input(
    contents: str | list[Any],
    image_parts: list[dict] | None = None,
    system_instruction: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """将现有消息格式转成 Responses API input。"""
    messages = _chat_client._normalize_messages(  # noqa: SLF001
        contents,
        image_parts=image_parts,
        system_instruction=system_instruction,
    )

    instructions_parts: list[str] = []
    response_input: list[dict[str, Any]] = []

    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower()
        content = msg.get("content")

        if role == "system":
            instructions_parts.append(_flatten_message_text(content))
            continue

        item: dict[str, Any] = {
            "type": "message",
            "role": role if role in {"user", "assistant", "developer"} else "user",
        }
        content_text_type = (
            "output_text" if item["role"] == "assistant" else "input_text"
        )

        if isinstance(content, str):
            item["content"] = (
                [{"type": content_text_type, "text": content}]
                if content else ""
            )
            response_input.append(item)
            continue

        content_parts: list[dict[str, Any]] = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in {"text", "input_text"}:
                    text = _coerce_text(part.get("text"))
                    if text:
                        content_parts.append(
                            {"type": content_text_type, "text": text}
                        )
                    continue
                if part_type == "output_text":
                    text = _coerce_text(part.get("text"))
                    if text:
                        content_parts.append(
                            {"type": content_text_type, "text": text}
                        )
                    continue
                if part_type in {"image_url", "input_image"}:
                    if item["role"] == "assistant":
                        # Assistant 历史消息不支持 input_image，直接跳过。
                        continue
                    image_url = part.get("image_url")
                    url = ""
                    if isinstance(image_url, dict):
                        url = str(image_url.get("url", "")).strip()
                    elif isinstance(image_url, str):
                        url = image_url.strip()
                    if url:
                        content_parts.append(
                            {
                                "type": "input_image",
                                "image_url": url,
                                "detail": "auto",
                            }
                        )

        item["content"] = content_parts if content_parts else ""
        response_input.append(item)

    instructions = "\n\n".join(part for part in instructions_parts if part).strip()
    return response_input, instructions or None


def _flatten_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"text", "input_text"}:
                text = _coerce_text(item.get("text"))
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return _coerce_text(content)


def _build_create_kwargs(
    *,
    model: str,
    response_input: list[dict[str, Any]],
    instructions: str | None,
    temperature: Optional[float],
    top_p: Optional[float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": response_input,
        "tools": list(_WEB_SEARCH_TOOL),
        "tool_choice": "auto",
        "store": False,
        "include": [
            "web_search_call.action.sources",
            "reasoning.encrypted_content",
        ],
        "text": {"verbosity": "high"},
    }
    if instructions:
        kwargs["instructions"] = instructions
    if temperature is not None:
        kwargs["temperature"] = temperature
    kwargs["top_p"] = DEFAULT_TOP_P if top_p is None else top_p
    if extra:
        kwargs.update(extra)
    return kwargs


def _extract_text_from_content_item(content_item: Any) -> str:
    """从单个 content item 中尽量提取文本。"""
    content_type = _get_attr(content_item, "type", "")
    if content_type in {"output_text", "text", "input_text", "reasoning_text"}:
        return _coerce_text(_get_attr(content_item, "text", ""))
    if isinstance(content_item, str):
        return content_item
    if isinstance(content_item, dict):
        text = content_item.get("text", "")
        if isinstance(text, str):
            return text
    return ""


def _extract_reasoning_text(item: Any) -> str:
    """从 reasoning item 中提取 reasoning 文本。"""
    parts: list[str] = []
    for content_item in _get_attr(item, "content", []) or []:
        if _get_attr(content_item, "type", "") == "reasoning_text":
            parts.append(_coerce_text(_get_attr(content_item, "text", "")))
    for summary_item in _get_attr(item, "summary", []) or []:
        if _get_attr(summary_item, "type", "") in {"summary_text", "reasoning_text"}:
            parts.append(_coerce_text(_get_attr(summary_item, "text", "")))
    return "".join(parts)


def _iter_possible_output_items(response: Any) -> list[Any]:
    """返回可能含正文的输出项列表，兼容不完全规范的中转。"""
    output = _get_attr(response, "output", None)
    if isinstance(output, list):
        return output
    if output is not None:
        return [output]

    content = _get_attr(response, "content", None)
    if isinstance(content, list):
        return [{"type": "message", "content": content}]
    if content is not None:
        return [{"type": "message", "content": [content]}]

    choices = _get_attr(response, "choices", []) or []
    if choices:
        message = _get_attr(choices[0], "message")
        if message is not None:
            msg_content = _get_attr(message, "content", None)
            if isinstance(msg_content, str):
                return [{"type": "message", "content": [{"type": "output_text", "text": msg_content}]}]
            if msg_content is not None:
                return [{"type": "message", "content": msg_content}]
    return []


def _extract_grounding_from_annotations(annotations: list[Any]) -> list[dict[str, str]]:
    grounding: list[dict[str, str]] = []
    for annotation in annotations:
        annotation_type = _get_attr(annotation, "type", "")
        if annotation_type != "url_citation":
            continue
        uri = _get_attr(annotation, "url", "")
        title = _get_attr(annotation, "title", "")
        entry: dict[str, str] = {}
        if title:
            entry["title"] = title
        if uri:
            entry["uri"] = uri
        if entry:
            grounding.append(entry)
    return grounding


def _dedup_grounding(chunks: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for chunk in chunks:
        key = (chunk.get("title", ""), chunk.get("uri", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def _extract_response_payload(response: Any) -> tuple[str, str, list[dict[str, str]]]:
    """从 Responses 最终结果中提取正文、推理和引用。"""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    grounding_chunks: list[dict[str, str]] = []

    output_text = _get_attr(response, "output_text", "")

    for item in _iter_possible_output_items(response):
        item_type = _get_attr(item, "type", "")
        if item_type == "message":
            content_list = _get_attr(item, "content", []) or []
            if isinstance(content_list, str):
                if content_list:
                    text_parts.append(content_list)
                continue
            for content_item in content_list:
                text = _extract_text_from_content_item(content_item)
                if text:
                    text_parts.append(text)
                annotations = _get_attr(content_item, "annotations", []) or []
                grounding_chunks.extend(
                    _extract_grounding_from_annotations(annotations)
                )
            continue

        if item_type == "output_text":
            text = _extract_text_from_content_item(item)
            if text:
                text_parts.append(text)
            continue

        if item_type == "reasoning":
            reasoning_parts.append(_extract_reasoning_text(item))

    text = "".join(text_parts)
    if not text and output_text:
        text = _coerce_text(output_text)
    if not text:
        text = _coerce_text(_get_attr(response, "text", ""))
    reasoning = "".join(reasoning_parts)
    return text, reasoning, _dedup_grounding(grounding_chunks)


def _append_delta(existing: str, incoming: str) -> tuple[str, str]:
    """合并流式增量，避免 done/completed 重复正文。"""
    if not incoming:
        return existing, ""
    if not existing:
        return incoming, incoming
    if incoming.startswith(existing):
        delta = incoming[len(existing):]
        return incoming, delta
    return existing + incoming, incoming


def _handle_stream_event(
    event: Any,
    state: dict[str, Any],
) -> list[tuple[str, str, list[dict[str, str]]]]:
    """解析单个 Responses 流事件，返回标准 chunk 列表。"""
    chunks: list[tuple[str, str, list[dict[str, str]]]] = []
    event_type = _get_attr(event, "type", "")

    if event_type == "response.output_text.delta":
        delta = _coerce_text(_get_attr(event, "delta", ""))
        if delta:
            state["text"], emitted = _append_delta(state["text"], delta)
            if emitted:
                chunks.append((emitted, "", []))
        return chunks

    if event_type == "response.output_text.done":
        text = _coerce_text(_get_attr(event, "text", ""))
        state["text"], emitted = _append_delta(state["text"], text)
        if emitted:
            chunks.append((emitted, "", []))
        return chunks

    if event_type == "response.reasoning_text.delta":
        delta = _coerce_text(_get_attr(event, "delta", ""))
        if delta:
            state["reasoning"], emitted = _append_delta(state["reasoning"], delta)
            if emitted:
                chunks.append(("", emitted, []))
        return chunks

    if event_type == "response.reasoning_text.done":
        text = _coerce_text(_get_attr(event, "text", ""))
        state["reasoning"], emitted = _append_delta(state["reasoning"], text)
        if emitted:
            chunks.append(("", emitted, []))
        return chunks

    if event_type == "response.content_part.done":
        part = _get_attr(event, "part")
        part_type = _get_attr(part, "type", "")
        if part_type in {"output_text", "text"}:
            text = _extract_text_from_content_item(part)
            state["text"], emitted = _append_delta(state["text"], text)
            if emitted:
                chunks.append((emitted, "", []))
        elif part_type == "reasoning_text":
            text = _extract_text_from_content_item(part)
            state["reasoning"], emitted = _append_delta(state["reasoning"], text)
            if emitted:
                chunks.append(("", emitted, []))
        return chunks

    if event_type == "response.output_text.annotation.added":
        annotation = _get_attr(event, "annotation")
        raw_chunks = _extract_grounding_from_annotations(
            [annotation] if annotation else []
        )
        new_chunks: list[dict[str, str]] = []
        for chunk in raw_chunks:
            key = (chunk.get("title", ""), chunk.get("uri", ""))
            if key in state["grounding_seen"]:
                continue
            state["grounding_seen"].add(key)
            state["grounding"].append(chunk)
            new_chunks.append(chunk)
        if new_chunks:
            chunks.append(("", "", new_chunks))
        return chunks

    if event_type == "response.completed":
        response = _get_attr(event, "response")
        if response is None:
            return chunks
        full_text, full_reasoning, grounding = _extract_response_payload(response)
        state["text"], emitted_text = _append_delta(state["text"], full_text)
        if emitted_text:
            chunks.append((emitted_text, "", []))
        state["reasoning"], emitted_reasoning = _append_delta(
            state["reasoning"], full_reasoning
        )
        if emitted_reasoning:
            chunks.append(("", emitted_reasoning, []))

        new_chunks = []
        for chunk in grounding:
            key = (chunk.get("title", ""), chunk.get("uri", ""))
            if key in state["grounding_seen"]:
                continue
            state["grounding_seen"].add(key)
            state["grounding"].append(chunk)
            new_chunks.append(chunk)
        if new_chunks:
            chunks.append(("", "", new_chunks))
        return chunks

    logger.debug("[OpenAI Responses] ignored stream event type: %s", event_type)
    return chunks


async def _create_streaming_response(
    client: AsyncOpenAI,
    kwargs: dict[str, Any],
) -> Any:
    """创建 Responses 流。"""
    stream_kwargs = dict(kwargs)
    stream_kwargs["stream"] = True

    async def _call():
        return await client.responses.create(**stream_kwargs)

    return await with_retry(
        _call,
        timeout=LLM_REQUEST_TIMEOUT,
        timeout_retries=LLM_TIMEOUT_RETRIES,
        network_retries=LLM_NETWORK_RETRIES,
    )


async def _create_and_consume_stream_with_retry(
    client: AsyncOpenAI,
    kwargs: dict[str, Any],
) -> tuple[str, str, list[dict[str, str]]]:
    """创建流并消费到完成，chunk 超时也纳入整请求重试。"""

    async def _call() -> tuple[str, str, list[dict[str, str]]]:
        stream = await _create_streaming_response(client, kwargs)
        return await _consume_stream_to_completion(stream)

    return await with_retry(
        _call,
        timeout=LLM_REQUEST_TIMEOUT,
        timeout_retries=LLM_TIMEOUT_RETRIES,
        network_retries=LLM_NETWORK_RETRIES,
    )


async def _consume_stream_to_completion(
    stream: Any,
) -> tuple[str, str, list[dict[str, str]]]:
    """消费完整流并聚合为最终结果。"""
    chunk_timeout = STREAM_CHUNK_TIMEOUT if STREAM_CHUNK_TIMEOUT > 0 else None
    aiter = stream.__aiter__()
    state: dict[str, Any] = {
        "text": "",
        "reasoning": "",
        "grounding": [],
        "grounding_seen": set(),
    }

    while True:
        try:
            if chunk_timeout:
                event = await asyncio.wait_for(
                    aiter.__anext__(), timeout=chunk_timeout
                )
            else:
                event = await aiter.__anext__()
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            logger.error(
                "[OpenAI Responses] stream single-chunk timeout (%.0fs), upstream may be disconnected",
                chunk_timeout,
            )
            raise

        _handle_stream_event(event, state)

    return state["text"], state["reasoning"], list(state["grounding"])


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
    """使用 Responses API 生成结构化 JSON。"""
    await _random_delay()
    client = get_client(provider)

    if thinking_budget > 0:
        logger.debug(
            "[OpenAI Responses] thinking_budget=%d requested but not enforced",
            thinking_budget,
        )

    response_input, instructions = _normalize_to_response_input(
        contents,
        image_parts=image_parts,
        system_instruction=system_instruction,
    )
    normalized_schema = _chat_client._lower_schema_types(response_schema)  # noqa: SLF001
    if json_via_prompt:
        guard = _chat_client._build_json_prompt_guard(normalized_schema)  # noqa: SLF001
        instructions = f"{instructions}\n\n{guard}".strip() if instructions else guard

    structured_kwargs = _build_create_kwargs(
        model=model,
        response_input=response_input,
        instructions=instructions,
        temperature=temperature,
        top_p=top_p,
        extra={
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "prisma_structured_output",
                    "schema": normalized_schema,
                    "strict": True,
                }
            }
        },
    )

    fallback_input = list(response_input)
    fallback_input.append(
        {
            "type": "message",
            "role": "user",
            "content": (
                _chat_client._build_json_prompt_guard(normalized_schema)  # noqa: SLF001
                if json_via_prompt
                else "Return only one valid JSON object. Do not use markdown code fences."
            ),
        }
    )
    fallback_kwargs = _build_create_kwargs(
        model=model,
        response_input=fallback_input,
        instructions=instructions,
        temperature=temperature,
        top_p=top_p,
    )

    async def _structured_call():
        return await client.responses.create(**structured_kwargs)

    async def _fallback_call():
        return await client.responses.create(**fallback_kwargs)

    raw_text = ""
    reasoning = ""
    grounding: list[dict[str, str]] = []
    try:
        if OPENAI_RESPONSES_USE_STREAM_FOR_NON_STREAM:
            raw_text, reasoning, grounding = await _create_and_consume_stream_with_retry(
                client,
                structured_kwargs,
            )
        else:
            response = await with_retry(
                _structured_call,
                timeout=LLM_REQUEST_TIMEOUT,
                timeout_retries=LLM_TIMEOUT_RETRIES,
                network_retries=LLM_NETWORK_RETRIES,
            )
            raw_text, reasoning, grounding = _extract_response_payload(response)
    except Exception as structured_exc:
        logger.warning(
            "[OpenAI Responses] structured JSON unavailable, using text fallback: %s",
            structured_exc,
        )
        if OPENAI_RESPONSES_USE_STREAM_FOR_NON_STREAM:
            raw_text, reasoning, grounding = await _create_and_consume_stream_with_retry(
                client,
                fallback_kwargs,
            )
        else:
            response = await with_retry(
                _fallback_call,
                timeout=LLM_REQUEST_TIMEOUT,
                timeout_retries=LLM_TIMEOUT_RETRIES,
                network_retries=LLM_NETWORK_RETRIES,
            )
            raw_text, reasoning, grounding = _extract_response_payload(response)
    logger.debug(
        "[OpenAI Responses] generate_json raw response (model=%s):\n%s",
        model,
        raw_text,
    )
    cleaned = _chat_client._clean_json_string(raw_text or "{}")  # noqa: SLF001
    if debug_info is not None:
        debug_info["client"] = "openai_responses"
        debug_info["model"] = model
        debug_info["provider"] = provider or LLM_PROVIDER
        debug_info["raw_text"] = raw_text or ""
        debug_info["cleaned_text"] = cleaned or ""
        debug_info["reasoning_text"] = reasoning or ""
        debug_info["grounding_chunks"] = grounding
    return _chat_client._parse_json_lenient(cleaned)  # noqa: SLF001


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
    """非流式内容生成。"""
    await _random_delay()
    client = get_client(provider)

    if thinking_budget > 0:
        logger.debug(
            "[OpenAI Responses] thinking_budget=%d requested but not enforced",
            thinking_budget,
        )

    response_input, instructions = _normalize_to_response_input(
        contents,
        image_parts=image_parts,
        system_instruction=system_instruction,
    )
    kwargs = _build_create_kwargs(
        model=model,
        response_input=response_input,
        instructions=instructions,
        temperature=temperature,
        top_p=top_p,
    )

    async def _call():
        return await client.responses.create(**kwargs)

    if OPENAI_RESPONSES_USE_STREAM_FOR_NON_STREAM:
        text, thoughts, grounding = await _create_and_consume_stream_with_retry(
            client,
            kwargs,
        )
    else:
        response = await with_retry(
            _call,
            timeout=LLM_REQUEST_TIMEOUT,
            timeout_retries=LLM_TIMEOUT_RETRIES,
            network_retries=LLM_NETWORK_RETRIES,
        )
        text, thoughts, grounding = _extract_response_payload(response)
    logger.debug(
        "[OpenAI Responses] generate_content complete (model=%s): %d chars text, %d chars thoughts, %d grounding",
        model,
        len(text),
        len(thoughts),
        len(grounding),
    )
    return text, thoughts, grounding


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
    """流式内容生成。"""
    await _random_delay()
    client = get_client(provider)

    if thinking_budget > 0:
        logger.debug(
            "[OpenAI Responses] thinking_budget=%d requested but not enforced",
            thinking_budget,
        )

    response_input, instructions = _normalize_to_response_input(
        contents,
        image_parts=image_parts,
        system_instruction=system_instruction,
    )
    kwargs = _build_create_kwargs(
        model=model,
        response_input=response_input,
        instructions=instructions,
        temperature=temperature,
        top_p=top_p,
        extra={"stream": True},
    )

    async def _call():
        return await client.responses.create(**kwargs)

    attempt = 0
    while True:
        stream = await with_retry(
            _call,
            timeout=LLM_REQUEST_TIMEOUT,
            timeout_retries=LLM_TIMEOUT_RETRIES,
            network_retries=LLM_NETWORK_RETRIES,
        )

        chunk_timeout = STREAM_CHUNK_TIMEOUT if STREAM_CHUNK_TIMEOUT > 0 else None
        aiter = stream.__aiter__()
        state: dict[str, Any] = {
            "text": "",
            "reasoning": "",
            "grounding": [],
            "grounding_seen": set(),
        }
        yielded_any = False

        try:
            while True:
                try:
                    if chunk_timeout:
                        event = await asyncio.wait_for(
                            aiter.__anext__(), timeout=chunk_timeout
                        )
                    else:
                        event = await aiter.__anext__()
                except StopAsyncIteration:
                    return
                except asyncio.TimeoutError:
                    logger.error(
                        "[OpenAI Responses] stream single-chunk timeout (%.0fs), upstream may be disconnected",
                        chunk_timeout,
                    )
                    raise

                for text_chunk, thought_chunk, grounding_chunks in _handle_stream_event(
                    event, state
                ):
                    if text_chunk or thought_chunk or grounding_chunks:
                        yielded_any = True
                        yield text_chunk, thought_chunk, grounding_chunks
        except Exception as exc:
            status = extract_status(exc)
            retryable = isinstance(exc, asyncio.TimeoutError) or is_retryable_error(
                status
            )
            if yielded_any or not retryable or attempt >= LLM_NETWORK_RETRIES:
                raise

            attempt += 1
            delay = 1.5 * attempt
            logger.warning(
                "[OpenAI Responses] stream failed before first chunk, retry %d/%d in %.1fs: %s",
                attempt,
                LLM_NETWORK_RETRIES,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
