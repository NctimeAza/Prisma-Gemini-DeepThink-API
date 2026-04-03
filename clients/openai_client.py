"""OpenAI-compatible LLM client wrapper.

Supports official OpenAI and OpenAI-compatible gateways via base_url.
Supports multiple provider instances via the ``provider`` argument.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
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
    STREAM_CHUNK_TIMEOUT,
    get_provider_config,
)
from utils.retry import with_retry

logger = logging.getLogger(__name__)

_clients: dict[str, AsyncOpenAI] = {}
_request_lock: Optional[asyncio.Lock] = None


def get_client(provider: str = "") -> AsyncOpenAI:
    """Get or create per-provider AsyncOpenAI client."""
    p = provider or LLM_PROVIDER
    if p not in _clients:
        cfg = get_provider_config(p)
        kwargs: dict[str, Any] = {"api_key": cfg.api_key}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
            logger.info("[OpenAI] Provider %s using base URL: %s", p, cfg.base_url)
        _clients[p] = AsyncOpenAI(**kwargs)
    return _clients[p]


async def _random_delay() -> None:
    """Optional queued random delay before each request."""
    global _request_lock
    if _request_lock is None:
        _request_lock = asyncio.Lock()

    if LLM_REQUEST_DELAY_MAX > 0:
        async with _request_lock:
            delay = random.uniform(LLM_REQUEST_DELAY_MIN, LLM_REQUEST_DELAY_MAX)
            if delay > 0:
                logger.debug(
                    "[OpenAI] Acquired request lock, queued delay %.3fs (throttling guard)",
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
        # Try common text-like fields first.
        for field in ("text", "content", "value"):
            v = value.get(field)
            if isinstance(v, str):
                return v
            if isinstance(v, list):
                joined = "".join(_coerce_text(item) for item in v)
                if joined:
                    return joined
        # OpenAI reasoning shape may include summary list.
        summary = value.get("summary")
        if isinstance(summary, list):
            joined = "".join(_coerce_text(item) for item in summary)
            if joined:
                return joined
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _parse_content_parts(content: Any) -> tuple[str, str]:
    text_chunks: list[str] = []
    reasoning_chunks: list[str] = []

    if isinstance(content, str):
        return content, ""

    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                text_chunks.append(part)
                continue

            if not isinstance(part, dict):
                continue

            part_type = str(part.get("type", "")).strip().lower()
            if part_type in {"text", "output_text"}:
                text_chunks.append(_coerce_text(part.get("text") or part.get("value")))
                continue
            if part_type in {
                "reasoning",
                "reasoning_text",
                "output_reasoning",
                "thinking",
            }:
                reasoning_chunks.append(
                    _coerce_text(
                        part.get("text")
                        or part.get("summary")
                        or part.get("value")
                        or part.get("content")
                    )
                )
                continue

            if "text" in part:
                text_chunks.append(_coerce_text(part.get("text")))
        return "".join(text_chunks), "".join(reasoning_chunks)

    if isinstance(content, dict):
        return (
            _coerce_text(content.get("text") or content.get("content")),
            _coerce_text(content.get("reasoning") or content.get("reasoning_content")),
        )

    return _coerce_text(content), ""


def _extract_message_text_and_reasoning(message: Any) -> tuple[str, str]:
    text, reasoning = _parse_content_parts(_get_attr(message, "content"))

    extra_reasoning = _coerce_text(_get_attr(message, "reasoning_content"))
    if extra_reasoning:
        reasoning += extra_reasoning

    extra_reasoning = _coerce_text(_get_attr(message, "reasoning"))
    if extra_reasoning:
        reasoning += extra_reasoning

    extra_reasoning = _coerce_text(_get_attr(message, "thinking"))
    if extra_reasoning:
        reasoning += extra_reasoning

    return text, reasoning


def _extract_delta_text_and_reasoning(delta: Any) -> tuple[str, str]:
    text, reasoning = _parse_content_parts(_get_attr(delta, "content"))

    extra_reasoning = _coerce_text(_get_attr(delta, "reasoning_content"))
    if extra_reasoning:
        reasoning += extra_reasoning

    extra_reasoning = _coerce_text(_get_attr(delta, "reasoning"))
    if extra_reasoning:
        reasoning += extra_reasoning

    extra_reasoning = _coerce_text(_get_attr(delta, "thinking"))
    if extra_reasoning:
        reasoning += extra_reasoning

    return text, reasoning


def _gemini_inline_data_to_image_part(inline_data: dict[str, Any]) -> dict[str, Any] | None:
    b64 = inline_data.get("data")
    if not b64:
        return None
    mime = inline_data.get("mime_type") or "image/png"
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def _gemini_parts_to_openai_content(parts: list[Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []

    for part in parts:
        if isinstance(part, str):
            content.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue

        text = part.get("text")
        if isinstance(text, str) and text:
            content.append({"type": "text", "text": text})

        inline_data = part.get("inline_data")
        if isinstance(inline_data, dict):
            image_part = _gemini_inline_data_to_image_part(inline_data)
            if image_part:
                content.append(image_part)
            continue

        # Pass through already OpenAI-style image parts if present.
        if part.get("type") == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict) and image_url.get("url"):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url["url"]},
                    }
                )

    return content


def _normalize_messages(
    contents: str | list[Any],
    image_parts: list[dict] | None = None,
    system_instruction: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    if isinstance(contents, str):
        if image_parts:
            user_content: list[dict[str, Any]] = [{"type": "text", "text": contents}]
            user_content.extend(_gemini_parts_to_openai_content(image_parts))
            messages.append({"role": "user", "content": user_content})
        else:
            # Keep plain-string content for better compatibility with older gateways.
            messages.append({"role": "user", "content": contents})
        return messages

    if not isinstance(contents, list):
        messages.append({"role": "user", "content": _coerce_text(contents)})
        return messages

    for item in contents:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue

        if not isinstance(item, dict):
            messages.append({"role": "user", "content": _coerce_text(item)})
            continue

        role = str(item.get("role", "user")).strip().lower()
        if role == "model":
            role = "assistant"
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"

        if isinstance(item.get("parts"), list):
            parsed_content = _gemini_parts_to_openai_content(item["parts"])
            if len(parsed_content) == 1 and parsed_content[0].get("type") == "text":
                messages.append({"role": role, "content": parsed_content[0]["text"]})
            elif parsed_content:
                messages.append({"role": role, "content": parsed_content})
            else:
                messages.append({"role": role, "content": ""})
            continue

        if "content" in item:
            messages.append({"role": role, "content": item["content"]})
            continue

        messages.append({"role": role, "content": _coerce_text(item.get("text"))})

    if image_parts:
        converted_images = _gemini_parts_to_openai_content(image_parts)
        if converted_images:
            for msg in reversed(messages):
                if msg.get("role") != "user":
                    continue
                existing = msg.get("content")
                if isinstance(existing, str):
                    merged: list[dict[str, Any]] = []
                    if existing:
                        merged.append({"type": "text", "text": existing})
                    merged.extend(converted_images)
                    msg["content"] = merged
                elif isinstance(existing, list):
                    msg["content"] = existing + converted_images
                else:
                    msg["content"] = converted_images
                break
            else:
                messages.append({"role": "user", "content": converted_images})

    return messages


def _lower_schema_types(value: Any) -> Any:
    if isinstance(value, list):
        return [_lower_schema_types(v) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.lower()
            else:
                out[k] = _lower_schema_types(v)
        return out
    return value


def _build_json_prompt_guard(schema: dict[str, Any]) -> str:
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return (
        "JSON output contract:\n"
        "1) Output exactly one JSON object and nothing else.\n"
        "2) Do not use markdown/code fences/explanations.\n"
        "3) The JSON must conform to this schema exactly:\n"
        f"{schema_text}"
    )


def _inject_json_prompt_guard(
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    guard = _build_json_prompt_guard(schema)
    out: list[dict[str, Any]] = [dict(msg) for msg in messages]

    for msg in out:
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = f"{content}\n\n{guard}"
        elif isinstance(content, list):
            msg["content"] = [*content, {"type": "text", "text": guard}]
        else:
            msg["content"] = guard
        return out

    out.insert(0, {"role": "system", "content": guard})
    return out


def _extract_response_text(response: Any) -> tuple[str, str]:
    choices = _get_attr(response, "choices", []) or []
    if not choices:
        return "", ""
    message = _get_attr(choices[0], "message")
    return _extract_message_text_and_reasoning(message)


def _chat_create_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: Optional[float],
    top_p: Optional[float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    effective_top_p = DEFAULT_TOP_P if top_p is None else top_p
    kwargs["top_p"] = effective_top_p
    if extra:
        kwargs.update(extra)
    return kwargs


async def generate_json(
    model: str,
    contents: str | list[Any],
    system_instruction: str,
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
    """Generate structured JSON with strict-first fallback policy."""
    await _random_delay()
    client = get_client(provider)

    if thinking_budget > 0:
        logger.debug(
            "[OpenAI] thinking_budget=%d requested but not enforced in OAI mode",
            thinking_budget,
        )

    messages = _normalize_messages(
        contents,
        image_parts=image_parts,
        system_instruction=system_instruction,
    )
    normalized_schema = _lower_schema_types(response_schema)
    if json_via_prompt:
        logger.info(
            "[OpenAI] json_via_prompt enabled for model=%s (provider=%s)",
            model,
            provider or LLM_PROVIDER,
        )
        messages = _inject_json_prompt_guard(messages, normalized_schema)

    strict_kwargs = _chat_create_kwargs(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        extra={
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "prisma_structured_output",
                    "schema": normalized_schema,
                    "strict": True,
                },
            }
        },
    )

    json_object_kwargs = _chat_create_kwargs(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        extra={"response_format": {"type": "json_object"}},
    )

    fallback_messages = list(messages)
    fallback_hint = "Return only one valid JSON object. Do not use markdown code fences."
    if json_via_prompt:
        fallback_hint = _build_json_prompt_guard(normalized_schema)
    fallback_messages.append(
        {
            "role": "user",
            "content": fallback_hint,
        }
    )
    text_fallback_kwargs = _chat_create_kwargs(
        model=model,
        messages=fallback_messages,
        temperature=temperature,
        top_p=top_p,
    )

    async def _strict_call():
        return await client.chat.completions.create(**strict_kwargs)

    async def _json_object_call():
        return await client.chat.completions.create(**json_object_kwargs)

    async def _text_fallback_call():
        return await client.chat.completions.create(**text_fallback_kwargs)

    response = None
    try:
        response = await with_retry(
            _strict_call,
            timeout=LLM_REQUEST_TIMEOUT,
            timeout_retries=LLM_TIMEOUT_RETRIES,
            network_retries=LLM_NETWORK_RETRIES,
        )
    except Exception as strict_exc:
        logger.warning(
            "[OpenAI] json_schema response_format unavailable, trying json_object fallback: %s",
            strict_exc,
        )
        try:
            response = await with_retry(
                _json_object_call,
                timeout=LLM_REQUEST_TIMEOUT,
                timeout_retries=LLM_TIMEOUT_RETRIES,
                network_retries=LLM_NETWORK_RETRIES,
            )
        except Exception as json_object_exc:
            logger.warning(
                "[OpenAI] json_object response_format unavailable, using text fallback: %s",
                json_object_exc,
            )
            response = await with_retry(
                _text_fallback_call,
                timeout=LLM_REQUEST_TIMEOUT,
                timeout_retries=LLM_TIMEOUT_RETRIES,
                network_retries=LLM_NETWORK_RETRIES,
            )

    raw_text, _ = _extract_response_text(response)
    logger.debug("[OpenAI] generate_json raw response (model=%s):\n%s", model, raw_text)
    cleaned = _clean_json_string(raw_text or "{}")
    if cleaned != raw_text:
        logger.debug("[OpenAI] cleaned JSON:\n%s", cleaned)
    if debug_info is not None:
        debug_info["client"] = "openai"
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
    """Non-streaming generation, returns (text, thoughts, grounding_chunks)."""
    await _random_delay()
    client = get_client(provider)

    if thinking_budget > 0:
        logger.debug(
            "[OpenAI] thinking_budget=%d requested but not enforced in OAI mode",
            thinking_budget,
        )

    messages = _normalize_messages(
        contents,
        image_parts=image_parts,
        system_instruction=system_instruction,
    )
    kwargs = _chat_create_kwargs(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
    )

    async def _call():
        return await client.chat.completions.create(**kwargs)

    response = await with_retry(
        _call,
        timeout=LLM_REQUEST_TIMEOUT,
        timeout_retries=LLM_TIMEOUT_RETRIES,
        network_retries=LLM_NETWORK_RETRIES,
    )

    text, thoughts = _extract_response_text(response)
    logger.debug(
        "[OpenAI] generate_content complete (model=%s): %d chars text, %d chars thoughts",
        model,
        len(text),
        len(thoughts),
    )
    return text, thoughts, []


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
    """Streaming generation, yields (text_chunk, thought_chunk, grounding_chunks)."""
    await _random_delay()
    client = get_client(provider)

    if thinking_budget > 0:
        logger.debug(
            "[OpenAI] thinking_budget=%d requested but not enforced in OAI mode",
            thinking_budget,
        )

    messages = _normalize_messages(
        contents,
        image_parts=image_parts,
        system_instruction=system_instruction,
    )
    kwargs = _chat_create_kwargs(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        extra={"stream": True},
    )

    async def _call():
        return await client.chat.completions.create(**kwargs)

    stream = await with_retry(
        _call,
        timeout=LLM_REQUEST_TIMEOUT,
        timeout_retries=LLM_TIMEOUT_RETRIES,
        network_retries=LLM_NETWORK_RETRIES,
    )

    chunk_timeout = STREAM_CHUNK_TIMEOUT if STREAM_CHUNK_TIMEOUT > 0 else None
    aiter = stream.__aiter__()
    while True:
        try:
            if chunk_timeout:
                chunk = await asyncio.wait_for(aiter.__anext__(), timeout=chunk_timeout)
            else:
                chunk = await aiter.__anext__()
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            logger.error(
                "[OpenAI] stream single-chunk timeout (%.0fs), upstream may be disconnected",
                chunk_timeout,
            )
            raise

        choices = _get_attr(chunk, "choices", []) or []
        if not choices:
            continue

        delta = _get_attr(choices[0], "delta")
        text_chunk, thought_chunk = _extract_delta_text_and_reasoning(delta)

        if text_chunk or thought_chunk:
            yield text_chunk, thought_chunk, []


def _clean_json_string(s: str) -> str:
    """Clean a JSON string that may be wrapped in markdown fences."""
    if not s:
        return "{}"

    md_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s)
    if md_match:
        return md_match.group(1).strip()

    first_open = s.find("{")
    last_close = s.rfind("}")
    if first_open != -1 and last_close != -1 and last_close > first_open:
        return s[first_open : last_close + 1]

    return s.strip() if s.strip().startswith("{") else "{}"
