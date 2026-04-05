"""Gemini-native API routes.

Provides generateContent and streamGenerateContent endpoints that accept
and return data in Gemini's native format, while internally routing through
the same DeepThink pipeline.

Gemini native format key differences from OpenAI:
  - Thinking content goes in parts with ``thought: true``
  - Grounding metadata is returned in ``groundingMetadata``
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import (
    CHECKPOINT_REPLAY_CHUNK_SIZE,
    CHECKPOINT_SCHEMA_VERSION,
    ENABLE_RECURSIVE_LOOP,
    LLM_PROVIDER,
    MAX_CONTEXT_MESSAGES,
    SSE_HEARTBEAT_INTERVAL,
    StageProviders,
    resolve_model,
    resolve_refinement_config,
    split_forced_model_suffix,
)
from engine.checkpoint_store import CheckpointStore, CheckpointStoreError
from engine.orchestrator import SYNTHESIS_FALLBACK_TEXT, run_deep_think
from models import (
    ChatCompletionRequest,
    ChatMessageContent,
    DeepThinkCheckpoint,
    DeepThinkConfig,
)
from prompts import REFINEMENT_FALLBACK_TEXT, RESUME_HINT_TEXT

logger = logging.getLogger(__name__)
router = APIRouter()

_CONTINUE_COMMAND = "!deepthink_continue"
_CONTINUE_ALIASES: tuple[str, ...] = (_CONTINUE_COMMAND, "/continue")
_CONTINUE_RE = re.compile(
    r"^(?:!deepthink_continue|/continue)\s+([A-Za-z0-9_-]+)\s*$"
)
_ACTIVE_RESUME_IDS: set[str] = set()
_ACTIVE_RESUME_LOCK = asyncio.Lock()


def _resume_hint(resume_id: str) -> str:
    """Build resume hint text identical to the OAI route."""
    return (
        f"[resume_id] {resume_id}\n"
        f"{RESUME_HINT_TEXT.format(command=_CONTINUE_COMMAND, resume_id=resume_id)}\n"
    )


def _parse_continue_command(query: str) -> tuple[bool, str | None, str | None]:
    """解析 Gemini 请求中的 continue 命令。"""
    text = (query or "").strip()
    if not text.startswith(_CONTINUE_ALIASES):
        return False, None, None

    match = _CONTINUE_RE.fullmatch(text)
    if not match:
        return True, None, f"invalid continue command, use: {_CONTINUE_COMMAND} <id>"
    return True, match.group(1), None


def _find_previous_user_query(history: list[dict[str, str]]) -> str:
    """从历史消息中回溯 continue 之前的最后一条 user 提问。"""
    for item in reversed(history):
        if item.get("role") == "user":
            return item.get("content", "")
    return ""


async def _acquire_resume_id(resume_id: str) -> bool:
    async with _ACTIVE_RESUME_LOCK:
        if resume_id in _ACTIVE_RESUME_IDS:
            return False
        _ACTIVE_RESUME_IDS.add(resume_id)
        return True


async def _release_resume_id(resume_id: str) -> None:
    async with _ACTIVE_RESUME_LOCK:
        _ACTIVE_RESUME_IDS.discard(resume_id)


def _iter_chunks(text: str) -> list[str]:
    """按配置切分回放内容，避免单包过大。"""
    if not text:
        return []
    size = max(64, CHECKPOINT_REPLAY_CHUNK_SIZE)
    return [text[i : i + size] for i in range(0, len(text), size)]


def _is_fallback_error_text(text: str) -> bool:
    stripped = (text or "").strip()
    return stripped == SYNTHESIS_FALLBACK_TEXT or stripped == REFINEMENT_FALLBACK_TEXT


def _is_completed_with_empty_output(checkpoint: DeepThinkCheckpoint) -> bool:
    """判断综合阶段是否出现“已完成但无正文输出”的异常检查点。"""
    return (
        checkpoint.status == "completed"
        and checkpoint.phase == "synthesis"
        and not (checkpoint.output_content or "").strip()
    )


def _repair_legacy_completed_checkpoint(checkpoint: DeepThinkCheckpoint) -> bool:
    """修复旧版错误状态：避免 continue 被误判为仅回放。"""
    is_legacy_fallback_case = (
        checkpoint.status == "completed"
        and checkpoint.phase == "synthesis"
        and _is_fallback_error_text(checkpoint.output_content)
    )
    if not is_legacy_fallback_case and not _is_completed_with_empty_output(checkpoint):
        return False

    checkpoint.status = "error"
    checkpoint.completed_at = None
    checkpoint.output_content = ""
    return True


def _resolve_request_config(
    model_id: str,
) -> tuple[str, str, str, DeepThinkConfig, str, StageProviders]:
    base_model_id, forced_prefill_suffix = split_forced_model_suffix(model_id)
    (
        real_model, mgr_model, syn_model,
        p_level, e_level, s_level,
        model_max_rounds, provider,
        planning_temp, expert_temp, review_temp, synthesis_temp,
        mode,
        json_via_prompt,
        stage_providers,
    ) = resolve_model(model_id)

    refinement_kwargs: dict[str, Any] = {}
    if mode == "refinement":
        ref_cfg = resolve_refinement_config(
            base_model_id, real_model, mgr_model, syn_model,
        )
        refinement_kwargs = {
            "refinement_max_rounds": ref_cfg.refinement_max_rounds,
            "pre_draft_review_rounds": ref_cfg.pre_draft_review_rounds,
            "enable_json_repair": ref_cfg.enable_json_repair,
            "enable_text_cleaner": ref_cfg.enable_text_cleaner,
            "draft_model": ref_cfg.draft_model,
            "review_model": ref_cfg.review_model,
            "merge_model": ref_cfg.merge_model,
            "json_repair_model": ref_cfg.json_repair_model,
        }

    config = DeepThinkConfig(
        mode=mode,
        planning_level=p_level,
        expert_level=e_level,
        synthesis_level=s_level,
        enable_recursive_loop=ENABLE_RECURSIVE_LOOP,
        max_rounds=model_max_rounds,
        max_context_messages=MAX_CONTEXT_MESSAGES,
        planning_temperature=planning_temp,
        expert_temperature=expert_temp,
        review_temperature=review_temp,
        synthesis_temperature=synthesis_temp,
        json_via_prompt=json_via_prompt,
        forced_prefill_suffix=forced_prefill_suffix,
        **refinement_kwargs,
    )
    return real_model, mgr_model, syn_model, config, provider, stage_providers


# ---------------------------------------------------------------------------
# Request parsing helpers
# ---------------------------------------------------------------------------

def _parse_gemini_request(body: dict[str, Any]) -> tuple[
    str, str, list[dict[str, str]], list[dict], str | None, float | None, float | None, bool
]:
    """Parse a Gemini generateContent request body.

    Returns:
        (model, query, history, image_parts, system_instruction, temperature, top_p,
         include_thoughts)
    """
    model = body.get("model", "")
    contents = body.get("contents", [])
    gen_config = body.get("generationConfig", {})
    temperature = gen_config.get("temperature")
    top_p = gen_config.get("topP")
    if top_p is None:
        top_p = gen_config.get("top_p")

    # 提取 thinkingConfig.includeThoughts
    # 没有 thinkingConfig -> 下游不关心思维链，默认 False
    # 有 thinkingConfig 但没指定 includeThoughts -> 默认 True
    thinking_config = gen_config.get("thinkingConfig")
    if thinking_config is not None:
        include_thoughts = thinking_config.get("includeThoughts", True)
    else:
        include_thoughts = False

    # system_instruction
    sys_instr = body.get("systemInstruction")
    system_text: str | None = None
    if isinstance(sys_instr, dict):
        parts = sys_instr.get("parts", [])
        system_text = "\n".join(
            p.get("text", "") for p in parts if isinstance(p, dict)
        )
    elif isinstance(sys_instr, str):
        system_text = sys_instr

    # Extract history and query from contents
    history: list[dict[str, str]] = []
    query = ""
    image_parts: list[dict] = []

    for item in contents:
        role = item.get("role", "user")
        parts = item.get("parts", [])

        texts: list[str] = []
        for part in parts:
            if isinstance(part, dict):
                if "text" in part:
                    texts.append(part["text"])
                elif "inlineData" in part:
                    inline = part["inlineData"]
                    image_parts.append({
                        "inline_data": {
                            "mime_type": inline.get("mimeType", "image/png"),
                            "data": inline.get("data", ""),
                        }
                    })

        combined_text = "\n".join(texts)
        mapped_role = "assistant" if role == "model" else "user"

        if mapped_role in ("user", "assistant"):
            history.append({"role": mapped_role, "content": combined_text})

    # Last user message is the query
    if history and history[-1]["role"] == "user":
        query = history[-1]["content"]
        history = history[:-1]

    return (
        model,
        query,
        history,
        image_parts,
        system_text,
        temperature,
        top_p,
        include_thoughts,
    )


def _build_gemini_response(
    *,
    model: str,
    text: str,
    reasoning: str,
    grounding_chunks: list[dict],
) -> dict[str, Any]:
    """Build a Gemini-native generateContent response."""
    parts: list[dict[str, Any]] = []

    # Thought parts first
    if reasoning:
        parts.append({"text": reasoning, "thought": True})

    # Text part
    if text:
        parts.append({"text": text})

    candidate: dict[str, Any] = {
        "content": {
            "role": "model",
            "parts": parts,
        },
        "finishReason": "STOP",
    }

    if grounding_chunks:
        candidate["groundingMetadata"] = {
            "groundingChunks": [
                {"web": chunk} for chunk in grounding_chunks
            ]
        }

    return {
        "candidates": [candidate],
    }


def _build_gemini_stream_chunk(
    *,
    text: str = "",
    thought: str = "",
    grounding_chunks: list[dict] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Build a single Gemini streaming response chunk."""
    parts: list[dict[str, Any]] = []

    if thought:
        parts.append({"text": thought, "thought": True})
    if text:
        parts.append({"text": text})

    candidate: dict[str, Any] = {
        "content": {
            "role": "model",
            "parts": parts,
        },
    }

    if finish_reason:
        candidate["finishReason"] = finish_reason

    if grounding_chunks:
        candidate["groundingMetadata"] = {
            "groundingChunks": [
                {"web": chunk} for chunk in grounding_chunks
            ]
        }

    return {"candidates": [candidate]}


def _dedup_grounding(chunks: list[dict]) -> list[dict]:
    """Deduplicate grounding chunks by URI."""
    seen: set[str] = set()
    result: list[dict] = []
    for item in chunks:
        uri = item.get("uri", "")
        if uri and uri in seen:
            continue
        if uri:
            seen.add(uri)
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Route: list / get models
# ---------------------------------------------------------------------------

@router.get("/v1beta/models")
async def list_models():
    """List available virtual models (Gemini native format)."""
    from config import VIRTUAL_MODELS

    return {
        "models": [
            {
                "name": f"models/{vm.id}",
                "displayName": vm.id,
                "description": vm.desc,
                "supportedGenerationMethods": [
                    "generateContent",
                    "streamGenerateContent",
                ],
            }
            for vm in VIRTUAL_MODELS
        ],
    }


@router.get("/v1beta/models/{model_name}")
async def get_model(model_name: str):
    """Get a single virtual model by name (Gemini native format)."""
    from config import VIRTUAL_MODELS

    for vm in VIRTUAL_MODELS:
        if vm.id == model_name:
            return {
                "name": f"models/{vm.id}",
                "displayName": vm.id,
                "description": vm.desc,
                "supportedGenerationMethods": [
                    "generateContent",
                    "streamGenerateContent",
                ],
            }
    return JSONResponse(
        status_code=404,
        content={"error": {"message": f"model not found: {model_name}", "code": 404}},
    )


# ---------------------------------------------------------------------------
# Route: streamGenerateContent
# ---------------------------------------------------------------------------

async def _gemini_sse_stream(
    body: dict[str, Any],
) -> AsyncGenerator[str, None]:
    """Generate Gemini-native SSE stream."""
    model_id, query, history, image_parts, system_text, temperature, top_p, include_thoughts = (
        _parse_gemini_request(body)
    )
    continue_mode, resume_id, continue_error = _parse_continue_command(query)
    if continue_error:
        error_data = {"error": {"message": continue_error, "code": 400}}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        return
    if continue_mode:
        resumed_query = _find_previous_user_query(history).strip()
        if not resumed_query:
            error_data = {
                "error": {
                    "message": f"missing user query after {_CONTINUE_COMMAND} command",
                    "code": 400,
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            return
        # 继续模式下，命中的上一条 user 消息应作为 query，不再保留在 history 中。
        for idx in range(len(history) - 1, -1, -1):
            if history[idx].get("role") == "user":
                del history[idx]
                break
        query = resumed_query

    if not query:
        error_data = {"error": {"message": "empty query", "code": 400}}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        return

    (
        real_model,
        mgr_model,
        syn_model,
        config,
        provider,
        stage_providers,
    ) = _resolve_request_config(model_id)

    checkpoint_store = CheckpointStore()
    now = int(time.time())
    replay_only = False
    locked_resume_id: str | None = None

    if continue_mode:
        if not resume_id:
            error_data = {
                "error": {
                    "message": f"missing resume id for {_CONTINUE_COMMAND} command",
                    "code": 400,
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            return
        try:
            checkpoint = checkpoint_store.load(resume_id)
        except FileNotFoundError:
            error_data = {
                "error": {"message": f"checkpoint not found: {resume_id}", "code": 404}
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            return
        except CheckpointStoreError as exc:
            error_data = {"error": {"message": str(exc), "code": 400}}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            return

        if _repair_legacy_completed_checkpoint(checkpoint):
            logger.warning(
                "[Checkpoint] repaired completed->error state for %s",
                checkpoint.resume_id,
            )

        replay_only = checkpoint.status == "completed"

        if checkpoint.pipeline_mode != config.mode:
            error_data = {
                "error": {
                    "message": (
                        "pipeline mode mismatch: checkpoint was created with "
                        f"mode='{checkpoint.pipeline_mode}' but current model uses "
                        f"mode='{config.mode}'. Cannot resume across different modes."
                    ),
                    "code": 400,
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            return

        checkpoint.request_model = model_id
        checkpoint.real_model = real_model
        checkpoint.manager_model = mgr_model
        checkpoint.synthesis_model = syn_model
        checkpoint.schema_version = CHECKPOINT_SCHEMA_VERSION
        checkpoint.updated_at = now
        if not replay_only:
            checkpoint.status = "running"
            checkpoint.error_message = ""
        checkpoint_store.save(checkpoint)

        acquired = await _acquire_resume_id(checkpoint.resume_id)
        if not acquired:
            error_data = {
                "error": {
                    "message": (
                        "resume id already has an active run, wait for completion "
                        "or disconnect before retrying"
                    ),
                    "code": 409,
                }
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            return
        locked_resume_id = checkpoint.resume_id
    else:
        new_resume_id = f"res_{uuid.uuid4().hex[:16]}"
        checkpoint = checkpoint_store.create(
            new_resume_id, schema_version=CHECKPOINT_SCHEMA_VERSION,
        )
        checkpoint.request_model = model_id
        checkpoint.real_model = real_model
        checkpoint.manager_model = mgr_model
        checkpoint.synthesis_model = syn_model
        checkpoint.phase = "planning"
        checkpoint.status = "running"
        checkpoint.current_round = 1
        checkpoint.reasoning_content = ""
        checkpoint.output_content = ""
        checkpoint.error_message = ""
        checkpoint.started_at = now
        checkpoint.updated_at = now
        checkpoint.completed_at = None
        checkpoint.pipeline_mode = config.mode
        checkpoint_store.save(checkpoint)

    async def _persist_event(_: str, __: dict) -> None:
        try:
            checkpoint.updated_at = int(time.time())
            checkpoint_store.save(checkpoint)
        except Exception:
            logger.exception(
                "[Checkpoint] failed to persist %s", checkpoint.resume_id
            )

    all_grounding: list[dict] = []

    try:
        # resume hint as the first thought chunk (only if thoughts requested)
        if include_thoughts:
            hint_chunk = _build_gemini_stream_chunk(
                thought=_resume_hint(checkpoint.resume_id),
            )
            yield f"data: {json.dumps(hint_chunk, ensure_ascii=False)}\n\n"

        if continue_mode:
            if include_thoughts:
                for thought_part in _iter_chunks(checkpoint.reasoning_content):
                    replay_chunk = _build_gemini_stream_chunk(thought=thought_part)
                    yield f"data: {json.dumps(replay_chunk, ensure_ascii=False)}\n\n"

            for text_part in _iter_chunks(checkpoint.output_content):
                replay_chunk = _build_gemini_stream_chunk(text=text_part)
                yield f"data: {json.dumps(replay_chunk, ensure_ascii=False)}\n\n"

            if replay_only:
                final_data = _build_gemini_stream_chunk(finish_reason="STOP")
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
                return

        async for text_chunk, thought_chunk, _phase, grounding in run_deep_think(
            query=query,
            history=history,
            image_parts=image_parts,
            model=real_model,
            manager_model=mgr_model,
            synthesis_model=syn_model,
            config=config,
            temperature=temperature,
            top_p=top_p if top_p is not None else config.top_p,
            system_prompt=system_text or "",
            resume_checkpoint=checkpoint,
            event_callback=_persist_event,
            resume_mode=continue_mode,
            stage_providers=stage_providers,
        ):
            if grounding:
                all_grounding.extend(grounding)

            # 如果不需要思维链，过滤掉 thought 内容
            effective_thought = thought_chunk if include_thoughts else ""

            if text_chunk or effective_thought:
                chunk_data = _build_gemini_stream_chunk(
                    text=text_chunk,
                    thought=effective_thought,
                )
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        # Final chunk with finish reason and grounding
        final_data = _build_gemini_stream_chunk(
            finish_reason="STOP",
            grounding_chunks=_dedup_grounding(all_grounding) if all_grounding else None,
        )
        yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

    except Exception as exc:
        logger.exception("[Gemini route] streaming failed")
        error_data = {"error": {"message": str(exc), "code": 500}}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
    finally:
        if locked_resume_id:
            await _release_resume_id(locked_resume_id)


@router.post("/v1beta/models/{model_name}:streamGenerateContent")
async def stream_generate_content(model_name: str, raw_request: Request):
    """Gemini-native streaming endpoint."""
    raw_body = await raw_request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "invalid JSON"})

    # Model from URL path takes precedence
    body["model"] = model_name

    logger.debug(
        "[Gemini API] streamGenerateContent request\n%s",
        json.dumps(body, ensure_ascii=False, indent=2)[:5000],
    )

    return StreamingResponse(
        _gemini_sse_stream(body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Route: generateContent (non-streaming)
# ---------------------------------------------------------------------------

@router.post("/v1beta/models/{model_name}:generateContent")
async def generate_content(model_name: str, raw_request: Request):
    """Gemini-native non-streaming endpoint."""
    raw_body = await raw_request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "invalid JSON"})

    body["model"] = model_name

    logger.debug(
        "[Gemini API] generateContent request\n%s",
        json.dumps(body, ensure_ascii=False, indent=2)[:5000],
    )

    model_id, query, history, image_parts, system_text, temperature, top_p, include_thoughts = (
        _parse_gemini_request(body)
    )
    continue_mode, resume_id, continue_error = _parse_continue_command(query)
    if continue_error:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": continue_error, "code": 400}},
        )
    if continue_mode:
        resumed_query = _find_previous_user_query(history).strip()
        if not resumed_query:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": (
                            f"missing user query after {_CONTINUE_COMMAND} command"
                        ),
                        "code": 400,
                    }
                },
            )
        # 继续模式下，命中的上一条 user 消息应作为 query，不再保留在 history 中。
        for idx in range(len(history) - 1, -1, -1):
            if history[idx].get("role") == "user":
                del history[idx]
                break
        query = resumed_query

    if not query:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "empty query", "code": 400}},
        )

    (
        real_model,
        mgr_model,
        syn_model,
        config,
        provider,
        stage_providers,
    ) = _resolve_request_config(model_id)

    checkpoint_store = CheckpointStore()
    now = int(time.time())
    replay_only = False
    locked_resume_id: str | None = None

    if continue_mode:
        if not resume_id:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"missing resume id for {_CONTINUE_COMMAND} command",
                        "code": 400,
                    }
                },
            )
        try:
            checkpoint = checkpoint_store.load(resume_id)
        except FileNotFoundError:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"checkpoint not found: {resume_id}", "code": 404}},
            )
        except CheckpointStoreError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "code": 400}},
            )

        if _repair_legacy_completed_checkpoint(checkpoint):
            logger.warning(
                "[Checkpoint] repaired completed->error state for %s",
                checkpoint.resume_id,
            )

        replay_only = checkpoint.status == "completed"
        if checkpoint.pipeline_mode != config.mode:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": (
                            "pipeline mode mismatch: checkpoint was created with "
                            f"mode='{checkpoint.pipeline_mode}' but current model uses "
                            f"mode='{config.mode}'. Cannot resume across different modes."
                        ),
                        "code": 400,
                    }
                },
            )

        checkpoint.request_model = model_id
        checkpoint.real_model = real_model
        checkpoint.manager_model = mgr_model
        checkpoint.synthesis_model = syn_model
        checkpoint.schema_version = CHECKPOINT_SCHEMA_VERSION
        checkpoint.updated_at = now
        if not replay_only:
            checkpoint.status = "running"
            checkpoint.error_message = ""
        checkpoint_store.save(checkpoint)

        acquired = await _acquire_resume_id(checkpoint.resume_id)
        if not acquired:
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "message": (
                            "resume id already has an active run, wait for completion "
                            "or disconnect before retrying"
                        ),
                        "code": 409,
                    }
                },
            )
        locked_resume_id = checkpoint.resume_id
    else:
        new_resume_id = f"res_{uuid.uuid4().hex[:16]}"
        checkpoint = checkpoint_store.create(
            new_resume_id, schema_version=CHECKPOINT_SCHEMA_VERSION,
        )
        checkpoint.request_model = model_id
        checkpoint.real_model = real_model
        checkpoint.manager_model = mgr_model
        checkpoint.synthesis_model = syn_model
        checkpoint.phase = "planning"
        checkpoint.status = "running"
        checkpoint.current_round = 1
        checkpoint.reasoning_content = ""
        checkpoint.output_content = ""
        checkpoint.error_message = ""
        checkpoint.started_at = now
        checkpoint.updated_at = now
        checkpoint.completed_at = None
        checkpoint.pipeline_mode = config.mode
        checkpoint_store.save(checkpoint)

    async def _persist_event(_: str, __: dict) -> None:
        try:
            checkpoint.updated_at = int(time.time())
            checkpoint_store.save(checkpoint)
        except Exception:
            logger.exception(
                "[Checkpoint] failed to persist %s", checkpoint.resume_id
            )

    try:
        full_text = checkpoint.output_content if continue_mode else ""
        full_reasoning = ""
        if include_thoughts:
            full_reasoning = _resume_hint(checkpoint.resume_id)
            if continue_mode:
                full_reasoning += checkpoint.reasoning_content
        all_grounding: list[dict] = []

        if not replay_only:
            async for text_chunk, thought_chunk, _phase, grounding in run_deep_think(
                query=query,
                history=history,
                image_parts=image_parts,
                model=real_model,
                manager_model=mgr_model,
                synthesis_model=syn_model,
                config=config,
                temperature=temperature,
                top_p=top_p if top_p is not None else config.top_p,
                system_prompt=system_text or "",
                resume_checkpoint=checkpoint,
                event_callback=_persist_event,
                resume_mode=continue_mode,
                stage_providers=stage_providers,
            ):
                full_text += text_chunk
                if include_thoughts:
                    full_reasoning += thought_chunk
                if grounding:
                    all_grounding.extend(grounding)

        return _build_gemini_response(
            model=model_id,
            text=full_text,
            reasoning=full_reasoning,
            grounding_chunks=_dedup_grounding(all_grounding),
        )
    finally:
        if locked_resume_id:
            await _release_resume_id(locked_resume_id)
