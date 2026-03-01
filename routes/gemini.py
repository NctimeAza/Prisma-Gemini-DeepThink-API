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
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import (
    CHECKPOINT_SCHEMA_VERSION,
    ENABLE_RECURSIVE_LOOP,
    LLM_PROVIDER,
    MAX_CONTEXT_MESSAGES,
    SSE_HEARTBEAT_INTERVAL,
    resolve_model,
    resolve_refinement_config,
)
from engine.checkpoint_store import CheckpointStore
from engine.orchestrator import run_deep_think
from models import (
    ChatCompletionRequest,
    ChatMessageContent,
    DeepThinkCheckpoint,
    DeepThinkConfig,
)
from prompts import RESUME_HINT_TEXT

logger = logging.getLogger(__name__)
router = APIRouter()

_CONTINUE_COMMAND = "!deepthink_continue"


def _resume_hint(resume_id: str) -> str:
    """Build resume hint text identical to the OAI route."""
    return (
        f"[resume_id] {resume_id}\n"
        f"{RESUME_HINT_TEXT.format(command=_CONTINUE_COMMAND, resume_id=resume_id)}\n"
    )


def _resolve_request_config(
    model_id: str,
) -> tuple[str, str, str, DeepThinkConfig, str]:
    (
        real_model, mgr_model, syn_model,
        p_level, e_level, s_level,
        model_max_rounds, provider,
        planning_temp, expert_temp, review_temp, synthesis_temp,
        mode,
    ) = resolve_model(model_id)

    refinement_kwargs: dict[str, Any] = {}
    if mode == "refinement":
        ref_cfg = resolve_refinement_config(
            model_id, real_model, mgr_model, syn_model,
        )
        refinement_kwargs = {
            "refinement_max_rounds": ref_cfg.refinement_max_rounds,
            "pre_draft_review_rounds": ref_cfg.pre_draft_review_rounds,
            "enable_json_repair": ref_cfg.enable_json_repair,
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
        **refinement_kwargs,
    )
    return real_model, mgr_model, syn_model, config, provider


# ---------------------------------------------------------------------------
# Request parsing helpers
# ---------------------------------------------------------------------------

def _parse_gemini_request(body: dict[str, Any]) -> tuple[
    str, str, list[dict[str, str]], list[dict], str | None, float | None
]:
    """Parse a Gemini generateContent request body.

    Returns:
        (model, query, history, image_parts, system_instruction, temperature)
    """
    model = body.get("model", "")
    contents = body.get("contents", [])
    gen_config = body.get("generationConfig", {})
    temperature = gen_config.get("temperature")

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

    return model, query, history, image_parts, system_text, temperature


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
    model_id, query, history, image_parts, system_text, temperature = (
        _parse_gemini_request(body)
    )

    if not query:
        yield f"data: {json.dumps({'error': 'empty query'})}\n\n"
        return

    real_model, mgr_model, syn_model, config, provider = _resolve_request_config(
        model_id
    )

    checkpoint_store = CheckpointStore()
    resume_id = f"res_{uuid.uuid4().hex[:16]}"
    now = int(time.time())
    checkpoint = checkpoint_store.create(
        resume_id, schema_version=CHECKPOINT_SCHEMA_VERSION,
    )
    checkpoint.request_model = model_id
    checkpoint.real_model = real_model
    checkpoint.manager_model = mgr_model
    checkpoint.synthesis_model = syn_model
    checkpoint.phase = "planning"
    checkpoint.status = "running"
    checkpoint.current_round = 1
    checkpoint.started_at = now
    checkpoint.updated_at = now
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
        # resume hint as the first thought chunk
        hint_chunk = _build_gemini_stream_chunk(
            thought=_resume_hint(checkpoint.resume_id),
        )
        yield f"data: {json.dumps(hint_chunk, ensure_ascii=False)}\n\n"

        async for text_chunk, thought_chunk, _phase, grounding in run_deep_think(
            query=query,
            history=history,
            image_parts=image_parts,
            model=real_model,
            manager_model=mgr_model,
            synthesis_model=syn_model,
            config=config,
            temperature=temperature,
            system_prompt=system_text or "",
            resume_checkpoint=checkpoint,
            event_callback=_persist_event,
            resume_mode=False,
            provider=provider,
        ):
            if grounding:
                all_grounding.extend(grounding)

            if text_chunk or thought_chunk:
                chunk_data = _build_gemini_stream_chunk(
                    text=text_chunk,
                    thought=thought_chunk,
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
        yield f"data: {json.dumps(error_data)}\n\n"


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

    model_id, query, history, image_parts, system_text, temperature = (
        _parse_gemini_request(body)
    )

    if not query:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "empty query", "code": 400}},
        )

    real_model, mgr_model, syn_model, config, provider = _resolve_request_config(
        model_id
    )

    checkpoint_store = CheckpointStore()
    resume_id = f"res_{uuid.uuid4().hex[:16]}"
    now = int(time.time())
    checkpoint = checkpoint_store.create(
        resume_id, schema_version=CHECKPOINT_SCHEMA_VERSION,
    )
    checkpoint.request_model = model_id
    checkpoint.real_model = real_model
    checkpoint.manager_model = mgr_model
    checkpoint.synthesis_model = syn_model
    checkpoint.phase = "planning"
    checkpoint.status = "running"
    checkpoint.current_round = 1
    checkpoint.started_at = now
    checkpoint.updated_at = now
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

    full_text = ""
    full_reasoning = _resume_hint(checkpoint.resume_id)
    all_grounding: list[dict] = []

    async for text_chunk, thought_chunk, _phase, grounding in run_deep_think(
        query=query,
        history=history,
        image_parts=image_parts,
        model=real_model,
        manager_model=mgr_model,
        synthesis_model=syn_model,
        config=config,
        temperature=temperature,
        system_prompt=system_text or "",
        resume_checkpoint=checkpoint,
        event_callback=_persist_event,
        resume_mode=False,
        provider=provider,
    ):
        full_text += text_chunk
        full_reasoning += thought_chunk
        if grounding:
            all_grounding.extend(grounding)

    return _build_gemini_response(
        model=model_id,
        text=full_text,
        reasoning=full_reasoning,
        grounding_chunks=_dedup_grounding(all_grounding),
    )
