"""OpenAI-compatible chat route with resume support."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import (
    CHECKPOINT_REPLAY_CHUNK_SIZE,
    CHECKPOINT_SCHEMA_VERSION,
    ENABLE_RECURSIVE_LOOP,
    MAX_CONTEXT_MESSAGES,
    SSE_HEARTBEAT_INTERVAL,
    resolve_model,
)
from engine.checkpoint_store import CheckpointStore, CheckpointStoreError
from engine.orchestrator import SYNTHESIS_FALLBACK_TEXT, run_deep_think
from prompts import REFINEMENT_FALLBACK_TEXT, RESUME_HINT_TEXT
from models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    DeepThinkCheckpoint,
    DeepThinkConfig,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_CONTINUE_COMMAND = "!deepthink_continue"
_CONTINUE_ALIASES: tuple[str, ...] = (_CONTINUE_COMMAND, "/continue")
_CONTINUE_RE = re.compile(
    r"^(?:!deepthink_continue|/continue)\s+([A-Za-z0-9_-]+)\s*$"
)
_ACTIVE_RESUME_IDS: set[str] = set()
_ACTIVE_RESUME_LOCK = asyncio.Lock()
_DISCONNECT_POLL_INTERVAL_SECONDS = 0.25
_CLIENT_CLOSED_STATUS_CODE = 499


def _error_response(status_code: int, message: str) -> JSONResponse:
    logger.error("[API] HTTP %d error: %s", status_code, message)
    return JSONResponse(status_code=status_code, content={"error": message})


def _build_history(request: ChatCompletionRequest) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for msg in request.messages[:-1]:
        if msg.role in ("user", "assistant"):
            history.append({"role": msg.role, "content": msg.text})
    return history


def _get_query(request: ChatCompletionRequest) -> str:
    for msg in reversed(request.messages):
        if msg.role == "user":
            return msg.text
    return ""


def _extract_image_parts(request: ChatCompletionRequest) -> list[dict]:
    for msg in reversed(request.messages):
        if msg.role == "user":
            return msg.image_parts
    return []


def _extract_system_prompt(request: ChatCompletionRequest) -> str:
    return "\n".join(msg.text for msg in request.messages if msg.role == "system")


def _resolve_request(
    request: ChatCompletionRequest,
) -> tuple[str, str, str, DeepThinkConfig, str]:
    (
        real_model, mgr_model, syn_model,
        p_level, e_level, s_level,
        model_max_rounds, provider,
        planning_temp, expert_temp, review_temp, synthesis_temp,
        mode,
    ) = resolve_model(request.model)

    if request.prisma_config:
        return real_model, mgr_model, syn_model, request.prisma_config, provider

    # 精修流程额外配置
    refinement_kwargs: dict = {}
    if mode == "refinement":
        from config import resolve_refinement_config
        ref_cfg = resolve_refinement_config(
            request.model, real_model, mgr_model, syn_model,
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


def _parse_continue_command(
    request: ChatCompletionRequest,
) -> tuple[bool, str | None, str | None]:
    """Detect and strip resume command if it is the last user message."""
    if not request.messages:
        return False, None, None

    last = request.messages[-1]
    if last.role != "user":
        return False, None, None

    text = last.text.strip()
    if not text.startswith(_CONTINUE_ALIASES):
        return False, None, None

    match = _CONTINUE_RE.fullmatch(text)
    if not match:
        return True, None, (
            f"invalid continue command, use: {_CONTINUE_COMMAND} <id>"
        )

    request.messages = request.messages[:-1]
    return True, match.group(1), None


async def _acquire_resume_id(resume_id: str) -> bool:
    async with _ACTIVE_RESUME_LOCK:
        if resume_id in _ACTIVE_RESUME_IDS:
            return False
        _ACTIVE_RESUME_IDS.add(resume_id)
        return True


async def _release_resume_id(resume_id: str) -> None:
    async with _ACTIVE_RESUME_LOCK:
        _ACTIVE_RESUME_IDS.discard(resume_id)


async def _wait_for_client_disconnect(request: Request) -> None:
    """Block until client disconnects."""
    while True:
        if await request.is_disconnected():
            return
        await asyncio.sleep(_DISCONNECT_POLL_INTERVAL_SECONDS)


def _iter_chunks(text: str) -> list[str]:
    if not text:
        return []
    size = max(64, CHECKPOINT_REPLAY_CHUNK_SIZE)
    return [text[i : i + size] for i in range(0, len(text), size)]


def _is_fallback_error_text(text: str) -> bool:
    stripped = (text or "").strip()
    return stripped == SYNTHESIS_FALLBACK_TEXT or stripped == REFINEMENT_FALLBACK_TEXT


def _resume_hint(resume_id: str) -> str:
    return (
        f"[resume_id] {resume_id}\n"
        f"{RESUME_HINT_TEXT.format(command=_CONTINUE_COMMAND, resume_id=resume_id)}\n"
    )


def _new_reasoning_chunk(
    completion_id: str,
    model: str,
    content: str,
) -> str:
    chunk = ChatCompletionChunk(
        id=completion_id,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(reasoning_content=content)
            )
        ],
    )
    return f"data: {chunk.model_dump_json()}\n\n"


def _new_content_chunk(
    completion_id: str,
    model: str,
    content: str,
) -> str:
    chunk = ChatCompletionChunk(
        id=completion_id,
        model=model,
        choices=[
            ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=content))
        ],
    )
    return f"data: {chunk.model_dump_json()}\n\n"


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


async def _generate_sse_stream(
    *,
    request: ChatCompletionRequest,
    query: str,
    history: list[dict[str, str]],
    system_prompt: str,
    image_parts: list[dict],
    real_model: str,
    mgr_model: str,
    syn_model: str,
    config: DeepThinkConfig,
    checkpoint: DeepThinkCheckpoint,
    checkpoint_store: CheckpointStore,
    resume_mode: bool,
    replay_only: bool,
    provider: str = "",
) -> AsyncGenerator[str, None]:
    """Stream OpenAI-compatible SSE chunks with optional checkpoint replay."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async def _persist_event(_: str, __: dict) -> None:
        try:
            checkpoint.updated_at = int(time.time())
            checkpoint_store.save(checkpoint)
        except Exception:
            logger.exception("[Checkpoint] failed to persist %s", checkpoint.resume_id)

    # role chunk first
    role_chunk = ChatCompletionChunk(
        id=completion_id,
        model=request.model,
        choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(role="assistant"))],
    )
    yield f"data: {role_chunk.model_dump_json()}\n\n"

    try:
        # ID is exposed only through reasoning stream as requested
        yield _new_reasoning_chunk(
            completion_id,
            request.model,
            _resume_hint(checkpoint.resume_id),
        )

        if resume_mode:
            for thought_part in _iter_chunks(checkpoint.reasoning_content):
                yield _new_reasoning_chunk(completion_id, request.model, thought_part)

            for text_part in _iter_chunks(checkpoint.output_content):
                yield _new_content_chunk(completion_id, request.model, text_part)

            if replay_only:
                finish_chunk = ChatCompletionChunk(
                    id=completion_id,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {finish_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

        sse_queue: asyncio.Queue = asyncio.Queue()
        sse_stop = asyncio.Event()
        all_grounding: list[dict] = []

        async def _data_producer() -> None:
            try:
                async for text_chunk, thought_chunk, _phase, grounding in run_deep_think(
                    query=query,
                    history=history,
                    image_parts=image_parts,
                    model=real_model,
                    manager_model=mgr_model,
                    synthesis_model=syn_model,
                    config=config,
                    temperature=request.temperature,
                    system_prompt=system_prompt,
                    resume_checkpoint=checkpoint,
                    event_callback=_persist_event,
                    resume_mode=resume_mode,
                    provider=provider,
                ):
                    await sse_queue.put(("data", text_chunk, thought_chunk, grounding))
            except Exception as exc:
                logger.error("[SSE] producer failed: %s", exc)
            finally:
                await sse_queue.put(None)

        async def _sse_heartbeat() -> None:
            while not sse_stop.is_set():
                try:
                    await asyncio.wait_for(
                        sse_stop.wait(), timeout=SSE_HEARTBEAT_INTERVAL
                    )
                    break
                except asyncio.TimeoutError:
                    await sse_queue.put(("heartbeat",))

        producer_task = asyncio.create_task(_data_producer())
        heartbeat_task = asyncio.create_task(_sse_heartbeat())

        try:
            while True:
                item = await sse_queue.get()
                if item is None:
                    break
                if item[0] == "heartbeat":
                    yield ": heartbeat\n\n"
                    continue

                _, text_chunk, thought_chunk, grounding = item
                if thought_chunk:
                    yield _new_reasoning_chunk(completion_id, request.model, thought_chunk)
                if text_chunk:
                    yield _new_content_chunk(completion_id, request.model, text_chunk)
                if grounding:
                    all_grounding.extend(grounding)

        finally:
            sse_stop.set()
            heartbeat_task.cancel()
            if not producer_task.done():
                producer_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            try:
                await producer_task
            except asyncio.CancelledError:
                pass

        # 在 finish chunk 前追加 grounding metadata
        if all_grounding:
            grounding_json = json.dumps(
                {"grounding_chunks": _dedup_grounding(all_grounding)},
                ensure_ascii=False,
            )
            yield _new_content_chunk(
                completion_id,
                request.model,
                f"\n\n<grounding>\n{grounding_json}\n</grounding>",
            )

        finish_chunk = ChatCompletionChunk(
            id=completion_id,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {finish_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    finally:
        await _release_resume_id(checkpoint.resume_id)


@router.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    """OpenAI-compatible chat completions endpoint."""
    raw_body = await raw_request.body()
    try:
        raw_json = json.loads(raw_body)
    except json.JSONDecodeError:
        raw_json = {"_raw": raw_body.decode("utf-8", errors="replace")}

    logger.debug(
        "[API] raw request\n%s",
        json.dumps(raw_json, ensure_ascii=False, indent=2)[:500000],
    )

    try:
        request = ChatCompletionRequest(**raw_json)
    except Exception as exc:
        return _error_response(422, f"request parse failed: {exc}")

    if request.messages and request.messages[-1].role == "assistant":
        logger.warning("[API] dropping trailing assistant prefill message")
        request.messages = request.messages[:-1]

    continue_mode, resume_id, continue_error = _parse_continue_command(request)
    if continue_error:
        return _error_response(400, continue_error)

    query = _get_query(request).strip()
    if not query:
        return _error_response(400, f"missing user query after {_CONTINUE_COMMAND} command")

    history = _build_history(request)
    system_prompt = _extract_system_prompt(request)
    image_parts = _extract_image_parts(request)
    real_model, mgr_model, syn_model, config, provider = _resolve_request(request)

    checkpoint_store = CheckpointStore()
    now = int(time.time())

    replay_only = False
    if continue_mode:
        if not resume_id:
            return _error_response(400, f"missing resume id for {_CONTINUE_COMMAND} command")
        try:
            checkpoint = checkpoint_store.load(resume_id)
        except FileNotFoundError:
            return _error_response(404, f"checkpoint not found: {resume_id}")
        except CheckpointStoreError as exc:
            return _error_response(400, str(exc))

        # Repair legacy checkpoints that were incorrectly marked completed
        # after synthesis failures and only contain fallback error output.
        if (
            checkpoint.status == "completed"
            and checkpoint.phase == "synthesis"
            and _is_fallback_error_text(checkpoint.output_content)
        ):
            logger.warning(
                "[Checkpoint] repaired completed->error state for %s",
                checkpoint.resume_id,
            )
            checkpoint.status = "error"
            checkpoint.completed_at = None
            checkpoint.output_content = ""

        replay_only = checkpoint.status == "completed"

        # 禁止跨模式 continue
        if checkpoint.pipeline_mode != config.mode:
            await _release_resume_id(checkpoint.resume_id)
            return _error_response(
                400,
                f"pipeline mode mismatch: checkpoint was created with "
                f"mode='{checkpoint.pipeline_mode}' but current model uses "
                f"mode='{config.mode}'. Cannot resume across different modes."
            )

        checkpoint.request_model = request.model
        checkpoint.real_model = real_model
        checkpoint.manager_model = mgr_model
        checkpoint.synthesis_model = syn_model
        checkpoint.schema_version = CHECKPOINT_SCHEMA_VERSION
        checkpoint.updated_at = now
        if not replay_only:
            checkpoint.status = "running"
            checkpoint.error_message = ""
        checkpoint_store.save(checkpoint)
    else:
        resume_id = f"res_{uuid.uuid4().hex[:16]}"
        checkpoint = checkpoint_store.create(
            resume_id,
            schema_version=CHECKPOINT_SCHEMA_VERSION,
        )
        checkpoint.request_model = request.model
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

    acquired = await _acquire_resume_id(checkpoint.resume_id)
    if not acquired:
        return _error_response(
            409,
            "resume id already has an active run, wait for completion "
            "or disconnect before retrying"
        )

    if request.stream:
        return StreamingResponse(
            _generate_sse_stream(
                request=request,
                query=query,
                history=history,
                system_prompt=system_prompt,
                image_parts=image_parts,
                real_model=real_model,
                mgr_model=mgr_model,
                syn_model=syn_model,
                config=config,
                checkpoint=checkpoint,
                checkpoint_store=checkpoint_store,
                resume_mode=continue_mode,
                replay_only=replay_only,
                provider=provider,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _persist_event(_: str, __: dict) -> None:
        try:
            checkpoint.updated_at = int(time.time())
            checkpoint_store.save(checkpoint)
        except Exception:
            logger.exception("[Checkpoint] failed to persist %s", checkpoint.resume_id)

    try:
        resume_hint = _resume_hint(checkpoint.resume_id)
        full_content = checkpoint.output_content if continue_mode else ""
        full_reasoning = resume_hint + (
            checkpoint.reasoning_content if continue_mode else ""
        )
        all_grounding_ns: list[dict] = []

        if not replay_only:
            deep_think_iter = run_deep_think(
                query=query,
                history=history,
                image_parts=image_parts,
                model=real_model,
                manager_model=mgr_model,
                synthesis_model=syn_model,
                config=config,
                temperature=request.temperature,
                system_prompt=system_prompt,
                resume_checkpoint=checkpoint,
                event_callback=_persist_event,
                resume_mode=continue_mode,
                provider=provider,
            )
            disconnect_task = asyncio.create_task(
                _wait_for_client_disconnect(raw_request)
            )
            try:
                while True:
                    next_chunk_task = asyncio.create_task(deep_think_iter.__anext__())
                    done, _ = await asyncio.wait(
                        [next_chunk_task, disconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if disconnect_task in done:
                        logger.info(
                            "[API] non-stream client disconnected; cancelling run %s",
                            checkpoint.resume_id,
                        )
                        next_chunk_task.cancel()
                        await asyncio.gather(next_chunk_task, return_exceptions=True)
                        return _error_response(_CLIENT_CLOSED_STATUS_CODE, "client disconnected")

                    try:
                        text_chunk, thought_chunk, _phase, grounding = next_chunk_task.result()
                    except StopAsyncIteration:
                        break

                    full_content += text_chunk
                    full_reasoning += thought_chunk
                    if grounding:
                        all_grounding_ns.extend(grounding)
            finally:
                disconnect_task.cancel()
                await asyncio.gather(disconnect_task, return_exceptions=True)

        message: dict[str, str] = {
            "role": "assistant",
            "content": full_content,
        }
        if full_reasoning:
            message["reasoning_content"] = full_reasoning
        if all_grounding_ns:
            message["grounding_chunks"] = _dedup_grounding(all_grounding_ns)

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=message,
                    finish_reason="stop",
                )
            ],
        )
    finally:
        await _release_resume_id(checkpoint.resume_id)
