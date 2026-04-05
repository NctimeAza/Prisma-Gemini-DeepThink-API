"""Core orchestration pipeline for DeepThink."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, AsyncGenerator, Optional

from config import LLM_PROVIDER, SSE_HEARTBEAT_INTERVAL, StageProviders, get_thinking_budget
from engine import expert, manager, synthesis
from models import (
    AnalysisResult,
    DeepThinkCheckpoint,
    DeepThinkConfig,
    ExpertConfig,
    ExpertResult,
    ReviewResult,
)
from prompts import (
    EXPERT_NAME_SEPARATOR,
    MSG_EXPERT_DONE,
    MSG_EXPERT_ERROR,
    MSG_EXPERT_START,
    MSG_EXPERTS_ASSIGNED,
    MSG_NEXT_ROUND,
    MSG_PIPELINE_START,
    MSG_PREPARING,
    MSG_REVIEW_ERROR,
    MSG_REVIEW_ACTION_DELETE,
    MSG_REVIEW_ACTION_ITERATE,
    MSG_REVIEW_NO_EXPERTS,
    MSG_REVIEW_PASSED,
    MSG_REVIEW_REJECTED_REASON,
    MSG_REVIEWING,
    MSG_ROUND_ASSIGNED,
    MSG_SYNTHESIS_START,
    SYNTHESIS_FALLBACK_TEXT,
    REFINEMENT_FALLBACK_TEXT,
    format_expert_task,
    reset_forced_prefill_suffix_enabled,
    set_forced_prefill_suffix_enabled,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = DeepThinkConfig()

EventCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]


def _now_ts() -> int:
    return int(time.time())


async def _emit_event(
    event_callback: EventCallback | None,
    event: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Emit orchestration events; callback errors must not break serving."""
    if not event_callback:
        return
    try:
        result = event_callback(event, payload or {})
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.exception("[Orchestrator] event callback failed: %s", event)


async def _heartbeat(queue: asyncio.Queue, stop_event: asyncio.Event) -> None:
    """Emit heartbeat thought chunks to keep SSE connections alive."""
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=SSE_HEARTBEAT_INTERVAL)
            break
        except asyncio.TimeoutError:
            await queue.put(("", ".", "synthesis", []))
            logger.debug("[Heartbeat] emitted")


def _build_recent_history(
    history: list[dict[str, str]], max_context_messages: int
) -> str:
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-max_context_messages:]
    )


def _ensure_fallback_analysis(query: str, analysis: AnalysisResult) -> AnalysisResult:
    if analysis.experts:
        return analysis
    logger.warning(
        "[Orchestrator] manager returned empty experts, injecting fallback expert"
    )
    analysis.experts = [
        ExpertConfig(
            role="General Analyst",
            description="General purpose analyst for broad user queries.",
            temperature=1.0,
            prompt=query,
        )
    ]
    return analysis


def _format_expert_names(experts: list[ExpertResult] | list[ExpertConfig]) -> str:
    if not experts:
        return "(none)"
    return EXPERT_NAME_SEPARATOR.join(
        f"{exp.role}(T={exp.temperature:.1f})" for exp in experts
    )


def _build_round_experts(
    query: str,
    expert_configs: list[ExpertConfig],
    round_no: int,
) -> list[ExpertResult]:
    experts: list[ExpertResult] = []
    for idx, exp_config in enumerate(expert_configs):
        experts.append(
            ExpertResult(
                id=f"expert-r{round_no}-{idx + 1}",
                role=exp_config.role,
                description=exp_config.description,
                temperature=exp_config.temperature,
                prompt=format_expert_task(query, exp_config.prompt),
                status="pending",
                round=round_no,
            )
        )
    return experts


def _upsert_review(reviews: list[ReviewResult], review: ReviewResult) -> None:
    for idx, existing in enumerate(reviews):
        if existing.round == review.round:
            reviews[idx] = review
            return
    reviews.append(review)


def _normalize_action_name(action_name: str) -> str:
    action = (action_name or "").strip().lower()
    if action in ("delete", "remove", "drop") or "删" in action or "移除" in action:
        return "delete"
    if action in ("iterate", "iteration", "refine", "improve") or "迭代" in action or "改进" in action:
        return "iterate"
    return "keep"


def _truncate_for_iteration_context(content: str, max_len: int = 20000) -> str:
    if len(content) <= max_len:
        return content
    return (
        f"{content[:max_len]}\n\n"
        f"[注：该段已截断，原始长度为 {len(content)} 字符]"
    )


def _build_iteration_prompt(
    *,
    target_expert: ExpertResult,
    previous_content: str,
    strict_prompt: str,
    improvement_suggestions: str,
    reason: str,
    base_prompt: str,
) -> str:
    strict_part = strict_prompt.strip() or "你必须严格修复上一轮专家回答中的问题。"
    suggestions_part = (
        improvement_suggestions.strip()
        or reason.strip()
        or "请针对审查指出的问题完成彻底改进。"
    )
    task_part = base_prompt.strip() or "输出对目标专家回答的高质量改写和增强版本。"
    source_part = _truncate_for_iteration_context(previous_content or "（无输出）")
    dimension_part = (
        f"【上一轮可用维度】\n{target_expert.usage_dimension}\n\n"
        if (target_expert.usage_dimension or "").strip()
        else ""
    )

    return (
        "你是被指派来进行“专家迭代改进”的新专家。\n"
        f"目标专家：{target_expert.role}（id={target_expert.id}）\n\n"
        f"{dimension_part}"
        "【上一轮目标专家原回复】\n"
        f"{source_part}\n\n"
        "【审查模型严厉指令】\n"
        f"{strict_part}\n\n"
        "【审查模型改进意见】\n"
        f"{suggestions_part}\n\n"
        "【你本轮必须完成的任务】\n"
        f"{task_part}"
    )


def _find_target_expert(
    experts: list[ExpertResult],
    target_expert_id: str,
    target_expert_role: str,
) -> ExpertResult | None:
    if target_expert_id:
        for expert_obj in experts:
            if expert_obj.id == target_expert_id:
                return expert_obj

    if target_expert_role:
        candidates = [e for e in experts if e.role == target_expert_role]
        if candidates:
            return max(candidates, key=lambda e: e.round)

    return None


def _apply_review_actions(
    review_result: ReviewResult,
    all_experts: list[ExpertResult],
) -> tuple[list[ExpertConfig], list[str]]:
    iterated_expert_configs: list[ExpertConfig] = []
    notices: list[str] = []
    processed_targets: set[str] = set()

    for action_obj in review_result.expert_actions:
        action = _normalize_action_name(action_obj.action)
        target = _find_target_expert(
            all_experts,
            action_obj.target_expert_id,
            action_obj.target_expert_role,
        )
        if not target:
            logger.warning(
                "[Orchestrator] review action target not found: id=%s role=%s action=%s",
                action_obj.target_expert_id,
                action_obj.target_expert_role,
                action,
            )
            continue

        if action == "keep":
            continue

        if target.id in processed_targets:
            logger.debug(
                "[Orchestrator] skip duplicate review action for expert %s (%s)",
                target.id,
                target.role,
            )
            continue
        processed_targets.add(target.id)

        if action == "delete":
            reason = (
                action_obj.reason.strip()
                or "审查模型判定该专家方向错误且难以修复，已移除。"
            )
            target.context_status = "deleted"
            target.context_note = reason
            target.content = (
                "该专家内容已被审查模型删除。\n"
                f"专家方向：{target.description or target.role}\n"
                f"删除原因：{reason}"
            )
            notices.append(MSG_REVIEW_ACTION_DELETE.format(expert_name=target.role))
            continue

        if action == "iterate":
            if not action_obj.iterated_expert:
                logger.warning(
                    "[Orchestrator] iterate action missing iterated_expert for %s(%s)",
                    target.role,
                    target.id,
                )
                continue

            reason = (
                action_obj.reason.strip()
                or "审查模型认可方向但要求继续迭代改进。"
            )
            original_content = target.content or "（无输出）"
            target.context_status = "iterated"
            target.context_note = reason
            target.content = (
                "该专家原回复已被审查模型移出后续上下文，转由下一轮迭代专家继续改进。\n"
                f"迭代原因：{reason}"
            )

            iter_cfg = ExpertConfig(**action_obj.iterated_expert.model_dump())
            iter_cfg.prompt = _build_iteration_prompt(
                target_expert=target,
                previous_content=original_content,
                strict_prompt=action_obj.strict_prompt,
                improvement_suggestions=action_obj.improvement_suggestions,
                reason=reason,
                base_prompt=iter_cfg.prompt,
            )
            iterated_expert_configs.append(iter_cfg)
            notices.append(
                MSG_REVIEW_ACTION_ITERATE.format(
                    expert_name=target.role,
                    next_expert_name=iter_cfg.role,
                )
            )

    return iterated_expert_configs, notices


async def _pipeline(
    queue: asyncio.Queue,
    query: str,
    history: list[dict[str, str]],
    model: str,
    mgr_model: str,
    syn_model: str,
    config: DeepThinkConfig,
    temperature: Optional[float],
    top_p: float | None = None,
    system_prompt: str = "",
    image_parts: list[dict] | None = None,
    resume_checkpoint: DeepThinkCheckpoint | None = None,
    event_callback: EventCallback | None = None,
    resume_mode: bool = False,
    stage_providers: StageProviders | None = None,
    provider: str = "",
) -> None:
    """Run manager/expert/review/synthesis pipeline and push chunks into queue."""

    stage_providers = stage_providers or StageProviders.from_single(
        provider or LLM_PROVIDER
    )
    manager_provider = stage_providers.manager
    expert_provider = stage_providers.expert
    synthesis_provider = stage_providers.synthesis

    # --- 精修模式分发 ---
    if config.mode == "refinement":
        from engine.refinement.pipeline import run_refinement_pipeline
        await run_refinement_pipeline(
            queue=queue,
            query=query,
            history=history,
            model=model,
            mgr_model=mgr_model,
            syn_model=syn_model,
            config=config,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            image_parts=image_parts,
            resume_checkpoint=resume_checkpoint,
            stage_providers=stage_providers,
            provider=manager_provider,
        )
        return
    _child_tasks: set[asyncio.Task] = set()

    def _spawn(coro: Awaitable[Any]) -> asyncio.Task:
        task = asyncio.create_task(coro)
        _child_tasks.add(task)
        task.add_done_callback(_child_tasks.discard)
        return task

    analysis: AnalysisResult | None = None
    all_experts: list[ExpertResult] = []
    all_reviews: list[ReviewResult] = []
    round_counter = 1

    async def _sync_checkpoint(event: str, payload: dict[str, Any] | None = None) -> None:
        if not resume_checkpoint:
            return
        resume_checkpoint.analysis = analysis
        resume_checkpoint.experts = all_experts
        resume_checkpoint.reviews = all_reviews
        resume_checkpoint.current_round = max(1, round_counter)
        resume_checkpoint.updated_at = _now_ts()
        await _emit_event(event_callback, event, payload)

    async def _set_phase(phase: str) -> None:
        if not resume_checkpoint:
            return
        resume_checkpoint.phase = phase
        await _sync_checkpoint("phase", {"phase": phase})

    async def _emit_text_notice(text: str) -> None:
        await queue.put(("", f"{text}\n", "synthesis", []))

    async def _run_experts(experts_to_run: list[ExpertResult], context: str, budget: int) -> None:
        if not experts_to_run:
            return

        # 如果虚拟模型定义了 expert_temperature，强制覆盖每个 Expert 的温度
        forced_expert_temp = config.expert_temperature

        async def _run_single(exp: ExpertResult) -> ExpertResult:
            if forced_expert_temp is not None:
                exp.temperature = forced_expert_temp
            await _emit_text_notice(MSG_EXPERT_START.format(expert_name=exp.role))
            exp.status = "thinking"
            await _sync_checkpoint(
                "expert_status",
                {"expert_id": exp.id, "role": exp.role, "status": exp.status},
            )

            result = await expert.run_expert(
                model,
                exp,
                context,
                budget,
                top_p=top_p,
                all_expert_roles=list(
                    dict.fromkeys(
                        e.role for e in all_experts
                        if e.round == exp.round and e.role
                    )
                ),
                user_system_prompt=system_prompt,
                image_parts=image_parts,
                provider=expert_provider,
            )

            if result.status == "completed":
                await _emit_text_notice(MSG_EXPERT_DONE.format(expert_name=result.role))
                if result.content:
                    await queue.put(("", f"\n```content\n{result.content}\n```\n", "experts", []))
            else:
                await _emit_text_notice(MSG_EXPERT_ERROR.format(expert_name=result.role))

            await _sync_checkpoint(
                "expert_status",
                {
                    "expert_id": result.id,
                    "role": result.role,
                    "status": result.status,
                },
            )
            return result

        await asyncio.gather(*[_spawn(_run_single(exp)) for exp in experts_to_run])

    try:
        recent_history = _build_recent_history(history, config.max_context_messages)

        planning_budget = get_thinking_budget(config.planning_level, model)
        expert_budget = get_thinking_budget(config.expert_level, model)
        synthesis_budget = get_thinking_budget(config.synthesis_level, model)

        if resume_mode and resume_checkpoint:
            analysis = resume_checkpoint.analysis
            all_experts = list(resume_checkpoint.experts)
            all_reviews = list(resume_checkpoint.reviews)
            round_counter = max(1, resume_checkpoint.current_round)
        else:
            await _emit_text_notice(MSG_PIPELINE_START)
            await asyncio.sleep(3)
            await _emit_text_notice(MSG_PREPARING)

        start_phase = (
            resume_checkpoint.phase
            if resume_mode and resume_checkpoint
            else "planning"
        )

        planning_needed = (
            (not resume_mode)
            or (start_phase == "planning")
            or (analysis is None)
        )

        if planning_needed:
            logger.info("[Orchestrator] phase=planning")
            await _set_phase("planning")

            if resume_checkpoint:
                analysis = None
                all_experts = []
                all_reviews = []
                round_counter = 1
                resume_checkpoint.current_round = 1
                await _sync_checkpoint("reset_for_planning")

            manager_task = _spawn(
                manager.analyze(
                    mgr_model,
                    query,
                    recent_history,
                    planning_budget,
                    temperature=(
                        config.planning_temperature
                        if config.planning_temperature is not None
                        else temperature
                    ),
                    top_p=top_p,
                    user_system_prompt=system_prompt,
                    image_parts=image_parts,
                    provider=manager_provider,
                    json_via_prompt=config.json_via_prompt,
                )
            )
            analysis = await manager_task
            analysis = _ensure_fallback_analysis(query, analysis)
            await _sync_checkpoint("manager_completed")

            round_counter = 1
            all_experts = _build_round_experts(query, analysis.experts, round_counter)
            await _set_phase("experts")

            await _emit_text_notice(
                MSG_EXPERTS_ASSIGNED.format(
                    total=len(all_experts),
                    names=_format_expert_names(all_experts),
                )
            )
            await _sync_checkpoint("round_assigned", {"round": round_counter})

        elif analysis and not all_experts:
            logger.info("[Orchestrator] rebuilding experts from stored analysis")
            round_counter = 1
            all_experts = _build_round_experts(query, analysis.experts, round_counter)
            await _set_phase("experts")
            await _sync_checkpoint("round_assigned", {"round": round_counter})
        elif all_experts:
            round_counter = max(round_counter, max(exp.round for exp in all_experts))

        if not (resume_mode and start_phase == "synthesis"):
            pending_experts = [exp for exp in all_experts if exp.status != "completed"]
            if pending_experts:
                logger.info(
                    "[Orchestrator] running %d pending experts",
                    len(pending_experts),
                )
                await _set_phase("experts")
                await _run_experts(pending_experts, recent_history, expert_budget)

            loop_active = config.enable_recursive_loop and len(all_experts) > 0
            max_rounds = config.max_rounds

            while loop_active and round_counter < max_rounds:
                logger.info("[Orchestrator] phase=review round=%d", round_counter)
                await _set_phase("review")
                await _emit_text_notice(MSG_REVIEWING)

                try:
                    remaining_rounds = (
                        max_rounds - round_counter
                        if config.planning_level in ("medium", "high")
                        else 0
                    )

                    review_task = _spawn(
                        manager.review(
                            mgr_model,
                            query,
                            all_experts,
                            planning_budget,
                            context=recent_history,
                            temperature=(
                                config.review_temperature
                                if config.review_temperature is not None
                                else 0.7
                            ),
                            top_p=top_p,
                            user_system_prompt=system_prompt,
                            image_parts=image_parts,
                            remaining_rounds=remaining_rounds,
                            previous_reviews=all_reviews,
                            provider=manager_provider,
                            json_via_prompt=config.json_via_prompt,
                        )
                    )
                    review_result = await review_task
                    review_result.round = round_counter
                    _upsert_review(all_reviews, review_result)
                    await _sync_checkpoint(
                        "review_completed",
                        {
                            "round": round_counter,
                            "satisfied": review_result.satisfied,
                        },
                    )

                    if review_result.satisfied:
                        await _emit_text_notice(MSG_REVIEW_PASSED)
                        loop_active = False
                        break

                    iterated_experts, action_notices = _apply_review_actions(
                        review_result,
                        all_experts,
                    )
                    for notice in action_notices:
                        await _emit_text_notice(notice)
                    await _sync_checkpoint(
                        "review_actions_applied",
                        {
                            "round": round_counter,
                            "action_count": len(review_result.expert_actions),
                            "iterated_count": len(iterated_experts),
                        },
                    )

                    round_counter += 1
                    await _sync_checkpoint(
                        "round_advanced", {"round": round_counter}
                    )

                    if review_result.overall_rejection_reason:
                        await _emit_text_notice(
                            MSG_REVIEW_REJECTED_REASON.format(
                                reason=review_result.overall_rejection_reason
                            )
                        )

                    await _emit_text_notice(MSG_NEXT_ROUND.format(round=round_counter))

                    next_round_configs = list(review_result.refined_experts)
                    next_round_configs.extend(iterated_experts)
                    next_round_experts = _build_round_experts(
                        query,
                        next_round_configs,
                        round_counter,
                    )
                    if not next_round_experts:
                        await _emit_text_notice(MSG_REVIEW_NO_EXPERTS)
                        loop_active = False
                        break

                    await _emit_text_notice(
                        MSG_ROUND_ASSIGNED.format(
                            round=round_counter,
                            count=len(next_round_experts),
                            names=_format_expert_names(next_round_experts),
                        )
                    )

                    all_experts.extend(next_round_experts)
                    await _set_phase("experts")
                    await _sync_checkpoint(
                        "round_assigned",
                        {"round": round_counter, "count": len(next_round_experts)},
                    )

                    await _run_experts(next_round_experts, recent_history, expert_budget)

                except Exception as exc:
                    logger.error("[Orchestrator] review failed: %s", exc)
                    await _emit_text_notice(MSG_REVIEW_ERROR)
                    await _sync_checkpoint(
                        "review_failed",
                        {"round": round_counter, "error": str(exc)},
                    )
                    loop_active = False

        logger.info("[Orchestrator] phase=synthesis")
        await _set_phase("synthesis")
        await _emit_text_notice(MSG_SYNTHESIS_START)
        await _sync_checkpoint("synthesis_started")

        try:
            async for text_chunk, thought_chunk, grounding_chunks in synthesis.stream_synthesis(
                model=syn_model,
                query=query,
                history_context=recent_history,
                expert_results=all_experts,
                review_results=all_reviews,
                budget=synthesis_budget,
                temperature=(
                    config.synthesis_temperature
                    if config.synthesis_temperature is not None
                    else temperature
                ),
                top_p=top_p,
                user_system_prompt=system_prompt,
                image_parts=image_parts,
                provider=synthesis_provider,
            ):
                await queue.put((text_chunk, thought_chunk, "synthesis", grounding_chunks))
        except Exception as exc:
            logger.exception("[Orchestrator] synthesis failed")
            if resume_checkpoint:
                resume_checkpoint.status = "error"
                resume_checkpoint.error_message = str(exc)
                resume_checkpoint.completed_at = None
                resume_checkpoint.updated_at = _now_ts()
            await queue.put((SYNTHESIS_FALLBACK_TEXT, "", "system_error", []))
            await _sync_checkpoint("synthesis_failed", {"error": str(exc)})
            return

        await _sync_checkpoint("pipeline_completed")

    except asyncio.CancelledError:
        logger.info("[Orchestrator] pipeline cancelled")
        raise
    finally:
        remaining = [task for task in _child_tasks if not task.done()]
        if remaining:
            logger.info(
                "[Orchestrator] cancelling %d child tasks",
                len(remaining),
            )
            for task in remaining:
                task.cancel()
            await asyncio.gather(*remaining, return_exceptions=True)


async def run_deep_think(
    query: str,
    history: list[dict[str, str]],
    model: str,
    manager_model: str | None = None,
    synthesis_model: str | None = None,
    config: DeepThinkConfig | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    system_prompt: str = "",
    image_parts: list[dict] | None = None,
    resume_checkpoint: DeepThinkCheckpoint | None = None,
    event_callback: EventCallback | None = None,
    resume_mode: bool = False,
    stage_providers: StageProviders | None = None,
    provider: str = "",
) -> AsyncGenerator[tuple[str, str, str, list[dict]], None]:
    """Run the complete DeepThink pipeline and stream text/thought chunks."""
    if not config:
        config = DEFAULT_CONFIG

    stage_providers = stage_providers or StageProviders.from_single(
        provider or LLM_PROVIDER
    )

    mgr_model = manager_model or model
    syn_model = synthesis_model or model

    if not query.strip():
        return

    prefill_suffix_token = set_forced_prefill_suffix_enabled(
        config.forced_prefill_suffix
    )

    if resume_checkpoint:
        resume_checkpoint.status = "running"
        resume_checkpoint.error_message = ""
        resume_checkpoint.updated_at = _now_ts()
        await _emit_event(event_callback, "run_started", {"resume_mode": resume_mode})

    queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()

    heartbeat_task = asyncio.create_task(_heartbeat(queue, stop_event))
    pipeline_task = asyncio.create_task(
        _pipeline(
            queue,
            query,
            history,
            model,
            mgr_model,
            syn_model,
            config,
            temperature,
            top_p,
            system_prompt,
            image_parts,
            resume_checkpoint,
            event_callback,
            resume_mode,
            stage_providers,
            provider,
        )
    )

    async def _record_chunk(item: tuple[str, str, str, list[dict]]) -> None:
        if not resume_checkpoint:
            return
        text_chunk, thought_chunk, phase, _grounding = item
        is_fallback_error_chunk = (
            phase == "system_error"
            and (text_chunk == SYNTHESIS_FALLBACK_TEXT
                 or text_chunk == REFINEMENT_FALLBACK_TEXT)
        )
        if text_chunk and not is_fallback_error_chunk:
            resume_checkpoint.output_content += text_chunk
        if thought_chunk:
            resume_checkpoint.reasoning_content += thought_chunk
        resume_checkpoint.updated_at = _now_ts()
        await _emit_event(
            event_callback,
            "chunk",
            {
                "phase": phase,
                "text_len": len(text_chunk),
                "thought_len": len(thought_chunk),
            },
        )

    get_task: asyncio.Task | None = None

    try:
        while True:
            get_task = asyncio.create_task(queue.get())
            try:
                done, _ = await asyncio.wait(
                    [get_task, pipeline_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                get_task.cancel()
                raise

            if get_task in done:
                item = get_task.result()
                await _record_chunk(item)
                yield item
            else:
                get_task.cancel()
                break

        while not queue.empty():
            item = queue.get_nowait()
            await _record_chunk(item)
            yield item

        if pipeline_task.done() and pipeline_task.exception():
            raise pipeline_task.exception()

    except asyncio.CancelledError:
        if resume_checkpoint:
            resume_checkpoint.status = "interrupted"
            resume_checkpoint.updated_at = _now_ts()
            await _emit_event(event_callback, "interrupted")
        raise
    except Exception as exc:
        if resume_checkpoint:
            resume_checkpoint.status = "error"
            resume_checkpoint.error_message = str(exc)
            resume_checkpoint.updated_at = _now_ts()
            await _emit_event(event_callback, "error", {"error": str(exc)})
        raise
    else:
        if resume_checkpoint:
            if resume_checkpoint.status == "running":
                has_final_output = bool((resume_checkpoint.output_content or "").strip())
                if has_final_output:
                    resume_checkpoint.status = "completed"
                    resume_checkpoint.phase = "synthesis"
                    resume_checkpoint.completed_at = _now_ts()
                    resume_checkpoint.updated_at = _now_ts()
                    await _emit_event(event_callback, "completed")
                else:
                    # 综合阶段没有正文输出时，不能标记为 completed。
                    resume_checkpoint.status = "error"
                    resume_checkpoint.error_message = (
                        "synthesis finished without output_content"
                    )
                    resume_checkpoint.completed_at = None
                    resume_checkpoint.updated_at = _now_ts()
                    await _emit_event(
                        event_callback,
                        "completed_without_output",
                        {"phase": resume_checkpoint.phase},
                    )
    finally:
        stop_event.set()
        heartbeat_task.cancel()
        if not pipeline_task.done():
            pipeline_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass
        reset_forced_prefill_suffix_enabled(prefill_suffix_token)
