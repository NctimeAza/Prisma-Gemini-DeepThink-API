"""Refinement pipeline implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from config import LLM_NETWORK_RETRIES, LLM_PROVIDER, StageProviders, get_thinking_budget
from clients.llm_client import generate_content
from engine import manager
from engine.orchestrator import _apply_review_actions
from engine.refinement import applier, cleaner, draft, improver, merger, planner, reviewer
from models import (
    DeepThinkCheckpoint,
    DeepThinkConfig,
    DiffOperation,
    ExpertConfig,
    ExpertResult,
    MergeDecision,
    RefinementExpertConfig,
    ReviewResult,
)
from prompts import (
    MSG_PIPELINE_START,
    MSG_REFINEMENT_APPLIED,
    MSG_REFINEMENT_CLEAN_DONE,
    MSG_REFINEMENT_CLEAN_ERROR,
    MSG_REFINEMENT_CLEAN_START,
    MSG_REFINEMENT_DRAFT_DONE,
    MSG_REFINEMENT_DRAFT_START,
    MSG_REFINEMENT_EXPERT_DONE,
    MSG_REFINEMENT_EXPERT_START,
    MSG_REFINEMENT_IMPROVER_DONE,
    MSG_REFINEMENT_IMPROVER_START,
    MSG_REFINEMENT_MERGE_DONE,
    MSG_REFINEMENT_MERGE_START,
    MSG_REFINEMENT_NEXT_ROUND,
    MSG_REFINEMENT_OUTPUT,
    MSG_REFINEMENT_PLANNING,
    MSG_REFINEMENT_PRE_DRAFT_NEXT_ROUND,
    MSG_REFINEMENT_PRE_DRAFT_REVIEW_APPROVED,
    MSG_REFINEMENT_PRE_DRAFT_REVIEW_REJECTED_REASON,
    MSG_REFINEMENT_PRE_DRAFT_REVIEW_START,
    MSG_REFINEMENT_PRE_DRAFT_ROUND_ASSIGNED,
    MSG_REFINEMENT_REVIEW_APPROVED,
    MSG_REFINEMENT_REVIEW_START,
    build_refinement_expert_contents,
    format_expert_task,
    get_refinement_expert_system_instruction,
)
from utils.retry import extract_status, is_retryable_error

logger = logging.getLogger(__name__)


def _now_ts() -> int:
    return int(time.time())


def _upsert_review(reviews: list[ReviewResult], review: ReviewResult) -> None:
    for idx, existing in enumerate(reviews):
        if existing.round == review.round:
            reviews[idx] = review
            return
    reviews.append(review)


def _to_refinement_configs(expert_configs: list[ExpertConfig]) -> list[RefinementExpertConfig]:
    all_roles = [cfg.role for cfg in expert_configs]
    return [
        RefinementExpertConfig(
            role=cfg.role,
            domain=cfg.description or cfg.role,
            prompt=cfg.prompt,
            temperature=cfg.temperature,
            all_expert_roles=all_roles,
        )
        for cfg in expert_configs
    ]


def _outputs_to_expert_results(
    outputs: list[dict[str, str]],
    round_no: int,
) -> list[ExpertResult]:
    return [
        ExpertResult(
            id=f"refinement-r{round_no}-{idx}",
            role=item.get("role", f"expert-{idx}"),
            description=item.get("domain", ""),
            temperature=1.0,
            prompt="",
            status="completed",
            content=item.get("content", ""),
            round=round_no,
        )
        for idx, item in enumerate(outputs, start=1)
    ]


def _collect_draft_inputs(
    experts: list[ExpertResult],
    fallback_outputs: list[dict[str, str]],
    query: str,
) -> list[dict[str, str]]:
    active_outputs = [
        {"role": exp.role, "domain": exp.description, "content": exp.content}
        for exp in experts
        if exp.context_status == "active" and (exp.content or "").strip()
    ]
    if active_outputs:
        return active_outputs

    rounds = sorted({exp.round for exp in experts}, reverse=True)
    for round_no in rounds:
        round_outputs = [
            {"role": exp.role, "domain": exp.description, "content": exp.content}
            for exp in experts
            if exp.round == round_no and (exp.content or "").strip()
        ]
        if round_outputs:
            return round_outputs

    if fallback_outputs:
        return fallback_outputs

    return [{"role": "fallback", "domain": "", "content": query}]


async def _run_single_expert(
    model: str,
    expert_cfg: RefinementExpertConfig,
    query: str,
    context: str,
    budget: int,
    top_p: float | None = None,
    user_system_prompt: str = "",
    image_parts: list[dict] | None = None,
    provider: str = "",
    forced_temperature: float | None = None,
) -> dict[str, str]:
    """Run one refinement expert and return {role, domain, content}."""
    system_instruction = get_refinement_expert_system_instruction(
        role=expert_cfg.role,
        domain=expert_cfg.domain,
        context=context,
        all_expert_roles=expert_cfg.all_expert_roles,
        user_system_prompt=user_system_prompt,
    )

    task_prompt = format_expert_task(query, expert_cfg.prompt)
    contents = build_refinement_expert_contents(
        task_prompt,
        image_parts=image_parts,
    )

    temperature = (
        forced_temperature if forced_temperature is not None else expert_cfg.temperature
    )

    for attempt in range(LLM_NETWORK_RETRIES + 1):
        try:
            full_content, _, _ = await generate_content(
                model=model,
                contents=contents,
                system_instruction=system_instruction,
                temperature=temperature,
                top_p=top_p,
                thinking_budget=budget,
                provider=provider,
            )
            if full_content.strip():
                return {
                    "role": expert_cfg.role,
                    "domain": expert_cfg.domain,
                    "content": full_content,
                }

            if attempt < LLM_NETWORK_RETRIES:
                delay = 1.5 * (attempt + 1)
                logger.warning(
                    "[RefinementExpert] %s empty response, retry %d/%d",
                    expert_cfg.role,
                    attempt + 1,
                    LLM_NETWORK_RETRIES,
                )
                await asyncio.sleep(delay)
                continue

            return {
                "role": expert_cfg.role,
                "domain": expert_cfg.domain,
                "content": "(expert generated empty output)",
            }

        except Exception as exc:
            status = extract_status(exc)
            retryable = is_retryable_error(status)
            if retryable and attempt < LLM_NETWORK_RETRIES:
                delay = 1.5 * (attempt + 1)
                logger.warning(
                    "[RefinementExpert] %s error (status=%s), retry %d/%d: %s",
                    expert_cfg.role,
                    status,
                    attempt + 1,
                    LLM_NETWORK_RETRIES,
                    exc,
                )
                await asyncio.sleep(delay)
                continue

            logger.error("[RefinementExpert] %s failed: %s", expert_cfg.role, exc)
            return {
                "role": expert_cfg.role,
                "domain": expert_cfg.domain,
                "content": f"(expert execution failed: {exc})",
            }

    return {
        "role": expert_cfg.role,
        "domain": expert_cfg.domain,
        "content": "(expert retries exhausted)",
    }


async def run_refinement_pipeline(
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
    stage_providers: StageProviders | None = None,
    provider: str = "",
) -> None:
    """Main entry point for refinement pipeline."""

    stage_providers = stage_providers or StageProviders.from_single(
        provider or LLM_PROVIDER
    )
    manager_provider = stage_providers.manager
    expert_provider = stage_providers.expert
    synthesis_provider = stage_providers.synthesis

    async def _emit(text: str) -> None:
        await queue.put(("", f"{text}\n", "refinement", []))

    def _set_refinement_phase(phase: str) -> None:
        if resume_checkpoint:
            resume_checkpoint.refinement_phase = phase
            resume_checkpoint.updated_at = _now_ts()

    async def _run_expert_batch(
        expert_cfgs: list[RefinementExpertConfig],
    ) -> list[dict[str, str]]:
        if not expert_cfgs:
            return []

        for expert_cfg in expert_cfgs:
            await _emit(
                MSG_REFINEMENT_EXPERT_START.format(
                    expert_name=expert_cfg.role,
                    domain=expert_cfg.domain,
                )
            )

        async def _run_one(expert_cfg: RefinementExpertConfig) -> dict[str, str]:
            output = await _run_single_expert(
                model=model,
                expert_cfg=expert_cfg,
                query=query,
                context=recent_history,
                budget=expert_budget,
                top_p=top_p,
                user_system_prompt=system_prompt,
                image_parts=image_parts,
                provider=expert_provider,
                forced_temperature=config.expert_temperature,
            )
            await _emit(MSG_REFINEMENT_EXPERT_DONE.format(expert_name=output["role"]))
            if output["content"]:
                await queue.put(
                    ("", f"\n```content\n{output['content']}\n```\n", "experts", [])
                )
            return output

        return list(await asyncio.gather(*[_run_one(cfg) for cfg in expert_cfgs]))

    planning_budget = get_thinking_budget(config.planning_level, model)
    expert_budget = get_thinking_budget(config.expert_level, model)
    synthesis_budget = get_thinking_budget(config.synthesis_level, model)

    draft_model = config.draft_model or model
    review_model = config.review_model or mgr_model
    merge_model = config.merge_model or syn_model
    json_repair_model = config.json_repair_model or "gemini-3-flash-preview"
    max_refinement_rounds = config.refinement_max_rounds
    pre_draft_review_rounds = max(0, config.pre_draft_review_rounds)
    enable_json_repair = config.enable_json_repair

    max_ctx = config.max_context_messages
    recent_history = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-max_ctx:]
    )

    start_phase = "planning"
    approved_outputs: list[dict[str, str]] = []
    draft_text = ""
    previous_merge_summary = ""
    start_round = 1

    if resume_checkpoint and resume_checkpoint.pipeline_mode == "refinement":
        start_phase = resume_checkpoint.refinement_phase or "planning"
        if resume_checkpoint.refinement_expert_outputs:
            approved_outputs = list(resume_checkpoint.refinement_expert_outputs)
        if resume_checkpoint.draft_content:
            draft_text = resume_checkpoint.draft_content
        if resume_checkpoint.refinement_merge_summary:
            previous_merge_summary = resume_checkpoint.refinement_merge_summary
        if resume_checkpoint.refinement_round > 0:
            start_round = resume_checkpoint.refinement_round

        logger.info(
            "[RefinementPipeline] resuming from phase=%s round=%d experts=%d draft_len=%d",
            start_phase,
            start_round,
            len(approved_outputs),
            len(draft_text),
        )

    phase_order = [
        "planning",
        "experts",
        "pre_draft_review",
        "draft",
        "review",
        "improvers",
        "merge",
        "apply",
        "cleanup",
        "output",
    ]

    def _should_skip(phase: str) -> bool:
        if start_phase == "planning":
            return False
        try:
            return phase_order.index(phase) < phase_order.index(start_phase)
        except ValueError:
            return False

    async def _run_draft_review(round_no: int):
        return await reviewer.review_draft(
            model=review_model,
            query=query,
            draft_text=draft_text,
            budget=planning_budget,
            refinement_round=round_no,
            previous_summary=previous_merge_summary,
            temperature=(
                config.review_temperature
                if config.review_temperature is not None
                else 0.7
            ),
            top_p=top_p,
            user_system_prompt=system_prompt,
            image_parts=image_parts,
            provider=manager_provider,
            enable_json_repair=enable_json_repair,
            json_repair_model=json_repair_model,
        )

    try:
        if not _should_skip("planning"):
            await _emit(MSG_PIPELINE_START)
            await asyncio.sleep(2)

        if not _should_skip("experts"):
            _set_refinement_phase("planning")
            await _emit(MSG_REFINEMENT_PLANNING)

            expert_configs = await planner.plan(
                model=mgr_model,
                query=query,
                context=recent_history,
                budget=planning_budget,
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

            if not expert_configs:
                expert_configs = [
                    RefinementExpertConfig(
                        role="General Analyst",
                        domain="General request analysis",
                        prompt=query,
                        all_expert_roles=["General Analyst"],
                    )
                ]

            await _emit(
                f"Assigned {len(expert_configs)} experts: "
                f"{', '.join(e.role for e in expert_configs)}"
            )

            _set_refinement_phase("experts")
            approved_outputs = await _run_expert_batch(expert_configs)

            if not approved_outputs:
                approved_outputs = [{"role": "fallback", "domain": "", "content": query}]

            if resume_checkpoint:
                resume_checkpoint.refinement_expert_outputs = [
                    {
                        "role": item["role"],
                        "domain": item.get("domain", ""),
                        "content": item["content"],
                    }
                    for item in approved_outputs
                ]
                resume_checkpoint.updated_at = _now_ts()

        if pre_draft_review_rounds > 0 and not _should_skip("pre_draft_review"):
            _set_refinement_phase("pre_draft_review")

            pre_draft_experts = _outputs_to_expert_results(approved_outputs, round_no=1)
            if not pre_draft_experts:
                pre_draft_experts = _outputs_to_expert_results(
                    [{"role": "fallback", "domain": "", "content": query}],
                    round_no=1,
                )

            pre_draft_reviews: list[ReviewResult] = []
            pre_draft_round = 1

            while pre_draft_round <= pre_draft_review_rounds:
                await _emit(
                    MSG_REFINEMENT_PRE_DRAFT_REVIEW_START.format(round=pre_draft_round)
                )

                pre_review = await manager.review(
                    model=mgr_model,
                    query=query,
                    current_experts=pre_draft_experts,
                    budget=planning_budget,
                    context=recent_history,
                    temperature=(
                        config.review_temperature
                        if config.review_temperature is not None
                        else 0.7
                    ),
                    top_p=top_p,
                    user_system_prompt=system_prompt,
                    image_parts=image_parts,
                    remaining_rounds=max(pre_draft_review_rounds - pre_draft_round, 0),
                    previous_reviews=pre_draft_reviews,
                    provider=manager_provider,
                    json_via_prompt=config.json_via_prompt,
                )
                pre_review.round = pre_draft_round
                _upsert_review(pre_draft_reviews, pre_review)

                if pre_review.satisfied:
                    await _emit(MSG_REFINEMENT_PRE_DRAFT_REVIEW_APPROVED)
                    break

                if pre_review.overall_rejection_reason:
                    await _emit(
                        MSG_REFINEMENT_PRE_DRAFT_REVIEW_REJECTED_REASON.format(
                            reason=pre_review.overall_rejection_reason
                        )
                    )

                iterated_experts, action_notices = _apply_review_actions(
                    pre_review,
                    pre_draft_experts,
                )
                for notice in action_notices:
                    await _emit(notice)

                next_round_configs = list(pre_review.refined_experts)
                next_round_configs.extend(iterated_experts)
                if not next_round_configs:
                    await _emit(MSG_REFINEMENT_PRE_DRAFT_REVIEW_APPROVED)
                    break

                if pre_draft_round >= pre_draft_review_rounds:
                    break

                next_refinement_configs = _to_refinement_configs(next_round_configs)
                next_round_no = pre_draft_round + 1
                await _emit(
                    MSG_REFINEMENT_PRE_DRAFT_ROUND_ASSIGNED.format(
                        round=next_round_no,
                        count=len(next_refinement_configs),
                        names=", ".join(cfg.role for cfg in next_refinement_configs),
                    )
                )

                next_outputs = await _run_expert_batch(next_refinement_configs)
                if not next_outputs:
                    break

                pre_draft_experts.extend(
                    _outputs_to_expert_results(next_outputs, round_no=next_round_no)
                )
                pre_draft_round = next_round_no

                if pre_draft_round <= pre_draft_review_rounds:
                    await _emit(
                        MSG_REFINEMENT_PRE_DRAFT_NEXT_ROUND.format(
                            round=pre_draft_round
                        )
                    )

            approved_outputs = _collect_draft_inputs(
                pre_draft_experts,
                approved_outputs,
                query,
            )

            if resume_checkpoint:
                resume_checkpoint.refinement_expert_outputs = [
                    {
                        "role": item["role"],
                        "domain": item.get("domain", ""),
                        "content": item["content"],
                    }
                    for item in approved_outputs
                ]
                resume_checkpoint.updated_at = _now_ts()

        if not approved_outputs:
            approved_outputs = [{"role": "fallback", "domain": "", "content": query}]

        if not _should_skip("draft"):
            _set_refinement_phase("draft")
            await _emit(MSG_REFINEMENT_DRAFT_START)

            draft_text = await draft.generate_draft(
                model=draft_model,
                query=query,
                context=recent_history,
                expert_outputs=approved_outputs,
                budget=synthesis_budget,
                temperature=(
                    config.synthesis_temperature
                    if config.synthesis_temperature is not None
                    else temperature
                ),
                top_p=top_p,
                user_system_prompt=system_prompt,
                image_parts=image_parts,
                provider=expert_provider,
            )

            await _emit(MSG_REFINEMENT_DRAFT_DONE)
            await queue.put(("", f"\n```content\n{draft_text}\n```\n", "draft", []))

            if resume_checkpoint:
                resume_checkpoint.draft_content = draft_text
                resume_checkpoint.updated_at = _now_ts()

        global_op_id_counter = 0

        for refinement_round in range(start_round, max_refinement_rounds + 1):
            if resume_checkpoint:
                resume_checkpoint.refinement_round = refinement_round
                resume_checkpoint.updated_at = _now_ts()

            review_analysis = None
            all_operations: list[DiffOperation] = []
            merge_result = None

            if not (_should_skip("review") and refinement_round == start_round):
                _set_refinement_phase("review")
                await _emit(MSG_REFINEMENT_REVIEW_START.format(round=refinement_round))
                review_analysis = await _run_draft_review(refinement_round)

                if review_analysis.approved and refinement_round >= 2:
                    await _emit(MSG_REFINEMENT_REVIEW_APPROVED)
                    break

                if not review_analysis.refinement_experts:
                    await _emit(MSG_REFINEMENT_REVIEW_APPROVED)
                    break

                if review_analysis.issues:
                    issues_text = "\n".join(f"- {issue}" for issue in review_analysis.issues)
                    await _emit(f"Review issues:\n{issues_text}")

            if not (_should_skip("improvers") and refinement_round == start_round):
                _set_refinement_phase("improvers")

                if review_analysis is None:
                    review_analysis = await _run_draft_review(refinement_round)
                    if review_analysis.approved and not review_analysis.refinement_experts:
                        await _emit(MSG_REFINEMENT_REVIEW_APPROVED)
                        break

                draft_lines_json = json.dumps(
                    reviewer.split_draft_to_lines(draft_text),
                    ensure_ascii=False,
                )

                improver_configs = review_analysis.refinement_experts
                await _emit(
                    f"Assigned {len(improver_configs)} improvement experts: "
                    f"{', '.join(e.role for e in improver_configs)}"
                )

                for improver_cfg in improver_configs:
                    await _emit(
                        MSG_REFINEMENT_IMPROVER_START.format(
                            expert_name=improver_cfg.role,
                            domain=improver_cfg.domain,
                        )
                    )

                improver_tasks = [
                    improver.run_improver(
                        model=model,
                        expert_config=cfg,
                        draft_lines_json=draft_lines_json,
                        budget=expert_budget,
                        top_p=top_p,
                        guidance=review_analysis.expert_guidance.get(cfg.role, ""),
                        user_system_prompt=system_prompt,
                        image_parts=image_parts,
                        provider=expert_provider,
                        enable_json_repair=enable_json_repair,
                        json_repair_model=json_repair_model,
                    )
                    for cfg in improver_configs
                ]
                improver_results = await asyncio.gather(*improver_tasks)

                for result in improver_results:
                    await _emit(
                        MSG_REFINEMENT_IMPROVER_DONE.format(
                            expert_name=result.role,
                            op_count=len(result.operations),
                        )
                    )
                    if result.analysis:
                        await _emit(
                            f"[{result.role}] analysis:\n"
                            f"```content\n{result.analysis[:5000]}\n```"
                        )

                    for operation in result.operations:
                        operation.op_id = global_op_id_counter
                        operation.expert_role = result.role
                        all_operations.append(operation)
                        global_op_id_counter += 1

                if resume_checkpoint:
                    resume_checkpoint.refinement_improver_results = [
                        {
                            "role": result.role,
                            "analysis": result.analysis,
                            "operations": [
                                operation.model_dump(mode="json")
                                for operation in result.operations
                            ],
                        }
                        for result in improver_results
                    ]
                    resume_checkpoint.updated_at = _now_ts()

                if not all_operations:
                    await _emit("No improvement operations returned. Skipping merge.")
                    break

            if not (_should_skip("merge") and refinement_round == start_round):
                _set_refinement_phase("merge")
                await _emit(MSG_REFINEMENT_MERGE_START)

                merge_result = await merger.merge_operations(
                    model=merge_model,
                    draft_text=draft_text,
                    operations=all_operations,
                    budget=synthesis_budget,
                    temperature=(
                        config.synthesis_temperature
                        if config.synthesis_temperature is not None
                        else 0.5
                    ),
                    top_p=top_p,
                    provider=synthesis_provider,
                    enable_json_repair=enable_json_repair,
                    json_repair_model=json_repair_model,
                )

                accepted = sum(
                    1 for decision in merge_result.decisions if decision.decision == "accept"
                )
                rejected = sum(
                    1 for decision in merge_result.decisions if decision.decision == "reject"
                )
                modified = sum(
                    1 for decision in merge_result.decisions if decision.decision == "modify"
                )
                await _emit(
                    MSG_REFINEMENT_MERGE_DONE.format(
                        accepted=accepted,
                        rejected=rejected,
                        modified=modified,
                    )
                )
                if merge_result.summary:
                    await _emit(f"Merge summary:\n```content\n{merge_result.summary}\n```")

                previous_merge_summary = merge_result.summary
                if resume_checkpoint:
                    resume_checkpoint.refinement_merge_summary = previous_merge_summary
                    resume_checkpoint.updated_at = _now_ts()

            if not (_should_skip("apply") and refinement_round == start_round):
                _set_refinement_phase("apply")
                if merge_result is None:
                    break
                draft_text = applier.apply_refinements(
                    draft_text,
                    all_operations,
                    merge_result.decisions,
                )
                await _emit(MSG_REFINEMENT_APPLIED)

                if resume_checkpoint:
                    resume_checkpoint.draft_content = draft_text
                    resume_checkpoint.updated_at = _now_ts()

            remaining = max_refinement_rounds - refinement_round
            if remaining <= 0:
                break

            await _emit(MSG_REFINEMENT_NEXT_ROUND.format(round=refinement_round + 1))

        if config.enable_text_cleaner and not _should_skip("cleanup"):
            _set_refinement_phase("cleanup")
            await _emit(MSG_REFINEMENT_CLEAN_START)

            try:
                draft_lines = reviewer.split_draft_to_lines(draft_text)
                draft_lines_json = json.dumps(
                    draft_lines,
                    ensure_ascii=False,
                )
                clean_analysis, clean_ops = await cleaner.run_text_cleaner(
                    model=merge_model,
                    query=query,
                    draft_lines_json=draft_lines_json,
                    budget=synthesis_budget,
                    max_line=len(draft_lines),
                    top_p=top_p,
                    user_system_prompt=system_prompt,
                    provider=synthesis_provider,
                    json_via_prompt=config.json_via_prompt,
                )

                if clean_analysis:
                    await _emit(
                        f"[TextCleaner] analysis:\n"
                        f"```content\n{clean_analysis[:5000]}\n```"
                    )

                removed = sum(1 for op in clean_ops if op.action == "remove")
                modified = sum(1 for op in clean_ops if op.action == "modify")

                if clean_ops:
                    for idx, op in enumerate(clean_ops):
                        op.op_id = idx
                        op.expert_role = op.expert_role or "TextCleaner"

                    decisions = [
                        MergeDecision(op_id=op.op_id, decision="accept")
                        for op in clean_ops
                    ]

                    draft_text = applier.apply_refinements(
                        draft_text,
                        clean_ops,
                        decisions,
                    )

                    if resume_checkpoint:
                        resume_checkpoint.draft_content = draft_text
                        resume_checkpoint.updated_at = _now_ts()

                await _emit(
                    MSG_REFINEMENT_CLEAN_DONE.format(
                        removed=removed, modified=modified
                    )
                )

            except Exception as exc:
                logger.warning("[RefinementPipeline] text cleaner failed: %s", exc)
                await _emit(MSG_REFINEMENT_CLEAN_ERROR)

        _set_refinement_phase("output")
        await _emit(MSG_REFINEMENT_OUTPUT)

        chunk_size = 200
        for idx in range(0, len(draft_text), chunk_size):
            await queue.put((draft_text[idx : idx + chunk_size], "", "synthesis", []))
            await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        logger.info("[RefinementPipeline] cancelled")
        raise
    except Exception:
        logger.exception("[RefinementPipeline] failed")
        from prompts import REFINEMENT_FALLBACK_TEXT

        await queue.put((REFINEMENT_FALLBACK_TEXT, "", "system_error", []))
