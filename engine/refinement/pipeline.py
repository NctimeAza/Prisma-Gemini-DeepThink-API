"""精修主流水线.

编排所有精修阶段: 规划 -> 专家执行 -> 规范审核 -> 初稿 -> 审查 ->
改进专家 -> 综合合并 -> 应用精修 -> 迭代或输出.
通过 asyncio.Queue 推送状态和最终输出, 与 orchestrator 对接.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from config import get_thinking_budget, LLM_NETWORK_RETRIES
from engine.refinement import (
    applier,
    compliance,
    draft,
    improver,
    merger,
    planner,
    reviewer,
)
from models import (
    DeepThinkCheckpoint,
    DeepThinkConfig,
    DiffOperation,
    RefinementExpertConfig,
)
from prompts import (
    MSG_REFINEMENT_APPLIED,
    MSG_REFINEMENT_COMPLIANCE_CHECK,
    MSG_REFINEMENT_COMPLIANCE_FAILED,
    MSG_REFINEMENT_COMPLIANCE_PASSED,
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
    MSG_REFINEMENT_REVIEW_APPROVED,
    MSG_REFINEMENT_REVIEW_START,
    MSG_PIPELINE_START,
    build_refinement_expert_contents,
    format_expert_task,
    get_refinement_expert_system_instruction,
)
from clients.llm_client import generate_content
from utils.retry import extract_status, is_retryable_error

logger = logging.getLogger(__name__)


def _now_ts() -> int:
    return int(time.time())


async def _run_single_expert(
    model: str,
    expert_cfg: RefinementExpertConfig,
    query: str,
    context: str,
    budget: int,
    user_system_prompt: str = "",
    image_parts: list[dict] | None = None,
    provider: str = "",
    forced_temperature: float | None = None,
) -> dict[str, str]:
    """执行单个精修专家, 返回 {role, domain, content}.

    使用 prefill 确认轮提高执行力.
    """
    system_instruction = get_refinement_expert_system_instruction(
        role=expert_cfg.role,
        domain=expert_cfg.domain,
        context=context,
        all_expert_roles=expert_cfg.all_expert_roles,
        user_system_prompt=user_system_prompt,
    )

    task_prompt = format_expert_task(query, expert_cfg.prompt)
    contents = build_refinement_expert_contents(
        task_prompt, image_parts=image_parts,
    )

    temperature = (
        forced_temperature
        if forced_temperature is not None
        else expert_cfg.temperature
    )

    max_retries = LLM_NETWORK_RETRIES

    for attempt in range(max_retries + 1):
        try:
            full_content, _, _ = await generate_content(
                model=model,
                contents=contents,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking_budget=budget,
                provider=provider,
            )

            if not full_content.strip():
                if attempt < max_retries:
                    delay = 1.5 * (attempt + 1)
                    logger.warning(
                        "[RefinementExpert] %s empty response, retry %d/%d",
                        expert_cfg.role, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(delay)
                    continue
                full_content = "（专家未生成有效内容）"

            return {
                "role": expert_cfg.role,
                "domain": expert_cfg.domain,
                "content": full_content,
            }

        except Exception as e:
            status = extract_status(e)
            retryable = is_retryable_error(status)
            if retryable and attempt < max_retries:
                delay = 1.5 * (attempt + 1)
                logger.warning(
                    "[RefinementExpert] %s error (status=%s), retry %d/%d: %s",
                    expert_cfg.role, status, attempt + 1, max_retries, e,
                )
                await asyncio.sleep(delay)
                continue

            logger.error(
                "[RefinementExpert] %s failed: %s", expert_cfg.role, e,
            )
            return {
                "role": expert_cfg.role,
                "domain": expert_cfg.domain,
                "content": f"（专家执行失败: {e}）",
            }

    return {
        "role": expert_cfg.role,
        "domain": expert_cfg.domain,
        "content": "（专家重试次数耗尽）",
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
    system_prompt: str = "",
    image_parts: list[dict] | None = None,
    resume_checkpoint: DeepThinkCheckpoint | None = None,
    provider: str = "",
) -> None:
    """精修流水线主入口.

    Args:
        queue: 输出 queue, 推送 (text, thought, phase, grounding) 元组.
        query: 用户原始问题.
        history: 对话历史.
        model: Expert 模型.
        mgr_model: Manager/规划 模型.
        syn_model: Synthesis 模型.
        config: 配置参数.
        temperature: 默认温度.
        system_prompt: 用户 system prompt.
        image_parts: 图片列表.
        resume_checkpoint: 断点恢复数据.
        provider: provider 标识符.
    """

    async def _emit(text: str) -> None:
        """推送思维链状态文本."""
        await queue.put(("", f"{text}\n", "refinement", []))

    # 计算各阶段预算
    planning_budget = get_thinking_budget(config.planning_level, model)
    expert_budget = get_thinking_budget(config.expert_level, model)
    synthesis_budget = get_thinking_budget(config.synthesis_level, model)

    # 获取精修专用配置
    compliance_model = config.compliance_model or "gemini-3-flash-preview"
    draft_model = config.draft_model or model
    review_model = config.review_model or mgr_model
    merge_model = config.merge_model or syn_model
    json_repair_model = config.json_repair_model or "gemini-3-flash-preview"
    max_refinement_rounds = config.refinement_max_rounds
    max_compliance_retries = config.compliance_check_max_retries
    enable_json_repair = config.enable_json_repair

    # 对话上下文
    max_ctx = config.max_context_messages
    recent_history = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-max_ctx:]
    )

    try:
        # ========== 阶段 1: 规划 ==========
        await _emit(MSG_PIPELINE_START)
        await asyncio.sleep(2)
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
            user_system_prompt=system_prompt,
            image_parts=image_parts,
            provider=provider,
        )

        if not expert_configs:
            # 兜底: 分配一个通用专家
            expert_configs = [RefinementExpertConfig(
                role="通用分析专家",
                domain="全面分析用户需求",
                prompt=query,
                all_expert_roles=["通用分析专家"],
            )]

        expert_names = "、".join(e.role for e in expert_configs)
        await _emit(f"已分配 {len(expert_configs)} 位专家：{expert_names}")

        # ========== 阶段 2: 专家并行执行 ==========
        for ec in expert_configs:
            await _emit(
                MSG_REFINEMENT_EXPERT_START.format(
                    expert_name=ec.role, domain=ec.domain,
                )
            )

        forced_expert_temp = config.expert_temperature

        expert_tasks = [
            _run_single_expert(
                model=model,
                expert_cfg=ec,
                query=query,
                context=recent_history,
                budget=expert_budget,
                user_system_prompt=system_prompt,
                image_parts=image_parts,
                provider=provider,
                forced_temperature=forced_expert_temp,
            )
            for ec in expert_configs
        ]
        expert_outputs = await asyncio.gather(*expert_tasks)

        for eo in expert_outputs:
            await _emit(MSG_REFINEMENT_EXPERT_DONE.format(expert_name=eo["role"]))
            if eo["content"]:
                await queue.put(
                    ("", f"```content\n{eo['content']}\n```\n", "experts", [])
                )

        # ========== 阶段 3: 基本规范审核 ==========
        approved_outputs: list[dict[str, str]] = []
        for idx, eo in enumerate(expert_outputs):
            ec = expert_configs[idx] if idx < len(expert_configs) else None

            await _emit(
                MSG_REFINEMENT_COMPLIANCE_CHECK.format(expert_name=eo["role"])
            )

            passed = False
            for retry in range(max_compliance_retries + 1):
                check_result = await compliance.check_compliance(
                    content=eo["content"],
                    role=eo["role"],
                    domain=eo.get("domain", ""),
                    task=ec.prompt if ec else query,
                    model=compliance_model,
                    provider=provider,
                    enable_json_repair=enable_json_repair,
                    json_repair_model=json_repair_model,
                )

                if check_result.passed:
                    passed = True
                    await _emit(
                        MSG_REFINEMENT_COMPLIANCE_PASSED.format(
                            expert_name=eo["role"],
                        )
                    )
                    break

                if retry < max_compliance_retries:
                    await _emit(
                        MSG_REFINEMENT_COMPLIANCE_FAILED.format(
                            expert_name=eo["role"],
                            reason=check_result.reason[:200],
                        )
                    )
                    # 重新生成专家输出
                    if ec:
                        eo_new = await _run_single_expert(
                            model=model,
                            expert_cfg=ec,
                            query=query,
                            context=recent_history,
                            budget=expert_budget,
                            user_system_prompt=system_prompt,
                            image_parts=image_parts,
                            provider=provider,
                            forced_temperature=forced_expert_temp,
                        )
                        eo["content"] = eo_new["content"]
                else:
                    # 超过重试次数, 强制放行
                    passed = True
                    await _emit(
                        MSG_REFINEMENT_COMPLIANCE_PASSED.format(
                            expert_name=eo["role"],
                        )
                    )

            if passed:
                approved_outputs.append(eo)

        if not approved_outputs:
            approved_outputs = list(expert_outputs)

        # ========== 阶段 4: 初稿生成 ==========
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
            user_system_prompt=system_prompt,
            image_parts=image_parts,
            provider=provider,
        )

        await _emit(MSG_REFINEMENT_DRAFT_DONE)
        await queue.put(("", f"```content\n{draft_text}\n```\n", "draft", []))

        # 保存初稿到 checkpoint
        if resume_checkpoint:
            resume_checkpoint.draft_content = draft_text
            resume_checkpoint.updated_at = _now_ts()

        # ========== 阶段 5-8: 精修迭代循环 ==========
        previous_merge_summary = ""
        global_op_id_counter = 0

        for refinement_round in range(1, max_refinement_rounds + 1):
            # 更新 checkpoint
            if resume_checkpoint:
                resume_checkpoint.refinement_round = refinement_round
                resume_checkpoint.updated_at = _now_ts()

            # --- 5. 审查 ---
            await _emit(
                MSG_REFINEMENT_REVIEW_START.format(round=refinement_round)
            )

            review_analysis = await reviewer.review_draft(
                model=review_model,
                query=query,
                draft_text=draft_text,
                budget=planning_budget,
                refinement_round=refinement_round,
                previous_summary=previous_merge_summary,
                temperature=(
                    config.review_temperature
                    if config.review_temperature is not None
                    else 0.7
                ),
                user_system_prompt=system_prompt,
                image_parts=image_parts,
                provider=provider,
                enable_json_repair=enable_json_repair,
                json_repair_model=json_repair_model,
            )

            # 迭代轮 (>= 2) 允许通过
            if review_analysis.approved and refinement_round >= 2:
                await _emit(MSG_REFINEMENT_REVIEW_APPROVED)
                break

            if not review_analysis.refinement_experts:
                await _emit(MSG_REFINEMENT_REVIEW_APPROVED)
                break

            # 输出审查发现的问题
            if review_analysis.issues:
                issues_text = "\n".join(
                    f"- {issue}" for issue in review_analysis.issues
                )
                await _emit(f"审查发现的问题：\n{issues_text}")

            # --- 6. 改进专家并行执行 ---
            draft_lines_json = json.dumps(
                reviewer.split_draft_to_lines(draft_text),
                ensure_ascii=False,
            )

            improver_configs = review_analysis.refinement_experts
            improver_names = "、".join(e.role for e in improver_configs)
            await _emit(f"已分配 {len(improver_configs)} 位改进专家：{improver_names}")

            for ic in improver_configs:
                await _emit(
                    MSG_REFINEMENT_IMPROVER_START.format(
                        expert_name=ic.role, domain=ic.domain,
                    )
                )

            improver_tasks = [
                improver.run_improver(
                    model=model,
                    expert_config=ic,
                    draft_lines_json=draft_lines_json,
                    budget=expert_budget,
                    guidance=review_analysis.expert_guidance.get(ic.role, ""),
                    user_system_prompt=system_prompt,
                    image_parts=image_parts,
                    provider=provider,
                    enable_json_repair=enable_json_repair,
                    json_repair_model=json_repair_model,
                )
                for ic in improver_configs
            ]
            improver_results = await asyncio.gather(*improver_tasks)

            # 合并所有操作并分配全局 op_id
            all_operations: list[DiffOperation] = []
            for ir in improver_results:
                await _emit(
                    MSG_REFINEMENT_IMPROVER_DONE.format(
                        expert_name=ir.role,
                        op_count=len(ir.operations),
                    )
                )
                # 推送改进专家的分析到思维链
                if ir.analysis:
                    await _emit(f"「{ir.role}」分析：{ir.analysis[:500]}")

                for op in ir.operations:
                    op.op_id = global_op_id_counter
                    op.expert_role = ir.role
                    all_operations.append(op)
                    global_op_id_counter += 1

            if not all_operations:
                await _emit("改进专家未提交任何操作, 跳过合并。")
                break

            # --- 7. 综合助手合并 ---
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
                provider=provider,
                enable_json_repair=enable_json_repair,
                json_repair_model=json_repair_model,
            )

            accepted = sum(
                1 for d in merge_result.decisions if d.decision == "accept"
            )
            rejected = sum(
                1 for d in merge_result.decisions if d.decision == "reject"
            )
            modified = sum(
                1 for d in merge_result.decisions if d.decision == "modify"
            )
            await _emit(
                MSG_REFINEMENT_MERGE_DONE.format(
                    accepted=accepted, rejected=rejected, modified=modified,
                )
            )
            if merge_result.summary:
                await _emit(f"综合简评：{merge_result.summary}")

            previous_merge_summary = merge_result.summary

            # --- 8. 应用精修 ---
            draft_text = applier.apply_refinements(
                draft_text, all_operations, merge_result.decisions,
            )
            await _emit(MSG_REFINEMENT_APPLIED)

            # 更新 checkpoint
            if resume_checkpoint:
                resume_checkpoint.draft_content = draft_text
                resume_checkpoint.updated_at = _now_ts()

            # 判断是否继续迭代
            remaining = max_refinement_rounds - refinement_round
            if remaining <= 0:
                break

            await _emit(
                MSG_REFINEMENT_NEXT_ROUND.format(round=refinement_round + 1)
            )

        # ========== 阶段 9: 输出最终结果 ==========
        await _emit(MSG_REFINEMENT_OUTPUT)

        # 流式输出精修后的最终文本
        # 分块推送, 模拟流式效果
        chunk_size = 200
        for i in range(0, len(draft_text), chunk_size):
            chunk = draft_text[i:i + chunk_size]
            await queue.put((chunk, "", "synthesis", []))
            await asyncio.sleep(0.01)  # 微小延迟让 SSE 更流畅

    except asyncio.CancelledError:
        logger.info("[RefinementPipeline] cancelled")
        raise
    except Exception as exc:
        logger.exception("[RefinementPipeline] failed")
        await queue.put(("精修流程出错, 请重试。", "", "system_error", []))
