"""Manager 模块：规划与审查.

迁移自原项目 manager.ts。
负责将用户问题分解为多个 Expert 任务，并在执行后审查质量。
"""

import json
import logging
from typing import Any

from clients.llm_client import generate_json
from models import AnalysisResult, ExpertResult, ReviewResult
from prompts import (
    MANAGER_SYSTEM_PROMPT,
    MANAGER_REVIEW_SYSTEM_PROMPT,
    ROUNDS_ENCOURAGEMENT,
    build_prefill_contents,
)

logger = logging.getLogger(__name__)

# Gemini 的 Schema 定义（对应原项目 Type.OBJECT 等）
ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "thought_process": {
            "type": "STRING",
            "description": (
                "Brief explanation of why these supplementary "
                "experts were chosen."
            ),
        },
        "experts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "role": {"type": "STRING"},
                    "description": {"type": "STRING"},
                    "temperature": {"type": "NUMBER"},
                    "prompt": {"type": "STRING"},
                },
                "required": ["role", "description", "temperature", "prompt"],
            },
        },
    },
    "required": ["thought_process", "experts"],
}

_EXPERT_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "role": {"type": "STRING"},
        "description": {"type": "STRING"},
        "temperature": {"type": "NUMBER"},
        "prompt": {"type": "STRING"},
    },
    "required": ["role", "description", "temperature", "prompt"],
}

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "satisfied": {
            "type": "BOOLEAN",
            "description": (
                "True if the experts have fully answered the query "
                "with high quality."
            ),
        },
        "review_critique": {
            "type": "STRING",
            "description": (
                "审查简评。对每个专家进行简评，给出4个等级（很满意、还不错、平庸、不满意）并简短叙述原因。"
            ),
        },
        "overall_rejection_reason": {
            "type": "STRING",
            "description": (
                "整体驳回理由。简短的描述为何不满意，如满意可不填。"
            ),
        },
        "critique": {
            "type": "STRING",
            "description": (
                "If not satisfied, explain why and what is missing."
            ),
        },
        "next_round_strategy": {
            "type": "STRING",
            "description": "Plan for the next iteration.",
        },
        "refined_experts": {
            "type": "ARRAY",
            "description": (
                "The list of experts for the next round. "
                "Can be the same roles or new ones."
            ),
            "items": _EXPERT_CONFIG_SCHEMA,
        },
        "expert_actions": {
            "type": "ARRAY",
            "description": (
                "Per-expert actions in this review round. "
                "Each item should choose keep/iterate/delete."
            ),
            "items": {
                "type": "OBJECT",
                "properties": {
                    "target_expert_id": {"type": "STRING"},
                    "target_expert_role": {"type": "STRING"},
                    "action": {
                        "type": "STRING",
                        "description": "One of keep / iterate / delete.",
                    },
                    "reason": {
                        "type": "STRING",
                        "description": (
                            "Decision reason. For delete, must include the expert "
                            "research direction, fatal mistake, and why removal is needed."
                        ),
                    },
                    "strict_prompt": {
                        "type": "STRING",
                        "description": (
                            "Required for iterate. Harsh directive for next-round fix."
                        ),
                    },
                    "improvement_suggestions": {
                        "type": "STRING",
                        "description": (
                            "Required for iterate. Concrete improvement points."
                        ),
                    },
                    "iterated_expert": _EXPERT_CONFIG_SCHEMA,
                },
                "required": [
                    "target_expert_id",
                    "target_expert_role",
                    "action",
                    "reason",
                ],
            },
        },
    },
    "required": [
        "satisfied",
        "review_critique",
        "overall_rejection_reason",
        "critique",
        "expert_actions",
    ],
}


def _normalize_action_name(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in ("delete", "remove", "drop"):
        return "delete"
    if raw in ("iterate", "iteration", "refine", "improve"):
        return "iterate"
    if "删" in raw or "移除" in raw:
        return "delete"
    if "迭代" in raw or "改进" in raw:
        return "iterate"
    return "keep"


def _normalize_review_actions(raw_actions: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_actions, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw_actions:
        if not isinstance(item, dict):
            continue
        cleaned = dict(item)
        cleaned["action"] = _normalize_action_name(item.get("action"))
        normalized.append(cleaned)
    return normalized


def _truncate_expert_content(content: str, max_len: int = 20000) -> str:
    if len(content) <= max_len:
        return content
    return (
        f"Output (因篇幅已截取前 {max_len} 字符，"
        f"专家实际输出共 {len(content)} 字符，请勿将截取视为回答不完整):\n"
        f"{content[:max_len]}"
    )


def _render_expert_node(expert: ExpertResult) -> str:
    content = _truncate_expert_content(expert.content or "（无输出）")
    dimension = (expert.usage_dimension or "").strip()
    dimension_part = (
        f"  <Usage_Dimension>{dimension}</Usage_Dimension>\n"
        if dimension else ""
    )
    return (
        f'  <Expert id="{expert.id}" name="{expert.role}" '
        f'context_status="{expert.context_status}">\n'
        f"{dimension_part}"
        f"{content}\n"
        "  </Expert>"
    )


async def analyze(
    model: str,
    query: str,
    context: str,
    budget: int,
    temperature: float | None = None,
    top_p: float | None = None,
    user_system_prompt: str = "",
    image_parts: list[dict] | None = None,
    provider: str = "",
    json_via_prompt: bool = False,
) -> AnalysisResult:
    """Manager 规划阶段：分析用户问题，分解为 Expert 任务.

    Args:
        model: 模型标识符.
        query: 用户当前问题.
        context: 最近对话上下文.
        budget: thinking token 预算.
        user_system_prompt: 下游客户端的 system prompt.
        image_parts: Gemini inlineData 格式的图片列表.

    Returns:
        AnalysisResult 包含 thought_process 和 experts 列表.
    """
    sys_section = (
        f"\n用户的重要指示：{user_system_prompt}" if user_system_prompt else ""
    )
    text_prompt = f'Context:\n{context}{sys_section}\n\nCurrent Query: "{query}"'
    contents = build_prefill_contents(
        text_prompt,
        image_parts=image_parts,
        leading_instruction=MANAGER_SYSTEM_PROMPT,
    )

    try:
        result = await generate_json(
            model=model,
            contents=contents,
            system_instruction="",
            response_schema=ANALYSIS_SCHEMA,
            thinking_budget=budget,
            temperature=temperature,
            top_p=top_p,
            image_parts=None,
            provider=provider,
            json_via_prompt=json_via_prompt,
        )
        logger.debug(
            "[Manager] analyze raw response:\n%s",
            json.dumps(result, ensure_ascii=False, indent=2),
        )

        analysis = AnalysisResult(**result)

        if not analysis.experts:
            logger.warning("[Manager] analyze returned an empty experts list")

        logger.info(
            "[Manager] analyze completed: %d experts, thought: %s",
            len(analysis.experts),
            analysis.thought_process[:2000],
        )
        for exp in analysis.experts:
            logger.debug(
                "[Manager]   Expert: %s | temp=%.1f | prompt=%s",
                exp.role, exp.temperature, exp.prompt[:1000],
            )
        return analysis

    except Exception as e:
        logger.error("[Manager] analyze failed: %s", e)
        return AnalysisResult(
            thought_process=(
                f"Direct processing fallback due to analysis error: {e}"
            ),
            experts=[],
        )


async def review(
    model: str,
    query: str,
    current_experts: list[ExpertResult],
    budget: int,
    context: str = "",
    temperature: float | None = None,
    top_p: float | None = None,
    user_system_prompt: str = "",
    image_parts: list[dict] | None = None,
    remaining_rounds: int = 0,
    previous_reviews: list[ReviewResult] | None = None,
    provider: str = "",
    json_via_prompt: bool = False,
) -> ReviewResult:
    """Manager 审查阶段：评估 Expert 输出质量.

    Args:
        model: 模型标识符.
        query: 用户原始问题.
        current_experts: 所有 Expert 的执行结果.
        budget: thinking token 预算.
        context: 对话历史上下文.
        user_system_prompt: 下游客户端的 system prompt.
        image_parts: Gemini inlineData 格式的图片列表.

    Returns:
        ReviewResult 包含 satisfied 状态和可能的下一轮 experts.
    """
    if previous_reviews is None:
        previous_reviews = []
        
    experts_by_round: dict[int, list[ExpertResult]] = {}
    for e in current_experts:
        experts_by_round.setdefault(e.round, []).append(e)

    reviews_by_round: dict[int, ReviewResult] = {r.round: r for r in previous_reviews}
    
    rounds_str_parts = []
    for r in sorted(experts_by_round.keys()):
        parts = []
        review_obj = reviews_by_round.get(r)
        
        status = "Unreviewed"
        if review_obj:
            status = "Approved" if review_obj.satisfied else "Rejected"
            
        parts.append(f'<Round id="{r}" status="{status}">')
        for e in experts_by_round[r]:
            parts.append(_render_expert_node(e))
        
        if review_obj and status != "Unreviewed":
            parts.append("  <Review_Critique>")
            parts.append(f"{review_obj.review_critique}")
            if status == "Rejected" and review_obj.overall_rejection_reason:
                parts.append(f"审查驳回理由：{review_obj.overall_rejection_reason}")
            if review_obj.expert_actions:
                parts.append("审查动作记录：")
                for action in review_obj.expert_actions:
                    target = (
                        action.target_expert_role
                        or action.target_expert_id
                        or "未知专家"
                    )
                    parts.append(
                        f"- {target}: {action.action} | 原因: {action.reason or '（未提供）'}"
                    )
            parts.append("  </Review_Critique>")
            
        parts.append("</Round>")
        rounds_str_parts.append("\n".join(parts))

    expert_outputs = "\n\n".join(rounds_str_parts)
    context_section = f"对话上下文：\n{context}\n\n" if context else ""
    sys_section = (
        f"用户的重要指示：{user_system_prompt}\n\n"
        if user_system_prompt else ""
    )

    rounds_encouragement = ""
    if remaining_rounds > 0:
        rounds_encouragement = ROUNDS_ENCOURAGEMENT.format(
            remaining_rounds=remaining_rounds
        )

    content = (
        f"{context_section}"
        f"{sys_section}"
        f"{rounds_encouragement}"
        f'用户查询："{query}"\n\n'
        f"当前专家输出：\n{expert_outputs}"
    )
    contents = build_prefill_contents(
        content,
        image_parts=image_parts,
        leading_instruction=MANAGER_REVIEW_SYSTEM_PROMPT,
    )

    try:
        result = await generate_json(
            model=model,
            contents=contents,
            system_instruction="",
            response_schema=REVIEW_SCHEMA,
            thinking_budget=budget,
            temperature=temperature,
            top_p=top_p,
            image_parts=None,
            provider=provider,
            json_via_prompt=json_via_prompt,
        )

        logger.debug(
            "[Manager] review raw response:\n%s",
            json.dumps(result, ensure_ascii=False, indent=2),
        )

        normalized = dict(result)
        normalized["expert_actions"] = _normalize_review_actions(
            result.get("expert_actions", [])
        )

        review_result = ReviewResult(**normalized)
        logger.info(
            "[Manager] review completed: satisfied=%s, critique=%s",
            review_result.satisfied,
            review_result.critique[:1000],
        )
        return review_result

    except Exception as e:
        logger.error("[Manager] review failed: %s", e)
        return ReviewResult(
            satisfied=True,
            critique="Processing Error, proceeding to synthesis.",
        )
