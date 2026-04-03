"""审查模块 (Reviewer).

将初稿按行切分, 分析存在的问题, 分配改进专家.
精修迭代时允许通过 (approved=true).
"""

import json
import logging
from typing import Any

from clients.llm_client import generate_content
from engine.refinement.json_repair import parse_json_with_repair
from models import RefinementExpertConfig, RefinementReviewAnalysis
from prompts import REFINEMENT_REVIEW_PROMPT

logger = logging.getLogger(__name__)


def split_draft_to_lines(draft_text: str) -> list[dict[str, Any]]:
    """将初稿按行切分为 JSON 数组格式.

    Args:
        draft_text: 初稿文本.

    Returns:
        [{"line": 1, "text": "..."}, ...] 格式的列表.
    """
    lines = draft_text.split("\n")
    return [{"line": i + 1, "text": line} for i, line in enumerate(lines)]


async def review_draft(
    model: str,
    query: str,
    draft_text: str,
    budget: int,
    refinement_round: int = 1,
    previous_summary: str = "",
    temperature: float | None = None,
    top_p: float | None = None,
    user_system_prompt: str = "",
    image_parts: list[dict] | None = None,
    provider: str = "",
    enable_json_repair: bool = False,
    json_repair_model: str = "",
) -> RefinementReviewAnalysis:
    """审查初稿, 分析问题并分配改进专家.

    Args:
        model: 审查模型.
        query: 用户原始问题.
        draft_text: 初稿文本.
        budget: thinking token 预算.
        refinement_round: 当前精修轮数.
        previous_summary: 上一轮综合助手的改动简评.
        temperature: 温度参数.
        user_system_prompt: 用户 system prompt.
        image_parts: 图片列表.
        provider: provider 标识符.
        enable_json_repair: 是否启用 JSON 修复.
        json_repair_model: JSON 修复模型.

    Returns:
        RefinementReviewAnalysis 审查结果.
    """
    lines_json = json.dumps(
        split_draft_to_lines(draft_text), ensure_ascii=False,
    )

    # 构建迭代备注
    iteration_note = ""
    if refinement_round > 1:
        iteration_note = (
            f"<Iteration_Info>\n"
            f"这是第 {refinement_round} 轮精修。\n"
            f"上一轮综合助手的改动简评：{previous_summary or '（无）'}\n"
            f"本轮你可以选择通过（approved=true，不分配改进专家）或继续精修。\n"
            f"</Iteration_Info>"
        )
    else:
        iteration_note = (
            "这是首轮审查，必须进行精修（approved 必须为 false），"
            "你需要分析问题并分配改进专家。"
        )

    sys_section = ""
    if user_system_prompt:
        sys_section = f"\n用户的重要指示：{user_system_prompt}\n"

    prompt = (
        f"{REFINEMENT_REVIEW_PROMPT.format(iteration_note=iteration_note)}\n\n"
        f"{sys_section}"
        f'用户原始需求："{query}"\n\n'
        f"初稿按行切分内容：\n{lines_json}"
    )

    try:
        raw_content, _, _ = await generate_content(
            model=model,
            contents=prompt,
            temperature=temperature or 0.7,
            top_p=top_p,
            thinking_budget=budget,
            image_parts=image_parts,
            provider=provider,
        )

        # 提取 JSON
        text = raw_content.strip()
        if text.startswith("```"):
            text_lines = text.split("\n")
            text_lines = text_lines[1:]
            if text_lines and text_lines[-1].strip() == "```":
                text_lines = text_lines[:-1]
            text = "\n".join(text_lines)

        parsed = await parse_json_with_repair(
            text,
            enable_repair=enable_json_repair,
            repair_model=json_repair_model,
            provider=provider,
            top_p=top_p,
        )

        # 解析专家配置
        experts_raw = parsed.get("refinement_experts", [])
        all_roles = [e.get("role", "") for e in experts_raw]
        guidance = parsed.get("expert_guidance", {})

        experts = []
        for e in experts_raw:
            cfg = RefinementExpertConfig(
                role=e["role"],
                domain=e.get("domain", ""),
                prompt=e.get("prompt", ""),
                temperature=e.get("temperature", 0.8),
                all_expert_roles=all_roles,
            )
            experts.append(cfg)

        result = RefinementReviewAnalysis(
            issues=parsed.get("issues", []),
            refinement_experts=experts,
            expert_guidance=guidance,
            approved=parsed.get("approved", False),
            approval_reason=parsed.get("approval_reason", ""),
        )

        logger.info(
            "[Reviewer] reviewed draft: approved=%s, issues=%d, experts=%d",
            result.approved, len(result.issues), len(result.refinement_experts),
        )
        return result

    except Exception as e:
        logger.error("[Reviewer] review failed: %s", e)
        # 审查失败时强制通过, 避免阻塞流程
        return RefinementReviewAnalysis(
            approved=True,
            approval_reason=f"审查失败, 强制通过: {e}",
        )
