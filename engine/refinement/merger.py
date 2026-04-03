"""综合助手 (Merger) 模块.

接收初稿和所有改进专家的 diff 操作 (带全局 op_id),
对每个操作做 accept/reject/modify 决策, 输出总体改动简评.
"""

import json
import logging

from clients.llm_client import generate_content
from engine.refinement.json_repair import parse_json_with_repair
from models import DiffOperation, MergeDecision, MergeResult
from prompts import REFINEMENT_MERGE_PROMPT

logger = logging.getLogger(__name__)


def _format_operations_for_merge(
    draft_text: str,
    operations: list[DiffOperation],
) -> str:
    """格式化初稿和操作列表供综合助手审阅.

    Args:
        draft_text: 初稿原文.
        operations: 带 op_id 的 diff 操作列表.

    Returns:
        格式化后的文本.
    """
    ops_list = []
    for op in operations:
        op_dict = {
            "op_id": op.op_id,
            "expert_role": op.expert_role,
            "action": op.action,
            "line": op.line,
            "reason": op.reason,
        }
        if op.action in ("modify", "add"):
            op_dict["content"] = op.content
        ops_list.append(op_dict)

    ops_json = json.dumps(ops_list, ensure_ascii=False, indent=2)

    return (
        f"初稿原文：\n```\n{draft_text}\n```\n\n"
        f"改进专家提交的操作列表：\n{ops_json}"
    )


async def merge_operations(
    model: str,
    draft_text: str,
    operations: list[DiffOperation],
    budget: int,
    temperature: float | None = None,
    top_p: float | None = None,
    provider: str = "",
    enable_json_repair: bool = False,
    json_repair_model: str = "",
) -> MergeResult:
    """综合助手合并改进操作.

    Args:
        model: 综合助手模型.
        draft_text: 初稿原文.
        operations: 所有改进专家的 diff 操作.
        budget: thinking token 预算.
        temperature: 温度参数.
        provider: provider 标识符.
        enable_json_repair: 是否启用 JSON 修复.
        json_repair_model: JSON 修复模型.

    Returns:
        MergeResult 合并结果.
    """
    content_prompt = _format_operations_for_merge(draft_text, operations)
    prompt = f"{REFINEMENT_MERGE_PROMPT}\n\n{content_prompt}"

    try:
        raw_content, _, _ = await generate_content(
            model=model,
            contents=prompt,
            temperature=temperature or 0.5,
            top_p=top_p,
            thinking_budget=budget,
            provider=provider,
        )

        # 提取 JSON
        text = raw_content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        parsed = await parse_json_with_repair(
            text,
            enable_repair=enable_json_repair,
            repair_model=json_repair_model,
            provider=provider,
            top_p=top_p,
        )

        decisions = []
        for d in parsed.get("decisions", []):
            decision = d.get("decision", "").strip().lower()
            if decision not in ("accept", "reject", "modify"):
                decision = "accept"  # 默认接受
            decisions.append(MergeDecision(
                op_id=d.get("op_id", 0),
                decision=decision,
                reason=d.get("reason", ""),
                modified_line=d.get("modified_line"),
                modified_content=d.get("modified_content"),
            ))

        result = MergeResult(
            decisions=decisions,
            summary=parsed.get("summary", ""),
        )

        accepted = sum(1 for d in decisions if d.decision == "accept")
        rejected = sum(1 for d in decisions if d.decision == "reject")
        modified = sum(1 for d in decisions if d.decision == "modify")

        logger.info(
            "[Merger] merge completed: %d accepted, %d rejected, %d modified",
            accepted, rejected, modified,
        )
        return result

    except Exception as e:
        logger.error("[Merger] merge failed: %s", e)
        # 合并失败时全部接受
        return MergeResult(
            decisions=[
                MergeDecision(op_id=op.op_id, decision="accept")
                for op in operations
            ],
            summary=f"合并失败 ({e}), 全部操作已自动接受。",
        )
