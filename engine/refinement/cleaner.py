"""文本清洗专家（末端去相邻重复）模块。

在 refinement 流程末端对 draft_text 做一次保守的相邻重复检查。
模型输出仅允许 remove / modify 两类行级 diff 操作，由程序自动全量接收并应用。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from clients.llm_client import generate_json
from models import DiffOperation
from prompts import REFINEMENT_CLEANER_PROMPT

logger = logging.getLogger(__name__)


REFINEMENT_CLEANER_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "analysis": {"type": "STRING"},
        "operations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "action": {"type": "STRING"},
                    "line": {"type": "NUMBER"},
                    "content": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
                "required": ["action", "line"],
            },
        },
    },
    "required": ["analysis", "operations"],
}


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except Exception:
        return 0


def _normalize_cleaner_operation(
    op_raw: Any,
    *,
    max_line: int,
    expert_role: str,
) -> DiffOperation | None:
    if not isinstance(op_raw, dict):
        return None

    action = str(op_raw.get("action", "")).strip().lower()
    if action not in ("remove", "modify"):
        return None

    line = _coerce_int(op_raw.get("line", 0))
    if line <= 0 or line > max_line:
        return None

    reason = op_raw.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        reason = str(reason)

    if action == "remove":
        return DiffOperation(
            expert_role=expert_role,
            action="remove",
            line=line,
            reason=reason,
        )

    # modify
    if "content" not in op_raw:
        return None

    content = op_raw.get("content")
    if content is None:
        return None
    if not isinstance(content, str):
        content = str(content)
    if "\n" in content or "\r" in content:
        return None

    return DiffOperation(
        expert_role=expert_role,
        action="modify",
        line=line,
        content=content,
        reason=reason,
    )


def parse_cleaner_result(
    parsed: Any,
    *,
    max_line: int,
    expert_role: str = "TextCleaner",
) -> tuple[str, list[DiffOperation]]:
    """解析并校验清洗专家输出，返回 (analysis, operations)。"""
    if not isinstance(parsed, dict):
        return "", []

    analysis = parsed.get("analysis", "")
    if analysis is None:
        analysis = ""
    if not isinstance(analysis, str):
        analysis = str(analysis)

    ops_raw = parsed.get("operations", [])
    if not isinstance(ops_raw, list):
        ops_raw = []

    # 以 line 为 key 去重：remove > modify
    by_line: dict[int, DiffOperation] = {}
    for raw in ops_raw:
        op = _normalize_cleaner_operation(
            raw, max_line=max_line, expert_role=expert_role
        )
        if op is None:
            continue

        existing = by_line.get(op.line)
        if existing and existing.action == "remove":
            continue
        if op.action == "remove":
            by_line[op.line] = op
            continue

        by_line[op.line] = op

    operations = [by_line[k] for k in sorted(by_line)]
    return analysis.strip(), operations


async def run_text_cleaner(
    model: str,
    query: str,
    draft_lines_json: str,
    budget: int,
    *,
    max_line: int | None = None,
    user_system_prompt: str = "",
    provider: str = "",
    json_via_prompt: bool = False,
) -> tuple[str, list[DiffOperation]]:
    """运行末端文本清洗专家，返回 (analysis, operations)。

    Args:
        model: 清洗模型（默认复用 merge_model）。
        query: 用户原始需求。
        draft_lines_json: [{"line": 1, "text": "..."}, ...] 的 JSON 字符串。
        budget: thinking token 预算。
        user_system_prompt: 用户 system prompt（可选）。
        provider: provider 标识符。
        json_via_prompt: 是否通过 prompt 额外约束 JSON 输出（兼容渠道）。
    """
    resolved_max_line = max(0, int(max_line or 0))
    if resolved_max_line <= 0:
        try:
            draft_lines = json.loads(draft_lines_json)
            if isinstance(draft_lines, list):
                resolved_max_line = len(draft_lines)
        except Exception:
            resolved_max_line = 0

    sys_section = ""
    if user_system_prompt:
        sys_section = f"\n用户的重要指示：{user_system_prompt}\n"

    contents = (
        f'{sys_section}用户原始需求："{query}"\n\n'
        f"正文按行切分（JSON）：\n{draft_lines_json}"
    )

    try:
        parsed = await generate_json(
            model=model,
            contents=contents,
            system_instruction=REFINEMENT_CLEANER_PROMPT,
            response_schema=REFINEMENT_CLEANER_SCHEMA,
            thinking_budget=budget,
            temperature=0.2,
            provider=provider,
            json_via_prompt=json_via_prompt,
        )
    except Exception as exc:
        logger.warning("[Cleaner] generation failed: %s", exc)
        return "", []

    analysis, operations = parse_cleaner_result(parsed, max_line=resolved_max_line)
    logger.info("[Cleaner] parsed operations: %d", len(operations))
    return analysis, operations
