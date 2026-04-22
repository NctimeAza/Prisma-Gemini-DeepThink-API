"""强力杀八股模块。

对专家文本或草稿执行行级八股清洗，只允许 remove / modify。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from clients.llm_client import generate_json
from models import DiffOperation, MergeDecision
from prompts import (
    REFINEMENT_NO_CLICHES_CLICHE_LIST,
    REFINEMENT_NO_CLICHES_PERSONA,
    REFINEMENT_NO_CLICHES_PROMPT,
    build_prefill_contents,
)

from engine.refinement import applier
from engine.refinement.cleaner import parse_cleaner_result

logger = logging.getLogger(__name__)


NO_CLICHES_SCHEMA: dict[str, Any] = {
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


def _split_text_to_lines_json(text: str) -> str:
    """将文本按行切分为 JSON。"""
    lines = text.split("\n")
    payload = [{"line": idx + 1, "text": line} for idx, line in enumerate(lines)]
    return json.dumps(payload, ensure_ascii=False)


def apply_no_cliches_operations(
    text: str,
    operations: list[DiffOperation],
) -> str:
    """应用八股清洗操作。"""
    if not operations:
        return text

    for idx, op in enumerate(operations):
        op.op_id = idx

    decisions = [
        MergeDecision(op_id=op.op_id, decision="accept")
        for op in operations
    ]
    return applier.apply_refinements(text, operations, decisions)


async def run_no_cliches(
    model: str,
    text: str,
    budget: int,
    *,
    provider: str = "",
    temperature: float = 0.2,
    top_p: float | None = None,
    json_via_prompt: bool = False,
    expert_role: str = "NoCliches",
) -> tuple[str, list[DiffOperation], str]:
    """执行强力杀八股。

    Args:
        model: 杀八股模型。
        text: 待清洗文本。
        budget: thinking token 预算。
        provider: provider 标识符。
        temperature: 采样温度。
        top_p: 采样参数。
        json_via_prompt: 是否启用 prompt 级 JSON 约束。
        expert_role: 记录到 DiffOperation 的角色名。

    Returns:
        (analysis, operations, cleaned_text)。
    """
    if not text.strip():
        return "", [], text

    lines_json = _split_text_to_lines_json(text)
    instruction = (
        f"{REFINEMENT_NO_CLICHES_PERSONA}\n\n"
        f"{REFINEMENT_NO_CLICHES_PROMPT}\n\n"
        f"八股表：\n{REFINEMENT_NO_CLICHES_CLICHE_LIST}"
    )
    contents = build_prefill_contents(
        f"待清洗文本（JSON）：\n{lines_json}",
        leading_instruction=instruction,
    )

    debug_info: dict[str, Any] = {}
    parsed = await generate_json(
        model=model,
        contents=contents,
        system_instruction="",
        response_schema=NO_CLICHES_SCHEMA,
        thinking_budget=budget,
        temperature=temperature,
        top_p=top_p,
        provider=provider,
        json_via_prompt=json_via_prompt,
        debug_info=debug_info,
    )

    analysis, operations = parse_cleaner_result(
        parsed,
        max_line=max(1, text.count("\n") + 1),
        expert_role=expert_role,
    )
    cleaned_text = apply_no_cliches_operations(text, operations)

    logger.info(
        "[NoCliches] parsed operations: %d (provider=%s, model=%s)",
        len(operations),
        provider,
        model,
    )
    return analysis, operations, cleaned_text
