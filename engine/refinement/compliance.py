"""基本规范审核模块.

用小模型对每个专家的输出进行基本规范检查:
有无无意义寒暄, 是否直入主题, 是否做到了prompt要求的事.
第一轮不通过可重试, 超过重试次数后强制放行.
"""

import json
import logging

from clients.llm_client import generate_content
from engine.refinement.json_repair import parse_json_with_repair
from models import ComplianceCheckResult
from prompts import COMPLIANCE_CHECK_PROMPT

logger = logging.getLogger(__name__)


async def check_compliance(
    content: str,
    role: str,
    domain: str,
    task: str,
    *,
    model: str,
    provider: str = "",
    thinking_budget: int = 1024,
    enable_json_repair: bool = False,
    json_repair_model: str = "",
) -> ComplianceCheckResult:
    """对单个专家输出进行基本规范审核.

    Args:
        content: 专家回复内容.
        role: 专家角色名.
        domain: 专家负责领域.
        task: 专家原始任务.
        model: 审核用模型.
        provider: provider 标识符.
        thinking_budget: thinking token 预算.
        enable_json_repair: 是否启用 JSON 修复.
        json_repair_model: JSON 修复模型.

    Returns:
        ComplianceCheckResult 审核结果.
    """
    prompt = COMPLIANCE_CHECK_PROMPT.format(
        role=role, domain=domain, task=task, content=content,
    )

    try:
        raw_content, _, _ = await generate_content(
            model=model,
            contents=prompt,
            temperature=0.0,
            thinking_budget=thinking_budget,
            provider=provider,
        )

        # 尝试从回复中提取 JSON
        text = raw_content.strip()
        # 去除可能的 markdown 包裹
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
        )

        result = ComplianceCheckResult(**parsed)
        logger.info(
            "[Compliance] %s: passed=%s reason=%s",
            role, result.passed, result.reason[:200] if result.reason else "",
        )
        return result

    except Exception as e:
        logger.warning("[Compliance] check failed for %s: %s, defaulting to pass", role, e)
        return ComplianceCheckResult(passed=True)
