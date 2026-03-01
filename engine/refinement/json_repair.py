"""JSON 格式修复模块.

当 JSON 解析失败时, 调用小模型尝试修复格式错误.
可通过虚拟模型配置的 enable_json_repair 开关启用.
"""

import json
import logging

from clients.llm_client import generate_content

logger = logging.getLogger(__name__)

_REPAIR_SYSTEM_INSTRUCTION = (
    "你是 JSON 格式修复工具。用户会给你一段格式有误的 JSON 文本，"
    "你必须修复其中的语法错误（如缺少引号、多余逗号、未闭合括号等），"
    "只输出修复后的合法 JSON，不要输出任何其他内容。"
)


async def try_repair_json(
    raw_text: str,
    *,
    model: str,
    provider: str = "",
    thinking_budget: int = 1024,
) -> dict | list | None:
    """尝试用小模型修复格式错误的 JSON.

    Args:
        raw_text: 格式有误的原始文本.
        model: 修复用模型.
        provider: provider 标识符.
        thinking_budget: thinking token 预算.

    Returns:
        修复后的 JSON 对象, 修复失败返回 None.
    """
    try:
        content, _, _ = await generate_content(
            model=model,
            contents=f"请修复以下 JSON：\n```\n{raw_text}\n```",
            system_instruction=_REPAIR_SYSTEM_INSTRUCTION,
            temperature=0.0,
            thinking_budget=thinking_budget,
            provider=provider,
        )
        # 尝试从回复中提取 JSON
        text = content.strip()
        # 去除可能的 markdown 包裹
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # 去掉 ```json 行
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        return json.loads(text)
    except Exception as e:
        logger.warning("[JSONRepair] repair failed: %s", e)
        return None


async def parse_json_with_repair(
    raw_text: str,
    *,
    enable_repair: bool = False,
    repair_model: str = "",
    provider: str = "",
) -> dict | list:
    """解析 JSON, 失败时可选调用修复模型.

    Args:
        raw_text: 原始 JSON 文本.
        enable_repair: 是否启用修复.
        repair_model: 修复用模型.
        provider: provider 标识符.

    Returns:
        解析后的 JSON 对象.

    Raises:
        json.JSONDecodeError: 解析和修复均失败.
    """
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        if not enable_repair or not repair_model:
            raise

        logger.info("[JSONRepair] attempting repair with model=%s", repair_model)
        repaired = await try_repair_json(
            raw_text, model=repair_model, provider=provider,
        )
        if repaired is not None:
            logger.info("[JSONRepair] repair succeeded")
            return repaired

        # 修复失败, 抛出原始错误
        raise
