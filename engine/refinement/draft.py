"""初稿生成模块.

收集通过审核的专家输出, XML 包裹后调用 LLM 生成初稿.
类似经典流程的 Synthesis 但定位为初稿阶段.
"""

import logging

from clients.llm_client import generate_content
from prompts import REFINEMENT_DRAFT_PROMPT

logger = logging.getLogger(__name__)


def _build_draft_prompt(
    query: str,
    context: str,
    expert_outputs: list[dict[str, str]],
    user_system_prompt: str = "",
) -> str:
    """构建初稿生成 prompt.

    Args:
        query: 用户原始问题.
        context: 对话上下文.
        expert_outputs: 专家输出列表, 每项含 role/domain/content.
        user_system_prompt: 用户 system prompt.

    Returns:
        完整的初稿 prompt.
    """
    expert_xml_parts = []
    for exp in expert_outputs:
        expert_xml_parts.append(
            f'  <Expert name="{exp["role"]}" domain="{exp["domain"]}">\n'
            f'{exp["content"]}\n'
            "  </Expert>"
        )
    experts_xml = "\n\n".join(expert_xml_parts)

    user_instruction = ""
    if user_system_prompt:
        user_instruction = f"\n\n用户的重要指示：\n{user_system_prompt}"

    return (
        f"{REFINEMENT_DRAFT_PROMPT}{user_instruction}\n\n"
        f"对话上下文：\n{context}\n\n"
        f'用户原始需求："{query}"\n\n'
        f"以下是各专家提供的领域素材：\n"
        f"{experts_xml}"
    )


async def generate_draft(
    model: str,
    query: str,
    context: str,
    expert_outputs: list[dict[str, str]],
    budget: int,
    temperature: float | None = None,
    user_system_prompt: str = "",
    image_parts: list[dict] | None = None,
    provider: str = "",
) -> str:
    """生成初稿.

    Args:
        model: 初稿生成模型.
        query: 用户原始问题.
        context: 对话上下文.
        expert_outputs: 专家输出列表.
        budget: thinking token 预算.
        temperature: 温度参数.
        user_system_prompt: 用户 system prompt.
        image_parts: 图片列表.
        provider: provider 标识符.

    Returns:
        初稿文本.
    """
    prompt = _build_draft_prompt(
        query, context, expert_outputs, user_system_prompt,
    )

    kwargs: dict = {
        "model": model,
        "contents": prompt,
        "thinking_budget": budget,
        "image_parts": image_parts,
        "provider": provider,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    content, thoughts, _ = await generate_content(**kwargs)

    logger.info(
        "[Draft] generated draft (%d chars, %d lines)",
        len(content), content.count("\n") + 1,
    )
    return content
