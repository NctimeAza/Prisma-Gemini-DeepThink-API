"""精修规划模块.

分析用户需求, 分配严格领域划分的专家团队.
每个专家被注入所有已分配专家信息以防越权.
"""

import json
import logging
from typing import Any

from clients.llm_client import generate_json
from models import RefinementExpertConfig
from prompts import REFINEMENT_PLANNER_PROMPT

logger = logging.getLogger(__name__)

# 规划阶段的 JSON Schema
REFINEMENT_PLANNING_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "thought_process": {
            "type": "STRING",
            "description": "分析用户需求后的拆解思路。",
        },
        "experts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "role": {"type": "STRING"},
                    "domain": {
                        "type": "STRING",
                        "description": "该专家严格负责的领域。",
                    },
                    "temperature": {"type": "NUMBER"},
                    "prompt": {"type": "STRING"},
                },
                "required": ["role", "domain", "temperature", "prompt"],
            },
        },
    },
    "required": ["thought_process", "experts"],
}


async def plan(
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
) -> list[RefinementExpertConfig]:
    """精修规划阶段: 分析需求并分配专家.

    Args:
        model: 规划用模型.
        query: 用户当前问题.
        context: 最近对话上下文.
        budget: thinking token 预算.
        temperature: 温度参数.
        user_system_prompt: 下游客户端 system prompt.
        image_parts: 图片列表.
        provider: provider 标识符.

    Returns:
        RefinementExpertConfig 列表, 每个专家已注入所有专家角色信息.
    """
    sys_section = (
        f"\n用户的重要指示：{user_system_prompt}" if user_system_prompt else ""
    )
    text_prompt = f'Context:\n{context}{sys_section}\n\nCurrent Query: "{query}"'

    try:
        result = await generate_json(
            model=model,
            contents=text_prompt,
            system_instruction=REFINEMENT_PLANNER_PROMPT,
            response_schema=REFINEMENT_PLANNING_SCHEMA,
            thinking_budget=budget,
            temperature=temperature,
            top_p=top_p,
            image_parts=image_parts,
            provider=provider,
            json_via_prompt=json_via_prompt,
        )
        logger.debug(
            "[RefinementPlanner] raw response:\n%s",
            json.dumps(result, ensure_ascii=False, indent=2),
        )

        experts_raw = result.get("experts", [])
        if not experts_raw:
            logger.warning("[RefinementPlanner] returned empty experts list")
            return []

        # 收集所有角色名
        all_roles = [e.get("role", "") for e in experts_raw]

        experts: list[RefinementExpertConfig] = []
        for e in experts_raw:
            cfg = RefinementExpertConfig(
                role=e["role"],
                domain=e.get("domain", ""),
                prompt=e.get("prompt", ""),
                temperature=e.get("temperature", 1.0),
                all_expert_roles=all_roles,
            )
            experts.append(cfg)

        logger.info(
            "[RefinementPlanner] planned %d experts: %s",
            len(experts),
            ", ".join(e.role for e in experts),
        )
        return experts

    except Exception as e:
        logger.error("[RefinementPlanner] planning failed: %s", e)
        return []
