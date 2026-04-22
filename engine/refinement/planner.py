"""精修规划模块.

分析用户需求, 分配严格领域划分的专家团队.
每个专家被注入所有已分配专家信息以防越权.
"""

import json
import logging
from typing import Any

from clients.llm_client import generate_json
from models import RefinementExpertConfig
from prompts import (
    REFINEMENT_PLANNER_PROMPT,
    build_prefill_contents,
    format_expert_model_pool_note,
)

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
                    "expert_model": {"type": "STRING"},
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
    expert_model_pool: list | None = None,
    enable_expert_model_selection: bool = False,
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
    text_prompt = f'Context:\n{context}\n\nCurrent Query: "{query}"'
    expert_model_note = ""
    if enable_expert_model_selection:
        expert_model_note = format_expert_model_pool_note(expert_model_pool)
    contents = build_prefill_contents(
        text_prompt,
        image_parts=image_parts,
        leading_instruction="\n\n".join(
            part
            for part in [REFINEMENT_PLANNER_PROMPT, expert_model_note]
            if part
        ),
    )
    debug_info: dict[str, Any] = {}

    try:
        result = await generate_json(
            model=model,
            contents=contents,
            system_instruction=user_system_prompt or None,
            response_schema=REFINEMENT_PLANNING_SCHEMA,
            thinking_budget=budget,
            temperature=temperature,
            top_p=top_p,
            image_parts=None,
            provider=provider,
            json_via_prompt=json_via_prompt,
            debug_info=debug_info,
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
                expert_model=str(e.get("expert_model", "")).strip(),
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
        raw_text = str(debug_info.get("raw_text", "") or "")
        cleaned_text = str(debug_info.get("cleaned_text", "") or "")
        if raw_text or cleaned_text:
            logger.error(
                "[RefinementPlanner] planning parse debug: provider=%s model=%s raw_len=%d cleaned_len=%d",
                debug_info.get("provider", provider),
                debug_info.get("model", model),
                len(raw_text),
                len(cleaned_text),
            )
            logger.error(
                "[RefinementPlanner] planning parse raw response:\n%s",
                raw_text,
            )
            logger.error(
                "[RefinementPlanner] planning parse cleaned response:\n%s",
                cleaned_text,
            )
        return []
