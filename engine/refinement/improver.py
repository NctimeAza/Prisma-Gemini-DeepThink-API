"""改进专家执行模块.

并行执行改进专家, 每个专家输出 JSON 格式的 diff 操作.
操作带全局递增 op_id, 复用 prefill 确认轮提高执行力.
"""

import asyncio
import json
import logging

from clients.llm_client import generate_content
from config import LLM_NETWORK_RETRIES
from engine.refinement.json_repair import parse_json_with_repair
from models import DiffOperation, RefinementExpertConfig, RefinementExpertResult
from prompts import (
    build_refinement_expert_contents,
    get_refinement_improver_system_instruction,
)
from utils.retry import extract_status, is_retryable_error

logger = logging.getLogger(__name__)


async def run_improver(
    model: str,
    expert_config: RefinementExpertConfig,
    draft_lines_json: str,
    budget: int,
    top_p: float | None = None,
    guidance: str = "",
    user_system_prompt: str = "",
    image_parts: list[dict] | None = None,
    provider: str = "",
    enable_json_repair: bool = False,
    json_repair_model: str = "",
) -> RefinementExpertResult:
    """执行单个改进专家.

    Args:
        model: 改进专家模型.
        expert_config: 改进专家配置.
        draft_lines_json: 初稿按行切分的 JSON 字符串.
        budget: thinking token 预算.
        guidance: 审查模型给的额外指导.
        user_system_prompt: 用户 system prompt.
        image_parts: 图片列表.
        provider: provider 标识符.
        enable_json_repair: 是否启用 JSON 修复.
        json_repair_model: JSON 修复模型.

    Returns:
        RefinementExpertResult 改进结果.
    """
    system_instruction = get_refinement_improver_system_instruction(
        role=expert_config.role,
        domain=expert_config.domain,
        all_expert_roles=expert_config.all_expert_roles,
        guidance=guidance,
        user_system_prompt=user_system_prompt,
    )

    task_prompt = (
        f"{expert_config.prompt}\n\n"
        f"初稿按行切分内容：\n{draft_lines_json}"
    )

    contents = build_refinement_expert_contents(
        task_prompt, image_parts=image_parts,
    )

    max_retries = LLM_NETWORK_RETRIES

    for attempt in range(max_retries + 1):
        try:
            full_content, _, _ = await generate_content(
                model=model,
                contents=contents,
                system_instruction=system_instruction,
                temperature=expert_config.temperature,
                top_p=top_p,
                thinking_budget=budget,
                provider=provider,
            )

            if not full_content.strip():
                if attempt < max_retries:
                    delay = 1.5 * (attempt + 1)
                    logger.warning(
                        "[Improver] %s returned empty, retry %d/%d in %.1fs",
                        expert_config.role, attempt + 1, max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                return RefinementExpertResult(
                    role=expert_config.role,
                    analysis="Empty response after retries",
                )

            # 提取 JSON
            text = full_content.strip()
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

            operations = []
            for op_raw in parsed.get("operations", []):
                action = op_raw.get("action", "").strip().lower()
                if action not in ("modify", "add", "remove"):
                    logger.warning(
                        "[Improver] %s: unknown action %r, skipping",
                        expert_config.role, action,
                    )
                    continue
                operations.append(DiffOperation(
                    expert_role=expert_config.role,
                    action=action,
                    line=op_raw.get("line", 0),
                    content=op_raw.get("content", ""),
                    reason=op_raw.get("reason", ""),
                ))

            result = RefinementExpertResult(
                role=expert_config.role,
                analysis=parsed.get("analysis", ""),
                operations=operations,
            )

            logger.info(
                "[Improver] %s completed: %d operations",
                expert_config.role, len(operations),
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(
                "[Improver] %s JSON parse failed: %s", expert_config.role, e,
            )
            return RefinementExpertResult(
                role=expert_config.role,
                analysis=f"JSON parse error: {e}",
            )

        except Exception as e:
            status = extract_status(e)
            retryable = is_retryable_error(status)

            if retryable and attempt < max_retries:
                delay = 1.5 * (attempt + 1)
                logger.warning(
                    "[Improver] %s failed (status=%s), retry %d/%d: %s",
                    expert_config.role, status, attempt + 1, max_retries, e,
                )
                await asyncio.sleep(delay)
                continue

            logger.error(
                "[Improver] %s failed after %d retries: %s",
                expert_config.role, attempt, e,
            )
            return RefinementExpertResult(
                role=expert_config.role,
                analysis=f"Error: {e}",
            )

    return RefinementExpertResult(
        role=expert_config.role,
        analysis="Max retries exceeded",
    )
