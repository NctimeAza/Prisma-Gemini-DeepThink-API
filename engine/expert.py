"""Expert 模块：专家角色非流式执行.

迁移自原项目 expert.ts。
每个 Expert 独立调用 LLM 获取完整结果。
使用非流式调用避免长连接稳定性问题，遇到可重试的错误时会自动重试。
"""

import asyncio
import logging

from clients.llm_client import generate_content
from config import LLM_NETWORK_RETRIES
from models import ExpertResult
from prompts import get_expert_system_instruction, build_expert_contents
from utils.retry import extract_status, is_retryable_error

logger = logging.getLogger(__name__)


async def run_expert(
    model: str,
    expert: ExpertResult,
    context: str,
    budget: int,
    all_expert_roles: list[str] | None = None,
    user_system_prompt: str = "",
    image_parts: list[dict] | None = None,
    provider: str = "",
) -> ExpertResult:
    """执行单个 Expert 的完整生命周期.

    非流式调用 LLM，一次性获取完整结果。
    with_retry 在底层覆盖整个请求，Expert 层重试处理空回和最终失败。

    Args:
        model: 模型标识符.
        expert: 待执行的 Expert 配置和状态.
        context: 对话上下文.
        budget: thinking token 预算.
        user_system_prompt: 下游客户端的 system prompt.
        image_parts: Gemini inlineData 格式的图片列表.

    Returns:
        更新了 content/thoughts/status 的 ExpertResult.
    """
    expert.status = "thinking"
    logger.info("[Expert] Starting: %s (Round %d)", expert.role, expert.round)

    system_instruction = get_expert_system_instruction(
        expert.role,
        expert.description,
        context,
        all_expert_roles=all_expert_roles,
        user_system_prompt=user_system_prompt,
    )

    max_retries = LLM_NETWORK_RETRIES
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            # 构建包含 prefill 确认轮的多轮对话 contents
            contents = build_expert_contents(
                expert.prompt, image_parts=image_parts,
            )

            full_content, full_thoughts, _grounding = await generate_content(
                model=model,
                contents=contents,
                system_instruction=system_instruction,
                temperature=expert.temperature,
                thinking_budget=budget,
                provider=provider,
            )

            expert.content = full_content
            expert.thoughts = full_thoughts

            # 空回检测：正常完成但内容为空，视为异常并重试
            if not full_content.strip():
                if attempt < max_retries:
                    delay = 1.5 * (attempt + 1)
                    logger.warning(
                        "[Expert] %s returned empty content, retry %d/%d in %.1fs",
                        expert.role, attempt + 1, max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        "[Expert] %s returned empty content (retried %d times), marking as error",
                        expert.role, attempt,
                    )
                    expert.status = "error"
                    expert.content = "**Error:** Expert returned empty content"
                    return expert

            expert.status = "completed"
            logger.info(
                "[Expert] Completed: %s (%d chars)", expert.role, len(full_content)
            )
            return expert

        except Exception as e:
            last_error = e
            status = extract_status(e)
            retryable = is_retryable_error(status)

            if retryable and attempt < max_retries:
                delay = 1.5 * (attempt + 1)
                logger.warning(
                    "[Expert] %s failed (status=%s), retry %d/%d in %.1fs: %s",
                    expert.role, status or "no_status_code",
                    attempt + 1, max_retries, delay, e,
                )
                await asyncio.sleep(delay)
                continue

            # 不可重试或重试次数耗尽
            logger.error(
                "[Expert] %s failed after %d retries: %s",
                expert.role, attempt, e,
            )
            expert.status = "error"
            expert.content = f"**Error:** {e}"
            return expert

    # 理论上不会到这里，但以防万一
    expert.status = "error"
    expert.content = f"**Error:** {last_error}"
    return expert
