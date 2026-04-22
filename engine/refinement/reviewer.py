"""审查模块 (Reviewer).

将初稿按行切分, 分析存在的问题, 分配改进专家.
精修迭代时允许通过 (approved=true).
"""

import json
import logging
from typing import Any

import clients.openai_client as _chat_client
from clients.llm_client import generate_content
from engine.refinement.json_repair import parse_json_with_repair
from models import RefinementExpertConfig, RefinementReviewAnalysis
from prompts import (
    REFINEMENT_REVIEW_PROMPT,
    build_prefill_contents,
    format_expert_model_pool_note,
)

logger = logging.getLogger(__name__)


def split_draft_to_lines(draft_text: str) -> list[dict[str, Any]]:
    """将初稿按行切分为 JSON 数组格式.

    Args:
        draft_text: 初稿文本.

    Returns:
        [{"line": 1, "text": "..."}, ...] 格式的列表.
    """
    lines = draft_text.split("\n")
    return [{"line": i + 1, "text": line} for i, line in enumerate(lines)]


def _stringify_field(value: Any) -> str:
    """将漂移字段稳态化为字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_stringify_field(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "value", "reason", "description"):
            text = _stringify_field(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _normalize_string_list(value: Any) -> list[str]:
    """将 issue 等字段规整为字符串数组。"""
    if value is None:
        return []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            text = _stringify_field(item)
            if text:
                result.append(text)
        return result
    text = _stringify_field(value)
    return [text] if text else []


def _coerce_temperature(value: Any, default: float = 0.8) -> float:
    """尽量把温度字段转为 float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_expert_payload(raw_item: Any) -> dict[str, Any] | None:
    """兼容 refinement reviewer 文本 fallback 的字段漂移。"""
    if not isinstance(raw_item, dict):
        return None

    role = _stringify_field(
        raw_item.get("role")
        or raw_item.get("name")
        or raw_item.get("expert_name")
    )
    if not role:
        return None

    return {
        "role": role,
        "domain": _stringify_field(
            raw_item.get("domain")
            or raw_item.get("description")
            or raw_item.get("focus")
            or role
        ),
        "prompt": _stringify_field(
            raw_item.get("prompt")
            or raw_item.get("task")
            or raw_item.get("instruction")
            or raw_item.get("mission")
        ),
        "temperature": _coerce_temperature(raw_item.get("temperature"), 0.8),
        "expert_model": _stringify_field(
            raw_item.get("expert_model")
            or raw_item.get("expert_model_id")
            or raw_item.get("model_id")
            or raw_item.get("model")
        ),
    }


def _normalize_guidance_map(value: Any) -> dict[str, str]:
    """兼容 expert_guidance 的对象/数组漂移。"""
    if isinstance(value, dict):
        result: dict[str, str] = {}
        for key, item in value.items():
            text = _stringify_field(item)
            if key and text:
                result[str(key)] = text
        return result

    if isinstance(value, list):
        result = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            role = _stringify_field(
                item.get("role") or item.get("name") or item.get("expert_name")
            )
            guidance = _stringify_field(
                item.get("guidance")
                or item.get("instruction")
                or item.get("prompt")
                or item.get("text")
            )
            if role and guidance:
                result[role] = guidance
        return result

    return {}


def _normalize_review_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    """兼容 reviewer 输出中的常见字段漂移。"""
    raw_experts = parsed.get("refinement_experts", [])
    normalized_experts: list[dict[str, Any]] = []
    if isinstance(raw_experts, list):
        for item in raw_experts:
            normalized = _normalize_expert_payload(item)
            if normalized is not None:
                normalized_experts.append(normalized)

    return {
        "issues": _normalize_string_list(parsed.get("issues", [])),
        "refinement_experts": normalized_experts,
        "expert_guidance": _normalize_guidance_map(
            parsed.get("expert_guidance", {})
        ),
        "approved": bool(parsed.get("approved", False)),
        "approval_reason": _stringify_field(parsed.get("approval_reason")),
    }


async def review_draft(
    model: str,
    query: str,
    draft_text: str,
    budget: int,
    refinement_round: int = 1,
    previous_summary: str = "",
    context: str = "",
    temperature: float | None = None,
    top_p: float | None = None,
    user_system_prompt: str = "",
    image_parts: list[dict] | None = None,
    provider: str = "",
    enable_json_repair: bool = False,
    json_repair_model: str = "",
    expert_model_pool: list | None = None,
    enable_expert_model_selection: bool = False,
) -> RefinementReviewAnalysis:
    """审查初稿, 分析问题并分配改进专家.

    Args:
        model: 审查模型.
        query: 用户原始问题.
        draft_text: 初稿文本.
        budget: thinking token 预算.
        refinement_round: 当前精修轮数.
        previous_summary: 上一轮综合助手的改动简评.
        context: 对话上下文.
        temperature: 温度参数.
        user_system_prompt: 用户 system prompt.
        image_parts: 图片列表.
        provider: provider 标识符.
        enable_json_repair: 是否启用 JSON 修复.
        json_repair_model: JSON 修复模型.

    Returns:
        RefinementReviewAnalysis 审查结果.
    """
    lines_json = json.dumps(
        split_draft_to_lines(draft_text), ensure_ascii=False,
    )

    # 构建迭代备注
    iteration_note = ""
    if refinement_round > 1:
        iteration_note = (
            f"<Iteration_Info>\n"
            f"这是第 {refinement_round} 轮精修。\n"
            f"上一轮综合助手的改动简评：{previous_summary or '（无）'}\n"
            f"本轮你可以选择通过（approved=true，不分配改进专家）或继续精修。\n"
            f"</Iteration_Info>"
        )
    else:
        iteration_note = (
            "这是首轮审查，必须进行精修（approved 必须为 false），"
            "你需要分析问题并分配改进专家。"
        )

    context_section = f"对话上下文：\n{context}\n\n" if context else ""
    expert_model_note = ""
    if enable_expert_model_selection:
        expert_model_note = format_expert_model_pool_note(expert_model_pool)

    prompt_parts = [
        REFINEMENT_REVIEW_PROMPT.format(iteration_note=iteration_note),
    ]
    if expert_model_note:
        prompt_parts.append(expert_model_note)
    if context_section:
        prompt_parts.append(context_section.rstrip())
    prompt_parts.append(f'用户原始需求："{query}"')
    prompt_parts.append(f"初稿按行切分内容：\n{lines_json}")
    prompt = "\n\n".join(prompt_parts)
    contents = build_prefill_contents(prompt, image_parts=image_parts)
    raw_content = ""
    cleaned_text = ""

    try:
        raw_content, _, _ = await generate_content(
            model=model,
            contents=contents,
            system_instruction=user_system_prompt or None,
            temperature=temperature or 0.7,
            top_p=top_p,
            thinking_budget=budget,
            image_parts=None,
            provider=provider,
        )

        # 提取 JSON
        text = raw_content.strip()
        if text.startswith("```"):
            text_lines = text.split("\n")
            text_lines = text_lines[1:]
            if text_lines and text_lines[-1].strip() == "```":
                text_lines = text_lines[:-1]
            text = "\n".join(text_lines)
        cleaned_text = _chat_client._clean_json_string(text)  # noqa: SLF001

        parsed = await parse_json_with_repair(
            cleaned_text,
            enable_repair=enable_json_repair,
            repair_model=json_repair_model,
            provider=provider,
            top_p=top_p,
        )
        normalized = _normalize_review_payload(parsed)

        # 解析专家配置
        experts_raw = normalized.get("refinement_experts", [])
        all_roles = [e.get("role", "") for e in experts_raw]
        guidance = normalized.get("expert_guidance", {})

        experts = []
        for e in experts_raw:
            cfg = RefinementExpertConfig(
                role=e["role"],
                domain=e.get("domain", ""),
                prompt=e.get("prompt", ""),
                temperature=e.get("temperature", 0.8),
                expert_model=e.get("expert_model", ""),
                all_expert_roles=all_roles,
            )
            experts.append(cfg)

        result = RefinementReviewAnalysis(
            issues=normalized.get("issues", []),
            refinement_experts=experts,
            expert_guidance=guidance,
            approved=normalized.get("approved", False),
            approval_reason=normalized.get("approval_reason", ""),
        )

        logger.info(
            "[Reviewer] reviewed draft: approved=%s, issues=%d, experts=%d",
            result.approved, len(result.issues), len(result.refinement_experts),
        )
        return result

    except Exception as e:
        logger.error("[Reviewer] review failed: %s", e)
        if raw_content or cleaned_text:
            logger.error(
                "[Reviewer] review parse debug: provider=%s model=%s raw_len=%d cleaned_len=%d",
                provider,
                model,
                len(raw_content or ""),
                len(cleaned_text or ""),
            )
            logger.error("[Reviewer] review raw response:\n%s", raw_content)
            logger.error("[Reviewer] review cleaned response:\n%s", cleaned_text)
        raise
