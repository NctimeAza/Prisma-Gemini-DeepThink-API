"""JSON 格式修复模块.

当 JSON 解析失败时, 调用小模型尝试修复格式错误.
可通过虚拟模型配置的 enable_json_repair 开关启用.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path

from clients.llm_client import generate_content
from config import (
    JSON_REPAIR_DEBUG_DIR,
    JSON_REPAIR_DEBUG_ENABLED,
    JSON_REPAIR_DEBUG_MAX_CHARS,
)
from prompts import build_prefill_contents

logger = logging.getLogger(__name__)

_REPAIR_SYSTEM_INSTRUCTION = (
    "你是 JSON 格式修复工具。用户会给你一段格式有误的 JSON 文本，"
    "你必须修复其中的语法错误（如缺少引号、多余逗号、未闭合括号等），"
    "只输出修复后的合法 JSON，不要输出任何其他内容。"
)


def _truncate_for_debug(text: str) -> tuple[str, bool]:
    """按配置截断调试文本。"""
    limit = JSON_REPAIR_DEBUG_MAX_CHARS
    if limit == 0 or len(text) <= limit:
        return text, False
    omitted = len(text) - limit
    tail = f"\n\n[TRUNCATED] omitted_chars={omitted}"
    return text[:limit] + tail, True


def _write_debug_bundle(
    *,
    phase: str,
    model: str,
    provider: str,
    top_p: float | None,
    thinking_budget: int,
    raw_input: str,
    raw_response: str,
    cleaned_response: str,
    error: str = "",
) -> None:
    """落盘 JSON 修复阶段专属调试信息。"""
    if not JSON_REPAIR_DEBUG_ENABLED:
        return

    try:
        root = Path(JSON_REPAIR_DEBUG_DIR)
        root.mkdir(parents=True, exist_ok=True)

        ts = int(time.time() * 1000)
        run_id = f"{ts}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        bundle_dir = root / run_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

        input_dump, input_truncated = _truncate_for_debug(raw_input)
        raw_rsp_dump, raw_rsp_truncated = _truncate_for_debug(raw_response)
        cleaned_dump, cleaned_truncated = _truncate_for_debug(cleaned_response)

        meta = {
            "phase": phase,
            "model": model,
            "provider": provider,
            "top_p": top_p,
            "thinking_budget": thinking_budget,
            "raw_input_chars": len(raw_input),
            "raw_response_chars": len(raw_response),
            "cleaned_response_chars": len(cleaned_response),
            "raw_response_tokens_approx": len(raw_response) // 4,
            "input_truncated": input_truncated,
            "raw_response_truncated": raw_rsp_truncated,
            "cleaned_response_truncated": cleaned_truncated,
            "error": error,
            "created_at_ms": ts,
        }

        (bundle_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (bundle_dir / "repair_input.txt").write_text(
            input_dump,
            encoding="utf-8",
        )
        (bundle_dir / "raw_response.txt").write_text(
            raw_rsp_dump,
            encoding="utf-8",
        )
        (bundle_dir / "cleaned_response.txt").write_text(
            cleaned_dump,
            encoding="utf-8",
        )
        logger.info("[JSONRepair][Debug] dump saved: %s", bundle_dir)
    except Exception as dump_err:
        logger.warning("[JSONRepair][Debug] dump failed: %s", dump_err)


async def try_repair_json(
    raw_text: str,
    *,
    model: str,
    provider: str = "",
    top_p: float | None = None,
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
    request_text = f"请修复以下 JSON：\n```\n{raw_text}\n```"
    prefilled_contents = build_prefill_contents(
        request_text,
        leading_instruction=_REPAIR_SYSTEM_INSTRUCTION,
    )
    raw_response = ""
    cleaned_response = ""

    try:
        content, _, _ = await generate_content(
            model=model,
            contents=prefilled_contents,
            temperature=0.0,
            top_p=top_p,
            thinking_budget=thinking_budget,
            provider=provider,
        )
        raw_response = content or ""
        # 尝试从回复中提取 JSON
        text = raw_response.strip()
        # 去除可能的 markdown 包裹
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # 去掉 ```json 行
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        cleaned_response = text

        parsed = json.loads(text)
        _write_debug_bundle(
            phase="success",
            model=model,
            provider=provider,
            top_p=top_p,
            thinking_budget=thinking_budget,
            raw_input=raw_text,
            raw_response=raw_response,
            cleaned_response=cleaned_response,
        )
        return parsed
    except Exception as e:
        _write_debug_bundle(
            phase="failed",
            model=model,
            provider=provider,
            top_p=top_p,
            thinking_budget=thinking_budget,
            raw_input=raw_text,
            raw_response=raw_response,
            cleaned_response=cleaned_response,
            error=str(e),
        )
        logger.warning("[JSONRepair] repair failed: %s", e)
        return None


async def parse_json_with_repair(
    raw_text: str,
    *,
    enable_repair: bool = False,
    repair_model: str = "",
    provider: str = "",
    top_p: float | None = None,
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
            raw_text, model=repair_model, provider=provider, top_p=top_p,
        )
        if repaired is not None:
            logger.info("[JSONRepair] repair succeeded")
            return repaired

        # 修复失败, 抛出原始错误
        raise
