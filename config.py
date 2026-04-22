"""Prisma DeepThink 配置模块.

模型注册表、Thinking Budget 计算、环境变量加载。
通过虚拟模型名（如 gemini-3-pro-deepthink-high）映射到实际模型 + 思考预算。
虚拟模型支持通过 .env 中的 VIRTUAL_MODELS_FILE 或 VIRTUAL_MODELS_EXTRA 自定义新增。
每个虚拟模型可指定 provider（gemini / openai），支持同时使用多个上游提供商。
"""

import json
import logging
import os
import re as _re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from models import ExpertModelProfile

load_dotenv()

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent

# --- 环境变量 ---

_SUPPORTED_LLM_PROVIDERS = {"gemini", "openai", "openai_responses"}
_LLM_PROVIDER_RAW = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
if _LLM_PROVIDER_RAW in _SUPPORTED_LLM_PROVIDERS:
    LLM_PROVIDER: str = _LLM_PROVIDER_RAW
else:
    LLM_PROVIDER = "gemini"
    logger.warning(
        "[Config] Invalid LLM_PROVIDER=%r; falling back to 'gemini'",
        _LLM_PROVIDER_RAW,
    )

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL: Optional[str] = os.getenv("GEMINI_BASE_URL") or None
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL") or None
OPENAI_RESPONSES_API_KEY: str = os.getenv(
    "OPENAI_RESPONSES_API_KEY", OPENAI_API_KEY
)
OPENAI_RESPONSES_BASE_URL: Optional[str] = (
    os.getenv("OPENAI_RESPONSES_BASE_URL")
    or OPENAI_BASE_URL
    or None
)


# --- Provider 配置注册表 ---


@dataclass
class ProviderConfig:
    """单个 LLM Provider 的连接配置."""

    name: str          # provider 标识符，如 "gemini", "openai", "deepseek"
    type: str          # 底层 API 类型: "gemini" / "openai" / "openai_responses"
    api_key: str = ""
    base_url: Optional[str] = None


def _load_provider_configs() -> dict[str, ProviderConfig]:
    """从环境变量加载所有 provider 配置.

    内置 provider (始终存在):
      - gemini: 使用 GEMINI_API_KEY / GEMINI_BASE_URL
      - openai: 使用 OPENAI_API_KEY / OPENAI_BASE_URL

    自定义 provider 通过环境变量命名约定注册:
      PROVIDER_<NAME>_API_KEY   (必填)
      PROVIDER_<NAME>_BASE_URL  (可选)
      PROVIDER_<NAME>_TYPE      (可选, 默认 "openai")

    例如:
      PROVIDER_DEEPSEEK_API_KEY=sk-xxx
      PROVIDER_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

    Returns:
        {provider_name: ProviderConfig} 字典.
    """
    configs: dict[str, ProviderConfig] = {
        "gemini": ProviderConfig(
            name="gemini",
            type="gemini",
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,
        ),
        "openai": ProviderConfig(
            name="openai",
            type="openai",
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        ),
        "openai_responses": ProviderConfig(
            name="openai_responses",
            type="openai_responses",
            api_key=OPENAI_RESPONSES_API_KEY,
            base_url=OPENAI_RESPONSES_BASE_URL,
        ),
    }

    # 扫描环境变量，发现 PROVIDER_<NAME>_API_KEY 就注册
    pattern = _re.compile(r"^PROVIDER_([A-Za-z0-9_]+)_API_KEY$")
    for key, value in os.environ.items():
        m = pattern.match(key)
        if not m or not value:
            continue
        name = m.group(1).lower()
        if name in configs:
            # 如果和内置 provider 同名，更新 api_key 和 base_url
            configs[name].api_key = value
            custom_base = os.getenv(f"PROVIDER_{m.group(1)}_BASE_URL")
            if custom_base:
                configs[name].base_url = custom_base
            continue
        provider_type = os.getenv(
            f"PROVIDER_{m.group(1)}_TYPE", "openai"
        ).strip().lower()
        if provider_type not in _SUPPORTED_LLM_PROVIDERS:
            provider_type = "openai"
        configs[name] = ProviderConfig(
            name=name,
            type=provider_type,
            api_key=value,
            base_url=os.getenv(f"PROVIDER_{m.group(1)}_BASE_URL") or None,
        )
        logger.info(
            "[Config] Registered custom provider: %s (type=%s, base_url=%s)",
            name, provider_type, configs[name].base_url or "(default)",
        )

    return configs


PROVIDER_CONFIGS: dict[str, ProviderConfig] = _load_provider_configs()


def get_provider_config(provider: str) -> ProviderConfig:
    """获取指定 provider 的配置，不存在则回退到全局默认.

    Args:
        provider: provider 名称.

    Returns:
        ProviderConfig 实例.
    """
    if provider in PROVIDER_CONFIGS:
        return PROVIDER_CONFIGS[provider]
    logger.warning(
        "[Config] Unknown provider %r, falling back to %r",
        provider, LLM_PROVIDER,
    )
    return PROVIDER_CONFIGS.get(LLM_PROVIDER, PROVIDER_CONFIGS["gemini"])


def _load_default_top_p() -> float:
    """加载全局默认 top_p，限制在 (0, 1] 区间。

    环境变量：
        DEFAULT_TOP_P: 默认 0.86。

    Returns:
        合法的 top_p 浮点值。
    """
    raw = os.getenv("DEFAULT_TOP_P", "0.86").strip()
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "[Config] Invalid DEFAULT_TOP_P=%r, fallback to 0.86",
            raw,
        )
        return 0.86

    if value <= 0.0 or value > 1.0:
        logger.warning(
            "[Config] DEFAULT_TOP_P out of range (0,1]: %s, fallback to 0.86",
            value,
        )
        return 0.86
    return value


HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
_SUPPORTED_APP_LANGUAGES = {"en", "zh"}
_APP_LANGUAGE_RAW = os.getenv("APP_LANGUAGE", "en").strip().lower()
if _APP_LANGUAGE_RAW in _SUPPORTED_APP_LANGUAGES:
    APP_LANGUAGE: str = _APP_LANGUAGE_RAW
else:
    APP_LANGUAGE = "en"
    logger.warning(
        "[Config] Invalid APP_LANGUAGE=%r; falling back to 'en'",
        _APP_LANGUAGE_RAW,
    )

DEFAULT_TOP_P: float = _load_default_top_p()

# --- DeepThink 流水线配置（.env 可覆盖）---

ENABLE_RECURSIVE_LOOP: bool = os.getenv(
    "ENABLE_RECURSIVE_LOOP", "true"
).lower() in ("true", "1", "yes")
MAX_ROUNDS: int = int(os.getenv("MAX_ROUNDS", "2"))
MAX_CONTEXT_MESSAGES: int = int(os.getenv("MAX_CONTEXT_MESSAGES", "10"))

# --- LLM 请求超时 & 重试 & 风控 ---

LLM_REQUEST_DELAY_MIN: float = float(os.getenv("LLM_REQUEST_DELAY_MIN", "0.0"))
LLM_REQUEST_DELAY_MAX: float = float(os.getenv("LLM_REQUEST_DELAY_MAX", "0.0"))

LLM_REQUEST_TIMEOUT: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "200"))
LLM_TIMEOUT_RETRIES: int = int(os.getenv("LLM_TIMEOUT_RETRIES", "1"))
LLM_NETWORK_RETRIES: int = int(os.getenv("LLM_NETWORK_RETRIES", "2"))

# --- SSE 保活 & 流式超时 ---

# SSE 心跳间隔（秒），定期发送 SSE 注释保持连接活跃，防止中间代理断开
SSE_HEARTBEAT_INTERVAL: int = int(os.getenv("SSE_HEARTBEAT_INTERVAL", "15"))
# 流式响应中等待单个 chunk 的超时（秒），超过则认为上游已断开
# 设为 0 表示不限制
STREAM_CHUNK_TIMEOUT: float = float(os.getenv("STREAM_CHUNK_TIMEOUT", "300"))

# OpenAI Responses 非流式调用是否内部改走流式聚合。
# 某些中转在 Responses 非流式模式下会空返回，开启后可绕过此问题。
OPENAI_RESPONSES_USE_STREAM_FOR_NON_STREAM: bool = os.getenv(
    "OPENAI_RESPONSES_USE_STREAM_FOR_NON_STREAM", "true"
).lower() in ("true", "1", "yes")

REFINEMENT_NO_CLICHES_ENABLED: bool = os.getenv(
    "REFINEMENT_NO_CLICHES_ENABLED", "false"
).lower() in ("true", "1", "yes")
REFINEMENT_NO_CLICHES_MODEL: str = os.getenv(
    "REFINEMENT_NO_CLICHES_MODEL", "gemini-3.1-pro-preview"
).strip() or "gemini-3.1-pro-preview"
REFINEMENT_NO_CLICHES_PROVIDER: str = os.getenv(
    "REFINEMENT_NO_CLICHES_PROVIDER", "gemini"
).strip().lower() or "gemini"


def _load_non_negative_int(name: str, default: int) -> int:
    """加载非负整数环境变量，非法时回退默认值。"""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        logger.warning("[Config] Invalid %s=%r, fallback to %d", name, raw, default)
        return default
    if value < 0:
        logger.warning("[Config] %s must be >= 0, got %d, fallback to %d", name, value, default)
        return default
    return value


# --- JSON 修复阶段专属调试 ---

JSON_REPAIR_DEBUG_ENABLED: bool = os.getenv(
    "JSON_REPAIR_DEBUG_ENABLED", "false"
).lower() in ("true", "1", "yes")

_json_repair_debug_dir_raw = os.getenv(
    "JSON_REPAIR_DEBUG_DIR", "logs/json_repair_debug"
)
if os.path.isabs(_json_repair_debug_dir_raw):
    JSON_REPAIR_DEBUG_DIR: str = _json_repair_debug_dir_raw
else:
    JSON_REPAIR_DEBUG_DIR = str(
        (_BASE_DIR / _json_repair_debug_dir_raw).resolve()
    )

# 0 表示不截断，完整落盘
JSON_REPAIR_DEBUG_MAX_CHARS: int = _load_non_negative_int(
    "JSON_REPAIR_DEBUG_MAX_CHARS",
    0,
)

# --- 文本清洗阶段专属调试 ---

TEXT_CLEANER_DEBUG_ENABLED: bool = os.getenv(
    "TEXT_CLEANER_DEBUG_ENABLED", "false"
).lower() in ("true", "1", "yes")

_text_cleaner_debug_dir_raw = os.getenv(
    "TEXT_CLEANER_DEBUG_DIR", "logs/text_cleaner_debug"
)
if os.path.isabs(_text_cleaner_debug_dir_raw):
    TEXT_CLEANER_DEBUG_DIR: str = _text_cleaner_debug_dir_raw
else:
    TEXT_CLEANER_DEBUG_DIR = str(
        (_BASE_DIR / _text_cleaner_debug_dir_raw).resolve()
    )

# 0 表示不截断，完整落盘
TEXT_CLEANER_DEBUG_MAX_CHARS: int = _load_non_negative_int(
    "TEXT_CLEANER_DEBUG_MAX_CHARS",
    0,
)

# --- 精修初始专家请求调试 ---

REFINEMENT_EXPERT_REQUEST_DEBUG_ENABLED: bool = os.getenv(
    "REFINEMENT_EXPERT_REQUEST_DEBUG_ENABLED", "false"
).lower() in ("true", "1", "yes")

_refinement_expert_request_debug_dir_raw = os.getenv(
    "REFINEMENT_EXPERT_REQUEST_DEBUG_DIR", "logs/refinement_expert_request_debug"
)
if os.path.isabs(_refinement_expert_request_debug_dir_raw):
    REFINEMENT_EXPERT_REQUEST_DEBUG_DIR: str = _refinement_expert_request_debug_dir_raw
else:
    REFINEMENT_EXPERT_REQUEST_DEBUG_DIR = str(
        (_BASE_DIR / _refinement_expert_request_debug_dir_raw).resolve()
    )

# 0 表示不截断，完整落盘
REFINEMENT_EXPERT_REQUEST_DEBUG_MAX_CHARS: int = _load_non_negative_int(
    "REFINEMENT_EXPERT_REQUEST_DEBUG_MAX_CHARS",
    0,
)

# --- Checkpoint / Resume ---
_checkpoint_dir_raw = os.getenv("CHECKPOINT_DIR", "checkpoints")
if os.path.isabs(_checkpoint_dir_raw):
    CHECKPOINT_DIR: str = _checkpoint_dir_raw
else:
    CHECKPOINT_DIR = str((_BASE_DIR / _checkpoint_dir_raw).resolve())

CHECKPOINT_SCHEMA_VERSION: int = int(
    os.getenv("CHECKPOINT_SCHEMA_VERSION", "1")
)
CHECKPOINT_REPLAY_CHUNK_SIZE: int = int(
    os.getenv("CHECKPOINT_REPLAY_CHUNK_SIZE", "800")
)


# --- 思考预算定义 ---

THINKING_BUDGETS = {
    "minimal": 15360,
    "low": 15360,  # \
    "medium": 15360,
    "high": 32768,
    "high_pro": 32768,  # \w
}


def get_thinking_budget(level: str, model: str) -> int:
    """根据 Thinking Level 和模型返回 token 预算.

    Args:
        level: thinking level 字符串.
        model: 实际模型标识符.

    Returns:
        token 预算数.
    """
    is_pro = "pro" in model
    if level == "high" and is_pro:
        return THINKING_BUDGETS["high_pro"]
    return THINKING_BUDGETS.get(level, 0)


# --- 虚拟模型注册表 ---


@dataclass
class StageProviders:
    """各阶段使用的 Provider."""

    manager: str
    expert: str
    synthesis: str

    @classmethod
    def from_single(cls, provider: str) -> "StageProviders":
        """使用同一个 provider 填充三个阶段."""
        return cls(manager=provider, expert=provider, synthesis=provider)


@dataclass
class ExpertRoutingConfig:
    """专家执行底模分配配置."""

    expert_model_pool: list[ExpertModelProfile] = field(default_factory=list)
    enable_manager_expert_model_selection: bool = False
    enable_review_expert_model_selection: bool = False


@dataclass
class VirtualModel:
    """虚拟模型定义：对外暴露的模型名 -> 实际模型 + 思考预算.

    温度覆盖字段（planning_temperature / expert_temperature /
    review_temperature / synthesis_temperature）设为具体数值后，
    该阶段的温度会被强制锁定，忽略请求温度和 Manager 分配的温度。
    保持 None 则沿用原有行为（请求温度 / Manager 分配温度）。
    """

    id: str                    # 对外暴露的虚拟模型名
    real_model: str            # Expert 使用的实际模型
    planning_level: str        # Manager 规划阶段的 thinking level
    expert_level: str          # Expert 执行阶段的 thinking level
    synthesis_level: str       # Synthesis 综合阶段的 thinking level
    desc: str                  # 模型描述
    max_rounds: int = MAX_ROUNDS  # 最大审查轮数（默认走 .env）
    manager_model: Optional[str] = None    # Manager 专用模型（None则复用 real_model）
    synthesis_model: Optional[str] = None  # Synthesis 专用模型（None则复用 real_model）
    provider: str = ""  # LLM provider 标识符（空字符串则使用全局 LLM_PROVIDER）
    manager_provider: Optional[str] = None  # 规划/Review阶段 provider
    expert_provider: Optional[str] = None   # Expert/执行阶段 provider
    synthesis_provider: Optional[str] = None  # 综合/合并阶段 provider
    # 各阶段温度覆盖（None = 不覆盖，使用请求温度或 Manager 分配温度）
    planning_temperature: Optional[float] = None
    expert_temperature: Optional[float] = None
    review_temperature: Optional[float] = None
    synthesis_temperature: Optional[float] = None
    # 开启后，会在结构化 JSON 请求里额外通过 prompt 强制约束输出格式
    # 用于兼容不稳定或不支持 response_format 的 OpenAI 兼容渠道
    json_via_prompt: bool = False
    # --- 专家执行底模分配 ---
    expert_model_pool: list[ExpertModelProfile] = field(default_factory=list)
    enable_manager_expert_model_selection: bool = False
    enable_review_expert_model_selection: bool = False
    # --- 精修流程专用字段 ---
    mode: str = "classic"  # "classic" 或 "refinement"
    refinement_planner_model: Optional[str] = None  # 精修规划模型
    refinement_planner_provider: Optional[str] = None  # 精修规划 provider
    pre_draft_expert_model: Optional[str] = None  # 初稿前专家模型
    pre_draft_expert_provider: Optional[str] = None  # 初稿前专家 provider
    pre_draft_review_model: Optional[str] = None  # 初稿前审查模型
    pre_draft_review_provider: Optional[str] = None  # 初稿前审查 provider
    draft_model: Optional[str] = None  # 初稿生成模型
    draft_provider: Optional[str] = None  # 初稿生成 provider
    review_model: Optional[str] = None  # 审查阶段模型
    review_provider: Optional[str] = None  # 审查阶段 provider
    improver_model: Optional[str] = None  # 改进专家模型
    improver_provider: Optional[str] = None  # 改进专家 provider
    merge_model: Optional[str] = None  # 综合助手模型
    merge_provider: Optional[str] = None  # 综合助手 provider
    text_cleaner_model: Optional[str] = None  # 文本清洗模型
    text_cleaner_provider: Optional[str] = None  # 文本清洗 provider
    json_repair_model: Optional[str] = None  # JSON 修复模型
    refinement_max_rounds: int = 2  # 精修最大迭代轮数
    pre_draft_review_rounds: int = 1  # pre-draft review rounds (0=disabled)
    enable_json_repair: bool = False  # 是否启用 JSON 修复
    enable_text_cleaner: bool = True  # 是否启用末端文本清洗专家（默认启用）
    enable_no_cliches: Optional[bool] = None  # 是否启用强力杀八股
    no_cliches_model: Optional[str] = None  # 杀八股模型
    no_cliches_provider: Optional[str] = None  # 杀八股 provider


# 注册所有虚拟模型（这里不包括env的）
VIRTUAL_MODELS: list[VirtualModel] = [
    # 快速测试用的
    VirtualModel(
        id="gemini-3-flash-deepthink-test",
        real_model="gemini-3-flash-preview",
        manager_model="gemini-3-flash-preview",
        synthesis_model="gemini-3-flash-preview",
        planning_level="high",
        expert_level="high",
        synthesis_level="high",
        max_rounds=2,
        desc="3 Flash + Low thinking budget. 测试用",
    ),
    # Kimi（k2.5官API不用1温就报400，此外对json格式遵循很差，经常返错的json）
    VirtualModel(
        id="kimi-k2.5-deepthink-test",
        real_model="kimi-k2.5",
        manager_model="kimi-k2.5",
        synthesis_model="kimi-k2.5",
        planning_level="high",
        expert_level="high",
        synthesis_level="high",
        provider="openai",
        max_rounds=2,
        planning_temperature=1,
        expert_temperature=1,
        review_temperature=1,
        synthesis_temperature=1,
        desc="Kimi K2.5 + Low thinking budget. 测试用",
    ),
    # Deepseek（废弃，DS官API根本不支持response_format）
    # VirtualModel(
    #     id="deepseek-v3.2-deepthink-test",
    #     real_model="deepseek-reasoner",
    #     manager_model="deepseek-reasoner",
    #     synthesis_model="deepseek-reasoner",
    #     planning_level="high",
    #     expert_level="high",
    #     synthesis_level="high",
    #     provider="openai",
    #     max_rounds=2,
    #     desc="Deepseek V3.2 + Low thinking budget. 测试用",
    # ),

    # --- Gemini 3.1 ---
    VirtualModel(
        id="gpt-5.4-deepthink-minimal",
        real_model="gpt-5.4",
        manager_model="gemini-3.1-pro-preview",
        synthesis_model="gemini-3.1-pro-preview", # 没事不用改了，就用它总结
        planning_level="medium",
        expert_level="low",
        synthesis_level="medium",
        max_rounds=1,
        provider="gemini",
        manager_provider="gemini",
        expert_provider="openai_responses",
        synthesis_provider="gemini",
        desc="GPT-5.4 联网搜索最小混合版。Manager/Review/Synthesis 走 Gemini，Expert 走 Responses。",
    ),
    VirtualModel(
        id="gemini-3.1-pro-deepthink-minimal",
        real_model="gemini-3.1-pro-preview",
        manager_model="gpt-5.4-xhigh",
        manager_provider="openai_responses",
        synthesis_model="gemini-3.1-pro-preview",
        planning_level="high",
        expert_level="high",
        synthesis_level="high",
        expert_model_pool=[
            ExpertModelProfile(
                id="gpt-5.4-high",
                model="gpt-5.4-high",
                provider="openai_responses",
                description=(
                    "逻辑强、创意一般、擅长抓bug、非创意类规划、代码审查、找逻辑漏洞（各种领域）、在线搜索。"
                    "前端审美和长文表达很一般。"
                ),
            ),
            ExpertModelProfile(
                id="gemini-3.1-pro-preview",
                model="gemini-3.1-pro-preview",
                provider="gemini",
                description=(
                    "参数量、知识量非常高创意强、头脑风暴强、前端审美、文字表达、整体呈现很强。可搜索但很难给出精确出处"
                    "小毛病偏多，代码稳定性略弱。"
                ),
            ),
        ],
        enable_manager_expert_model_selection=True,
        enable_review_expert_model_selection=True,
        max_rounds=1,
        desc="3.1 Pro + Low thinking budget. 单轮直出，不审查。",
    ),
    VirtualModel(
        id="gemini-3.1-pro-deepthink-low-no-cliches",
        real_model="gemini-3.1-pro-preview",
        manager_model="gemini-3.1-pro-preview",
        synthesis_model="gemini-3.1-pro-preview",
        planning_level="medium",
        expert_level="medium",
        synthesis_level="high",
        max_rounds=2,
        enable_no_cliches=True,
        no_cliches_model="gemini-3.1-pro-preview",
        no_cliches_provider="gemini",
        desc="3.1 Pro + Low thinking budget. 1轮审查+去八股",
    ),
    VirtualModel(
        id="gemini-3.1-pro-deepthink-low",
        real_model="gemini-3.1-pro-preview",
        manager_model="gpt-5.4-xhigh",
        manager_provider="openai_responses",
        synthesis_model="gemini-3.1-pro-preview",
        planning_level="high",
        expert_level="high",
        synthesis_level="high",
        expert_model_pool=[
            ExpertModelProfile(
                id="gpt-5.4-high",
                model="gpt-5.4-high",
                provider="openai_responses",
                description=(
                    "逻辑强、创意一般、擅长抓bug、非创意类规划、代码审查、找逻辑漏洞（各种领域）、在线搜索。"
                    "前端审美和长文表达很一般。"
                ),
            ),
            ExpertModelProfile(
                id="gemini-3.1-pro-preview",
                model="gemini-3.1-pro-preview",
                provider="gemini",
                description=(
                    "参数量、知识量非常高创意强、头脑风暴强、前端审美、文字表达、整体呈现很强。可搜索但很难给出精确出处"
                    "小毛病偏多，代码稳定性略弱。"
                ),
            ),
        ],
        enable_manager_expert_model_selection=True,
        enable_review_expert_model_selection=True,
        max_rounds=2,
        desc="3.1 Pro + Low thinking budget. 1轮审查，合适日常任务用",
    ),
    VirtualModel(
        id="gemini-3.1-pro-deepthink-medium",
        real_model="gemini-3.1-pro-preview",
        manager_model="gemini-3.1-pro-preview",
        synthesis_model="gemini-3.1-pro-preview",
        planning_level="high",
        expert_level="medium",
        synthesis_level="high",
        max_rounds=3,
        desc="3.1 Pro + Medium thinking budget. 2轮审查，合适中等任务用",
    ),
    VirtualModel(
        id="gemini-3.1-pro-deepthink-high",
        real_model="gemini-3.1-pro-preview",
        manager_model="gemini-3.1-pro-preview",
        synthesis_model="gemini-3.1-pro-preview",
        planning_level="high",
        expert_level="high",
        synthesis_level="high",
        max_rounds=5,
        desc="3.1 Pro + High thinking budget. 最多5轮深度审查。合适高难任务",
    ),
    VirtualModel(
        id="gemini-3.1-pro-deepthink-extra",
        real_model="gemini-3.1-pro-preview",
        manager_model="gemini-3.1-pro-preview",
        synthesis_model="gemini-3.1-pro-preview",
        planning_level="high",
        expert_level="high",
        synthesis_level="high",
        max_rounds=12,
        desc="3.1 Pro + High budget + 最多12轮极限审查。慎用，耗时可能很长。",
    ),
    VirtualModel(
        id="gemini-3.1-pro-deepthink-refinement-low",
        real_model="gemini-3.1-pro-preview",
        manager_model="gemini-3.1-pro-preview",
        synthesis_model="gemini-3.1-pro-preview",
        json_repair_model="gemini-3-flash-preview",
        mode="refinement",
        draft_model="claude-opus-4-6-thinking",
        planning_level="high",
        expert_level="high",
        synthesis_level="high",
        refinement_max_rounds=1,
        pre_draft_review_rounds=1,
        enable_json_repair=False,
        max_rounds=1,
        desc="3.1 Pro 精修流程实验模式，侧重写作精修改进"
    ),
    VirtualModel(
        id="gemini-3.1-pro-deepthink-refinement-medium",
        real_model="gemini-3.1-pro-preview",
        manager_model="gemini-3.1-pro-preview",
        synthesis_model="gemini-3.1-pro-preview",
        json_repair_model="gemini-3-flash-preview",
        mode="refinement",
        draft_model="claude-opus-4-6-thinking",
        planning_level="high",
        expert_level="high",
        synthesis_level="high",
        refinement_max_rounds=2,
        pre_draft_review_rounds=2,
        enable_json_repair=False,
        max_rounds=1,
        desc="3.1 Pro 精修流程实验模式，侧重写作精修改进"
    ),
    VirtualModel(
        id="gemini-3.1-pro-deepthink-refinement-high",
        real_model="gemini-3.1-pro-preview",
        manager_model="gemini-3.1-pro-preview",
        synthesis_model="gemini-3.1-pro-preview",
        json_repair_model="gemini-3-flash-preview",
        mode="refinement",
        draft_model="claude-opus-4-6-thinking",
        planning_level="high",
        expert_level="high",
        synthesis_level="high",
        refinement_max_rounds=5,
        pre_draft_review_rounds=3,
        enable_json_repair=False,
        max_rounds=1,
        desc="3.1 Pro 精修流程实验模式，侧重写作精修改进"
    ),
    # gpt文本能力不行，但纠错能力极强，故仅用于审查和分配专家
    VirtualModel(
        id="gemini-gpt-deepthink-refinement-high",
        real_model="gemini-3.1-pro-preview",
        manager_model="gpt-5.4-high",
        synthesis_model="gemini-3.1-pro-preview",
        mode="refinement",
        refinement_planner_model="gpt-5.4-high",
        refinement_planner_provider="openai_responses",
        pre_draft_expert_model="gemini-3.1-pro-preview",
        pre_draft_expert_provider="gemini",
        pre_draft_review_model="gpt-5.4-high",
        pre_draft_review_provider="openai_responses",
        draft_model="claude-opus-4-6-thinking",
        draft_provider="gemini",
        review_model="gpt-5.4-high",
        review_provider="openai_responses",
        improver_model="gemini-3.1-pro-preview",
        improver_provider="gemini",
        merge_model="gemini-3.1-pro-preview",
        merge_provider="gemini",
        text_cleaner_model="gemini-3.1-pro-preview",
        text_cleaner_provider="gemini",
        json_repair_model="gemini-3-flash-preview",
        planning_level="high",
        expert_level="high",
        synthesis_level="high",
        expert_temperature=2.0,
        refinement_max_rounds=2,
        pre_draft_review_rounds=4,
        enable_json_repair=False,
        max_rounds=1,
        provider="gemini",
        manager_provider="openai_responses",
        expert_provider="gemini",
        synthesis_provider="gemini",
        enable_no_cliches=True,
        no_cliches_model="gemini-2.5-pro",
        no_cliches_provider="gemini",
        desc="混合精修高档：规划/预审/初稿审查走 GPT-5.4，初稿前专家、改进与清洗走 Gemini，初稿写作走 Claude。",
    ),
    VirtualModel(
        id="gemini-gpt-deepthink-refinement-code-xhigh",
        real_model="gemini-3.1-pro-preview",
        manager_model="gpt-5.4-high",
        synthesis_model="gemini-3.1-pro-preview",
        mode="refinement",
        refinement_planner_model="gpt-5.4-high",
        refinement_planner_provider="openai_responses",
        pre_draft_expert_model="gemini-3.1-pro-preview",
        pre_draft_expert_provider="gemini",
        pre_draft_review_model="gpt-5.4-high",
        pre_draft_review_provider="openai_responses",
        draft_model="gemini-3.1-pro-preview",
        draft_provider="gemini",
        review_model="gpt-5.4-high",
        review_provider="openai_responses",
        improver_model="gemini-3.1-pro-preview",
        improver_provider="gemini",
        merge_model="gemini-3.1-pro-preview",
        merge_provider="gemini",
        text_cleaner_model="gemini-3.1-pro-preview",
        text_cleaner_provider="gemini",
        json_repair_model="gemini-3-flash-preview",
        planning_level="high",
        expert_level="high",
        synthesis_level="high",
        expert_temperature=1.0,
        refinement_max_rounds=5,
        pre_draft_review_rounds=3,
        enable_json_repair=False,
        max_rounds=1,
        provider="gemini",
        manager_provider="openai_responses",
        expert_provider="gemini",
        synthesis_provider="gemini",
        expert_model_pool=[
            ExpertModelProfile(
                id="gpt-5.4-high",
                model="gpt-5.4-high",
                provider="openai_responses",
                description=(
                    "创意一般、擅长抓bug、非创意类规划、代码审查、找逻辑漏洞（各种领域）、在线搜索。"
                    "前端审美和长文表达很一般。"
                ),
            ),
            ExpertModelProfile(
                id="gemini-3.1-pro-preview",
                model="gemini-3.1-pro-preview",
                provider="gemini",
                description=(
                    "创意强、头脑风暴强、前端审美、文字表达、整体呈现很强。"
                    "但小毛病偏多，代码稳定性略弱。"
                ),
            ),
        ],
        enable_manager_expert_model_selection=True,
        enable_review_expert_model_selection=True,
        desc="混合精修高档，代码专精：规划/预审/初稿审查走 GPT-5.4，初稿前专家、改进与清洗走 Gemini",
    ),
    VirtualModel(
        id="gemini-3.1-pro-deepthink-refinement-extra",
        real_model="gemini-3.1-pro-preview",
        manager_model="gemini-3.1-pro-preview",
        synthesis_model="gemini-3.1-pro-preview",
        json_repair_model="gemini-3-flash-preview",
        mode="refinement",
        draft_model="claude-opus-4-6-thinking",
        planning_level="high",
        expert_level="high",
        synthesis_level="high",
        refinement_max_rounds=10,
        pre_draft_review_rounds=5,
        enable_json_repair=False,
        max_rounds=1,
        desc="3.1 Pro 精修流程实验模式，侧重写作精修改进"
    ),
    # 快速精修测试flash
    VirtualModel(
        id="gemini-3-flash-deepthink-refinement-medium",
        real_model="gemini-3-flash-preview",
        manager_model="gemini-3-flash-preview",
        synthesis_model="gemini-3-flash-preview",
        json_repair_model="gemini-3-flash-preview",
        mode="refinement",
        draft_model="claude-opus-4-6-thinking",
        planning_level="high",
        expert_level="high",
        synthesis_level="high",
        refinement_max_rounds=2,
        pre_draft_review_rounds=1,
        enable_json_repair=False,
        max_rounds=1,
        desc="3.1 Flash 精修流程实验模式，侧重写作精修改进"
    ),
]


# --- 加载用户自定义虚拟模型 ---


def _parse_expert_model_pool(
    raw_pool: object,
) -> list[ExpertModelProfile]:
    """解析 expert_model_pool 字段."""
    if raw_pool is None:
        return []

    pool_items: list[object]
    if isinstance(raw_pool, dict):
        pool_items = []
        for item_id, item_value in raw_pool.items():
            if isinstance(item_value, dict):
                merged = dict(item_value)
                merged.setdefault("id", str(item_id))
                merged.setdefault("model", str(item_id))
                pool_items.append(merged)
            else:
                pool_items.append({
                    "id": str(item_id),
                    "model": str(item_id),
                    "description": str(item_value),
                })
    elif isinstance(raw_pool, list):
        pool_items = raw_pool
    else:
        raise ValueError("expert_model_pool must be an array or object")

    parsed: list[ExpertModelProfile] = []
    for idx, item in enumerate(pool_items, start=1):
        if isinstance(item, str):
            text = item.strip()
            if not text:
                raise ValueError(f"expert_model_pool item #{idx} is empty")
            parsed.append(ExpertModelProfile(id=text, model=text))
            continue
        if not isinstance(item, dict):
            raise ValueError(
                f"expert_model_pool item #{idx} must be an object or string"
            )

        item_id = str(item.get("id") or item.get("model") or "").strip()
        model_name = str(
            item.get("model") or item.get("real_model") or item_id
        ).strip()
        if not item_id:
            raise ValueError(f"expert_model_pool item #{idx} missing id/model")
        if not model_name:
            raise ValueError(f"expert_model_pool item #{idx} missing model")

        parsed.append(
            ExpertModelProfile(
                id=item_id,
                model=model_name,
                provider=str(item.get("provider") or "").strip().lower(),
                description=str(
                    item.get("description")
                    or item.get("desc")
                    or item.get("summary")
                    or ""
                ).strip(),
            )
        )
    return parsed

def _load_extra_virtual_models() -> list[VirtualModel]:
    """从 .env 配置加载用户自定义的虚拟模型.

    支持两种方式：
      1. VIRTUAL_MODELS_FILE: 指向 JSON 文件路径（相对或绝对）
      2. VIRTUAL_MODELS_EXTRA: 直接写 JSON 数组字符串
    _FILE 优先于 _EXTRA。

    JSON 格式示例::

        [
            {
                "id": "my-custom-deepthink",
                "real_model": "gemini-3-flash-preview",
                "planning_level": "medium",
                "expert_level": "medium",
                "synthesis_level": "medium",
                "desc": "自定义模型描述",
                "max_rounds": 3,
                "manager_model": null,
                "synthesis_model": null,
                "expert_temperature": 1.0
            }
        ]

    其中 max_rounds / manager_model / synthesis_model / provider 为可选字段。
    planning_temperature / expert_temperature / review_temperature /
    synthesis_temperature 也为可选字段，设为具体数值后该阶段温度会被锁定。

    Returns:
        解析出的 VirtualModel 列表，解析失败返回空列表。
    """
    raw_json = None

    # 方式 1：从文件加载
    file_path = os.getenv("VIRTUAL_MODELS_FILE")
    if file_path:
        resolved = (
            _BASE_DIR / file_path
            if not os.path.isabs(file_path)
            else Path(file_path)
        )
        try:
            raw_json = resolved.read_text(encoding="utf-8").strip()
            logger.info(
                "[Config] Loaded custom virtual models from file: %s", resolved
            )
        except FileNotFoundError:
            logger.warning(
                "[Config] VIRTUAL_MODELS_FILE does not exist: %s",
                resolved,
            )
        except Exception as e:
            logger.warning(
                "[Config] Failed to read VIRTUAL_MODELS_FILE: %s", e
            )

    # 方式 2：从环境变量直接读取 JSON
    if raw_json is None:
        raw_json = os.getenv("VIRTUAL_MODELS_EXTRA")
        if raw_json:
            logger.info(
                "[Config] Loaded custom virtual models from env (%d chars)",
                len(raw_json),
            )

    if not raw_json:
        return []

    try:
        items = json.loads(raw_json)
        if not isinstance(items, list):
            logger.error(
                "[Config] Custom virtual model JSON must be an array, got: %s",
                type(items).__name__,
            )
            return []
    except json.JSONDecodeError as e:
        logger.error("[Config] Failed to parse custom virtual model JSON: %s", e)
        return []

    models: list[VirtualModel] = []
    for idx, item in enumerate(items):
        try:
            # 必填字段检查
            for field in ("id", "real_model", "planning_level",
                          "expert_level", "synthesis_level", "desc"):
                if field not in item:
                    raise ValueError(f"Missing required field: {field}")

            vm = VirtualModel(
                id=item["id"],
                real_model=item["real_model"],
                planning_level=item["planning_level"],
                expert_level=item["expert_level"],
                synthesis_level=item["synthesis_level"],
                desc=item["desc"],
                max_rounds=item.get("max_rounds", MAX_ROUNDS),
                manager_model=item.get("manager_model"),
                synthesis_model=item.get("synthesis_model"),
                provider=item.get("provider", ""),
                manager_provider=item.get("manager_provider"),
                expert_provider=item.get("expert_provider"),
                synthesis_provider=item.get("synthesis_provider"),
                planning_temperature=item.get("planning_temperature"),
                expert_temperature=item.get("expert_temperature"),
                review_temperature=item.get("review_temperature"),
                synthesis_temperature=item.get("synthesis_temperature"),
                json_via_prompt=item.get("json_via_prompt", False),
                expert_model_pool=_parse_expert_model_pool(
                    item.get("expert_model_pool", item.get("expert_models"))
                ),
                enable_manager_expert_model_selection=item.get(
                    "enable_manager_expert_model_selection",
                    False,
                ),
                enable_review_expert_model_selection=item.get(
                    "enable_review_expert_model_selection",
                    False,
                ),
                mode=item.get("mode", "classic"),
                refinement_planner_model=item.get("refinement_planner_model"),
                refinement_planner_provider=item.get("refinement_planner_provider"),
                pre_draft_expert_model=item.get("pre_draft_expert_model"),
                pre_draft_expert_provider=item.get("pre_draft_expert_provider"),
                pre_draft_review_model=item.get("pre_draft_review_model"),
                pre_draft_review_provider=item.get("pre_draft_review_provider"),
                draft_model=item.get("draft_model"),
                draft_provider=item.get("draft_provider"),
                review_model=item.get("review_model"),
                review_provider=item.get("review_provider"),
                improver_model=item.get("improver_model"),
                improver_provider=item.get("improver_provider"),
                merge_model=item.get("merge_model"),
                merge_provider=item.get("merge_provider"),
                text_cleaner_model=item.get("text_cleaner_model"),
                text_cleaner_provider=item.get("text_cleaner_provider"),
                json_repair_model=item.get("json_repair_model"),
                refinement_max_rounds=item.get("refinement_max_rounds", 2),
                pre_draft_review_rounds=item.get("pre_draft_review_rounds", 1),
                enable_json_repair=item.get("enable_json_repair", False),
                enable_text_cleaner=item.get("enable_text_cleaner", True),
                enable_no_cliches=item.get("enable_no_cliches"),
                no_cliches_model=item.get("no_cliches_model"),
                no_cliches_provider=item.get("no_cliches_provider"),
            )
            models.append(vm)
            logger.info(
                "[Config] Loaded custom virtual model: %s -> %s",
                vm.id, vm.real_model,
            )
        except Exception as e:
            logger.error(
                "[Config] Failed to parse custom virtual model #%d: %s",
                idx + 1, e,
            )

    return models


def _merge_virtual_models(
    defaults: list[VirtualModel],
    extras: list[VirtualModel],
) -> list[VirtualModel]:
    """合并默认和自定义虚拟模型列表.

    如果自定义模型的 id 与默认模型冲突，自定义的会覆盖默认的。

    Args:
        defaults: 硬编码的默认模型列表.
        extras: 用户自定义的模型列表.

    Returns:
        合并后的完整模型列表.
    """
    extra_ids = {vm.id for vm in extras}
    # 保留不被覆盖的默认模型
    merged = [vm for vm in defaults if vm.id not in extra_ids]
    merged.extend(extras)
    if extra_ids:
        overridden = extra_ids & {vm.id for vm in defaults}
        if overridden:
            logger.info(
                "[Config] Default models overridden by custom config: %s",
                ", ".join(sorted(overridden)),
            )
    return merged


_extra_models = _load_extra_virtual_models()
if _extra_models:
    VIRTUAL_MODELS = _merge_virtual_models(VIRTUAL_MODELS, _extra_models)
    logger.info(
        "[Config] Total virtual models: %d (default %d + custom %d)",
        len(VIRTUAL_MODELS),
        len(VIRTUAL_MODELS) - len(_extra_models),
        len(_extra_models),
    )

# 快速查找表
_VIRTUAL_MODEL_MAP: dict[str, VirtualModel] = {
    vm.id: vm for vm in VIRTUAL_MODELS
}

FORCED_MODEL_SUFFIX = "-forced"


def split_forced_model_suffix(model_id: str) -> tuple[str, bool]:
    """拆分模型名末尾的 forced 后缀。

    Args:
        model_id: 请求传入的模型名。

    Returns:
        (base_model_id, forced_enabled) 元组。
    """
    if model_id.endswith(FORCED_MODEL_SUFFIX):
        base = model_id[: -len(FORCED_MODEL_SUFFIX)]
        if base:
            return base, True
    return model_id, False


def split_provider_model_prefix(model_id: str) -> tuple[str, str] | None:
    """拆分 provider/model 直通语法。

    Args:
        model_id: 请求传入的模型名。

    Returns:
        命中时返回 (provider, real_model)，否则返回 None。
    """
    provider, sep, real_model = model_id.partition("/")
    if not sep:
        return None
    provider = provider.strip().lower()
    real_model = real_model.strip()
    if not provider or not real_model:
        return None
    if provider not in PROVIDER_CONFIGS:
        return None
    return provider, real_model


# resolve_model 返回类型
_ResolveResult = tuple[
    str, str, str,               # real_model, manager_model, synthesis_model
    str, str, str,               # planning_level, expert_level, synthesis_level
    int, str,                    # max_rounds, legacy single provider
    Optional[float],             # planning_temperature
    Optional[float],             # expert_temperature
    Optional[float],             # review_temperature
    Optional[float],             # synthesis_temperature
    str,                         # mode ("classic" / "refinement")
    bool,                        # json_via_prompt
    StageProviders,              # stage_providers (manager/expert/synthesis)
]


def resolve_model(model_id: str) -> _ResolveResult:
    """解析虚拟模型名，返回各阶段实际模型、思考预算、最大轮数、provider、温度覆盖、mode、JSON 提示增强开关和阶段 provider.

    Args:
        model_id: 虚拟模型名或实际模型名.

    Returns:
        (real_model, manager_model, synthesis_model,
         planning_level, expert_level, synthesis_level,
         max_rounds, provider,
         planning_temperature, expert_temperature,
         review_temperature, synthesis_temperature,
         mode, json_via_prompt) 元组.
    """
    base_model_id, _ = split_forced_model_suffix(model_id)
    vm = _VIRTUAL_MODEL_MAP.get(base_model_id)
    if vm:
        mgr_model = vm.manager_model or vm.real_model
        syn_model = vm.synthesis_model or vm.real_model
        provider = vm.provider or LLM_PROVIDER
        stage_providers = StageProviders(
            manager=vm.manager_provider or provider,
            expert=vm.expert_provider or provider,
            synthesis=vm.synthesis_provider or provider,
        )
        return (
            vm.real_model, mgr_model, syn_model,
            vm.planning_level, vm.expert_level, vm.synthesis_level,
            vm.max_rounds, provider,
            vm.planning_temperature, vm.expert_temperature,
            vm.review_temperature, vm.synthesis_temperature,
            vm.mode,
            vm.json_via_prompt,
            stage_providers,
        )

    direct_provider = split_provider_model_prefix(base_model_id)
    if direct_provider:
        provider, real_model = direct_provider
        stage_providers = StageProviders.from_single(provider)
        return (
            real_model, real_model, real_model,
            "high", "high", "high",
            MAX_ROUNDS, provider,
            None, None, None, None,
            "classic",
            False,
            stage_providers,
        )

    # 未注册的模型名，直接透传，默认 high + .env 的 MAX_ROUNDS + 全局 provider + 无温度覆盖 + classic
    stage_providers = StageProviders.from_single(LLM_PROVIDER)
    return (
        base_model_id, base_model_id, base_model_id,
        "high", "high", "high",
        MAX_ROUNDS, LLM_PROVIDER,
        None, None, None, None,
        "classic",
        False,
        stage_providers,
    )


@dataclass
class RefinementModelConfig:
    """精修流程各阶段模型配置."""
    refinement_planner_model: str  # 精修规划模型
    refinement_planner_provider: str  # 精修规划 provider
    pre_draft_expert_model: str  # 初稿前专家模型
    pre_draft_expert_provider: str  # 初稿前专家 provider
    pre_draft_review_model: str  # 初稿前审查模型
    pre_draft_review_provider: str  # 初稿前审查 provider
    draft_model: str       # 初稿生成模型
    draft_provider: str  # 初稿生成 provider
    review_model: str      # 审查模型
    review_provider: str  # 审查 provider
    improver_model: str  # 改进专家模型
    improver_provider: str  # 改进专家 provider
    merge_model: str       # 综合助手模型
    merge_provider: str  # 综合助手 provider
    text_cleaner_model: str  # 文本清洗模型
    text_cleaner_provider: str  # 文本清洗 provider
    json_repair_model: str  # JSON 修复模型
    refinement_max_rounds: int = 2
    pre_draft_review_rounds: int = 1  # pre-draft review rounds (0=disabled)
    enable_json_repair: bool = False
    enable_text_cleaner: bool = True
    enable_no_cliches: bool = False
    no_cliches_model: str = REFINEMENT_NO_CLICHES_MODEL
    no_cliches_provider: str = REFINEMENT_NO_CLICHES_PROVIDER


@dataclass
class NoClichesConfig:
    """强力杀八股配置。"""

    enable_no_cliches: bool
    no_cliches_model: str
    no_cliches_provider: str


def resolve_no_cliches_config(
    model_id: str,
    real_model: str,
    stage_providers: StageProviders | None = None,
) -> NoClichesConfig:
    """解析强力杀八股配置。"""
    del real_model
    del stage_providers
    base_model_id, _ = split_forced_model_suffix(model_id)
    vm = _VIRTUAL_MODEL_MAP.get(base_model_id)

    if vm:
        return NoClichesConfig(
            enable_no_cliches=(
                vm.enable_no_cliches
                if vm.enable_no_cliches is not None
                else REFINEMENT_NO_CLICHES_ENABLED
            ),
            no_cliches_model=vm.no_cliches_model or REFINEMENT_NO_CLICHES_MODEL,
            no_cliches_provider=(
                vm.no_cliches_provider or REFINEMENT_NO_CLICHES_PROVIDER
            ),
        )

    return NoClichesConfig(
        enable_no_cliches=REFINEMENT_NO_CLICHES_ENABLED,
        no_cliches_model=REFINEMENT_NO_CLICHES_MODEL,
        no_cliches_provider=REFINEMENT_NO_CLICHES_PROVIDER,
    )


def resolve_expert_routing_config(
    model_id: str,
) -> ExpertRoutingConfig:
    """解析虚拟模型的专家执行底模分配配置。"""
    base_model_id, _ = split_forced_model_suffix(model_id)
    vm = _VIRTUAL_MODEL_MAP.get(base_model_id)
    if not vm:
        return ExpertRoutingConfig()

    return ExpertRoutingConfig(
        expert_model_pool=list(vm.expert_model_pool),
        enable_manager_expert_model_selection=(
            vm.enable_manager_expert_model_selection
        ),
        enable_review_expert_model_selection=(
            vm.enable_review_expert_model_selection
        ),
    )


def resolve_expert_model_selection(
    expert_model: str,
    default_model: str,
    default_provider: str,
    expert_model_pool: list[ExpertModelProfile] | None = None,
) -> tuple[str, str, str]:
    """将 expert_model 解析为实际执行的 model/provider。"""
    selected = (expert_model or "").strip()
    if not selected:
        return "", default_model, default_provider

    for profile in expert_model_pool or []:
        if selected == profile.id or selected == profile.model:
            return (
                profile.id,
                profile.model,
                profile.provider or default_provider,
            )

    direct_provider = split_provider_model_prefix(selected)
    if direct_provider:
        provider, real_model = direct_provider
        return selected, real_model, provider

    logger.warning(
        "[Config] Unknown expert_model=%r, fallback to default %s@%s",
        selected,
        default_model,
        default_provider,
    )
    return selected, default_model, default_provider


def resolve_refinement_config(
    model_id: str,
    real_model: str,
    mgr_model: str,
    syn_model: str,
    stage_providers: StageProviders | None = None,
) -> RefinementModelConfig:
    """解析虚拟模型精修流程配置.

    Args:
        model_id: 虚拟模型名.
        real_model: 已解析的 Expert 模型.
        mgr_model: 已解析的 Manager 模型.
        syn_model: 已解析的 Synthesis 模型.
        stage_providers: 已解析的阶段 provider.

    Returns:
        RefinementModelConfig 实例.
    """
    base_model_id, _ = split_forced_model_suffix(model_id)
    vm = _VIRTUAL_MODEL_MAP.get(base_model_id)
    default_small = "gemini-3-flash-preview"
    stage_providers = stage_providers or StageProviders.from_single(LLM_PROVIDER)

    if vm:
        return RefinementModelConfig(
            refinement_planner_model=vm.refinement_planner_model or mgr_model,
            refinement_planner_provider=(
                vm.refinement_planner_provider or stage_providers.manager
            ),
            pre_draft_expert_model=vm.pre_draft_expert_model or real_model,
            pre_draft_expert_provider=(
                vm.pre_draft_expert_provider or stage_providers.expert
            ),
            pre_draft_review_model=vm.pre_draft_review_model or mgr_model,
            pre_draft_review_provider=(
                vm.pre_draft_review_provider or stage_providers.manager
            ),
            draft_model=vm.draft_model or real_model,
            draft_provider=vm.draft_provider or stage_providers.expert,
            review_model=vm.review_model or mgr_model,
            review_provider=vm.review_provider or stage_providers.manager,
            improver_model=vm.improver_model or real_model,
            improver_provider=vm.improver_provider or stage_providers.expert,
            merge_model=vm.merge_model or syn_model,
            merge_provider=vm.merge_provider or stage_providers.synthesis,
            text_cleaner_model=vm.text_cleaner_model or (vm.merge_model or syn_model),
            text_cleaner_provider=(
                vm.text_cleaner_provider
                or vm.merge_provider
                or stage_providers.synthesis
            ),
            json_repair_model=vm.json_repair_model or default_small,
            refinement_max_rounds=vm.refinement_max_rounds,
            pre_draft_review_rounds=vm.pre_draft_review_rounds,
            enable_json_repair=vm.enable_json_repair,
            enable_text_cleaner=vm.enable_text_cleaner,
            enable_no_cliches=(
                vm.enable_no_cliches
                if vm.enable_no_cliches is not None
                else REFINEMENT_NO_CLICHES_ENABLED
            ),
            no_cliches_model=vm.no_cliches_model or REFINEMENT_NO_CLICHES_MODEL,
            no_cliches_provider=(
                vm.no_cliches_provider or REFINEMENT_NO_CLICHES_PROVIDER
            ),
        )

    return RefinementModelConfig(
        refinement_planner_model=mgr_model,
        refinement_planner_provider=stage_providers.manager,
        pre_draft_expert_model=real_model,
        pre_draft_expert_provider=stage_providers.expert,
        pre_draft_review_model=mgr_model,
        pre_draft_review_provider=stage_providers.manager,
        draft_model=real_model,
        draft_provider=stage_providers.expert,
        review_model=mgr_model,
        review_provider=stage_providers.manager,
        improver_model=real_model,
        improver_provider=stage_providers.expert,
        merge_model=syn_model,
        merge_provider=stage_providers.synthesis,
        text_cleaner_model=syn_model,
        text_cleaner_provider=stage_providers.synthesis,
        json_repair_model=default_small,
        enable_no_cliches=REFINEMENT_NO_CLICHES_ENABLED,
        no_cliches_model=REFINEMENT_NO_CLICHES_MODEL,
        no_cliches_provider=REFINEMENT_NO_CLICHES_PROVIDER,
    )


