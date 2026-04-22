"""Prisma DeepThink API entrypoint."""

import logging
import logging.handlers
import os
import sys
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import (
    HOST,
    LLM_PROVIDER,
    PORT,
    PROVIDER_CONFIGS,
)
from routes.chat import router as chat_router
from routes.gemini import router as gemini_router

# Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for h in logging.getLogger().handlers:
    if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
        h.setLevel(logging.INFO)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"prisma_{datetime.now().strftime('%Y-%m-%d')}.log")
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger("prisma")

app = FastAPI(
    title="Prisma DeepThink API",
    description=(
        "Multi-agent deep reasoning API with OpenAI-compatible "
        "and Gemini-native endpoints."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(gemini_router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Prisma DeepThink API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/v1/models")
async def list_models():
    """List available virtual models (OpenAI compatible)."""
    from config import VIRTUAL_MODELS

    return {
        "object": "list",
        "data": [
            {
                "id": vm.id,
                "object": "model",
                "owned_by": "prisma",
                "description": vm.desc,
                "provider": vm.provider or LLM_PROVIDER,
                "stage_providers": {
                    "manager": vm.manager_provider or vm.provider or LLM_PROVIDER,
                    "expert": vm.expert_provider or vm.provider or LLM_PROVIDER,
                    "synthesis": vm.synthesis_provider or vm.provider or LLM_PROVIDER,
                },
            }
            for vm in VIRTUAL_MODELS
        ],
    }


def main():
    """Start API service."""
    # 检查全局默认 provider 的 API Key
    default_cfg = PROVIDER_CONFIGS.get(LLM_PROVIDER)
    if default_cfg and not default_cfg.api_key:
        logger.error(
            "Default provider %r has no API key configured. "
            "Please set the corresponding env var in .env.",
            LLM_PROVIDER,
        )
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("Starting Prisma DeepThink API...")
    logger.info("Default LLM Provider: %s", LLM_PROVIDER)

    # 打印所有已注册的 provider
    for name, cfg in sorted(PROVIDER_CONFIGS.items()):
        key_preview = (
            f"{cfg.api_key[:4]}...{cfg.api_key[-4:]}"
            if len(cfg.api_key) >= 8
            else "(not set)"
        )
        logger.info(
            "  Provider: %-12s  type=%-6s  key=%s  base_url=%s",
            name, cfg.type, key_preview, cfg.base_url or "(default)",
        )

    logger.info("Listening on: %s:%d", HOST, PORT)
    logger.info("=" * 50)

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
