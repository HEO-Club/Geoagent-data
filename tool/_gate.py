"""真实外网闸门：默认关闭；正式跑或 MCP 宿主可开。"""

from __future__ import annotations

import os


def allow_real_api() -> bool:
    """ALLOW_REAL_API 为 1/true/yes/on 时才允许真实 HTTP。"""

    raw = os.environ.get("ALLOW_REAL_API", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def load_tool_dotenv() -> None:
    """加载仓库根 .env，不覆盖已有环境变量。"""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)
