"""真实 Tool 外网闸门：默认关闭，与数据流水线的 LLM 闸门隔离。"""

from __future__ import annotations

import os


def allow_real_api() -> bool:
    """仅在显式打开 ``ALLOW_REAL_TOOL_API`` 时允许 Tool 访问外网。

    ``ALLOW_REAL_API`` 属于 Stage 2/3 模型调用，不能隐式授权 Tool 上传图片、
    查询位置或下载任意媒体。函数名保留是为了兼容各 Tool 现有导入路径。
    """

    raw = os.environ.get("ALLOW_REAL_TOOL_API", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def load_tool_dotenv() -> None:
    """加载仓库根 .env，不覆盖已有环境变量。"""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)
