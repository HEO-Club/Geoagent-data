"""ocr_read.recognize：识别图片区域中的自然语言文字和数字"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.ocr_read._ocr import execute_recognize


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 ocr_read.recognize：返回原始文字、框与分数，不改写证据。"""

    return execute_recognize(purpose=purpose, inputs=inputs, ctx=ctx)
