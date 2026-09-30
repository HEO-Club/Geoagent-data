"""ocr_read.decode：解码二维码、嵌入式代码或特殊字符"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.ocr_read._ocr import execute_decode


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 ocr_read.decode：本地解码二维码与 OpenCV 支持的一维商品条码。"""

    return execute_decode(purpose=purpose, inputs=inputs, ctx=ctx)
