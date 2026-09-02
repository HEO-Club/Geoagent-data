"""ocr_read.decode：解码二维码、嵌入式代码或特殊字符"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 ocr_read.decode。尚未接入真实执行器。"""

    return not_implemented(
        'ocr_read',
        'decode',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
