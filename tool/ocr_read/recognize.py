"""ocr_read.recognize：识别图片区域中的自然语言文字和数字"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 ocr_read.recognize。尚未接入真实执行器。"""

    return not_implemented(
        'ocr_read',
        'recognize',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
