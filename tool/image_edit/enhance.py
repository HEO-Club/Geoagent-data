"""image_edit.enhance：调整亮度、阴影、对比度或清晰度以显现暗部和细节"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.image_edit._transform import execute_enhance


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 image_edit.enhance：确定性点运算，结果可复现。"""

    return execute_enhance(purpose=purpose, inputs=inputs, ctx=ctx)
