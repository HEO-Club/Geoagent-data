"""image_compare.compare：对两张或多张图片执行程序化视觉差异比较"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.image_compare._compare import execute_compare


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 image_compare.compare：特征/像素/直方图/几何比较，返回可核验证据。"""

    return execute_compare(purpose=purpose, inputs=inputs, ctx=ctx)
