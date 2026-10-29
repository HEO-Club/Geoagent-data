"""location_verify.verify：验证最终定位答案是否可信"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.location_verify._verify import execute_verify


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 location_verify.verify：街景图像比对 + OCR 文字一致性，输出 verdict。"""

    return execute_verify(purpose=purpose, inputs=inputs, ctx=ctx)
