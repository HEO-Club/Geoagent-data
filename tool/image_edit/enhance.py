"""image_edit.enhance：调整亮度、阴影、对比度或清晰度以显现暗部和细节"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 image_edit.enhance。尚未接入真实执行器。"""

    return not_implemented(
        'image_edit',
        'enhance',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
