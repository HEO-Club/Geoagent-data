"""image_edit.zoom：放大指定区域而不改变语义"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 image_edit.zoom。尚未接入真实执行器。"""

    return not_implemented(
        'image_edit',
        'zoom',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
