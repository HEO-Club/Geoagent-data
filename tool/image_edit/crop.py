"""image_edit.crop：裁剪指定区域供后续检查或检索"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 image_edit.crop。尚未接入真实执行器。"""

    return not_implemented(
        'image_edit',
        'crop',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
