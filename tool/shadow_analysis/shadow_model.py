"""shadow_analysis.shadow_model：根据太阳位置计算理论阴影方向或长度"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 shadow_analysis.shadow_model。尚未接入真实执行器。"""

    return not_implemented(
        'shadow_analysis',
        'shadow_model',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
