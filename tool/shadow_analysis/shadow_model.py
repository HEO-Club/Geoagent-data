"""shadow_analysis.shadow_model：根据太阳位置计算理论阴影方向或长度"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.shadow_analysis._geometry import execute_shadow_model


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 shadow_analysis.shadow_model：竖直物体与局部平面的正向阴影。"""

    return execute_shadow_model(purpose=purpose, inputs=inputs, ctx=ctx)
