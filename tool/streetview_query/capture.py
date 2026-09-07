"""streetview_query.capture：获取指定视角的街景画面"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 streetview_query.capture。尚未接入真实执行器。"""

    return not_implemented(
        'streetview_query',
        'capture',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
