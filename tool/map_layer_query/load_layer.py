"""map_layer_query.load_layer：加载水系、地形、行政区或其他地图图层"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 map_layer_query.load_layer。尚未接入真实执行器。"""

    return not_implemented(
        'map_layer_query',
        'load_layer',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
