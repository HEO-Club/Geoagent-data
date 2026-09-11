"""map_layer_query.load_layer：加载水系、地形、行政区或其他地图图层"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.map_layer_query._layers import execute_load_layer


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 map_layer_query.load_layer：OGC WMS 出图，可选 WFS 矢量。"""

    return execute_load_layer(purpose=purpose, inputs=inputs, ctx=ctx)
