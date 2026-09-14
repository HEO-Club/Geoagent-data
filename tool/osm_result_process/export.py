"""osm_result_process.export：把查询结果导出为 GeoJSON、矢量图层或其他格式"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.osm_result_process._processor import execute_export


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """把已有 OSM payload 安全导出为 JSON、GeoJSON、CSV 或 KML。"""

    return execute_export(
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
