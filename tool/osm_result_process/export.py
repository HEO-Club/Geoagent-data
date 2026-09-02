"""osm_result_process.export：把查询结果导出为 GeoJSON、矢量图层或其他格式"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 osm_result_process.export。尚未接入真实执行器。"""

    return not_implemented(
        'osm_result_process',
        'export',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
