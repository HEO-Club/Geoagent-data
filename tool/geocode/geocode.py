"""geocode.geocode：在地名、地址和坐标表达之间查询映射"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.geocode._nominatim import execute_geocode


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行地名/地址与 WGS84 坐标之间的候选映射。"""

    return execute_geocode(
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
