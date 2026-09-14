"""geocode.geocode：在地名、地址和坐标表达之间查询映射"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行地名/地址与 WGS84 坐标之间的候选映射。"""

    extras = ctx.extras if ctx is not None else {}
    if extras.get("geocode_client") is not None:
        from tool.geocode._nominatim import execute_geocode

        return execute_geocode(purpose=purpose, inputs=inputs, ctx=ctx)
    from tool.geocode._geocode import execute_geocode

    return execute_geocode(purpose=purpose, inputs=inputs, ctx=ctx)
