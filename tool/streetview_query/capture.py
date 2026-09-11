"""streetview_query.capture：获取指定视角的街景画面"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.streetview_query._streetview import execute_capture


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 streetview_query.capture：下载缩略图，全景按 heading/pitch/fov 投影。"""

    return execute_capture(purpose=purpose, inputs=inputs, ctx=ctx)
