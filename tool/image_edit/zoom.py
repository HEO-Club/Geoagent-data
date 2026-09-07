"""image_edit.zoom：放大指定区域而不改变语义"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.image_edit._transform import execute_zoom


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 image_edit.zoom：裁剪后 Lanczos 放大，不发明细节。"""

    return execute_zoom(purpose=purpose, inputs=inputs, ctx=ctx)
