"""image_edit.crop：裁剪指定区域供后续检查或检索"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.image_edit._transform import execute_crop


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 image_edit.crop：按 region 裁剪并返回派生图 ID。"""

    return execute_crop(purpose=purpose, inputs=inputs, ctx=ctx)
