"""media_metadata_read.exif：读取图片 EXIF 和 GPS 元数据"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, not_implemented


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 media_metadata_read.exif。尚未接入真实执行器。"""

    return not_implemented(
        'media_metadata_read',
        'exif',
        purpose=purpose,
        inputs=inputs,
        ctx=ctx,
    )
