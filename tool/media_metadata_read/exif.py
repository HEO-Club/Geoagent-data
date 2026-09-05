"""media_metadata_read.exif：读取图片 EXIF 和 GPS 元数据"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.media_metadata_read._metadata import execute_exif


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 media_metadata_read.exif：返回 EXIF/GPS 字段，缺失不回退。"""

    return execute_exif(purpose=purpose, inputs=inputs, ctx=ctx)
