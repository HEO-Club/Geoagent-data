"""media_metadata_read.file：读取文件容器、编码、创建时间等通用元数据"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.media_metadata_read._metadata import execute_file


def execute(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """执行 media_metadata_read.file：容器与文件系统元数据，不冒充拍摄时间。"""

    return execute_file(purpose=purpose, inputs=inputs, ctx=ctx)
