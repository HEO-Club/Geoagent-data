"""satellite_imagery_query：获取或切换 Copernicus 正射预览。不提供斜拍。"""

from __future__ import annotations

from tool.satellite_imagery_query._query import restore_runtime_context
from tool.satellite_imagery_query.change_time import execute as change_time
from tool.satellite_imagery_query.retrieve import execute as retrieve

OPERATIONS = {
    'retrieve': retrieve,
    'change_time': change_time,
}

__all__ = [
    "OPERATIONS",
    'change_time',
    "restore_runtime_context",
    'retrieve',
]
