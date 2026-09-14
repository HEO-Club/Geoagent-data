"""satellite_imagery_query：获取、切换或查看卫星/航片影像。"""

from __future__ import annotations

from tool.satellite_imagery_query._query import restore_runtime_context
from tool.satellite_imagery_query.change_time import execute as change_time
from tool.satellite_imagery_query.oblique_view import execute as oblique_view
from tool.satellite_imagery_query.retrieve import execute as retrieve

OPERATIONS = {
    'retrieve': retrieve,
    'change_time': change_time,
    'oblique_view': oblique_view,
}

__all__ = [
    "OPERATIONS",
    'change_time',
    'oblique_view',
    "restore_runtime_context",
    'retrieve',
]
