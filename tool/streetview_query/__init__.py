"""streetview_query：打开、导航、切换或截取街景会话。"""

from __future__ import annotations

from tool.streetview_query._streetview import restore_runtime_context
from tool.streetview_query.capture import execute as capture
from tool.streetview_query.change_time import execute as change_time
from tool.streetview_query.navigate import execute as navigate
from tool.streetview_query.open import execute as open

OPERATIONS = {
    'open': open,
    'navigate': navigate,
    'change_time': change_time,
    'capture': capture,
}

__all__ = [
    "OPERATIONS",
    'capture',
    'change_time',
    'navigate',
    'open',
    "restore_runtime_context",
]
