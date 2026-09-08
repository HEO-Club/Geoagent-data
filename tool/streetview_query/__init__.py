"""streetview_query：打开、导航、切换或截取街景会话。"""

from __future__ import annotations

from tool.streetview_query.open import execute as open
from tool.streetview_query.navigate import execute as navigate
from tool.streetview_query.change_time import execute as change_time
from tool.streetview_query.capture import execute as capture

OPERATIONS = {
    'open': open,
    'navigate': navigate,
    'change_time': change_time,
    'capture': capture,
}

__all__ = [
    "OPERATIONS",
    'open',
    'navigate',
    'change_time',
    'capture',
]
