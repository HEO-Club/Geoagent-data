"""web_page_read：打开并抽取已找到网页的内容。"""

from __future__ import annotations

from tool.web_page_read.open_result import execute as open_result

OPERATIONS = {
    'open_result': open_result,
}

__all__ = [
    "OPERATIONS",
    'open_result',
]
