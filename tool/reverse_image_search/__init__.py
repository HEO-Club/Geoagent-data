"""reverse_image_search：提交整图或裁剪图进行相似图片检索。"""

from __future__ import annotations

from tool.reverse_image_search.search import execute as search
from tool.reverse_image_search.search_crop import execute as search_crop

OPERATIONS = {
    'search': search,
    'search_crop': search_crop,
}

__all__ = [
    "OPERATIONS",
    'search',
    'search_crop',
]
