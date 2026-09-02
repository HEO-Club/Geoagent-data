"""media_search：搜索外部视频或照片。"""

from __future__ import annotations

from tool.media_search.video_search import execute as video_search
from tool.media_search.photo_search import execute as photo_search

OPERATIONS = {
    'video_search': video_search,
    'photo_search': photo_search,
}

__all__ = [
    "OPERATIONS",
    'video_search',
    'photo_search',
]
