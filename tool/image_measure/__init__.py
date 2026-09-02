"""image_measure：测量图片中的距离、角度、比例或颜色量。"""

from __future__ import annotations

from tool.image_measure.measure import execute as measure

OPERATIONS = {
    'measure': measure,
}

__all__ = [
    "OPERATIONS",
    'measure',
]
