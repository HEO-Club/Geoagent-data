"""image_compare：对多张图片执行特征、像素或几何比较。"""

from __future__ import annotations

from tool.image_compare.compare import execute as compare

OPERATIONS = {
    'compare': compare,
}

__all__ = [
    "OPERATIONS",
    'compare',
]
