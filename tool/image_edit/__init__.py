"""image_edit：裁剪、缩放或增强输入图片。"""

from __future__ import annotations

from tool.image_edit.enhance import execute as enhance
from tool.image_edit.crop import execute as crop
from tool.image_edit.zoom import execute as zoom

OPERATIONS = {
    'enhance': enhance,
    'crop': crop,
    'zoom': zoom,
}

__all__ = [
    "OPERATIONS",
    'enhance',
    'crop',
    'zoom',
]
