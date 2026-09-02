"""ocr_read：识别或解码图中文字和编码。"""

from __future__ import annotations

from tool.ocr_read.recognize import execute as recognize
from tool.ocr_read.decode import execute as decode

OPERATIONS = {
    'recognize': recognize,
    'decode': decode,
}

__all__ = [
    "OPERATIONS",
    'recognize',
    'decode',
]
