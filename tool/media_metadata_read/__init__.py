"""media_metadata_read：读取 EXIF、GPS 和通用媒体元数据。"""

from __future__ import annotations

from tool.media_metadata_read.exif import execute as exif
from tool.media_metadata_read.file import execute as file

OPERATIONS = {
    'exif': exif,
    'file': file,
}

__all__ = [
    "OPERATIONS",
    'exif',
    'file',
]
