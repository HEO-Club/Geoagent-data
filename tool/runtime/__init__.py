"""运行时共享依赖（图片仓库等），由调用方注入 RuntimeContext。"""

from __future__ import annotations

from tool.runtime.image_store import (
    FilesystemImageStore,
    ImageResolveError,
    ImageStore,
    put_image,
    resolve_image_ref,
)

__all__ = [
    "FilesystemImageStore",
    "ImageResolveError",
    "ImageStore",
    "put_image",
    "resolve_image_ref",
]
