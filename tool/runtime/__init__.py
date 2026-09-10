"""运行时共享依赖（图片仓库等），由调用方注入 RuntimeContext。"""

from __future__ import annotations

from tool.runtime.harness import HarnessError, ToolHarness, ToolHarnessConfig
from tool.runtime.harness_models import HarnessReport, ToolExecutionRecord
from tool.runtime.image_store import (
    FilesystemImageStore,
    ImageResolveError,
    ImageStore,
    put_image,
    resolve_image_ref,
)
from tool.runtime.result_store import (
    FilesystemResultStore,
    InMemoryResultStore,
    ResultStore,
)

__all__ = [
    "FilesystemImageStore",
    "FilesystemResultStore",
    "HarnessError",
    "HarnessReport",
    "ImageResolveError",
    "ImageStore",
    "InMemoryResultStore",
    "ResultStore",
    "ToolExecutionRecord",
    "ToolHarness",
    "ToolHarnessConfig",
    "put_image",
    "resolve_image_ref",
]
