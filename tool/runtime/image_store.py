"""会话内图片 ID ↔ 文件路径的仓库。

蒸馏 pipeline 不调用本模块；供日后运行时 / MCP 注入 `RuntimeContext.image_store`。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from tool.contract import RuntimeContext

CURRENT_IMAGE_REF = "$current_image"
_FORMAT_BY_SUFFIX = {
    "png": "PNG",
    "jpeg": "JPEG",
    "jpg": "JPEG",
    "webp": "WEBP",
}


class ImageResolveError(Exception):
    """无法把 image 引用解析为可读文件。"""

    def __init__(self, message: str, error_code: str = "image_not_found") -> None:
        super().__init__(message)
        self.error_code = error_code


@runtime_checkable
class ImageStore(Protocol):
    """图片引用仓库：登记已有文件、写出派生图、按 ID 取路径。"""

    def resolve(self, ref: str) -> Path:
        """把图片 ID 或已知路径解析为本地文件。未知引用时抛出 FileNotFoundError。"""

    def put(
        self,
        image: PILImage,
        *,
        source_id: str,
        suffix: str,
    ) -> str:
        """保存派生图并返回新图片 ID。"""

    def path_for(self, image_id: str) -> Path:
        """返回已登记图片 ID 的文件路径。"""

    def register(self, path: Path, image_id: str | None = None) -> str:
        """把已有文件登记为图片 ID；未指定 ID 时自动分配。"""


class FilesystemImageStore:
    """把派生图写到会话目录，ID 形如 ``img_0001``。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._paths: dict[str, Path] = {}
        self._counter = 0

    def resolve(self, ref: str) -> Path:
        if ref in self._paths:
            return self._paths[ref]
        path = Path(ref)
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(ref)

    def path_for(self, image_id: str) -> Path:
        if image_id not in self._paths:
            raise FileNotFoundError(image_id)
        return self._paths[image_id]

    def register(self, path: Path, image_id: str | None = None) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(str(path))
        assigned = image_id or self._next_id()
        self._paths[assigned] = resolved
        return assigned

    def put(
        self,
        image: PILImage,
        *,
        source_id: str,
        suffix: str,
    ) -> str:
        del source_id
        image_id = self._next_id()
        path = self.root / f"{image_id}.{_file_suffix(suffix)}"
        _save_pil(image, path, suffix)
        self._paths[image_id] = path
        return image_id

    def _next_id(self) -> str:
        while True:
            self._counter += 1
            image_id = f"img_{self._counter:04d}"
            if image_id not in self._paths:
                return image_id


def resolve_image_ref(
    ref: str,
    ctx: RuntimeContext | None,
) -> tuple[str, Path]:
    """按 ``$current_image`` → 当前图 → store ID → 现存路径 解析，返回 (image_id, path)。"""

    token = ref.strip()
    if not token:
        raise ImageResolveError("image 不能为空", "missing_input")
    if token == CURRENT_IMAGE_REF:
        current = ctx.current_image if ctx is not None else None
        if not current:
            raise ImageResolveError(
                "未设置 $current_image",
                "image_not_found",
            )
        return resolve_image_ref(current, ctx)

    store = ctx.image_store if ctx is not None else None
    if store is not None:
        try:
            path = store.resolve(token)
            return token, path
        except FileNotFoundError:
            pass

    path = Path(token)
    if path.is_file():
        resolved = path.resolve()
        image_id = token
        if store is not None:
            image_id = store.register(resolved, image_id=token)
        return image_id, resolved

    raise ImageResolveError(f"找不到图片: {token}", "image_not_found")


def put_image(
    image: PILImage,
    *,
    source_id: str,
    suffix: str,
    ctx: RuntimeContext | None,
) -> tuple[str, Path]:
    """把派生图写入 store 或 artifact 目录，返回 (image_id, path)。"""

    store = ctx.image_store if ctx is not None else None
    if store is not None:
        image_id = store.put(image, source_id=source_id, suffix=suffix)
        return image_id, store.path_for(image_id)

    artifact_dir = _artifact_dir(ctx)
    image_id = _fallback_image_id(artifact_dir)
    path = artifact_dir / f"{image_id}.{_file_suffix(suffix)}"
    _save_pil(image, path, suffix)
    return image_id, path


def _artifact_dir(ctx: RuntimeContext | None) -> Path:
    extras = ctx.extras if ctx is not None else {}
    raw = extras.get("artifact_dir")
    if raw:
        directory = Path(str(raw))
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    return Path(tempfile.mkdtemp(prefix="image_edit_"))


def _fallback_image_id(directory: Path) -> str:
    existing = {path.stem for path in directory.iterdir()} if directory.is_dir() else set()
    index = 1
    while True:
        image_id = f"img_{index:04d}"
        if image_id not in existing:
            return image_id
        index += 1


def _file_suffix(suffix: str) -> str:
    normalized = suffix.lower().lstrip(".")
    if normalized == "jpg":
        return "jpeg"
    if normalized not in _FORMAT_BY_SUFFIX:
        return "png"
    return normalized


def _save_pil(image: PILImage, path: Path, suffix: str) -> None:
    fmt = _FORMAT_BY_SUFFIX.get(_file_suffix(suffix), "PNG")
    payload = image
    if fmt == "JPEG" and payload.mode in {"RGBA", "LA", "P"}:
        payload = payload.convert("RGB")
    elif payload.mode == "P":
        payload = payload.convert("RGBA" if "transparency" in payload.info else "RGB")
    save_kwargs: dict[str, object] = {}
    if fmt == "JPEG":
        save_kwargs["quality"] = 95
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.save(path, format=fmt, **save_kwargs)
