"""OCR adapter（PaddleOCR）；引擎可注入/mock。"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from pipeline.config import get_settings


class OcrEngine(Protocol):
    def run(self, image_path: str, bbox: list[float] | None) -> list[str]: ...


_engine: Optional[OcrEngine] = None


def set_engine(engine: Optional[OcrEngine]) -> None:
    """测试用：注入或清除 OCR 引擎。"""
    global _engine
    _engine = engine


def _default_engine() -> OcrEngine:
    settings = get_settings()
    if settings.APP_ENV == "test" and not settings.ALLOW_REAL_API:
        # OCR 为本地模型，但仍避免在未注入时拉起重依赖
        raise RuntimeError("test 环境请通过 set_engine 注入 OCR 引擎")

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    class _PaddleAdapter:
        def run(self, image_path: str, bbox: list[float] | None) -> list[str]:
            _ = bbox
            raw = ocr.ocr(image_path, cls=True)
            texts: list[str] = []
            if not raw:
                return texts
            for block in raw:
                if not block:
                    continue
                for line in block:
                    if line and len(line) >= 2 and line[1]:
                        texts.append(str(line[1][0]))
            return texts

    return _PaddleAdapter()


def execute(params: dict[str, Any], image_path: str) -> dict[str, Any]:
    """执行 ocr。"""
    bbox = params.get("bbox")
    try:
        engine = _engine if _engine is not None else _default_engine()
        texts = engine.run(image_path, bbox)
        return {"status": "success", "error_message": None, "texts": texts}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_message": str(exc), "texts": []}
