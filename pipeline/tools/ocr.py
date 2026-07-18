"""OCR adapter（PaddleOCR 3.x + paddlepaddle）；引擎可注入/mock。"""

from __future__ import annotations

import os
from typing import Any, Optional, Protocol

from pipeline.config import get_settings

_ocr_singleton: Any = None


class OcrEngine(Protocol):
    def run(self, image_path: str, bbox: list[float] | None) -> list[str]: ...


_engine: Optional[OcrEngine] = None


def set_engine(engine: Optional[OcrEngine]) -> None:
    """测试用：注入或清除 OCR 引擎。"""
    global _engine
    _engine = engine


def _extract_texts(result: Any) -> list[str]:
    """从 PaddleOCR 3.x predict 返回值中提取文字列表。"""
    texts: list[str] = []
    if result is None:
        return texts

    if not isinstance(result, list):
        result = [result]

    for item in result:
        if item is None:
            continue

        rec_texts: Any = None
        # OCRResult 同时支持映射访问与 .json
        if hasattr(item, "get"):
            rec_texts = item.get("rec_texts")
        if rec_texts is None and hasattr(item, "__getitem__"):
            try:
                rec_texts = item["rec_texts"]
            except Exception:  # noqa: BLE001
                rec_texts = None
        if rec_texts is None and hasattr(item, "json"):
            payload = item.json
            if callable(payload):
                payload = payload()
            if isinstance(payload, dict):
                nested = payload.get("res") if isinstance(payload.get("res"), dict) else payload
                rec_texts = nested.get("rec_texts") or nested.get("texts")

        if isinstance(rec_texts, list):
            texts.extend(str(t) for t in rec_texts if t is not None and str(t).strip())
            continue

        # 兼容 2.x 风格：[[[box], (text, score)], ...]
        if isinstance(item, list):
            for line in item:
                if not line or len(line) < 2:
                    continue
                pair = line[1]
                if isinstance(pair, (list, tuple)) and pair:
                    texts.append(str(pair[0]))
                elif isinstance(pair, str):
                    texts.append(pair)
    return texts


def _get_paddle_ocr() -> Any:
    """懒加载并缓存 PaddleOCR 实例（模型较重）。"""
    global _ocr_singleton
    if _ocr_singleton is not None:
        return _ocr_singleton

    # Windows/CPU 上规避 oneDNN + PIR 不兼容问题
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    from paddleocr import PaddleOCR

    # PP-OCRv4 + disable mkldnn：在当前 Windows CPU 环境可稳定推理
    init_kwargs: dict[str, Any] = {
        "lang": "en",
        "ocr_version": "PP-OCRv4",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    try:
        _ocr_singleton = PaddleOCR(**init_kwargs, enable_mkldnn=False)
    except TypeError:
        _ocr_singleton = PaddleOCR(**init_kwargs)
    return _ocr_singleton


def _default_engine() -> OcrEngine:
    settings = get_settings()
    if settings.APP_ENV == "test" and not settings.ALLOW_REAL_API:
        # OCR 为本地模型，但仍避免在未注入时拉起重依赖
        raise RuntimeError("test 环境请通过 set_engine 注入 OCR 引擎")

    ocr = _get_paddle_ocr()

    class _PaddleAdapter:
        def run(self, image_path: str, bbox: list[float] | None) -> list[str]:
            # bbox 裁剪留待后续增强；当前按全图识别（与种子 schema 一致）
            _ = bbox
            if hasattr(ocr, "predict"):
                raw = ocr.predict(image_path)
            else:
                raw = ocr.ocr(image_path)
            return _extract_texts(raw)

    return _PaddleAdapter()


def execute(params: dict[str, Any], image_path: str) -> dict[str, Any]:
    """执行 ocr。"""
    bbox = params.get("bbox")
    try:
        engine = _engine if _engine is not None else _default_engine()
        texts = engine.run(image_path, bbox)
        if not texts:
            return {"status": "empty", "error_message": None, "texts": []}
        return {"status": "success", "error_message": None, "texts": texts}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_message": str(exc), "texts": []}
