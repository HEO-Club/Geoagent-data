"""ocr_read 的本地识别与解码：原始证据、框与分数，不做纠错。"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tool.contract import Observation, RuntimeContext
from tool.image_edit._transform import RegionError, _parse_region, _try_json
from tool.runtime.image_store import ImageResolveError, put_image, resolve_image_ref

_TEXT_KINDS = frozenset({"natural_text", "number", "road_sign", "address", "auto"})
_DEFAULT_TEXT_KIND = "auto"
_DEFAULT_LANGUAGES = ["zh"]
_DEFAULT_CODE_TYPES = ["qr"]
_QR_ALIASES = frozenset({"qr", "qrcode", "qr_code"})
_LANG_ALIASES = {
    "zh": "ch",
    "zh-cn": "ch",
    "zh_cn": "ch",
    "chi": "ch",
    "ch": "ch",
    "chinese": "ch",
    "en": "en",
    "eng": "en",
    "english": "en",
}
_NUMBER_KEEP = re.compile(r"[0-9]")
_NUMBER_STRIP = re.compile(r"[\s\-#/.,，。、+()（）]")
_ENGINE_LOCK = threading.Lock()
_RAPID_ENGINES: dict[str, RapidOcrEngine] = {}


class OcrInputError(Exception):
    """languages / text_kind / code_types 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class EngineUnavailableError(Exception):
    """默认 OCR 引擎无法导入或初始化。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class OcrHit:
    """ROI 坐标系下的一条原始识别结果。"""

    text: str
    confidence: float
    quad: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class LoadedRoi:
    """已解析的原图与待识别 ROI。"""

    image_id: str
    image: Image.Image
    region: tuple[int, int, int, int]

    def roi_rgb(self) -> Image.Image:
        """返回 ROI 的 RGB 拷贝。"""

        return self.image.crop(self.region).convert("RGB")


@runtime_checkable
class OcrEngine(Protocol):
    """可注入的文字识别引擎；测试用 extras['ocr_engine'] 替换 RapidOCR。"""

    def detect(self, image_rgb: Image.Image) -> list[Any]:
        """在 ROI 图上识别，返回 text/confidence/quad（ROI 坐标）。"""


class RapidOcrEngine:
    """RapidOCR（PP-OCR ONNX）适配器；只透传原始识别结果。"""

    name = "rapidocr"

    def __init__(self, lang_key: str) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise EngineUnavailableError(
                "未安装 rapidocr，无法执行 ocr_read.recognize",
            ) from exc
        try:
            params = _rapidocr_params(lang_key)
            self._engine = RapidOCR(params=params) if params else RapidOCR()
        except Exception as exc:
            raise EngineUnavailableError(
                f"RapidOCR 初始化失败: {exc}",
            ) from exc

    def detect(self, image_rgb: Image.Image) -> list[OcrHit]:
        array = np.asarray(image_rgb.convert("RGB"))
        try:
            raw = self._engine(array)
        except Exception as exc:
            raise EngineUnavailableError(f"RapidOCR 识别失败: {exc}") from exc
        return _normalize_ocr_output(raw)


def execute_recognize(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """识别 ROI 中的原始文字；不改写、不补全、不调用 LLM。"""

    del purpose
    try:
        loaded = _load_roi(inputs, ctx)
        languages, lang_key, ignored_langs = _parse_languages(inputs.get("languages"))
        text_kind = _parse_text_kind(inputs.get("text_kind"))
        engine, engine_name = _resolve_engine(lang_key, ctx)
        hits = [_coerce_hit(item) for item in engine.detect(loaded.roi_rgb())]
    except ImageResolveError as exc:
        return _fail(str(exc), exc.error_code)
    except RegionError as exc:
        return _fail(str(exc), exc.error_code)
    except OcrInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

    if text_kind == "number":
        hits = [hit for hit in hits if _is_number_text(hit.text)]
    items = [_hit_to_item(hit, loaded.region) for hit in hits]
    applied: dict[str, Any] = {
        "region": list(loaded.region),
        "languages": languages,
        "text_kind": text_kind,
        "engine": engine_name,
    }
    if ignored_langs:
        applied["ignored"] = {"languages": ignored_langs}
    return _ok_observation(loaded, "recognize", items, applied, ctx)


def execute_decode(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """解码 ROI 中的二维码；一维条码本阶段不支持。"""

    del purpose
    try:
        loaded = _load_roi(inputs, ctx)
        code_types, qr_requested, unsupported = _parse_code_types(inputs.get("code_types"))
    except ImageResolveError as exc:
        return _fail(str(exc), exc.error_code)
    except RegionError as exc:
        return _fail(str(exc), exc.error_code)
    except OcrInputError as exc:
        return _fail(str(exc), exc.error_code)

    if not qr_requested:
        return _fail("当前仅支持 code_types=qr", "unsupported_code_type")

    hits = _decode_qr(loaded.roi_rgb())
    items = [_hit_to_item(hit, loaded.region, code_type="qr") for hit in hits]
    applied: dict[str, Any] = {
        "region": list(loaded.region),
        "code_types": code_types,
        "engine": "opencv_qr",
    }
    if unsupported:
        applied["unsupported_code_types"] = unsupported
    return _ok_observation(loaded, "decode", items, applied, ctx)


def _load_roi(inputs: dict[str, Any], ctx: RuntimeContext | None) -> LoadedRoi:
    image_ref = inputs.get("image")
    if not isinstance(image_ref, str) or not image_ref.strip():
        raise OcrInputError("缺少必填输入 image", "missing_input")

    source_id, source_path = resolve_image_ref(image_ref, ctx)
    try:
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
    except OSError as exc:
        raise ImageResolveError(f"无法读取图片: {exc}", "image_not_found") from exc

    width, height = source.size
    raw_region = inputs.get("region")
    if raw_region is None:
        box = (0, 0, width, height)
    else:
        box = _parse_region(raw_region, width, height, ctx)
    return LoadedRoi(image_id=source_id, image=source, region=box)


def _parse_text_kind(raw: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DEFAULT_TEXT_KIND
    if not isinstance(raw, str):
        raise OcrInputError("text_kind 必须是字符串", "invalid_text_kind")
    kind = raw.strip().lower()
    if kind not in _TEXT_KINDS:
        raise OcrInputError(f"不支持的 text_kind: {raw}", "invalid_text_kind")
    return kind


def _parse_languages(raw: Any) -> tuple[list[str], str, list[str]]:
    values = _parse_string_list(raw, field="languages", error_code="invalid_languages")
    if not values:
        return list(_DEFAULT_LANGUAGES), "ch", []
    mapped: list[str] = []
    ignored: list[str] = []
    seen: set[str] = set()
    for item in values:
        key = _LANG_ALIASES.get(item.lower())
        if key is None:
            ignored.append(item)
            continue
        if key not in seen:
            mapped.append(key)
            seen.add(key)
    display = [item.lower() for item in values]
    if not mapped:
        return display, "ch", ignored
    lang_key = "ch" if "ch" in mapped else mapped[0]
    return display, lang_key, ignored


def _parse_code_types(raw: Any) -> tuple[list[str], bool, list[str]]:
    values = _parse_string_list(raw, field="code_types", error_code="invalid_code_types")
    if not values:
        return list(_DEFAULT_CODE_TYPES), True, []
    normalized = [item.strip().lower() for item in values if item.strip()]
    if not normalized:
        return list(_DEFAULT_CODE_TYPES), True, []
    qr_requested = any(item in _QR_ALIASES for item in normalized)
    unsupported = [item for item in normalized if item not in _QR_ALIASES]
    return normalized, qr_requested, unsupported


def _parse_string_list(raw: Any, *, field: str, error_code: str) -> list[str]:
    value: Any = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        loaded = _try_json(stripped)
        value = loaded if loaded is not None else ([stripped] if stripped else [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        raise OcrInputError(f"{field} 必须是字符串列表", error_code)
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise OcrInputError(f"{field} 必须是字符串列表", error_code)
        if item.strip():
            items.append(item.strip())
    return items


def _resolve_engine(lang_key: str, ctx: RuntimeContext | None) -> tuple[OcrEngine, str]:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("ocr_engine")
    if injected is not None:
        if not isinstance(injected, OcrEngine):
            raise OcrInputError("ocr_engine 必须提供 detect(image_rgb)", "engine_unavailable")
        name = str(getattr(injected, "name", "injected"))
        return injected, name
    return _get_rapid_engine(lang_key), "rapidocr"


def _get_rapid_engine(lang_key: str) -> RapidOcrEngine:
    with _ENGINE_LOCK:
        cached = _RAPID_ENGINES.get(lang_key)
        if cached is not None:
            return cached
        engine = RapidOcrEngine(lang_key)
        _RAPID_ENGINES[lang_key] = engine
        return engine


def _rapidocr_params(lang_key: str) -> dict[str, Any]:
    try:
        from rapidocr import LangRec
    except ImportError:
        return {}
    rec = getattr(LangRec, "EN" if lang_key == "en" else "CH", None)
    if rec is None:
        return {}
    return {"Rec.lang_type": rec}


def _normalize_ocr_output(raw: Any) -> list[OcrHit]:
    payload = raw
    if isinstance(raw, tuple) and raw:
        payload = raw[0]
    if payload is None:
        return []
    boxes = getattr(payload, "boxes", None)
    txts = getattr(payload, "txts", None)
    if boxes is not None and txts is not None:
        scores = getattr(payload, "scores", None)
        items: list[OcrHit] = []
        for index, text in enumerate(txts):
            score = 1.0
            if scores is not None and index < len(scores):
                score = float(scores[index])
            box = boxes[index] if index < len(boxes) else None
            hit = _hit_from_parts(text, score, box)
            if hit is not None:
                items.append(hit)
        return items
    if not isinstance(payload, list):
        return []
    items = []
    for row in payload:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            box = row[0]
            text = row[1]
            score = float(row[2]) if len(row) > 2 else 1.0
            hit = _hit_from_parts(text, score, box)
            if hit is not None:
                items.append(hit)
            continue
        coerced = _coerce_hit(row)
        if coerced.text:
            items.append(coerced)
    return items


def _hit_from_parts(text: Any, score: Any, box: Any) -> OcrHit | None:
    if text is None:
        return None
    payload = str(text)
    if not payload:
        return None
    try:
        confidence = float(score)
    except (TypeError, ValueError):
        confidence = 1.0
    quad = _parse_quad(box)
    if quad is None:
        return None
    return OcrHit(text=payload, confidence=confidence, quad=quad)


def _coerce_hit(raw: Any) -> OcrHit:
    if isinstance(raw, OcrHit):
        return raw
    if isinstance(raw, dict):
        text = raw.get("text")
        if text is None:
            text = raw.get("payload")
        quad = _parse_quad(raw.get("quad", raw.get("bbox")))
        if text is None or quad is None:
            raise OcrInputError("识别结果缺少 text/quad", "engine_unavailable")
        try:
            confidence = float(raw.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        return OcrHit(text=str(text), confidence=confidence, quad=quad)
    raise OcrInputError("识别结果格式无法解析", "engine_unavailable")


def _parse_quad(
    raw: Any,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)) and len(raw) == 4 and all(
        isinstance(item, (int, float)) for item in raw
    ):
        x1, y1, x2, y2 = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        points: list[tuple[float, float]] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    points.append((float(item[0]), float(item[1])))
                except (TypeError, ValueError):
                    return None
            else:
                return None
        return (points[0], points[1], points[2], points[3])
    return None


def _decode_qr(roi: Image.Image) -> list[OcrHit]:
    bgr = cv2.cvtColor(np.asarray(roi.convert("RGB")), cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    hits = _decode_qr_multi(detector, bgr)
    if hits:
        return hits
    text, points, _straight = detector.detectAndDecode(bgr)
    if not text or points is None:
        return []
    quad = _parse_quad(np.asarray(points).reshape(-1, 2))
    if quad is None:
        return []
    return [OcrHit(text=str(text), confidence=1.0, quad=quad)]


def _decode_qr_multi(detector: cv2.QRCodeDetector, bgr: np.ndarray) -> list[OcrHit]:
    try:
        unpacked = detector.detectAndDecodeMulti(bgr)
    except cv2.error:
        return []
    if not unpacked:
        return []
    retval = unpacked[0]
    decoded = unpacked[1] if len(unpacked) > 1 else ()
    points = unpacked[2] if len(unpacked) > 2 else None
    if not retval or decoded is None or points is None:
        return []
    hits: list[OcrHit] = []
    for text, quad_raw in zip(decoded, points, strict=False):
        if not text:
            continue
        quad = _parse_quad(quad_raw)
        if quad is None:
            continue
        hits.append(OcrHit(text=str(text), confidence=1.0, quad=quad))
    return hits


def _hit_to_item(
    hit: OcrHit,
    region: tuple[int, int, int, int],
    *,
    code_type: str | None = None,
) -> dict[str, Any]:
    ox, oy = region[0], region[1]
    quad = [
        [int(round(x + ox)), int(round(y + oy))]
        for x, y in hit.quad
    ]
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    item: dict[str, Any] = {
        "text": hit.text,
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "quad": quad,
        "confidence": float(hit.confidence),
    }
    if code_type is not None:
        item["code_type"] = code_type
        item["payload"] = hit.text
    return item


def _is_number_text(text: str) -> bool:
    compact = _NUMBER_STRIP.sub("", text)
    if not compact or not _NUMBER_KEEP.search(compact):
        return False
    digits = sum(ch.isdigit() for ch in compact)
    return digits / len(compact) >= 0.6


def _ok_observation(
    loaded: LoadedRoi,
    operation: str,
    items: list[dict[str, Any]],
    applied: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    full_text = "\n".join(
        str(item.get("text") or item.get("payload") or "")
        for item in items
        if item.get("text") or item.get("payload")
    )
    result: dict[str, Any] = {
        "operation": operation,
        "image_id": loaded.image_id,
        "full_text": full_text,
        "items": items,
        "applied": applied,
    }
    artifacts: dict[str, Any] = {}
    if items:
        vis_id, vis_path = _annotate(loaded, items, ctx)
        artifacts = {"annotated_image_id": vis_id, "image_path": str(vis_path)}
        result["annotated_image_id"] = vis_id
    return Observation(ok=True, result=result, artifacts=artifacts)


def _annotate(
    loaded: LoadedRoi,
    items: list[dict[str, Any]],
    ctx: RuntimeContext | None,
) -> tuple[str, Any]:
    canvas = loaded.image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for item in items:
        quad = item.get("quad")
        if isinstance(quad, list) and len(quad) == 4:
            points = [(int(point[0]), int(point[1])) for point in quad]
            draw.polygon(points, outline=(0, 220, 80), width=2)
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            draw.rectangle(
                (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                outline=(0, 180, 255),
                width=1,
            )
    return put_image(canvas, source_id=loaded.image_id, suffix="png", ctx=ctx)


def _fail(error: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=error, error_code=error_code)
