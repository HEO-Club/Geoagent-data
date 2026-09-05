"""image_edit 的本地 Pillow 变换：裁剪、放大、可复现增强。"""

from __future__ import annotations

import json
from typing import Any

from PIL import Image, ImageEnhance

from tool.contract import Observation, RuntimeContext
from tool.runtime.image_store import ImageResolveError, put_image, resolve_image_ref

_ENHANCE_KEYS = frozenset({"brightness", "contrast", "sharpness", "shadows"})
_OUTPUT_FORMATS = frozenset({"png", "jpeg", "webp"})
_DEFAULT_ZOOM_SCALE = 2.0
_FACTOR_MIN = 0.1
_FACTOR_MAX = 3.0


class RegionError(Exception):
    """区域无法解析为像素框。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def execute_crop(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """裁剪指定区域，写出派生图。"""

    del purpose
    return _run_edit("crop", inputs, ctx)


def execute_zoom(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """裁剪区域后按 Lanczos 放大；不发明原图没有的细节。"""

    del purpose
    return _run_edit("zoom", inputs, ctx)


def execute_enhance(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按增量调整亮度/对比度/锐度/阴影，结果可复现。"""

    del purpose
    return _run_edit("enhance", inputs, ctx)


def _run_edit(
    operation: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    image_ref = inputs.get("image")
    if not isinstance(image_ref, str) or not image_ref.strip():
        return _fail("缺少必填输入 image", "missing_input")

    try:
        source_id, source_path = resolve_image_ref(image_ref, ctx)
    except ImageResolveError as exc:
        return _fail(str(exc), exc.error_code)

    try:
        with Image.open(source_path) as opened:
            source = opened.convert("RGBA") if opened.mode == "P" else opened.copy()
    except OSError as exc:
        return _fail(f"无法读取图片: {exc}", "image_not_found")

    width, height = source.size
    region_required = operation in {"crop", "zoom"}
    raw_region = inputs.get("region")
    if raw_region is None:
        if region_required:
            return _fail("缺少必填输入 region", "missing_input")
        box: tuple[int, int, int, int] | None = None
    else:
        try:
            box = _parse_region(raw_region, width, height, ctx)
            padding = inputs.get("padding") if operation == "crop" else None
            if padding is not None:
                box = _apply_padding(box, padding, width, height)
        except RegionError as exc:
            return _fail(str(exc), exc.error_code)

    applied: dict[str, Any] = {"operation": operation}
    ignored: dict[str, Any] = {}

    if operation == "crop":
        assert box is not None
        result_image = source.crop(box)
        applied["region"] = list(box)
        if inputs.get("padding") is not None:
            applied["padding"] = inputs["padding"]
    elif operation == "zoom":
        assert box is not None
        try:
            scale = _parse_scale(inputs.get("scale"))
        except ValueError as exc:
            return _fail(str(exc), "invalid_scale")
        cropped = source.crop(box)
        new_size = (
            max(1, round(cropped.size[0] * scale)),
            max(1, round(cropped.size[1] * scale)),
        )
        result_image = cropped.resize(new_size, Image.Resampling.LANCZOS)
        applied["region"] = list(box)
        applied["scale"] = scale
    else:
        parsed = _parse_adjustments(inputs.get("adjustments"))
        if parsed is None:
            return _fail("adjustments 必须是对象", "invalid_adjustments")
        deltas, extra = parsed
        if box is not None:
            result_image = _enhance_image(source.crop(box), deltas)
            applied["region"] = list(box)
        else:
            result_image = _enhance_image(source, deltas)
        applied["adjustments"] = deltas
        if extra:
            ignored.update(extra)

    suffix = _parse_output_format(inputs.get("output_format"))
    if suffix is None:
        return _fail("output_format 必须是 png、jpeg 或 webp", "invalid_adjustments")
    if inputs.get("output_format") is not None:
        applied["output_format"] = suffix
    if ignored:
        applied["ignored"] = ignored

    image_id, output_path = put_image(
        result_image,
        source_id=source_id,
        suffix=suffix,
        ctx=ctx,
    )
    return Observation(
        ok=True,
        result={
            "image_id": image_id,
            "source_image_id": source_id,
            "width": result_image.size[0],
            "height": result_image.size[1],
            "applied": applied,
            "detail_invented": False,
        },
        artifacts={
            "image_id": image_id,
            "image_path": str(output_path),
        },
    )


def _parse_region(
    raw: Any,
    width: int,
    height: int,
    ctx: RuntimeContext | None,
) -> tuple[int, int, int, int]:
    value: Any = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        loaded = _try_json(stripped)
        if loaded is not None:
            value = loaded
        else:
            named = _named_regions(ctx).get(stripped)
            if named is None:
                raise RegionError(
                    f"无法解析命名区域: {stripped}",
                    "unresolved_region",
                )
            value = named

    coords = _coords_from_value(value)
    if coords is None:
        raise RegionError("region 必须是 [x1,y1,x2,y2] 或等价对象", "invalid_region")

    x1, y1, x2, y2 = coords
    if _looks_normalized(coords):
        x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height

    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    box = _clamp_box(
        (int(round(left)), int(round(top)), int(round(right)), int(round(bottom))),
        width,
        height,
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise RegionError("region 裁剪后为空", "invalid_region")
    return box


def _apply_padding(
    box: tuple[int, int, int, int],
    padding: Any,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    try:
        pad = float(padding)
    except (TypeError, ValueError) as exc:
        raise RegionError("padding 必须是非负数字", "invalid_region") from exc
    if pad < 0:
        raise RegionError("padding 必须是非负数字", "invalid_region")
    x1, y1, x2, y2 = box
    if pad >= 1:
        pad_x = pad_y = pad
    elif pad == 0:
        return box
    else:
        pad_x = pad * (x2 - x1)
        pad_y = pad * (y2 - y1)
    padded = _clamp_box(
        (
            int(round(x1 - pad_x)),
            int(round(y1 - pad_y)),
            int(round(x2 + pad_x)),
            int(round(y2 + pad_y)),
        ),
        width,
        height,
    )
    if padded[2] <= padded[0] or padded[3] <= padded[1]:
        raise RegionError("region 加 padding 后为空", "invalid_region")
    return padded


def _parse_scale(raw: Any) -> float:
    if raw is None:
        return _DEFAULT_ZOOM_SCALE
    try:
        scale = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("scale 必须是 >= 1 的数字") from exc
    if scale < 1:
        raise ValueError("scale 必须 >= 1")
    return scale


def _parse_adjustments(
    raw: Any,
) -> tuple[dict[str, float], dict[str, Any]] | None:
    if raw is None:
        return {}, {}
    if not isinstance(raw, dict):
        return None
    deltas: dict[str, float] = {}
    ignored: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key)
        if name not in _ENHANCE_KEYS:
            ignored[name] = value
            continue
        try:
            deltas[name] = float(value)
        except (TypeError, ValueError):
            ignored[name] = value
    return deltas, ignored


def _parse_output_format(raw: Any) -> str | None:
    if raw is None:
        return "png"
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().lower()
    if normalized == "jpg":
        normalized = "jpeg"
    if normalized not in _OUTPUT_FORMATS:
        return None
    return normalized


def _enhance_image(image: Image.Image, deltas: dict[str, float]) -> Image.Image:
    result = image
    brightness = deltas.get("brightness")
    if brightness:
        result = ImageEnhance.Brightness(result).enhance(_factor(brightness))
    contrast = deltas.get("contrast")
    if contrast:
        result = ImageEnhance.Contrast(result).enhance(_factor(contrast))
    sharpness = deltas.get("sharpness")
    if sharpness:
        result = ImageEnhance.Sharpness(result).enhance(_factor(sharpness))
    shadows = deltas.get("shadows")
    if shadows:
        result = _lift_shadows(result, shadows)
    return result


def _factor(delta: float) -> float:
    return min(_FACTOR_MAX, max(_FACTOR_MIN, 1.0 + delta))


def _lift_shadows(image: Image.Image, amount: float) -> Image.Image:
    lift = min(1.0, max(0.0, amount))
    if lift == 0:
        return image
    lut = [_shadow_value(index, lift) for index in range(256)]
    if image.mode == "RGB":
        return image.point(lut * 3)
    if image.mode == "RGBA":
        rgb = image.convert("RGB").point(lut * 3)
        alpha = image.getchannel("A")
        merged = rgb.convert("RGBA")
        merged.putalpha(alpha)
        return merged
    if image.mode == "L":
        return image.point(lut)
    rgb = image.convert("RGB").point(lut * 3)
    return rgb.convert(image.mode) if image.mode not in {"RGB", "RGBA"} else rgb


def _shadow_value(index: int, amount: float) -> int:
    sample = index / 255.0
    lifted = sample + amount * (1.0 - sample) ** 2
    return int(round(min(1.0, max(0.0, lifted)) * 255))


def _coords_from_value(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        try:
            return (
                float(value["x1"]),
                float(value["y1"]),
                float(value["x2"]),
                float(value["y2"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _looks_normalized(coords: tuple[float, float, float, float]) -> bool:
    return all(0.0 <= value <= 1.0 for value in coords)


def _clamp_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def _named_regions(ctx: RuntimeContext | None) -> dict[str, Any]:
    if ctx is None:
        return {}
    raw = ctx.extras.get("named_regions")
    return raw if isinstance(raw, dict) else {}


def _try_json(text: str) -> Any:
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _fail(error: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=error, error_code=error_code)
