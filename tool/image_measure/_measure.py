"""image_measure 的本地量测：像素几何、比例与可选均匀比例尺。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image, ImageStat

from tool.contract import Observation, RuntimeContext
from tool.image_edit._transform import RegionError, _named_regions, _parse_region, _try_json
from tool.runtime.image_store import ImageResolveError, resolve_image_ref

_MEASUREMENTS = frozenset({"distance", "angle", "ratio", "area", "color"})
_AXES = frozenset({"horizontal", "vertical", "diagonal"})
Axis = Literal["horizontal", "vertical", "diagonal"]


class ReferenceError(Exception):
    """reference 无法解析为合法尺度或比较几何。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class BoxGeom:
    """轴对齐像素框。"""

    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class SegmentGeom:
    """像素线段。"""

    p1: tuple[int, int]
    p2: tuple[int, int]


Geom = BoxGeom | SegmentGeom


@dataclass(frozen=True)
class ScaleRef:
    """可换算的均匀比例尺参照。"""

    geom: Geom
    known_length: float
    unit: str
    axis: Axis | None


@dataclass(frozen=True)
class ParsedReference:
    """reference 解析结果：尺度与/或比较几何。"""

    scale: ScaleRef | None
    geom: Geom | None


def execute_measure(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按 measurement 计算像素量，仅在 reference 可解析时换算实尺。"""

    del purpose
    image_ref = inputs.get("image")
    if not isinstance(image_ref, str) or not image_ref.strip():
        return _fail("缺少必填输入 image", "missing_input")

    raw_measurement = inputs.get("measurement")
    if raw_measurement is None or (isinstance(raw_measurement, str) and not raw_measurement.strip()):
        return _fail("缺少必填输入 measurement", "missing_input")
    if not isinstance(raw_measurement, str) or raw_measurement not in _MEASUREMENTS:
        return _fail(
            "measurement 必须是 distance、angle、ratio、area 或 color",
            "invalid_measurement",
        )
    measurement = raw_measurement

    try:
        _source_id, source_path = resolve_image_ref(image_ref, ctx)
    except ImageResolveError as exc:
        return _fail(str(exc), exc.error_code)

    try:
        with Image.open(source_path) as opened:
            width, height = opened.size
            rgb = opened.convert("RGB") if measurement == "color" else None
    except OSError as exc:
        return _fail(f"无法读取图片: {exc}", "image_not_found")

    try:
        axis = _parse_axis(inputs.get("axis"))
        geoms = _parse_region_geoms(
            inputs.get("region"),
            width,
            height,
            ctx,
            required=measurement != "color",
        )
        parsed_ref = _parse_reference(inputs.get("reference"), width, height, ctx, axis)
        return _compute(
            measurement=measurement,
            geoms=geoms,
            parsed_ref=parsed_ref,
            axis=axis,
            rgb=rgb,
            width=width,
            height=height,
        )
    except RegionError as exc:
        return _fail(str(exc), exc.error_code)
    except ReferenceError as exc:
        return _fail(str(exc), exc.error_code)


def _compute(
    *,
    measurement: str,
    geoms: list[Geom],
    parsed_ref: ParsedReference,
    axis: Axis | None,
    rgb: Image.Image | None,
    width: int,
    height: int,
) -> Observation:
    assumptions: list[str] = []
    applied: dict[str, Any] = _applied_from_geoms(geoms, axis)
    comparison = parsed_ref.geom
    if comparison is not None:
        applied["reference_region"] = _serialize_geom(comparison)

    if measurement == "color":
        return _measure_color(rgb, geoms, parsed_ref, applied, width, height)
    if measurement == "angle":
        return _measure_angle(geoms, comparison, parsed_ref, applied)
    if measurement == "ratio":
        return _measure_ratio(geoms, comparison, axis, applied)
    if measurement == "area":
        return _measure_area(geoms, parsed_ref, applied, assumptions)
    return _measure_distance(geoms, parsed_ref, axis, applied, assumptions)


def _measure_distance(
    geoms: list[Geom],
    parsed_ref: ParsedReference,
    axis: Axis | None,
    applied: dict[str, Any],
    assumptions: list[str],
) -> Observation:
    geom = _require_single_geom(geoms, "distance")
    pixel_value, method, components = _pixel_length(geom, axis)
    if axis is None and isinstance(geom, BoxGeom):
        assumptions.append("单框距离默认取较长边")
    return _finish_linear(
        measurement="distance",
        pixel_value=pixel_value,
        method=method,
        applied=applied,
        components=components,
        parsed_ref=parsed_ref,
        assumptions=assumptions,
    )


def _measure_area(
    geoms: list[Geom],
    parsed_ref: ParsedReference,
    applied: dict[str, Any],
    assumptions: list[str],
) -> Observation:
    geom = _require_single_geom(geoms, "area")
    if not isinstance(geom, BoxGeom):
        raise RegionError("area 需要轴对齐框，不能是线段", "invalid_region")
    x1, y1, x2, y2 = geom.box
    width_px = float(x2 - x1)
    height_px = float(y2 - y1)
    pixel_value = width_px * height_px
    components = {"width_px": width_px, "height_px": height_px, "area_px": pixel_value}
    return _finish_area(
        pixel_value=pixel_value,
        applied=applied,
        components=components,
        parsed_ref=parsed_ref,
        assumptions=assumptions,
    )


def _measure_ratio(
    geoms: list[Geom],
    comparison: Geom | None,
    axis: Axis | None,
    applied: dict[str, Any],
) -> Observation:
    pieces = list(geoms)
    if comparison is not None:
        pieces.append(comparison)
    if len(pieces) == 2:
        first, method_a, components_a = _pixel_length(pieces[0], axis)
        second, method_b, components_b = _pixel_length(pieces[1], axis)
        if second == 0:
            raise RegionError("比较段像素长度为 0，无法计算比例", "invalid_region")
        value = first / second
        assumptions = ["相对比例为两段像素长度之比，未做实尺换算"]
        return _ok(
            {
                "value": value,
                "unit": "ratio",
                "method": "relative_proportion",
                "assumptions": assumptions,
                "measurement": "ratio",
                "applied": applied,
                "pixel_value": value,
                "components": {
                    "numerator_px": first,
                    "denominator_px": second,
                    "numerator_method": method_a,
                    "denominator_method": method_b,
                    **{f"a_{key}": val for key, val in components_a.items()},
                    **{f"b_{key}": val for key, val in components_b.items()},
                },
            }
        )
    if len(pieces) != 1 or not isinstance(pieces[0], BoxGeom):
        raise RegionError("ratio 需要一个框（宽高比）或两段可量长度的几何", "invalid_region")
    x1, y1, x2, y2 = pieces[0].box
    width_px = float(x2 - x1)
    height_px = float(y2 - y1)
    if height_px == 0:
        raise RegionError("框高度为 0，无法计算宽高比", "invalid_region")
    value = width_px / height_px
    return _ok(
        {
            "value": value,
            "unit": "ratio",
            "method": "aspect_ratio",
            "assumptions": ["单框比例为宽/高，未做实尺换算"],
            "measurement": "ratio",
            "applied": applied,
            "pixel_value": value,
            "components": {"width_px": width_px, "height_px": height_px},
        }
    )


def _measure_angle(
    geoms: list[Geom],
    comparison: Geom | None,
    parsed_ref: ParsedReference,
    applied: dict[str, Any],
) -> Observation:
    pieces = list(geoms)
    if comparison is not None:
        pieces.append(comparison)
    lines = [geom for geom in pieces if isinstance(geom, SegmentGeom)]
    if len(lines) != 2 or len(pieces) != 2:
        raise RegionError("angle 必须提供两条线，不能只用一个框", "invalid_region")
    value = _angle_deg(lines[0], lines[1])
    assumptions = ["夹角取两线较小角，范围 [0, 90] 度"]
    if parsed_ref.scale is not None:
        assumptions.append("该量与比例尺无关，已忽略尺度参照")
    return _ok(
        {
            "value": value,
            "unit": "deg",
            "method": "line_angle",
            "assumptions": assumptions,
            "measurement": "angle",
            "applied": applied,
            "pixel_value": value,
            "components": {
                "line_a": _serialize_geom(lines[0]),
                "line_b": _serialize_geom(lines[1]),
            },
        }
    )


def _measure_color(
    rgb: Image.Image | None,
    geoms: list[Geom],
    parsed_ref: ParsedReference,
    applied: dict[str, Any],
    width: int,
    height: int,
) -> Observation:
    if rgb is None:
        return _fail("无法读取图片颜色", "image_not_found")
    if len(geoms) > 1:
        raise RegionError("color 只能使用一个区域", "invalid_region")
    if geoms and not isinstance(geoms[0], BoxGeom):
        raise RegionError("color 需要轴对齐框或整图", "invalid_region")
    sample = rgb
    if geoms:
        box = geoms[0].box
        sample = rgb.crop(box)
        applied["region"] = list(box)
    else:
        applied["region"] = [0, 0, width, height]
        applied["full_image"] = True
    stats = ImageStat.Stat(sample)
    channels = [_clamp_channel(value) for value in stats.mean[:3]]
    red, green, blue = channels
    assumptions = ["对区域内像素取均值，未做白平衡"]
    if parsed_ref.scale is not None:
        assumptions.append("该量与比例尺无关，已忽略尺度参照")
    return _ok(
        {
            "value": {"r": red, "g": green, "b": blue, "hex": f"#{red:02x}{green:02x}{blue:02x}"},
            "unit": "rgb",
            "method": "mean_rgb",
            "assumptions": assumptions,
            "measurement": "color",
            "applied": applied,
        }
    )


def _finish_linear(
    *,
    measurement: str,
    pixel_value: float,
    method: str,
    applied: dict[str, Any],
    components: dict[str, float],
    parsed_ref: ParsedReference,
    assumptions: list[str],
) -> Observation:
    if parsed_ref.scale is None:
        assumptions.append("无尺度参照，结果保持像素单位")
        return _ok(
            {
                "value": pixel_value,
                "unit": "px",
                "method": method,
                "assumptions": assumptions,
                "measurement": measurement,
                "applied": applied,
                "pixel_value": pixel_value,
                "components": components,
            }
        )
    scale, scale_info = _scale_from_ref(parsed_ref.scale)
    assumptions.append("假定像平面比例尺均匀，未做透视/标定")
    applied["pixel_method"] = method
    return _ok(
        {
            "value": pixel_value * scale,
            "unit": parsed_ref.scale.unit,
            "method": "reference_scale",
            "assumptions": assumptions,
            "measurement": measurement,
            "applied": applied,
            "pixel_value": pixel_value,
            "components": components,
            "scale": scale,
            "reference_applied": scale_info,
        }
    )


def _finish_area(
    *,
    pixel_value: float,
    applied: dict[str, Any],
    components: dict[str, float],
    parsed_ref: ParsedReference,
    assumptions: list[str],
) -> Observation:
    if parsed_ref.scale is None:
        assumptions.append("无尺度参照，结果保持像素单位")
        return _ok(
            {
                "value": pixel_value,
                "unit": "px",
                "method": "bbox_area",
                "assumptions": assumptions,
                "measurement": "area",
                "applied": applied,
                "pixel_value": pixel_value,
                "components": components,
            }
        )
    scale, scale_info = _scale_from_ref(parsed_ref.scale)
    assumptions.append("假定像平面比例尺均匀，未做透视/标定")
    applied["pixel_method"] = "bbox_area"
    return _ok(
        {
            "value": pixel_value * scale * scale,
            "unit": parsed_ref.scale.unit,
            "method": "reference_scale",
            "assumptions": assumptions,
            "measurement": "area",
            "applied": applied,
            "pixel_value": pixel_value,
            "components": components,
            "scale": scale,
            "reference_applied": scale_info,
        }
    )


def _scale_from_ref(scale_ref: ScaleRef) -> tuple[float, dict[str, Any]]:
    ref_px, ref_method, ref_components = _pixel_length(scale_ref.geom, scale_ref.axis)
    if ref_px <= 0:
        raise ReferenceError("参照区域像素长度为 0，无法建立比例尺", "unresolved_reference")
    scale = scale_ref.known_length / ref_px
    info = {
        "region": _serialize_geom(scale_ref.geom),
        "known_length": scale_ref.known_length,
        "unit": scale_ref.unit,
        "pixel_length": ref_px,
        "method": ref_method,
        "components": ref_components,
    }
    if scale_ref.axis is not None:
        info["axis"] = scale_ref.axis
    return scale, info


def _parse_region_geoms(
    raw: Any,
    width: int,
    height: int,
    ctx: RuntimeContext | None,
    *,
    required: bool,
) -> list[Geom]:
    if raw is None:
        if required:
            raise RegionError("缺少必填输入 region", "missing_input")
        return []
    value = _unwrap_region_value(raw, ctx)
    if _is_pair_of_geoms(value):
        return [
            _parse_one_geom(value[0], width, height, ctx),
            _parse_one_geom(value[1], width, height, ctx),
        ]
    return [_parse_one_geom(value, width, height, ctx)]


def _parse_reference(
    raw: Any,
    width: int,
    height: int,
    ctx: RuntimeContext | None,
    fallback_axis: Axis | None,
) -> ParsedReference:
    if raw is None:
        return ParsedReference(scale=None, geom=None)

    value: Any = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        loaded = _try_json(stripped)
        if loaded is not None:
            value = loaded
        else:
            named = _named_regions(ctx).get(stripped)
            if named is None:
                raise ReferenceError(
                    "reference 无法解析为参照区域、已知长度与单位",
                    "unresolved_reference",
                )
            return ParsedReference(scale=None, geom=_parse_one_geom(named, width, height, ctx))

    if isinstance(value, dict) and _looks_like_scale(value):
        return ParsedReference(
            scale=_parse_scale_ref(value, width, height, ctx, fallback_axis),
            geom=_parse_one_geom(value["region"], width, height, ctx),
        )
    if isinstance(value, dict) and "region" in value:
        return ParsedReference(
            scale=None,
            geom=_parse_one_geom(value["region"], width, height, ctx),
        )
    return ParsedReference(scale=None, geom=_parse_one_geom(value, width, height, ctx))


def _parse_scale_ref(
    value: dict[str, Any],
    width: int,
    height: int,
    ctx: RuntimeContext | None,
    fallback_axis: Axis | None,
) -> ScaleRef:
    if "region" not in value or "known_length" not in value or "unit" not in value:
        raise ReferenceError(
            "reference 必须同时包含 region、known_length 与 unit",
            "unresolved_reference",
        )
    unit = value.get("unit")
    if not isinstance(unit, str) or not unit.strip():
        raise ReferenceError("reference.unit 必须是非空字符串", "unresolved_reference")
    try:
        known_length = float(value["known_length"])
    except (TypeError, ValueError) as exc:
        raise ReferenceError("reference.known_length 必须是数字", "unresolved_reference") from exc
    if known_length <= 0:
        raise ReferenceError("reference.known_length 必须为正数", "unresolved_reference")
    raw_axis = value.get("axis", fallback_axis)
    axis = _parse_axis(raw_axis)
    geom = _parse_one_geom(value["region"], width, height, ctx)
    return ScaleRef(geom=geom, known_length=known_length, unit=unit.strip(), axis=axis)


def _looks_like_scale(value: dict[str, Any]) -> bool:
    return any(key in value for key in ("known_length", "unit"))


def _parse_one_geom(
    raw: Any,
    width: int,
    height: int,
    ctx: RuntimeContext | None,
) -> Geom:
    value = _unwrap_region_value(raw, ctx)
    if _is_segment_like(value):
        return _parse_segment(value, width, height)
    if _is_box_like(value) and _is_degenerate_box(value):
        return _parse_segment(_box_as_segment_points(value), width, height)
    return BoxGeom(_parse_region(value, width, height, ctx))


def _unwrap_region_value(raw: Any, ctx: RuntimeContext | None) -> Any:
    if not isinstance(raw, str):
        return raw
    stripped = raw.strip()
    loaded = _try_json(stripped)
    if loaded is not None:
        return loaded
    named = _named_regions(ctx).get(stripped)
    if named is None:
        raise RegionError(f"无法解析命名区域: {stripped}", "unresolved_region")
    return named


def _parse_segment(raw: Any, width: int, height: int) -> SegmentGeom:
    points = _segment_points(raw)
    if points is None:
        raise RegionError("线段必须是 [[x,y],[x,y]] 或等价对象", "invalid_region")
    (x1, y1), (x2, y2) = points
    if _looks_normalized_points(points):
        x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
    p1 = (_clamp_int(x1, width), _clamp_int(y1, height))
    p2 = (_clamp_int(x2, width), _clamp_int(y2, height))
    if p1 == p2:
        raise RegionError("线段长度为 0", "invalid_region")
    return SegmentGeom(p1, p2)


def _segment_points(raw: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if isinstance(raw, dict):
        if "p1" in raw and "p2" in raw:
            raw = [raw["p1"], raw["p2"]]
        elif "x1" in raw and "y1" in raw and "x2" in raw and "y2" in raw:
            raw = [[raw["x1"], raw["y1"]], [raw["x2"], raw["y2"]]]
        else:
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    start = _as_point(raw[0])
    end = _as_point(raw[1])
    if start is None or end is None:
        return None
    return start, end


def _as_point(raw: Any) -> tuple[float, float] | None:
    if isinstance(raw, dict) and "x" in raw and "y" in raw:
        try:
            return (float(raw["x"]), float(raw["y"]))
        except (TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            return (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError):
            return None
    return None


def _is_segment_like(value: Any) -> bool:
    return _segment_points(value) is not None and not _is_box_like(value)


def _is_box_like(value: Any) -> bool:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            float(value[0])
            float(value[1])
            float(value[2])
            float(value[3])
        except (TypeError, ValueError):
            return False
        return True
    if isinstance(value, dict):
        return all(key in value for key in ("x1", "y1", "x2", "y2"))
    return False


def _is_degenerate_box(value: Any) -> bool:
    coords = _box_coords(value)
    if coords is None:
        return False
    x1, y1, x2, y2 = coords
    return (x1 == x2) != (y1 == y2)


def _box_as_segment_points(value: Any) -> list[list[float]]:
    coords = _box_coords(value)
    assert coords is not None
    x1, y1, x2, y2 = coords
    return [[x1, y1], [x2, y2]]


def _box_coords(value: Any) -> tuple[float, float, float, float] | None:
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
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        except (TypeError, ValueError):
            return None
    return None


def _is_pair_of_geoms(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    return all(_is_box_like(item) or _is_segment_like(item) for item in value)


def _parse_axis(raw: Any) -> Axis | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or raw not in _AXES:
        raise RegionError("axis 必须是 horizontal、vertical 或 diagonal", "invalid_region")
    return raw


def _pixel_length(geom: Geom, axis: Axis | None) -> tuple[float, str, dict[str, float]]:
    if isinstance(geom, SegmentGeom):
        length = math.hypot(geom.p2[0] - geom.p1[0], geom.p2[1] - geom.p1[1])
        return length, "segment_length", {"length_px": length}
    x1, y1, x2, y2 = geom.box
    width_px = float(x2 - x1)
    height_px = float(y2 - y1)
    diagonal_px = math.hypot(width_px, height_px)
    components = {
        "width_px": width_px,
        "height_px": height_px,
        "diagonal_px": diagonal_px,
    }
    if axis == "horizontal":
        return width_px, "bbox_horizontal", components
    if axis == "vertical":
        return height_px, "bbox_vertical", components
    if axis == "diagonal":
        return diagonal_px, "bbox_diagonal", components
    longer = width_px if width_px >= height_px else height_px
    return longer, "bbox_longer_side", components


def _angle_deg(first: SegmentGeom, second: SegmentGeom) -> float:
    vec_a = (first.p2[0] - first.p1[0], first.p2[1] - first.p1[1])
    vec_b = (second.p2[0] - second.p1[0], second.p2[1] - second.p1[1])
    norm_a = math.hypot(*vec_a)
    norm_b = math.hypot(*vec_b)
    if norm_a == 0 or norm_b == 0:
        raise RegionError("线段长度为 0", "invalid_region")
    cosine = (vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]) / (norm_a * norm_b)
    cosine = min(1.0, max(-1.0, cosine))
    degrees = math.degrees(math.acos(cosine))
    return 180.0 - degrees if degrees > 90.0 else degrees


def _require_single_geom(geoms: list[Geom], measurement: str) -> Geom:
    if len(geoms) != 1:
        raise RegionError(f"{measurement} 需要恰好一段几何", "invalid_region")
    return geoms[0]


def _applied_from_geoms(geoms: list[Geom], axis: Axis | None) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    if len(geoms) == 1:
        applied["region"] = _serialize_geom(geoms[0])
    elif geoms:
        applied["region"] = [_serialize_geom(geom) for geom in geoms]
    if axis is not None:
        applied["axis"] = axis
    return applied


def _serialize_geom(geom: Geom) -> list[Any]:
    if isinstance(geom, BoxGeom):
        return list(geom.box)
    return [list(geom.p1), list(geom.p2)]


def _looks_normalized_points(points: tuple[tuple[float, float], tuple[float, float]]) -> bool:
    return all(0.0 <= coord <= 1.0 for point in points for coord in point)


def _clamp_int(value: float, limit: int) -> int:
    return max(0, min(limit, int(round(value))))


def _clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _ok(result: dict[str, Any]) -> Observation:
    return Observation(ok=True, result=result)


def _fail(error: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=error, error_code=error_code)
