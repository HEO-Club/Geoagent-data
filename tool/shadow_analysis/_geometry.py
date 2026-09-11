"""shadow_analysis 正向阴影：复用 solar_ephemeris，竖直物体与局部平面求交。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from tool.contract import Observation, RuntimeContext, declared_inputs
from tool.solar_ephemeris._ephemeris import execute_sun_position

_OP = "shadow_model"
_METHOD = "vertical_object_plane_intersection"
_LIBRARY = "pvlib"
_STATUS_OK = "computed_under_inputs"
_STATUS_DIRECTION = "direction_only"
_STATUS_BELOW = "sun_below_horizon"
_STATUS_MISS = "no_intersection"
_LOW_ELEV_DEG = 5.0
_PARALLEL_EPS = 1e-12
_RASTER_SUFFIXES = (".tif", ".tiff", ".geotiff", ".img", ".asc", ".vrt")
_SUN_PASSTHROUGH = (
    "area",
    "datetime",
    "timezone",
    "altitude_m",
    "candidate",
    "hypothesis",
    "status",
)
_HORIZONTAL_TOKENS = frozenset(
    {
        "flat",
        "horizontal",
        "level",
        "plane",
        "平地",
        "水平",
        "水平地面",
        "水平面",
    }
)
_SLOPE_KEYS = ("slope_deg", "slope", "dip_deg", "坡度")
_ASPECT_KEYS = ("aspect_deg", "aspect", "facing", "坡向")
_TERRAIN_KEYS = frozenset(
    {
        "dem",
        "dsm",
        "geotiff",
        "grid",
        "layer",
        "path",
        "raster",
        "terrain",
        "tif",
        "tiff",
        "uri",
        "url",
        "values",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "body",
        "confirmed_location",
        "confirmed_place",
        "content",
        "full_text",
        "location_confirmed",
        "markdown",
        "raw_content",
        "taken_at",
    }
)
_ASSUMPTION_METHOD = (
    "阴影按竖直物体与局部平面射线求交；太阳位置按 NREL SPA（pvlib nrel_numpy）"
    "本地计算，未调用在线历算 API，未做 DEM 或三维地形光线追踪"
)
_ASSUMPTION_VERTICAL = "物体按重力竖直，不是坡面法向"
_ASSUMPTION_HORIZONTAL = "地面按水平面；未提供坡度坡向时不编造复杂地形"
_ASSUMPTION_AZIMUTH = "阴影方位角以真北为 0°、顺时针增加，表示物体指向影尖"
_ASSUMPTION_NO_HEIGHT = "未提供物体高度，只返回阴影方向，未编造阴影长度"
_ASSUMPTION_BELOW = "太阳视高度角不高于地平线，地面无精确阴影长度"
_ASSUMPTION_MISS = "太阳光线与给定坡面无前向交点，未编造阴影长度"
_ASSUMPTION_LOW = "低太阳高度角，阴影长度对高度和地面假设极敏感"
_ASSUMPTION_CANDIDATE = "计算结果以给定地点与时刻为条件，不表示该地点或拍摄时间已被确认"


class ShadowInputError(Exception):
    """object_height_m / surface 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class SurfaceSpec:
    """局部地面：水平面或给定坡度/坡向的平面。"""

    mode: str
    slope_deg: float
    aspect_deg: float | None
    assumed_default: bool


@dataclass(frozen=True)
class ShadowHit:
    """射线–平面求交结果；长度为 None 时不得当作已测长度。"""

    status: str
    azimuth_deg: float
    length_m: float | None
    east_m: float | None
    north_m: float | None


def execute_shadow_model(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按地点、时刻与可选物体/地面参数计算理论阴影方向或长度。"""

    try:
        declared = declared_inputs(
            inputs, "area", "datetime", "object_height_m", "surface"
        )
        height_m = _parse_height(declared.get("object_height_m"))
        surface = _parse_surface(declared.get("surface"))
        sun_obs = execute_sun_position(
            purpose=purpose,
            inputs=_sun_inputs(inputs, declared),
            ctx=ctx,
        )
        if not sun_obs.ok or sun_obs.result is None:
            return sun_obs
        return _ok(_build_result(sun_obs.result, height_m=height_m, surface=surface))
    except ShadowInputError as exc:
        return _fail(str(exc), exc.error_code)


def _sun_inputs(inputs: dict[str, Any], declared: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in _SUN_PASSTHROUGH:
        if key in declared:
            payload[key] = declared[key]
        elif key in inputs:
            payload[key] = inputs[key]
    return payload


def _build_result(
    sun: dict[str, Any],
    *,
    height_m: float | None,
    surface: SurfaceSpec,
) -> dict[str, Any]:
    elevation = _require_angle(sun.get("apparent_elevation_deg"), field="apparent_elevation_deg")
    geometric = _require_angle(sun.get("elevation_deg"), field="elevation_deg")
    azimuth = _norm_azimuth(_require_angle(sun.get("azimuth_deg"), field="azimuth_deg"))
    hit = _cast_shadow(
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        height_m=height_m,
        surface=surface,
    )
    applied = dict(sun.get("applied") or {})
    applied["object_height_m"] = height_m
    applied["surface"] = _surface_payload(surface)
    payload: dict[str, Any] = {
        "operation": _OP,
        "sun": {
            "elevation_deg": geometric,
            "apparent_elevation_deg": elevation,
            "azimuth_deg": azimuth,
        },
        "shadow_azimuth_deg": hit.azimuth_deg,
        "shadow_length_m": hit.length_m,
        "object_height_m": height_m,
        "surface": _surface_payload(surface),
        "method": _METHOD,
        "library": _LIBRARY,
        "status": hit.status,
        "applied": applied,
        "assumptions": _assumptions(
            sun_assumptions=sun.get("assumptions"),
            surface=surface,
            hit=hit,
            elevation_deg=elevation,
            height_m=height_m,
        ),
    }
    if sun.get("zenith_deg") is not None:
        payload["sun"]["zenith_deg"] = sun["zenith_deg"]
    if hit.east_m is not None and hit.north_m is not None:
        payload["offset"] = {"east_m": hit.east_m, "north_m": hit.north_m}
    if sun.get("condition") is not None:
        payload["condition"] = sun["condition"]
    if sun.get("hypothesis") is True:
        payload["hypothesis"] = True
    return payload


def _cast_shadow(
    *,
    azimuth_deg: float,
    elevation_deg: float,
    height_m: float | None,
    surface: SurfaceSpec,
) -> ShadowHit:
    opposite = _norm_azimuth(azimuth_deg + 180.0)
    if elevation_deg <= 0.0:
        return ShadowHit(
            status=_STATUS_BELOW,
            azimuth_deg=opposite,
            length_m=None,
            east_m=None,
            north_m=None,
        )
    direction_only = height_m is None
    if height_m == 0.0:
        return ShadowHit(
            status=_STATUS_OK,
            azimuth_deg=opposite,
            length_m=0.0,
            east_m=0.0,
            north_m=0.0,
        )
    scale = 1.0 if direction_only else float(height_m or 1.0)
    intersection = _intersect_plane(
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        height_m=scale,
        surface=surface,
    )
    if intersection is None:
        return ShadowHit(
            status=_STATUS_MISS,
            azimuth_deg=opposite,
            length_m=None,
            east_m=None,
            north_m=None,
        )
    east_m, north_m, length_m = intersection
    azimuth = _azimuth_from_offset(east_m, north_m, fallback=opposite)
    if direction_only:
        return ShadowHit(
            status=_STATUS_DIRECTION,
            azimuth_deg=azimuth,
            length_m=None,
            east_m=None,
            north_m=None,
        )
    return ShadowHit(
        status=_STATUS_OK,
        azimuth_deg=azimuth,
        length_m=length_m,
        east_m=east_m,
        north_m=north_m,
    )


def _intersect_plane(
    *,
    azimuth_deg: float,
    elevation_deg: float,
    height_m: float,
    surface: SurfaceSpec,
) -> tuple[float, float, float] | None:
    azimuth_rad = math.radians(azimuth_deg)
    elev_rad = math.radians(elevation_deg)
    cos_e = math.cos(elev_rad)
    sin_e = math.sin(elev_rad)
    ray_e = -math.sin(azimuth_rad) * cos_e
    ray_n = -math.cos(azimuth_rad) * cos_e
    ray_u = -sin_e
    slope_rad = math.radians(surface.slope_deg)
    aspect_rad = math.radians(surface.aspect_deg or 0.0)
    sin_s = math.sin(slope_rad)
    normal_e = sin_s * math.sin(aspect_rad)
    normal_n = sin_s * math.cos(aspect_rad)
    normal_u = math.cos(slope_rad)
    denom = normal_e * ray_e + normal_n * ray_n + normal_u * ray_u
    if abs(denom) < _PARALLEL_EPS:
        return None
    t = -(normal_u * height_m) / denom
    if t <= 0.0:
        return None
    east_m = t * ray_e
    north_m = t * ray_n
    up_m = height_m + t * ray_u
    length_m = math.sqrt(east_m * east_m + north_m * north_m + up_m * up_m)
    return east_m, north_m, length_m


def _parse_height(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    value = _as_float(raw)
    if value is None:
        raise ShadowInputError("object_height_m 必须是数字", "invalid_input")
    if value < 0.0:
        raise ShadowInputError("object_height_m 必须 >= 0", "invalid_input")
    return value


def _parse_surface(raw: Any) -> SurfaceSpec:
    if raw in (None, ""):
        return SurfaceSpec(
            mode="horizontal",
            slope_deg=0.0,
            aspect_deg=None,
            assumed_default=True,
        )
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return SurfaceSpec(
                mode="horizontal",
                slope_deg=0.0,
                aspect_deg=None,
                assumed_default=True,
            )
        parsed = _maybe_json(text)
        if parsed is not text:
            return _parse_surface(parsed)
        if text.casefold() in _HORIZONTAL_TOKENS:
            return SurfaceSpec(
                mode="horizontal",
                slope_deg=0.0,
                aspect_deg=None,
                assumed_default=False,
            )
        if _looks_like_terrain_ref(text):
            raise ShadowInputError(
                "surface 只接受水平面或坡度/坡向，复杂地形请先 terrain_analysis",
                "invalid_input",
            )
        raise ShadowInputError(
            "surface 无法解析为水平面或坡度/坡向",
            "invalid_input",
        )
    if isinstance(raw, dict):
        if _has_terrain_ref(raw):
            raise ShadowInputError(
                "surface 只接受水平面或坡度/坡向，复杂地形请先 terrain_analysis",
                "invalid_input",
            )
        slope = _first_number(raw, _SLOPE_KEYS)
        aspect = _first_number(raw, _ASPECT_KEYS)
        if slope is None or slope == 0.0:
            return SurfaceSpec(
                mode="horizontal",
                slope_deg=0.0,
                aspect_deg=None if aspect is None else _norm_azimuth(aspect),
                assumed_default=False,
            )
        if slope < 0.0 or slope >= 90.0:
            raise ShadowInputError("surface.slope_deg 必须在 [0, 90) 度", "invalid_input")
        if aspect is None:
            raise ShadowInputError(
                "surface 坡度非 0 时需要 aspect_deg（坡向）",
                "missing_input",
            )
        return SurfaceSpec(
            mode="inclined",
            slope_deg=float(slope),
            aspect_deg=_norm_azimuth(aspect),
            assumed_default=False,
        )
    raise ShadowInputError("surface 必须是文本或含坡度/坡向的对象", "invalid_input")


def _surface_payload(surface: SurfaceSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": surface.mode,
        "slope_deg": surface.slope_deg,
    }
    if surface.aspect_deg is not None:
        payload["aspect_deg"] = surface.aspect_deg
    return payload


def _assumptions(
    *,
    sun_assumptions: Any,
    surface: SurfaceSpec,
    hit: ShadowHit,
    elevation_deg: float,
    height_m: float | None,
) -> list[str]:
    items: list[str] = []
    if isinstance(sun_assumptions, list):
        items.extend(str(item) for item in sun_assumptions if item)
    extras = [
        _ASSUMPTION_METHOD,
        _ASSUMPTION_VERTICAL,
        _ASSUMPTION_AZIMUTH,
        _ASSUMPTION_CANDIDATE,
    ]
    if surface.mode == "horizontal":
        extras.append(_ASSUMPTION_HORIZONTAL)
    else:
        extras.append(
            f"地面按坡度 {surface.slope_deg}°、坡向 {surface.aspect_deg}° 的局部平面，不是完整地形"
        )
    if height_m is None:
        extras.append(_ASSUMPTION_NO_HEIGHT)
    if hit.status == _STATUS_BELOW:
        extras.append(_ASSUMPTION_BELOW)
    if hit.status == _STATUS_MISS:
        extras.append(_ASSUMPTION_MISS)
    if (
        hit.length_m is not None
        and 0.0 < elevation_deg < _LOW_ELEV_DEG
    ):
        extras.append(_ASSUMPTION_LOW)
    for item in extras:
        if item not in items:
            items.append(item)
    return items


def _has_terrain_ref(raw: dict[str, Any]) -> bool:
    return any(key in raw and raw[key] not in (None, "", False) for key in _TERRAIN_KEYS)


def _looks_like_terrain_ref(text: str) -> bool:
    lowered = text.casefold()
    if any(lowered.endswith(suffix) for suffix in _RASTER_SUFFIXES):
        return True
    if "/" in text or "\\" in text:
        return True
    return any(token in lowered for token in ("dem", "dsm", "geotiff"))


def _first_number(raw: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in raw:
            continue
        value = _as_float(raw[key])
        if value is None and raw[key] not in (None, ""):
            raise ShadowInputError(f"surface.{key} 必须是数字", "invalid_input")
        if value is not None:
            return value
    return None


def _require_angle(raw: Any, *, field: str) -> float:
    value = _as_float(raw)
    if value is None or not math.isfinite(value):
        raise ShadowInputError(f"太阳位置缺少有效 {field}", "engine_unavailable")
    return value


def _azimuth_from_offset(east_m: float, north_m: float, *, fallback: float) -> float:
    if east_m == 0.0 and north_m == 0.0:
        return fallback
    return _norm_azimuth(math.degrees(math.atan2(east_m, north_m)))


def _maybe_json(raw: str) -> Any:
    text = raw.strip()
    if not text or text[0] not in "{[":
        return raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw


def _as_float(raw: Any) -> float | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            return float(raw.strip())
        except ValueError:
            return None
    return None


def _norm_azimuth(azimuth_deg: float) -> float:
    return float(azimuth_deg) % 360.0


def _ok(result: dict[str, Any]) -> Observation:
    return Observation(ok=True, result=_strip_forbidden(result))


def _fail(message: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=message, error_code=error_code)


def _strip_forbidden(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_forbidden(item)
            for key, item in value.items()
            if key not in _FORBIDDEN_KEYS
        }
    if isinstance(value, list):
        return [_strip_forbidden(item) for item in value]
    return value
