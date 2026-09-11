"""distance_bearing_calculator 共享执行器：WGS84 椭球测地距离与真北方位角。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from pyproj import CRS, Geod, Transformer
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from tool._crs import gcj02_to_wgs84, is_gcj02
from tool.contract import Observation, RuntimeContext, declared_inputs

_OP_DISTANCE = "distance"
_OP_BEARING = "bearing"
_ELLIPSOID = "WGS84"
_CRS_WGS84 = "wgs84"
_GEOD = Geod(ellps=_ELLIPSOID)
_PREVIOUS = "$previous_tool_result"
_DEFAULT_UNIT = "m"
_UNITS = frozenset({"m", "km", "mile"})
_METERS_PER_MILE = 1609.344
_MODE_GEODESIC = "geodesic"
_MODE_ROUTE = "route"
_MODE_GEOMETRY = "geometry"
_MODE_WIDTH = "width"
_MODES = frozenset({_MODE_GEODESIC, _MODE_ROUTE, _MODE_GEOMETRY, _MODE_WIDTH})
_REF_TRUE_NORTH = "true_north"
_REF_MAGNETIC = "magnetic_north"
_REF_IMAGE_AXIS = "image_axis"
_REFERENCES = frozenset({_REF_TRUE_NORTH, _REF_MAGNETIC, _REF_IMAGE_AXIS})
_COMPASS_8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
_COORD_SPLIT_RE = re.compile(r"[,;，、\s]+")
_POINT_GEOM_TYPES = frozenset({"Point", "MultiPoint"})
_LINE_GEOM_TYPES = frozenset({"LineString", "MultiLineString"})
_POLY_GEOM_TYPES = frozenset({"Polygon", "MultiPolygon"})
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
_GEODESIC_ASSUMPTION = "距离与方位角在 WGS84 椭球上用测地线计算，未把经纬度当平面坐标相减"
_NO_DATUM_ASSUMPTION = "测地计算在 WGS84 椭球上进行；GCJ-02 输入会先近似转换"
_COORD_ORDER_ASSUMPTION = "无法唯一判定时，坐标串按 lon,lat 解析"
_NO_SESSION_ASSUMPTION = "本步只返回测地计算结果，未打开街景或地图会话"
_PATH_ASSUMPTION = "多于两个点时 value 是相邻测地段之和，不是首尾直线距离"
_GEOMETRY_ASSUMPTION = "沿几何长度按顶点测地段累加，未加密插值"
_WIDTH_POINTS_ASSUMPTION = "两点或两要素的宽度按最近测地距解释"
_WIDTH_POLYGON_ASSUMPTION = "面要素宽度取局部 AEQD 投影下最小外接矩形短边"
_NEAREST_PROJECTED_ASSUMPTION = "非点要素的最近距在局部 AEQD 投影面计算，未做椭球最近点迭代"


class GeodesicInputError(Exception):
    """points / features / origin / target / mode / unit 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class GeoPoint:
    """地理点；GCJ-02 会先转到 WGS84 再测地。"""

    lat: float
    lon: float
    crs: str = _CRS_WGS84


@dataclass(frozen=True)
class ParsedFeature:
    """已解析的真实几何，坐标顺序为 GeoJSON lon,lat。"""

    geometry: BaseGeometry
    crs: str = _CRS_WGS84


def execute_distance(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按 mode 计算点或要素之间的测地 / 沿几何 / 宽度距离。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "points", "features", "unit", "mode")
        unit = _parse_unit(inputs.get("unit"))
        mode = _parse_mode(inputs.get("mode"))
        if mode == _MODE_ROUTE:
            raise GeodesicInputError(
                "mode=route 是路网距离，请改用 route_query，本工具不做路径规划",
                "unsupported_mode",
            )
        points = _parse_points_field(inputs.get("points"), ctx)
        features = _parse_features_field(inputs.get("features"), ctx)
        if not points and not features:
            raise GeodesicInputError("缺少必填输入 points 或 features", "missing_input")
        if mode == _MODE_WIDTH:
            return _distance_width(points, features, unit=unit)
        if mode == _MODE_GEOMETRY:
            return _distance_geometry(points, features, unit=unit)
        return _distance_geodesic(points, features, unit=unit)
    except GeodesicInputError as exc:
        return _fail(str(exc), exc.error_code)


def execute_bearing(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """计算起点到终点的真北方位角；磁北与图像轴不在本执行器内。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "origin", "target", "reference")
        origin = _require_single_point(inputs.get("origin"), ctx, field="origin")
        target = _require_single_point(inputs.get("target"), ctx, field="target")
        reference = _parse_reference(inputs.get("reference"))
        if reference == _REF_MAGNETIC:
            raise GeodesicInputError(
                "reference=magnetic_north 需要磁偏角模型，本工具不计算磁北",
                "unsupported_reference",
            )
        if reference == _REF_IMAGE_AXIS:
            raise GeodesicInputError(
                "reference=image_axis 属于图像量测，请改用 image_measure",
                "unsupported_reference",
            )
        fwd, back, _distance_m = _geod_inv(origin, target)
        crs = _result_crs([origin.crs, target.crs])
        azimuth = _norm_azimuth(fwd)
        back_azimuth = _norm_azimuth(back)
        applied = {
            "origin": _point_applied(origin),
            "target": _point_applied(target),
            "reference": reference,
        }
        return _ok(
            {
                "operation": _OP_BEARING,
                "value": azimuth,
                "unit": "deg",
                "azimuth_deg": azimuth,
                "back_azimuth_deg": back_azimuth,
                "compass": _compass8(azimuth),
                "reference": reference,
                "method": "geodesic",
                "ellipsoid": _ELLIPSOID,
                "crs": crs,
                "applied": applied,
                "assumptions": _assumptions(crs=crs),
            }
        )
    except GeodesicInputError as exc:
        return _fail(str(exc), exc.error_code)


def _distance_geodesic(
    points: list[GeoPoint],
    features: list[ParsedFeature],
    *,
    unit: str,
) -> Observation:
    path = points if len(points) >= 2 else _points_from_features(features)
    if len(path) < 2:
        if features and all(item.geometry.geom_type in _LINE_GEOM_TYPES for item in features):
            raise GeodesicInputError(
                "geodesic 需要至少两个点；沿线长度请用 mode=geometry",
                "invalid_input",
            )
        if features and all(item.geometry.geom_type in _POLY_GEOM_TYPES for item in features):
            raise GeodesicInputError(
                "geodesic 需要至少两个点；面要素宽度请用 mode=width",
                "invalid_input",
            )
        raise GeodesicInputError("geodesic 需要至少两个可解析坐标点", "invalid_input")
    return _path_observation(
        path,
        unit=unit,
        mode=_MODE_GEODESIC,
        method="geodesic" if len(path) == 2 else "geodesic_path",
        extra_assumptions=[_PATH_ASSUMPTION] if len(path) > 2 else [],
    )


def _distance_geometry(
    points: list[GeoPoint],
    features: list[ParsedFeature],
    *,
    unit: str,
) -> Observation:
    line_features = [item for item in features if item.geometry.geom_type in _LINE_GEOM_TYPES]
    if line_features:
        vertices: list[GeoPoint] = []
        crs_tags: list[str] = []
        for item in line_features:
            crs_tags.append(item.crs)
            vertices.extend(_points_along_geom(item.geometry, crs=item.crs))
        if len(vertices) < 2:
            raise GeodesicInputError("geometry 需要至少两个顶点的线要素", "invalid_input")
        extra = [_GEOMETRY_ASSUMPTION]
        return _path_observation(
            vertices,
            unit=unit,
            mode=_MODE_GEOMETRY,
            method="geometry_length",
            extra_assumptions=extra,
            crs=_result_crs(crs_tags),
        )
    if any(item.geometry.geom_type in _POLY_GEOM_TYPES for item in features):
        raise GeodesicInputError(
            "mode=geometry 只沿 LineString / 折线顶点累加；面要素请用 mode=width",
            "invalid_input",
        )
    path = points if len(points) >= 2 else _points_from_features(features)
    if len(path) < 2:
        raise GeodesicInputError("geometry 需要 LineString 或至少两个坐标点", "invalid_input")
    return _path_observation(
        path,
        unit=unit,
        mode=_MODE_GEOMETRY,
        method="geometry_length",
        extra_assumptions=[_GEOMETRY_ASSUMPTION],
    )


def _distance_width(
    points: list[GeoPoint],
    features: list[ParsedFeature],
    *,
    unit: str,
) -> Observation:
    polygons = [item for item in features if item.geometry.geom_type in _POLY_GEOM_TYPES]
    if len(features) == 1 and features[0].geometry.geom_type in _LINE_GEOM_TYPES:
        raise GeodesicInputError(
            "单个 LineString 无法定宽，请提供两个点、两个要素或面要素",
            "invalid_input",
        )
    if len(features) >= 2:
        return _nearest_observation(features[0], features[1], unit=unit)
    if len(polygons) == 1 and len(features) == 1:
        return _polygon_width_observation(polygons[0], unit=unit)
    if len(points) >= 2:
        return _path_observation(
            points[:2],
            unit=unit,
            mode=_MODE_WIDTH,
            method="nearest_geodesic",
            extra_assumptions=[_WIDTH_POINTS_ASSUMPTION],
        )
    feature_points = _points_from_features(features)
    if len(feature_points) >= 2:
        return _path_observation(
            feature_points[:2],
            unit=unit,
            mode=_MODE_WIDTH,
            method="nearest_geodesic",
            extra_assumptions=[_WIDTH_POINTS_ASSUMPTION],
        )
    raise GeodesicInputError(
        "width 需要两个点、两个要素，或一个 Polygon",
        "invalid_input",
    )


def _path_observation(
    path: list[GeoPoint],
    *,
    unit: str,
    mode: str,
    method: str,
    extra_assumptions: list[str] | None = None,
    crs: str | None = None,
) -> Observation:
    segments: list[dict[str, Any]] = []
    total_m = 0.0
    for start, end in zip(path, path[1:], strict=False):
        _fwd, _back, dist_m = _geod_inv(start, end)
        total_m += dist_m
        segments.append(
            {
                "from": _point_applied(start),
                "to": _point_applied(end),
                "distance_m": dist_m,
            }
        )
    resolved_crs = crs or _result_crs([item.crs for item in path])
    applied: dict[str, Any] = {
        "mode": mode,
        "unit": unit,
        "point_count": len(path),
        "points": [_point_applied(item) for item in path],
    }
    assumptions = _assumptions(crs=resolved_crs, extra=extra_assumptions)
    payload: dict[str, Any] = {
        "operation": _OP_DISTANCE,
        "value": _from_meters(total_m, unit),
        "unit": unit,
        "distance_m": total_m,
        "method": method,
        "ellipsoid": _ELLIPSOID,
        "crs": resolved_crs,
        "applied": applied,
        "assumptions": assumptions,
    }
    if len(segments) > 1:
        payload["segments"] = segments
    return _ok(payload)


def _nearest_observation(left: ParsedFeature, right: ParsedFeature, *, unit: str) -> Observation:
    if left.geometry.geom_type == "Point" and right.geometry.geom_type == "Point":
        start = GeoPoint(lat=float(left.geometry.y), lon=float(left.geometry.x), crs=left.crs)
        end = GeoPoint(lat=float(right.geometry.y), lon=float(right.geometry.x), crs=right.crs)
        return _path_observation(
            [start, end],
            unit=unit,
            mode=_MODE_WIDTH,
            method="nearest_geodesic",
            extra_assumptions=[_WIDTH_POINTS_ASSUMPTION],
        )
    if left.geometry.is_empty or right.geometry.is_empty:
        raise GeodesicInputError("width 要素几何为空", "invalid_input")
    lon_0, lat_0 = _mid_centroid(left.geometry, right.geometry)
    projected_left = _project_aeqd(left.geometry, lon_0=lon_0, lat_0=lat_0)
    projected_right = _project_aeqd(right.geometry, lon_0=lon_0, lat_0=lat_0)
    dist_m = float(projected_left.distance(projected_right))
    crs = _result_crs([left.crs, right.crs])
    applied = {
        "mode": _MODE_WIDTH,
        "unit": unit,
        "features": [_geom_applied(left), _geom_applied(right)],
        "projection": "aeqd",
    }
    return _ok(
        {
            "operation": _OP_DISTANCE,
            "value": _from_meters(dist_m, unit),
            "unit": unit,
            "distance_m": dist_m,
            "method": "nearest_geodesic",
            "ellipsoid": _ELLIPSOID,
            "crs": crs,
            "applied": applied,
            "assumptions": _assumptions(
                crs=crs,
                extra=[_WIDTH_POINTS_ASSUMPTION, _NEAREST_PROJECTED_ASSUMPTION],
            ),
        }
    )


def _polygon_width_observation(feature: ParsedFeature, *, unit: str) -> Observation:
    geom = feature.geometry
    if geom.geom_type == "MultiPolygon":
        geom = unary_union(geom)
    if geom.is_empty or geom.geom_type not in _POLY_GEOM_TYPES:
        raise GeodesicInputError("width 需要有效 Polygon", "invalid_input")
    centroid = geom.centroid
    projected = _project_aeqd(geom, lon_0=float(centroid.x), lat_0=float(centroid.y))
    mrr = projected.minimum_rotated_rectangle
    width_m = _mrr_short_side_m(mrr)
    if width_m <= 0:
        raise GeodesicInputError("面要素退化，无法计算宽度", "invalid_input")
    applied = {
        "mode": _MODE_WIDTH,
        "unit": unit,
        "features": [_geom_applied(feature)],
        "projection": "aeqd",
    }
    return _ok(
        {
            "operation": _OP_DISTANCE,
            "value": _from_meters(width_m, unit),
            "unit": unit,
            "distance_m": width_m,
            "method": "local_aeqd_mrr_width",
            "ellipsoid": _ELLIPSOID,
            "crs": feature.crs,
            "applied": applied,
            "assumptions": _assumptions(
                crs=feature.crs,
                extra=[_WIDTH_POLYGON_ASSUMPTION],
            ),
        }
    )


def _parse_mode(raw: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _MODE_GEODESIC
    if not isinstance(raw, str):
        raise GeodesicInputError(
            "mode 必须是 geodesic、route、geometry 或 width",
            "invalid_mode",
        )
    value = raw.strip().lower()
    if value not in _MODES:
        raise GeodesicInputError(
            "mode 必须是 geodesic、route、geometry 或 width",
            "invalid_mode",
        )
    return value


def _parse_unit(raw: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DEFAULT_UNIT
    if not isinstance(raw, str):
        raise GeodesicInputError("unit 必须是 m、km 或 mile", "invalid_unit")
    value = raw.strip().lower()
    if value in {"metres", "meters", "meter", "metre"}:
        value = "m"
    if value in {"kilometers", "kilometre", "kilometer", "kms"}:
        value = "km"
    if value in {"miles", "mi"}:
        value = "mile"
    if value not in _UNITS:
        raise GeodesicInputError("unit 必须是 m、km 或 mile", "invalid_unit")
    return value


def _parse_reference(raw: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _REF_TRUE_NORTH
    if not isinstance(raw, str):
        raise GeodesicInputError(
            "reference 必须是 true_north、magnetic_north 或 image_axis",
            "invalid_reference",
        )
    value = raw.strip().lower()
    if value not in _REFERENCES:
        raise GeodesicInputError(
            "reference 必须是 true_north、magnetic_north 或 image_axis",
            "invalid_reference",
        )
    return value


def _parse_points_field(raw: Any, ctx: RuntimeContext | None) -> list[GeoPoint]:
    if raw is None or raw == "":
        return []
    resolved = _resolve_ref(raw, ctx, field="points")
    if isinstance(resolved, str):
        resolved = _maybe_json(resolved)
        if isinstance(resolved, str):
            point = _parse_point(resolved)
            if point is None:
                raise GeodesicInputError(
                    "points 必须是真实坐标，不能用地名替代",
                    "invalid_input",
                )
            return [point]
    collected = _collect_points(resolved)
    if collected:
        return collected
    if _looks_like_place_name(raw):
        raise GeodesicInputError("points 必须是真实坐标，不能用地名替代", "invalid_input")
    raise GeodesicInputError("points 无法解析为坐标", "invalid_input")


def _parse_features_field(raw: Any, ctx: RuntimeContext | None) -> list[ParsedFeature]:
    if raw is None or raw == "":
        return []
    resolved = _resolve_ref(raw, ctx, field="features")
    if isinstance(resolved, str):
        resolved = _maybe_json(resolved)
        if isinstance(resolved, str):
            point = _parse_point(resolved)
            if point is not None:
                return [_feature_from_point(point)]
            raise GeodesicInputError(
                "features 必须是真实几何或坐标，不能用地名替代",
                "invalid_input",
            )
    features = _collect_features(resolved)
    if features:
        return features
    if _looks_like_place_name(raw):
        raise GeodesicInputError(
            "features 必须是真实几何或坐标，不能用地名替代",
            "invalid_input",
        )
    raise GeodesicInputError("features 无法解析为几何", "invalid_input")


def _require_single_point(raw: Any, ctx: RuntimeContext | None, *, field: str) -> GeoPoint:
    if raw is None or raw == "":
        raise GeodesicInputError(f"缺少必填输入 {field}", "missing_input")
    resolved = _resolve_ref(raw, ctx, field=field)
    if isinstance(resolved, str):
        resolved = _maybe_json(resolved)
        if isinstance(resolved, str):
            point = _parse_point(resolved)
            if point is None:
                raise GeodesicInputError(
                    f"{field} 必须是真实坐标，不能用地名替代",
                    "invalid_input",
                )
            return point
    point = _parse_point(resolved)
    if point is not None:
        return point
    points = _collect_points(resolved)
    if len(points) == 1:
        return points[0]
    if len(points) > 1:
        raise GeodesicInputError(
            f"无法从 {field} 唯一确定一个点",
            "invalid_input",
        )
    features = _collect_features(resolved)
    if len(features) == 1 and features[0].geometry.geom_type == "Point":
        geom = features[0].geometry
        return GeoPoint(lat=float(geom.y), lon=float(geom.x), crs=features[0].crs)
    if _looks_like_place_name(raw):
        raise GeodesicInputError(
            f"{field} 必须是真实坐标，不能用地名替代",
            "invalid_input",
        )
    raise GeodesicInputError(f"{field} 无法解析为坐标", "invalid_input")


def _resolve_ref(raw: Any, ctx: RuntimeContext | None, *, field: str) -> Any:
    if raw != _PREVIOUS:
        return raw
    previous = ctx.previous_tool_result if ctx is not None else None
    if previous is None:
        raise GeodesicInputError(f"无法解析 {field} 的 $previous_tool_result", "missing_input")
    return _unwrap_previous(previous)


def _unwrap_previous(raw: Any) -> Any:
    if isinstance(raw, dict) and isinstance(raw.get("result"), (dict, list)) and (
        "ok" in raw or "error_code" in raw or "artifacts" in raw
    ):
        return raw["result"]
    return raw


def _collect_points(raw: Any) -> list[GeoPoint]:
    found: list[GeoPoint] = []
    _walk_points(raw, found, depth=0)
    return found


def _walk_points(raw: Any, found: list[GeoPoint], *, depth: int) -> None:
    if depth > 6 or raw is None:
        return
    point = _parse_point(raw)
    if point is not None and not _is_bbox(raw):
        found.append(point)
        return
    if isinstance(raw, list):
        if raw and all(isinstance(item, (int, float, str)) for item in raw):
            return
        for item in raw:
            _walk_points(item, found, depth=depth + 1)
        return
    if not isinstance(raw, dict):
        return
    for key in ("location", "origin", "destination", "center", "point"):
        if key in raw:
            nested = _parse_point(raw[key])
            if nested is not None:
                found.append(_as_wgs84_point(nested))
    geom = raw.get("geometry")
    if isinstance(geom, dict) and geom.get("type") == "Point":
        parsed = _parse_point(geom.get("coordinates"))
        if parsed is not None:
            crs = _crs_of(raw) or _crs_of(geom) or parsed.crs
            found.append(_as_wgs84_point(GeoPoint(lat=parsed.lat, lon=parsed.lon, crs=crs)))
    for key in ("results", "elements", "features", "points", "hits"):
        nested_list = raw.get(key)
        if isinstance(nested_list, list):
            for item in nested_list:
                _walk_points(item, found, depth=depth + 1)


def _collect_features(raw: Any) -> list[ParsedFeature]:
    found: list[ParsedFeature] = []
    _walk_features(raw, found, depth=0)
    return found


def _walk_features(raw: Any, found: list[ParsedFeature], *, depth: int) -> None:
    if depth > 6 or raw is None:
        return
    feature = _parse_feature(raw)
    if feature is not None:
        found.append(feature)
        return
    if isinstance(raw, list):
        if _is_coord_pair(raw):
            point = _parse_point(raw)
            if point is not None:
                found.append(_feature_from_point(point))
            return
        if raw and all(isinstance(item, (list, tuple)) and _is_coord_pair(item) for item in raw):
            geom = _shape_from_mapping({"type": "LineString", "coordinates": raw})
            if geom is not None:
                found.append(_as_wgs84_feature(ParsedFeature(geometry=geom)))
            return
        for item in raw:
            _walk_features(item, found, depth=depth + 1)
        return
    if not isinstance(raw, dict):
        return
    for key in ("features", "elements", "results", "geometries"):
        nested = raw.get(key)
        if isinstance(nested, list):
            for item in nested:
                _walk_features(item, found, depth=depth + 1)


def _parse_feature(raw: Any) -> ParsedFeature | None:
    if isinstance(raw, dict):
        crs = _crs_of(raw)
        if raw.get("type") == "Feature":
            geom = _shape_from_mapping(raw.get("geometry"))
            if geom is not None:
                return _as_wgs84_feature(ParsedFeature(geometry=geom, crs=crs or _CRS_WGS84))
            return None
        if raw.get("type") in {
            "Point",
            "MultiPoint",
            "LineString",
            "MultiLineString",
            "Polygon",
            "MultiPolygon",
            "GeometryCollection",
        }:
            geom = _shape_from_mapping(raw)
            if geom is not None:
                return _as_wgs84_feature(ParsedFeature(geometry=geom, crs=crs or _CRS_WGS84))
            return None
        nested = raw.get("geometry")
        if isinstance(nested, dict):
            geom = _shape_from_mapping(nested)
            if geom is not None:
                return _as_wgs84_feature(
                    ParsedFeature(geometry=geom, crs=crs or _crs_of(nested) or _CRS_WGS84)
                )
        point = _parse_point(raw)
        if point is not None:
            return _feature_from_point(point)
        return None
    point = _parse_point(raw)
    if point is not None:
        return _feature_from_point(point)
    return None


def _shape_from_mapping(raw: Any) -> BaseGeometry | None:
    if not isinstance(raw, dict) or "type" not in raw:
        return None
    try:
        geom = shape(raw)
    except (ValueError, TypeError, AttributeError):
        return None
    if geom.is_empty:
        return None
    return geom


def _feature_from_point(point: GeoPoint) -> ParsedFeature:
    converted = _as_wgs84_point(point)
    geom = shape({"type": "Point", "coordinates": [converted.lon, converted.lat]})
    return ParsedFeature(geometry=geom, crs=converted.crs)


def _as_wgs84_point(point: GeoPoint) -> GeoPoint:
    if not is_gcj02(point.crs):
        return point
    lon, lat = gcj02_to_wgs84(point.lon, point.lat)
    return GeoPoint(lat=lat, lon=lon, crs=_CRS_WGS84)


def _as_wgs84_feature(feature: ParsedFeature) -> ParsedFeature:
    if not is_gcj02(feature.crs):
        return feature

    def _project(x: float, y: float, z: float | None = None) -> tuple[float, ...]:
        lon, lat = gcj02_to_wgs84(x, y)
        if z is None:
            return lon, lat
        return lon, lat, z

    return ParsedFeature(geometry=shapely_transform(_project, feature.geometry), crs=_CRS_WGS84)


def _parse_point(raw: Any) -> GeoPoint | None:
    if isinstance(raw, GeoPoint):
        return _as_wgs84_point(raw)
    if isinstance(raw, str):
        return _parse_coordinate_query(raw)
    if isinstance(raw, (list, tuple)):
        if len(raw) < 2:
            return None
        if any(isinstance(item, (list, dict)) for item in raw):
            return None
        first = _as_float(raw[0])
        second = _as_float(raw[1])
        if first is None or second is None:
            return None
        return _point_from_pair(first, second)
    if isinstance(raw, dict):
        crs = _crs_of(raw) or _CRS_WGS84
        lat = _as_float(raw.get("lat", raw.get("latitude")))
        lon = _as_float(raw.get("lon", raw.get("lng", raw.get("longitude"))))
        nested = raw.get("location")
        if isinstance(nested, dict) and (lat is None or lon is None):
            lat = _as_float(nested.get("lat", nested.get("latitude")))
            lon = _as_float(nested.get("lon", nested.get("lng", nested.get("longitude"))))
            crs = _crs_of(nested) or crs
        elif isinstance(nested, str) and (lat is None or lon is None):
            parsed = _parse_coordinate_query(nested)
            return None if parsed is None else _as_wgs84_point(
                GeoPoint(lat=parsed.lat, lon=parsed.lon, crs=crs)
            )
        elif isinstance(nested, (list, tuple)) and (lat is None or lon is None):
            parsed = _parse_point(nested)
            return None if parsed is None else _as_wgs84_point(
                GeoPoint(lat=parsed.lat, lon=parsed.lon, crs=crs)
            )
        if lat is None or lon is None:
            coords = raw.get("coordinates")
            if isinstance(coords, (list, tuple)):
                parsed = _parse_point(coords)
                if parsed is not None:
                    return _as_wgs84_point(GeoPoint(lat=parsed.lat, lon=parsed.lon, crs=crs))
            return None
        if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
            return None
        return _as_wgs84_point(GeoPoint(lat=lat, lon=lon, crs=crs))
    return None


def _parse_coordinate_query(raw: str) -> GeoPoint | None:
    text = raw.strip()
    if not text or text == _PREVIOUS:
        return None
    parts = [part for part in _COORD_SPLIT_RE.split(text) if part]
    if len(parts) != 2:
        return None
    first = _as_float(parts[0])
    second = _as_float(parts[1])
    if first is None or second is None:
        return None
    return _point_from_pair(first, second)


def _point_from_pair(first: float, second: float, *, crs: str = _CRS_WGS84) -> GeoPoint | None:
    if abs(first) > 90.0 and abs(second) <= 90.0:
        lon, lat = first, second
    elif abs(second) > 90.0 and abs(first) <= 90.0:
        lat, lon = first, second
    else:
        lon, lat = first, second
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return _as_wgs84_point(GeoPoint(lat=lat, lon=lon, crs=crs))


def _points_from_features(features: list[ParsedFeature]) -> list[GeoPoint]:
    points: list[GeoPoint] = []
    for item in features:
        if item.geometry.geom_type == "Point":
            points.append(GeoPoint(lat=float(item.geometry.y), lon=float(item.geometry.x), crs=item.crs))
        elif item.geometry.geom_type == "MultiPoint":
            for geom in item.geometry.geoms:
                points.append(GeoPoint(lat=float(geom.y), lon=float(geom.x), crs=item.crs))
    return points


def _points_along_geom(geom: BaseGeometry, *, crs: str) -> list[GeoPoint]:
    if geom.geom_type == "LineString":
        return [GeoPoint(lat=float(y), lon=float(x), crs=crs) for x, y in geom.coords]
    if geom.geom_type == "MultiLineString":
        points: list[GeoPoint] = []
        for part in geom.geoms:
            points.extend(_points_along_geom(part, crs=crs))
        return points
    return []


def _geod_inv(origin: GeoPoint, target: GeoPoint) -> tuple[float, float, float]:
    fwd, back, dist_m = _GEOD.inv(origin.lon, origin.lat, target.lon, target.lat)
    return float(fwd), float(back), float(dist_m)


def _project_aeqd(geom: BaseGeometry, *, lon_0: float, lat_0: float) -> BaseGeometry:
    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)

    def _project(x: float, y: float, z: float | None = None) -> tuple[float, ...]:
        east, north = transformer.transform(x, y)
        if z is None:
            return east, north
        return east, north, z

    return shapely_transform(_project, geom)


def _mid_centroid(left: BaseGeometry, right: BaseGeometry) -> tuple[float, float]:
    lon = (float(left.centroid.x) + float(right.centroid.x)) / 2.0
    lat = (float(left.centroid.y) + float(right.centroid.y)) / 2.0
    return lon, lat


def _mrr_short_side_m(mrr: BaseGeometry) -> float:
    coords = list(mrr.exterior.coords)
    if len(coords) < 3:
        return 0.0
    sides: list[float] = []
    for start, end in zip(coords, coords[1:], strict=False):
        sides.append(math.hypot(end[0] - start[0], end[1] - start[1]))
        if len(sides) == 4:
            break
    if len(sides) < 2:
        return 0.0
    return min(sides[0], sides[1])


def _from_meters(distance_m: float, unit: str) -> float:
    if unit == "km":
        return distance_m / 1000.0
    if unit == "mile":
        return distance_m / _METERS_PER_MILE
    return distance_m


def _norm_azimuth(azimuth_deg: float) -> float:
    return float(azimuth_deg) % 360.0


def _compass8(azimuth_deg: float) -> str:
    idx = int((azimuth_deg + 22.5) // 45.0) % 8
    return _COMPASS_8[idx]


def _result_crs(tags: list[str]) -> str:
    unique = {item.strip().lower() for item in tags if item and item.strip()}
    if len(unique) == 1:
        return unique.pop()
    if not unique:
        return _CRS_WGS84
    return "mixed"


def _crs_of(raw: dict[str, Any]) -> str | None:
    value = raw.get("crs", raw.get("datum"))
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    if isinstance(value, dict):
        name = value.get("properties", value) if isinstance(value.get("properties"), dict) else value
        if isinstance(name, dict):
            text = name.get("name", name.get("code"))
            if isinstance(text, str) and text.strip():
                return text.strip().lower()
    return None


def _point_applied(point: GeoPoint) -> dict[str, float | str]:
    return {"lon": point.lon, "lat": point.lat, "crs": point.crs}


def _geom_applied(feature: ParsedFeature) -> dict[str, Any]:
    payload = mapping(feature.geometry)
    return {"type": payload.get("type"), "crs": feature.crs}


def _assumptions(*, crs: str, extra: list[str] | None = None) -> list[str]:
    items = [
        _GEODESIC_ASSUMPTION,
        _NO_DATUM_ASSUMPTION,
        _COORD_ORDER_ASSUMPTION,
        _NO_SESSION_ASSUMPTION,
    ]
    if crs == "mixed":
        items.insert(2, "输入含多种 crs 标签，未做转换，按给定经纬度直接测地")
    elif crs != _CRS_WGS84:
        items.insert(2, f"输入 crs 为 {crs}，未转换到 WGS84")
    if extra:
        items.extend(extra)
    return items


def _maybe_json(raw: str) -> Any:
    text = raw.strip()
    if not text or text[0] not in "{[":
        return raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw


def _looks_like_place_name(raw: Any) -> bool:
    if not isinstance(raw, str):
        return False
    text = raw.strip()
    if not text or text == _PREVIOUS:
        return False
    if text[0] in "{[":
        return False
    return _parse_coordinate_query(text) is None


def _is_coord_pair(raw: Any) -> bool:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return False
    return _as_float(raw[0]) is not None and _as_float(raw[1]) is not None


def _is_bbox(raw: Any) -> bool:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return False
    return all(_as_float(item) is not None for item in raw)


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
