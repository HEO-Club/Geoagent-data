"""本地地理 Tool 共用的 WGS84 坐标、几何和结果引用工具。"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any

from pyproj import CRS, Geod, Transformer
from shapely.geometry import LineString, Point, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union

from tool.runtime.result_store import resolve_result

WGS84 = "EPSG:4326"
GEOD = Geod(ellps="WGS84")
_MAX_SOURCE_ITEMS = 100_000
_MAX_LOCAL_DISTANCE_M = 2_000_000.0
_MAX_LOCAL_SPAN_DEGREES = 30.0


class GeoRuntimeError(ValueError):
    """本地地理输入缺失、歧义或几何无效。"""

    def __init__(self, message: str, error_code: str = "invalid_input") -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class GeoPoint:
    latitude: float
    longitude: float
    label: str = ""
    derived_from: str = "coordinate"

    def as_dict(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "label": self.label,
            "derived_from": self.derived_from,
        }


def resolve_structured_source(
    raw: Any,
    ctx: Any,
    *,
    field: str = "source_result",
) -> dict[str, Any]:
    if raw is None:
        raise GeoRuntimeError(f"{field} 为必填", "missing_input")
    source = resolve_result(raw, ctx)
    if source is None:
        raise GeoRuntimeError(
            f"{field} 尚未解析；请传入真实 result_id 或结构化结果",
            "needs_acquisition",
        )
    return source


def source_items(source: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    for key in ("elements", "features", "candidates", "results"):
        raw_items = source.get(key)
        if raw_items is None:
            continue
        if not isinstance(raw_items, list):
            raise GeoRuntimeError(f"source_result.{key} 必须是数组", "invalid_source_result")
        if len(raw_items) > _MAX_SOURCE_ITEMS:
            raise GeoRuntimeError(
                f"source_result.{key} 超过 {_MAX_SOURCE_ITEMS} 条本地处理上限",
                "source_too_large",
            )
        items = [copy.deepcopy(item) for item in raw_items if isinstance(item, dict)]
        if len(items) != len(raw_items):
            raise GeoRuntimeError(
                f"source_result.{key} 必须只包含对象",
                "invalid_source_result",
            )
        return items, key
    raise GeoRuntimeError(
        "source_result 不包含 elements/features/candidates/results",
        "invalid_source_result",
    )


def require_wgs84(source: dict[str, Any], *, field: str = "source_result") -> None:
    raw = source.get("crs")
    if raw is None:
        return
    if isinstance(raw, dict):
        properties = raw.get("properties")
        raw = properties.get("name") if isinstance(properties, dict) else raw.get("name")
    normalized = str(raw).strip().upper().replace(" ", "")
    accepted = {
        "EPSG:4326",
        "WGS84",
        "WGS-84",
        "OGC:CRS84",
        "URN:OGC:DEF:CRS:OGC:1.3:CRS84",
        "URN:OGC:DEF:CRS:EPSG::4326",
    }
    if normalized not in accepted:
        raise GeoRuntimeError(
            f"{field}.crs={raw!r} 不是 WGS84；请先显式转换为 EPSG:4326",
            "unsupported_crs",
        )


def source_label(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        result_id = raw.get("result_id")
        if isinstance(result_id, str):
            return result_id
    return "inline"


def point_from_value(raw: Any, ctx: Any, *, field: str) -> GeoPoint:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise GeoRuntimeError(f"{field} 不能为空", "missing_input")
        if text.startswith(("{", "[")):
            try:
                return point_from_value(json.loads(text), ctx, field=field)
            except json.JSONDecodeError as exc:
                raise GeoRuntimeError(f"{field} 不是有效 JSON", "invalid_coordinate") from exc
        source = resolve_result(text, ctx)
        if source is None:
            raise GeoRuntimeError(
                f"{field}={text!r} 不是可解析坐标或 result_id",
                "needs_acquisition",
            )
        return point_from_value(source, ctx, field=field)

    if isinstance(raw, (list, tuple)):
        if len(raw) != 2:
            raise GeoRuntimeError(
                f"{field} 坐标数组必须是 [lat,lon]",
                "invalid_coordinate",
            )
        return _point(raw[0], raw[1], field=field)

    if not isinstance(raw, dict):
        raise GeoRuntimeError(f"{field} 必须是坐标对象或 result_id", "invalid_coordinate")

    if set(raw) == {"result_id"}:
        source = resolve_result(raw, ctx)
        if source is None:
            raise GeoRuntimeError(f"{field}.result_id 无法解析", "needs_acquisition")
        return point_from_value(source, ctx, field=field)

    label = str(raw.get("name") or raw.get("display_name") or raw.get("label") or "")
    if "latitude" in raw and "longitude" in raw:
        return _point(raw["latitude"], raw["longitude"], field=field, label=label)
    if "lat" in raw and ("lon" in raw or "lng" in raw):
        return _point(raw["lat"], raw.get("lon", raw.get("lng")), field=field, label=label)
    center = raw.get("center")
    if isinstance(center, dict):
        centered = point_from_value(center, ctx, field=field)
        return GeoPoint(
            centered.latitude,
            centered.longitude,
            label=label or centered.label,
            derived_from="center",
        )
    if raw.get("type") == "Point" and isinstance(raw.get("coordinates"), (list, tuple)):
        coordinates = raw["coordinates"]
        if len(coordinates) != 2:
            raise GeoRuntimeError(f"{field} GeoJSON Point 无效", "invalid_coordinate")
        return _point(
            coordinates[1],
            coordinates[0],
            field=field,
            label=label,
            derived_from="geojson_point",
        )
    geometry = raw.get("geometry")
    if isinstance(geometry, dict) and geometry.get("type"):
        geom = _validated_shape(geometry, field=field)
        centroid = geom.centroid
        return _point(
            centroid.y,
            centroid.x,
            field=field,
            label=label,
            derived_from="geometry_centroid",
        )
    for key in ("candidates", "elements", "features", "results"):
        values = raw.get(key)
        if isinstance(values, list):
            if len(values) != 1:
                raise GeoRuntimeError(
                    f"{field}.{key} 含 {len(values)} 个候选；必须先明确选择一个",
                    "ambiguous_coordinate",
                )
            return point_from_value(values[0], ctx, field=field)
    geom, source = geometry_from_item(raw)
    if geom is not None:
        centroid = geom.centroid
        return _point(
            centroid.y,
            centroid.x,
            field=field,
            label=label,
            derived_from=f"{source}_centroid",
        )
    raise GeoRuntimeError(f"{field} 不包含可用坐标", "needs_acquisition")


def points_from_value(raw: Any, ctx: Any, *, field: str = "points") -> list[GeoPoint]:
    if isinstance(raw, (str, dict)):
        source = resolve_result(raw, ctx)
        value = source if source is not None else raw
        if isinstance(value, dict):
            for key in ("points", "candidates", "elements", "features", "results"):
                items = value.get(key)
                if isinstance(items, list):
                    return [point_from_value(item, ctx, field=f"{field}[{index}]") for index, item in enumerate(items)]
        return [point_from_value(value, ctx, field=field)]
    if not isinstance(raw, (list, tuple)):
        raise GeoRuntimeError(f"{field} 必须是坐标数组", "invalid_coordinate")
    if len(raw) == 2 and all(isinstance(value, (int, float)) for value in raw):
        return [point_from_value(raw, ctx, field=field)]
    return [point_from_value(item, ctx, field=f"{field}[{index}]") for index, item in enumerate(raw)]


def geometry_from_reference(raw: Any, ctx: Any, *, field: str = "geometry") -> BaseGeometry:
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith(("{", "[")):
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise GeoRuntimeError(f"{field} 不是有效 JSON", "invalid_geometry") from exc
        else:
            source = resolve_result(text, ctx)
            if source is None:
                raise GeoRuntimeError(
                    f"{field} 不是可解析 result_id",
                    "needs_acquisition",
                )
            raw = source

    if isinstance(raw, (list, tuple)):
        if len(raw) == 4:
            west, south, east, north = _bbox(raw, field=field)
            return box(west, south, east, north)
        if len(raw) == 2:
            point = point_from_value(raw, ctx, field=field)
            return Point(point.longitude, point.latitude)
        raise GeoRuntimeError(
            f"{field} 数组必须是 [lat,lon] 或 [west,south,east,north]",
            "invalid_geometry",
        )
    if not isinstance(raw, dict):
        raise GeoRuntimeError(f"{field} 必须是 GeoJSON、bbox 或 result_id", "invalid_geometry")

    require_wgs84(raw, field=field)

    if set(raw) == {"result_id"}:
        source = resolve_result(raw, ctx)
        if source is None:
            raise GeoRuntimeError(f"{field}.result_id 无法解析", "needs_acquisition")
        return geometry_from_reference(source, ctx, field=field)
    if raw.get("type") == "Feature":
        return _validated_shape(raw.get("geometry"), field=field)
    if raw.get("type") == "FeatureCollection":
        features = raw.get("features")
        if not isinstance(features, list):
            raise GeoRuntimeError(f"{field}.features 必须是数组", "invalid_geometry")
        return _union_geometries(features, field=field)
    if isinstance(raw.get("type"), str) and "coordinates" in raw:
        return _validated_shape(raw, field=field)
    if "bbox" in raw:
        west, south, east, north = _bbox(raw["bbox"], field=field)
        return box(west, south, east, north)
    for key in ("elements", "features", "candidates", "results"):
        items = raw.get(key)
        if isinstance(items, list):
            return _union_geometries(items, field=field)
    geom, _ = geometry_from_item(raw)
    if geom is not None:
        return geom
    point = point_from_value(raw, ctx, field=field)
    return Point(point.longitude, point.latitude)


def geometry_from_item(item: dict[str, Any]) -> tuple[BaseGeometry | None, str]:
    if item.get("type") == "Feature":
        geometry = item.get("geometry")
        if geometry is None:
            return None, "missing"
        return _validated_shape(geometry, field="feature.geometry"), "geojson"
    geojson = item.get("geojson")
    if isinstance(geojson, dict) and geojson.get("type"):
        return _validated_shape(geojson, field="geojson"), "geojson"
    raw_geometry = item.get("geometry")
    if isinstance(raw_geometry, dict) and raw_geometry.get("type"):
        return _validated_shape(raw_geometry, field="geometry"), "geojson"
    if isinstance(raw_geometry, list):
        coordinates = _coordinate_pairs(raw_geometry, field="geometry")
        if len(coordinates) == 1:
            return Point(coordinates[0]), "osm_geometry"
        if len(coordinates) >= 4 and coordinates[0] == coordinates[-1]:
            polygon = Polygon(coordinates)
            return _validate_geometry(polygon, field="geometry"), "osm_geometry"
        if len(coordinates) >= 2:
            return LineString(coordinates), "osm_geometry"
    for latitude_key, longitude_key in (("latitude", "longitude"), ("lat", "lon")):
        if latitude_key in item and longitude_key in item:
            point = _point(item[latitude_key], item[longitude_key], field="element")
            return Point(point.longitude, point.latitude), "coordinate"
    center = item.get("center")
    if isinstance(center, dict) and "lat" in center and "lon" in center:
        point = _point(center["lat"], center["lon"], field="element.center")
        return Point(point.longitude, point.latitude), "center"
    bbox_raw = item.get("bbox")
    if bbox_raw is not None:
        west, south, east, north = _bbox(bbox_raw, field="element.bbox")
        return box(west, south, east, north), "bbox"
    return None, "missing"


def filter_items_spatially(
    items: list[dict[str, Any]],
    *,
    relation: str,
    reference: BaseGeometry,
    distance_m: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_relation = relation.strip().lower()
    if normalized_relation not in {"within", "intersects", "contains", "near", "crosses"}:
        raise GeoRuntimeError(f"不支持 relation={relation!r}", "unsupported_relation")
    threshold: float | None = None
    if normalized_relation == "near":
        threshold = _nonnegative(distance_m, field="distance_m")
        if threshold is None:
            raise GeoRuntimeError("relation=near 必须提供 distance_m", "missing_input")
        if threshold > _MAX_LOCAL_DISTANCE_M:
            raise GeoRuntimeError(
                f"distance_m 超过局部 near 上限 {_MAX_LOCAL_DISTANCE_M:g} 米",
                "distance_too_large",
            )

    matched: list[dict[str, Any]] = []
    missing_geometry = 0
    metric_projection: str | None = None
    metric_transformer: Transformer | None = None
    reference_metric: BaseGeometry | None = None
    if normalized_relation == "near":
        metric_transformer, metric_projection = _local_metric_transformer(reference)
        reference_metric = transform(metric_transformer.transform, reference)
    for item in items:
        candidate, geometry_source = geometry_from_item(item)
        if candidate is None:
            missing_geometry += 1
            continue
        distance_value: float | None = None
        if normalized_relation == "within":
            keep = candidate.within(reference)
        elif normalized_relation == "intersects":
            keep = candidate.intersects(reference)
        elif normalized_relation == "contains":
            keep = candidate.contains(reference)
        elif normalized_relation == "crosses":
            keep = candidate.crosses(reference)
        else:
            assert metric_transformer is not None
            assert reference_metric is not None
            assert threshold is not None
            candidate_metric = transform(metric_transformer.transform, candidate)
            distance_value = float(candidate_metric.distance(reference_metric))
            keep = distance_value <= float(threshold)
        if keep:
            copied = copy.deepcopy(item)
            copied["spatial_match"] = {
                "relation": normalized_relation,
                "geometry_source": geometry_source,
                "distance_m": round(distance_value, 6) if distance_value is not None else None,
            }
            matched.append(copied)
    stats = {
        "input_count": len(items),
        "matched_count": len(matched),
        "rejected_count": len(items) - len(matched),
        "missing_geometry_count": missing_geometry,
        "relation": normalized_relation,
        "distance_m": threshold,
        "metric_projection": metric_projection,
    }
    return matched, stats


def project_local_metric(
    first: BaseGeometry,
    second: BaseGeometry,
) -> tuple[BaseGeometry, BaseGeometry, str]:
    transformer, projection = _local_metric_transformer(
        second if not second.is_empty else first
    )
    return (
        transform(transformer.transform, first),
        transform(transformer.transform, second),
        projection,
    )


def _local_metric_transformer(reference: BaseGeometry) -> tuple[Transformer, str]:
    center = reference.centroid
    west, south, east, north = reference.bounds
    if east - west > _MAX_LOCAL_SPAN_DEGREES or north - south > _MAX_LOCAL_SPAN_DEGREES:
        raise GeoRuntimeError(
            "参照几何跨度超过 30°，不适合局部 AEQD 米制计算",
            "geometry_too_large",
        )
    longitude = float(center.x)
    latitude = float(center.y)
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise GeoRuntimeError("几何超出 WGS84 范围", "invalid_geometry")
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={latitude:.12g} +lon_0={longitude:.12g} "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(WGS84, local, always_xy=True)
    return transformer, local.to_string()


def geometry_mapping(geometry: BaseGeometry) -> dict[str, Any]:
    return mapping(geometry)


def _union_geometries(items: list[Any], *, field: str) -> BaseGeometry:
    geometries: list[BaseGeometry] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise GeoRuntimeError(f"{field}[{index}] 必须是对象", "invalid_geometry")
        geom, _ = geometry_from_item(item)
        if geom is not None:
            geometries.append(geom)
    if not geometries:
        raise GeoRuntimeError(f"{field} 不包含可用几何", "needs_acquisition")
    return _validate_geometry(unary_union(geometries), field=field)


def _validated_shape(raw: Any, *, field: str) -> BaseGeometry:
    if not isinstance(raw, dict):
        raise GeoRuntimeError(f"{field} 必须是 GeoJSON geometry", "invalid_geometry")
    try:
        return _validate_geometry(shape(raw), field=field)
    except (TypeError, ValueError, KeyError) as exc:
        raise GeoRuntimeError(f"{field} 不是有效 GeoJSON", "invalid_geometry") from exc


def _validate_geometry(geometry: BaseGeometry, *, field: str) -> BaseGeometry:
    if geometry.is_empty:
        raise GeoRuntimeError(f"{field} 是空几何", "invalid_geometry")
    if not geometry.is_valid:
        repaired = geometry.buffer(0)
        if repaired.is_empty or not repaired.is_valid:
            raise GeoRuntimeError(f"{field} 是无效几何", "invalid_geometry")
        geometry = repaired
    west, south, east, north = geometry.bounds
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise GeoRuntimeError(f"{field} 包含非有限坐标", "invalid_geometry")
    if west < -180 or east > 180 or south < -90 or north > 90:
        raise GeoRuntimeError(f"{field} 超出 WGS84 范围", "invalid_geometry")
    return geometry


def _point(
    latitude_raw: Any,
    longitude_raw: Any,
    *,
    field: str,
    label: str = "",
    derived_from: str = "coordinate",
) -> GeoPoint:
    try:
        latitude = float(latitude_raw)
        longitude = float(longitude_raw)
    except (TypeError, ValueError) as exc:
        raise GeoRuntimeError(f"{field} 坐标必须是数字", "invalid_coordinate") from exc
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        raise GeoRuntimeError(f"{field} 坐标必须是有限数字", "invalid_coordinate")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise GeoRuntimeError(f"{field} 超出 WGS84 范围", "invalid_coordinate")
    return GeoPoint(latitude, longitude, label=label, derived_from=derived_from)


def _bbox(raw: Any, *, field: str) -> tuple[float, float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise GeoRuntimeError(
            f"{field} bbox 必须是 [west,south,east,north]",
            "invalid_geometry",
        )
    try:
        west, south, east, north = (float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise GeoRuntimeError(f"{field} bbox 必须由数字组成", "invalid_geometry") from exc
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise GeoRuntimeError(f"{field} bbox 必须是有限数字", "invalid_geometry")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise GeoRuntimeError(f"{field} bbox 范围或顺序错误", "invalid_geometry")
    return west, south, east, north


def _coordinate_pairs(raw: list[Any], *, field: str) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for index, point in enumerate(raw):
        if isinstance(point, dict) and "lat" in point and "lon" in point:
            parsed = _point(point["lat"], point["lon"], field=f"{field}[{index}]")
            result.append((parsed.longitude, parsed.latitude))
            continue
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                longitude, latitude = float(point[0]), float(point[1])
            except (TypeError, ValueError) as exc:
                raise GeoRuntimeError(
                    f"{field}[{index}] 坐标必须是数字",
                    "invalid_geometry",
                ) from exc
            parsed = _point(latitude, longitude, field=f"{field}[{index}]")
            result.append((parsed.longitude, parsed.latitude))
            continue
        raise GeoRuntimeError(f"{field}[{index}] 不是坐标", "invalid_geometry")
    return result


def _nonnegative(raw: Any, *, field: str) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise GeoRuntimeError(f"{field} 必须是数字", "invalid_input") from exc
    if not math.isfinite(value) or value < 0:
        raise GeoRuntimeError(f"{field} 必须是非负有限数字", "invalid_input")
    return value


__all__ = [
    "GEOD",
    "WGS84",
    "GeoPoint",
    "GeoRuntimeError",
    "filter_items_spatially",
    "geometry_from_item",
    "geometry_from_reference",
    "geometry_mapping",
    "point_from_value",
    "points_from_value",
    "project_local_metric",
    "require_wgs84",
    "resolve_structured_source",
    "source_items",
    "source_label",
]
