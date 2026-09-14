"""WGS84 测地距离、几何长度/宽度与真北方位角本地计算。"""

from __future__ import annotations

import copy
import math
from itertools import pairwise
from typing import Any

from shapely.geometry import Polygon

from tool.contract import Observation, RuntimeContext
from tool.runtime.geo import (
    GEOD,
    GeoPoint,
    GeoRuntimeError,
    geometry_from_item,
    point_from_value,
    points_from_value,
    project_local_metric,
    require_wgs84,
    resolve_structured_source,
    source_items,
    source_label,
)
from tool.runtime.result_store import store_result

_UNIT_FACTORS = {"m": 1.0, "km": 1000.0, "mile": 1609.344}
_COMPASS_16 = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)


def execute_distance(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    del purpose
    try:
        points_raw = inputs.get("points")
        features_raw = inputs.get("features")
        if points_raw is not None and features_raw is not None:
            raise GeoRuntimeError("points 与 features 不能同时提供", "conflicting_inputs")
        if points_raw is None and features_raw is None:
            raise GeoRuntimeError("至少需要 points 或 features", "missing_input")
        unit = _unit(inputs.get("unit"))
        mode = _mode(inputs.get("mode"))
        if mode == "route":
            raise GeoRuntimeError(
                "route 距离需要 route_query 的真实路线 geometry/distance 结果；本工具不调用路线 API",
                "needs_acquisition",
            )
        if points_raw is not None:
            points = points_from_value(points_raw, ctx)
            result = _distance_points(points, unit=unit, mode=mode)
            result["source"] = "points"
        else:
            items, label = _feature_items(features_raw, ctx)
            if mode in {"geometry", "width"}:
                result = _distance_geometries(items, unit=unit, mode=mode)
            else:
                points = [point_from_value(item, ctx, field=f"features[{index}]") for index, item in enumerate(items)]
                result = _distance_points(points, unit=unit, mode="geodesic")
            result["source"] = label
    except GeoRuntimeError as exc:
        return _fail(str(exc), exc.error_code)

    result["operation"] = "distance"
    result["crs"] = "EPSG:4326"
    result["ellipsoid"] = "WGS84"
    result.setdefault("assumptions", []).extend(
        [
            "geodesic 使用 WGS84 椭球，不是道路、步行或驾车距离",
            "由面或线推导出的中心点/宽度会在记录中明确标注，不能冒充实测",
        ]
    )
    result_id = store_result(result, namespace="distance", ctx=ctx)
    if result_id:
        result["result_id"] = result_id
    return Observation(ok=True, result=result)


def execute_bearing(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    del purpose
    try:
        if "origin" not in inputs or "target" not in inputs:
            raise GeoRuntimeError("origin 和 target 均为必填", "missing_input")
        reference = _reference(inputs.get("reference"))
        if reference == "magnetic_north":
            raise GeoRuntimeError(
                "磁北方位需要日期、高度和磁偏角模型；请先获取这些参数",
                "needs_acquisition",
            )
        if reference == "image_axis":
            raise GeoRuntimeError(
                "image_axis 需要相机朝向或已标定图像轴，不能由两点坐标自动推断",
                "needs_acquisition",
            )
        origin = point_from_value(inputs["origin"], ctx, field="origin")
        target = point_from_value(inputs["target"], ctx, field="target")
        forward, backward, distance_m = GEOD.inv(
            origin.longitude,
            origin.latitude,
            target.longitude,
            target.latitude,
        )
        if distance_m <= 1e-9:
            raise GeoRuntimeError("origin 与 target 重合，方位角未定义", "coincident_points")
    except GeoRuntimeError as exc:
        return _fail(str(exc), exc.error_code)

    forward_normalized = _azimuth(forward)
    backward_normalized = _azimuth(backward)
    result: dict[str, Any] = {
        "operation": "bearing",
        "origin": origin.as_dict(),
        "target": target.as_dict(),
        "reference": "true_north",
        "forward_azimuth_deg": round(forward_normalized, 9),
        "back_azimuth_deg": round(backward_normalized, 9),
        "compass_direction": _compass(forward_normalized),
        "distance_m": round(float(distance_m), 6),
        "method": "pyproj.Geod.inv",
        "crs": "EPSG:4326",
        "ellipsoid": "WGS84",
        "assumptions": [
            "方位角以真北为 0°，顺时针增加",
            "返回的是起点处初始测地方位，不代表道路转向或图像朝向",
        ],
    }
    result_id = store_result(result, namespace="bearing", ctx=ctx)
    if result_id:
        result["result_id"] = result_id
    return Observation(ok=True, result=result)


def _distance_points(
    points: list[GeoPoint],
    *,
    unit: str,
    mode: str,
) -> dict[str, Any]:
    if len(points) < 2:
        raise GeoRuntimeError("距离计算至少需要两个点", "too_few_points")
    if mode not in {"geodesic", "geometry"}:
        raise GeoRuntimeError(f"points 不支持 mode={mode}", "unsupported_mode")
    pairs: list[dict[str, Any]] = []
    total_m = 0.0
    for index, (origin, target) in enumerate(pairwise(points)):
        forward, backward, distance_m = GEOD.inv(
            origin.longitude,
            origin.latitude,
            target.longitude,
            target.latitude,
        )
        total_m += float(distance_m)
        pairs.append(
            {
                "from_index": index,
                "to_index": index + 1,
                "origin": origin.as_dict(),
                "target": target.as_dict(),
                "distance_m": round(float(distance_m), 6),
                "distance": round(_convert(float(distance_m), unit), 9),
                "unit": unit,
                "forward_azimuth_deg": round(_azimuth(forward), 9),
                "back_azimuth_deg": round(_azimuth(backward), 9),
            }
        )
    return {
        "mode": "geodesic" if mode == "geodesic" else "geometry_vertices",
        "point_count": len(points),
        "pair_count": len(pairs),
        "pairs": pairs,
        "total_m": round(total_m, 6),
        "total": round(_convert(total_m, unit), 9),
        "unit": unit,
        "method": "pyproj.Geod.inv",
    }


def _distance_geometries(
    items: list[dict[str, Any]],
    *,
    unit: str,
    mode: str,
) -> dict[str, Any]:
    measurements: list[dict[str, Any]] = []
    total_m = 0.0
    for index, item in enumerate(items):
        geometry, geometry_source = geometry_from_item(item)
        if geometry is None:
            raise GeoRuntimeError(
                f"features[{index}] 缺少 geometry/coordinates/bbox",
                "needs_acquisition",
            )
        if mode == "geometry":
            value_m = abs(float(GEOD.geometry_length(geometry)))
            projection = None
            method = "pyproj.Geod.geometry_length"
        else:
            metric, _, projection = project_local_metric(geometry, geometry)
            if not isinstance(metric, Polygon):
                raise GeoRuntimeError(
                    f"features[{index}] 不是 Polygon，无法定义 width",
                    "unsupported_geometry",
                )
            rectangle = metric.minimum_rotated_rectangle
            coordinates = list(rectangle.exterior.coords)
            sides = [
                math.hypot(right[0] - left[0], right[1] - left[1])
                for left, right in pairwise(coordinates)
            ]
            positive = [value for value in sides if value > 1e-9]
            if not positive:
                raise GeoRuntimeError(
                    f"features[{index}] 无法得到非零宽度",
                    "invalid_geometry",
                )
            value_m = min(positive)
            method = "local_aeqd_minimum_rotated_rectangle"
        total_m += value_m
        measurements.append(
            {
                "index": index,
                "osm_type": item.get("osm_type"),
                "osm_id": item.get("osm_id"),
                "name": item.get("name"),
                "geometry_source": geometry_source,
                "value_m": round(value_m, 6),
                "value": round(_convert(value_m, unit), 9),
                "unit": unit,
                "projection": projection,
                "method": method,
            }
        )
    if not measurements:
        raise GeoRuntimeError("features 中没有可测量要素", "too_few_features")
    return {
        "mode": mode,
        "feature_count": len(measurements),
        "measurements": measurements,
        "total_m": round(total_m, 6),
        "total": round(_convert(total_m, unit), 9),
        "unit": unit,
        "method": measurements[0]["method"],
    }


def _feature_items(raw: Any, ctx: RuntimeContext | None) -> tuple[list[dict[str, Any]], str]:
    if isinstance(raw, list):
        if not raw or not all(isinstance(item, dict) for item in raw):
            raise GeoRuntimeError("features 数组必须包含至少一个对象", "invalid_source_result")
        return copy.deepcopy(raw), "inline_features"
    source = resolve_structured_source(raw, ctx, field="features")
    require_wgs84(source, field="features")
    items, _ = source_items(source)
    return items, source_label(raw)


def _unit(raw: Any) -> str:
    if raw is None:
        return "m"
    if not isinstance(raw, str) or raw.strip().lower() not in _UNIT_FACTORS:
        raise GeoRuntimeError("unit 必须是 m、km 或 mile", "invalid_unit")
    return raw.strip().lower()


def _mode(raw: Any) -> str:
    if raw is None:
        return "geodesic"
    if not isinstance(raw, str) or raw.strip().lower() not in {
        "geodesic",
        "route",
        "geometry",
        "width",
    }:
        raise GeoRuntimeError(
            "mode 必须是 geodesic、route、geometry 或 width",
            "invalid_mode",
        )
    return raw.strip().lower()


def _reference(raw: Any) -> str:
    if raw is None:
        return "true_north"
    if not isinstance(raw, str) or raw.strip().lower() not in {
        "true_north",
        "magnetic_north",
        "image_axis",
    }:
        raise GeoRuntimeError(
            "reference 必须是 true_north、magnetic_north 或 image_axis",
            "invalid_reference",
        )
    return raw.strip().lower()


def _convert(meters: float, unit: str) -> float:
    return meters / _UNIT_FACTORS[unit]


def _azimuth(value: float) -> float:
    return float(value) % 360.0


def _compass(azimuth: float) -> str:
    return _COMPASS_16[int((azimuth + 11.25) // 22.5) % 16]


def _fail(error: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=error, error_code=error_code)


__all__ = ["execute_bearing", "execute_distance"]
