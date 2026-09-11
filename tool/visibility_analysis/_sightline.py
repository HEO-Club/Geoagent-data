"""visibility_analysis 点到点视线：测地加密采样 DEM/DSM，不做 viewshed。"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from pyproj import Geod, Transformer

from tool.contract import Observation, RuntimeContext, declared_inputs

_OP = "sightline"
_ELLIPSOID = "WGS84"
_CRS_WGS84 = "wgs84"
_GEOD = Geod(ellps=_ELLIPSOID)
_PREVIOUS = "$previous_tool_result"
_EARTH_R_M = 6_371_000.0
_DEFAULT_K = 1.33
_EPS_M = 0.05
_PROFILE_MAX = 50
_MAX_SAMPLES = 50_000
_COORD_SPLIT_RE = re.compile(r"[,;，、\s]+")
_RASTER_SUFFIXES = frozenset({".tif", ".tiff", ".geotiff", ".img", ".asc", ".vrt"})
_KIND_DEM = "dem"
_KIND_DSM = "dsm"
_KIND_UNKNOWN = "unknown"
_METHOD = "geodesic_dem_sample"
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
_ASSUMPTION_METHOD = "点到点视线沿 WGS84 测地线加密后对地形栅格双线性采样，未计算区域可视域"
_ASSUMPTION_DEM = "裸地 DEM 未计入建筑和树木遮挡；需要这些因素时应使用合适的 DSM 或三维模型"
_ASSUMPTION_DSM = "DSM 可能含部分地表附着物，不能声称已覆盖全部建筑和树木"
_ASSUMPTION_COORD = "无法唯一判定时，坐标串按 lon,lat 解析"
_ASSUMPTION_NO_VIEWSHED = "本步只返回点到点视线，未生成 viewshed 栅格"
_ASSUMPTION_CURVATURE = "地球曲率按 4/3 大气折射（k=1.33）用中点凸起 d*(D-d)/(2 k R) 扣除，端点高程保持不变"
_ASSUMPTION_NO_CURVATURE = "未扣除地球曲率与大气折射"
_ASSUMPTION_NO_DOWNLOAD = "未下载或编造 DEM，只使用调用方提供的地形引用"


class SightlineInputError(Exception):
    """observer / target / through_points / terrain 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class EngineUnavailableError(Exception):
    """地形栅格无法读取或采样点超出覆盖范围。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class SightPoint:
    """视线端点；height_m 为离地，elevation_m 为绝对高程。"""

    lon: float
    lat: float
    height_m: float | None = None
    elevation_m: float | None = None
    crs: str = _CRS_WGS84


@dataclass(frozen=True)
class SamplePoint:
    """加密路径上的一个采样点。"""

    lon: float
    lat: float
    distance_m: float
    terrain_m: float
    los_m: float
    clearance_m: float


@runtime_checkable
class TerrainSampler(Protocol):
    """按 WGS84 经纬度采样高程。"""

    kind: str
    crs: str

    def sample(self, lon: float, lat: float) -> float | None:
        """返回该点高程（米）；超出覆盖或 nodata 时为 None。"""

    def resolution_m(self) -> float:
        """近似像元边长（米），用于加密步长。"""

    def bbox(self) -> tuple[float, float, float, float]:
        """覆盖范围 west, south, east, north（栅格坐标单位）。"""


@dataclass
class GridTerrainSampler:
    """内存规则网格；仿射为 GDAL/rasterio 的 (a,b,c,d,e,f)。"""

    values: np.ndarray
    transform: tuple[float, float, float, float, float, float]
    kind: str = _KIND_UNKNOWN
    crs: str = _CRS_WGS84
    nodata: float | None = None
    to_raster: Callable[[float, float], tuple[float, float]] | None = None

    def sample(self, lon: float, lat: float) -> float | None:
        """双线性采样；越界或 nodata 返回 None。"""

        x, y = (lon, lat) if self.to_raster is None else self.to_raster(lon, lat)
        col, row = _world_to_pixel(self.transform, x, y)
        rows, cols = self.values.shape
        if col < 0.0 or row < 0.0 or col > cols or row > rows:
            return None
        if cols == 1 or rows == 1:
            c_i = min(max(int(col), 0), cols - 1)
            r_i = min(max(int(row), 0), rows - 1)
            return _finite_elev(self.values[r_i, c_i], self.nodata)
        c_f = col - 0.5
        r_f = row - 0.5
        c0 = int(math.floor(c_f))
        r0 = int(math.floor(r_f))
        c1 = c0 + 1
        r1 = r0 + 1
        if c0 < 0 or r0 < 0 or c1 >= cols or r1 >= rows:
            c_i = min(max(int(math.floor(col)), 0), cols - 1)
            r_i = min(max(int(math.floor(row)), 0), rows - 1)
            return _finite_elev(self.values[r_i, c_i], self.nodata)
        dc = c_f - c0
        dr = r_f - r0
        v00 = _finite_elev(self.values[r0, c0], self.nodata)
        v10 = _finite_elev(self.values[r0, c1], self.nodata)
        v01 = _finite_elev(self.values[r1, c0], self.nodata)
        v11 = _finite_elev(self.values[r1, c1], self.nodata)
        if v00 is None or v10 is None or v01 is None or v11 is None:
            return None
        top = v00 * (1.0 - dc) + v10 * dc
        bottom = v01 * (1.0 - dc) + v11 * dc
        return top * (1.0 - dr) + bottom * dr

    def resolution_m(self) -> float:
        """由仿射与纬度估算像元米制边长。"""

        a, b, _c, d, e, f = self.transform
        px = math.hypot(a, d)
        py = math.hypot(b, e)
        native = (px + py) / 2.0 if (px + py) > 0 else 1.0
        if _crs_is_geographic(self.crs):
            mid_lat = f + e * (self.values.shape[0] / 2.0)
            return max(native * 111_320.0 * max(math.cos(math.radians(mid_lat)), 0.2), 1.0)
        return max(native, 1.0)

    def bbox(self) -> tuple[float, float, float, float]:
        """像素角点围成的轴对齐范围。"""

        a, b, c, d, e, f = self.transform
        rows, cols = self.values.shape
        xs = [c, c + a * cols, c + b * rows, c + a * cols + b * rows]
        ys = [f, f + d * cols, f + e * rows, f + d * cols + e * rows]
        return (min(xs), min(ys), max(xs), max(ys))


def execute_sightline(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """计算观察点到目标的点到点地形视线；区域可视域不在本执行器内。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "observer", "target", "through_points", "terrain")
        _reject_viewshed_request(inputs)
        observer = _require_point(inputs.get("observer"), ctx, field="observer")
        target = _require_point(inputs.get("target"), ctx, field="target")
        through = _parse_through_points(inputs.get("through_points"), ctx)
        sampler, terrain_source = _resolve_sampler(inputs.get("terrain"), ctx)
        curvature, curvature_k = _parse_curvature()
        waypoints = [observer, *through, target]
        segments: list[list[SamplePoint]] = []
        total_m = 0.0
        for start, end in zip(waypoints, waypoints[1:], strict=False):
            samples, dist_m = _sample_segment(
                start,
                end,
                sampler,
                curvature=curvature,
                curvature_k=curvature_k,
                distance_offset_m=total_m,
            )
            segments.append(samples)
            total_m += dist_m
        path = [item for seg in segments for item in seg]
        obstruction = _first_obstruction(segments)
        visible = obstruction is None
        profile = _downsample_profile(path)
        kind = _normalize_kind(sampler.kind)
        applied = {
            "observer": _point_applied(observer, sampler),
            "target": _point_applied(target, sampler),
            "through_points": [_point_applied(item, sampler) for item in through],
            "sample_step_m": 0.5 * sampler.resolution_m(),
            "sample_count": len(path),
            "curvature": curvature,
            "curvature_k": curvature_k if curvature else None,
            "terrain_source": terrain_source,
        }
        west, south, east, north = sampler.bbox()
        coverage = {
            "crs": sampler.crs,
            "bbox": [west, south, east, north],
            "kind": kind,
            "accounts_for_buildings": False,
            "accounts_for_trees": False,
        }
        return _ok(
            {
                "operation": _OP,
                "visible": visible,
                "method": _METHOD,
                "distance_m": total_m,
                "first_obstruction": obstruction,
                "profile": profile,
                "terrain_kind": kind,
                "coverage": coverage,
                "ellipsoid": _ELLIPSOID,
                "crs": _result_crs([observer.crs, target.crs, *[item.crs for item in through]]),
                "applied": applied,
                "assumptions": _assumptions(kind=kind, curvature=curvature),
            }
        )
    except SightlineInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)


def _sample_segment(
    start: SightPoint,
    end: SightPoint,
    sampler: TerrainSampler,
    *,
    curvature: bool,
    curvature_k: float,
    distance_offset_m: float,
) -> tuple[list[SamplePoint], float]:
    az12, _az21, dist_m = _GEOD.inv(start.lon, start.lat, end.lon, end.lat)
    dist_m = float(dist_m)
    z_start = _absolute_z(start, sampler, field="视线起点")
    z_end = _absolute_z(end, sampler, field="视线终点")
    if dist_m <= _EPS_M:
        sample = _make_sample(
            start.lon,
            start.lat,
            distance_m=distance_offset_m,
            terrain_m=_require_sample(sampler, start.lon, start.lat, field="observer"),
            los_m=z_start,
        )
        return [sample], 0.0
    step_m = max(0.5 * sampler.resolution_m(), 1.0)
    n_seg = max(2, int(math.ceil(dist_m / step_m)))
    n_pts = min(n_seg + 1, _MAX_SAMPLES)
    intermediates = n_pts - 2
    coords: list[tuple[float, float]] = [(start.lon, start.lat)]
    if intermediates > 0:
        mids = _GEOD.npts(start.lon, start.lat, end.lon, end.lat, intermediates)
        coords.extend((float(lon), float(lat)) for lon, lat in mids)
    coords.append((end.lon, end.lat))
    del az12
    samples: list[SamplePoint] = []
    for index, (lon, lat) in enumerate(coords):
        if index == 0:
            d_seg = 0.0
        elif index == len(coords) - 1:
            d_seg = dist_m
        else:
            _fwd, _back, d_seg = _GEOD.inv(start.lon, start.lat, lon, lat)
            d_seg = float(d_seg)
        terrain_z = _require_sample(sampler, lon, lat, field="视线点")
        frac = 0.0 if dist_m <= 0.0 else min(max(d_seg / dist_m, 0.0), 1.0)
        los = z_start + (z_end - z_start) * frac
        if curvature:
            los -= _bulge_m(d_seg, dist_m, curvature_k)
        samples.append(
            _make_sample(
                lon,
                lat,
                distance_m=distance_offset_m + d_seg,
                terrain_m=terrain_z,
                los_m=los,
            )
        )
    return samples, dist_m


def _first_obstruction(segments: list[list[SamplePoint]]) -> dict[str, float] | None:
    for samples in segments:
        interior = samples[1:-1] if len(samples) > 2 else []
        for item in interior:
            if item.terrain_m > item.los_m + _EPS_M:
                return {
                    "lon": item.lon,
                    "lat": item.lat,
                    "distance_m": item.distance_m,
                    "terrain_m": item.terrain_m,
                    "los_m": item.los_m,
                }
    return None


def _absolute_z(point: SightPoint, sampler: TerrainSampler, *, field: str) -> float:
    if point.elevation_m is not None:
        return point.elevation_m
    ground = _require_sample(sampler, point.lon, point.lat, field=field)
    return ground + (point.height_m or 0.0)


def _require_sample(sampler: TerrainSampler, lon: float, lat: float, *, field: str) -> float:
    value = sampler.sample(lon, lat)
    if value is None:
        raise EngineUnavailableError(
            f"{field} 不在地形覆盖范围内或为 nodata",
            "engine_unavailable",
        )
    return value


def _make_sample(
    lon: float,
    lat: float,
    *,
    distance_m: float,
    terrain_m: float,
    los_m: float,
) -> SamplePoint:
    return SamplePoint(
        lon=lon,
        lat=lat,
        distance_m=distance_m,
        terrain_m=terrain_m,
        los_m=los_m,
        clearance_m=los_m - terrain_m,
    )


def _bulge_m(distance_m: float, total_m: float, k: float) -> float:
    if total_m <= 0.0 or k <= 0.0:
        return 0.0
    return (distance_m * (total_m - distance_m)) / (2.0 * k * _EARTH_R_M)


def _downsample_profile(path: list[SamplePoint]) -> list[dict[str, float]]:
    if len(path) <= _PROFILE_MAX:
        return [_sample_payload(item) for item in path]
    last_index = len(path) - 1
    chosen = {0, last_index}
    for index in range(1, _PROFILE_MAX - 1):
        chosen.add(round(index * last_index / (_PROFILE_MAX - 1)))
    return [_sample_payload(path[index]) for index in sorted(chosen)]


def _sample_payload(item: SamplePoint) -> dict[str, float]:
    return {
        "lon": item.lon,
        "lat": item.lat,
        "distance_m": item.distance_m,
        "terrain_m": item.terrain_m,
        "los_m": item.los_m,
        "clearance_m": item.clearance_m,
    }


def _reject_viewshed_request(inputs: dict[str, Any]) -> None:
    for field in ("observer", "target"):
        raw = inputs.get(field)
        if _is_area_geometry(raw):
            raise SightlineInputError(
                f"{field} 是面或 bbox，区域可视域本阶段未实现",
                "unsupported_mode",
            )


def _is_area_geometry(raw: Any) -> bool:
    if _is_bbox(raw):
        return True
    if isinstance(raw, dict):
        geom_type = raw.get("type")
        if geom_type in {"Polygon", "MultiPolygon"}:
            return True
        nested = raw.get("geometry")
        if isinstance(nested, dict) and nested.get("type") in {"Polygon", "MultiPolygon"}:
            return True
    return False


def _require_point(raw: Any, ctx: RuntimeContext | None, *, field: str) -> SightPoint:
    if raw is None or raw == "":
        raise SightlineInputError(f"缺少必填输入 {field}", "missing_input")
    resolved = _resolve_ref(raw, ctx, field=field)
    if _is_area_geometry(resolved):
        raise SightlineInputError(
            f"{field} 是面或 bbox，区域可视域本阶段未实现",
            "unsupported_mode",
        )
    point = _parse_sight_point(resolved)
    if point is not None:
        return point
    if _looks_like_place_name(raw) or _looks_like_place_name(resolved):
        raise SightlineInputError(f"{field} 必须是真实坐标，不能用地名替代", "invalid_input")
    raise SightlineInputError(f"{field} 无法解析为坐标点", "invalid_input")


def _parse_through_points(raw: Any, ctx: RuntimeContext | None) -> list[SightPoint]:
    if raw is None or raw == "":
        return []
    resolved = _resolve_ref(raw, ctx, field="through_points")
    if isinstance(resolved, str):
        resolved = _maybe_json(resolved)
    point = _parse_sight_point(resolved)
    if point is not None:
        return [point]
    if isinstance(resolved, list):
        if _is_bbox(resolved):
            raise SightlineInputError(
                "through_points 不能是 bbox，区域可视域本阶段未实现",
                "unsupported_mode",
            )
        points: list[SightPoint] = []
        for item in resolved:
            parsed = _parse_sight_point(item)
            if parsed is None:
                raise SightlineInputError("through_points 必须是真实坐标点", "invalid_input")
            points.append(parsed)
        return points
    if isinstance(resolved, dict):
        geom_type = resolved.get("type")
        if geom_type == "MultiPoint":
            coords = resolved.get("coordinates")
            if isinstance(coords, list):
                return [_require_parsed_point(item, field="through_points") for item in coords]
        if geom_type == "LineString":
            coords = resolved.get("coordinates")
            if isinstance(coords, list):
                return [_require_parsed_point(item, field="through_points") for item in coords]
        if geom_type in {"Polygon", "MultiPolygon"}:
            raise SightlineInputError(
                "through_points 不能是面几何，区域可视域本阶段未实现",
                "unsupported_mode",
            )
        nested = resolved.get("points") or resolved.get("coordinates")
        if isinstance(nested, list) and nested and not _is_coord_pair(nested):
            return [_require_parsed_point(item, field="through_points") for item in nested]
    if _looks_like_place_name(raw) or _looks_like_place_name(resolved):
        raise SightlineInputError("through_points 必须是真实坐标，不能用地名替代", "invalid_input")
    raise SightlineInputError("through_points 无法解析为坐标点", "invalid_input")


def _require_parsed_point(raw: Any, *, field: str) -> SightPoint:
    point = _parse_sight_point(raw)
    if point is None:
        raise SightlineInputError(f"{field} 必须是真实坐标点", "invalid_input")
    return point


def _parse_sight_point(raw: Any) -> SightPoint | None:
    if isinstance(raw, SightPoint):
        return raw
    if isinstance(raw, str):
        raw = _maybe_json(raw.strip())
        if isinstance(raw, str):
            return _parse_coordinate_query(raw)
    if isinstance(raw, (list, tuple)):
        if len(raw) < 2 or any(isinstance(item, (list, dict)) for item in raw[:2]):
            return None
        first = _as_float(raw[0])
        second = _as_float(raw[1])
        if first is None or second is None:
            return None
        height = _as_float(raw[2]) if len(raw) >= 3 else None
        pair = _point_from_pair(first, second)
        if pair is None:
            return None
        return SightPoint(lon=pair.lon, lat=pair.lat, height_m=height, crs=pair.crs)
    if not isinstance(raw, dict):
        return None
    crs = _crs_of(raw) or _CRS_WGS84
    height = _first_float(raw, ("height_m", "observer_height_m", "offset_m", "agl_m"))
    elevation = _first_float(raw, ("elevation_m", "altitude_m", "elev_m", "z_m", "z"))
    lat = _as_float(raw.get("lat", raw.get("latitude")))
    lon = _as_float(raw.get("lon", raw.get("lng", raw.get("longitude"))))
    nested = raw.get("location")
    if isinstance(nested, dict) and (lat is None or lon is None):
        nested_point = _parse_sight_point(nested)
        if nested_point is not None:
            return SightPoint(
                lon=nested_point.lon,
                lat=nested_point.lat,
                height_m=height if height is not None else nested_point.height_m,
                elevation_m=elevation if elevation is not None else nested_point.elevation_m,
                crs=_crs_of(nested) or nested_point.crs,
            )
    elif isinstance(nested, (list, tuple, str)) and (lat is None or lon is None):
        nested_point = _parse_sight_point(nested)
        if nested_point is not None:
            return SightPoint(
                lon=nested_point.lon,
                lat=nested_point.lat,
                height_m=height if height is not None else nested_point.height_m,
                elevation_m=elevation if elevation is not None else nested_point.elevation_m,
                crs=crs,
            )
    geom = raw.get("geometry")
    if isinstance(geom, dict) and geom.get("type") == "Point" and (lat is None or lon is None):
        nested_point = _parse_sight_point(geom.get("coordinates"))
        if nested_point is not None:
            return SightPoint(
                lon=nested_point.lon,
                lat=nested_point.lat,
                height_m=height,
                elevation_m=elevation,
                crs=_crs_of(geom) or crs,
            )
    if lat is None or lon is None:
        coords = raw.get("coordinates")
        if isinstance(coords, (list, tuple)):
            nested_point = _parse_sight_point(coords)
            if nested_point is not None:
                return SightPoint(
                    lon=nested_point.lon,
                    lat=nested_point.lat,
                    height_m=height if height is not None else nested_point.height_m,
                    elevation_m=elevation,
                    crs=crs,
                )
        return None
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return SightPoint(lon=lon, lat=lat, height_m=height, elevation_m=elevation, crs=crs)


def _parse_coordinate_query(raw: str) -> SightPoint | None:
    text = raw.strip()
    if not text or text == _PREVIOUS:
        return None
    parts = [part for part in _COORD_SPLIT_RE.split(text) if part]
    if len(parts) not in {2, 3}:
        return None
    first = _as_float(parts[0])
    second = _as_float(parts[1])
    if first is None or second is None:
        return None
    height = _as_float(parts[2]) if len(parts) == 3 else None
    pair = _point_from_pair(first, second)
    if pair is None:
        return None
    return SightPoint(lon=pair.lon, lat=pair.lat, height_m=height, crs=pair.crs)


def _point_from_pair(first: float, second: float, *, crs: str = _CRS_WGS84) -> SightPoint | None:
    if abs(first) > 90.0 and abs(second) <= 90.0:
        lon, lat = first, second
    elif abs(second) > 90.0 and abs(first) <= 90.0:
        lat, lon = first, second
    else:
        lon, lat = first, second
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return SightPoint(lon=lon, lat=lat, crs=crs)


def _resolve_sampler(
    raw: Any,
    ctx: RuntimeContext | None,
) -> tuple[TerrainSampler, str]:
    injected = None if ctx is None else ctx.extras.get("visibility_terrain")
    if injected is not None:
        if not isinstance(injected, TerrainSampler):
            raise SightlineInputError("visibility_terrain 注入对象必须实现 TerrainSampler", "invalid_input")
        return injected, "injected"
    if raw is None or raw == "":
        raise SightlineInputError("缺少必填输入 terrain", "missing_input")
    resolved = _resolve_ref(raw, ctx, field="terrain")
    if isinstance(resolved, str):
        resolved = _maybe_json(resolved)
        if isinstance(resolved, str):
            return _sampler_from_path(resolved)
    if isinstance(resolved, TerrainSampler):
        return resolved, "injected"
    if isinstance(resolved, dict):
        if "result" in resolved and isinstance(resolved.get("result"), dict):
            inner = resolved["result"]
            if _looks_like_grid(inner):
                return _grid_from_mapping(inner), "grid"
        if _looks_like_grid(resolved):
            return _grid_from_mapping(resolved), "grid"
        path = resolved.get("path") or resolved.get("uri") or resolved.get("file")
        if isinstance(path, str) and path.strip():
            kind = _normalize_kind(resolved.get("kind") or resolved.get("type"))
            sampler, _source = _sampler_from_path(path.strip(), kind=kind)
            return sampler, "geotiff"
    if _looks_like_place_name(raw) or _looks_like_place_name(resolved):
        raise SightlineInputError("terrain 必须是地形网格或本地栅格，不能用地名替代", "invalid_input")
    raise SightlineInputError("terrain 无法解析为地形网格或栅格路径", "invalid_input")


def _sampler_from_path(raw: str, *, kind: str | None = None) -> tuple[TerrainSampler, str]:
    text = raw.strip()
    lower = text.lower()
    if lower.startswith(("http://", "https://")):
        raise SightlineInputError("不下载远程 DEM，请提供本地栅格或内存网格", "invalid_input")
    if not _looks_like_raster_path(text):
        raise SightlineInputError("terrain 必须是地形网格或本地栅格，不能用地名替代", "invalid_input")
    path = Path(text)
    if not path.is_file():
        raise EngineUnavailableError(f"地形栅格不存在: {text}", "engine_unavailable")
    return _load_geotiff(path, kind=kind or _KIND_UNKNOWN), "geotiff"


def _looks_like_raster_path(raw: str) -> bool:
    suffix = Path(raw).suffix.lower()
    if suffix in _RASTER_SUFFIXES:
        return True
    return ("/" in raw or "\\" in raw) and bool(suffix)


def _load_geotiff(path: Path, *, kind: str) -> GridTerrainSampler:
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:
        raise EngineUnavailableError("读取 GeoTIFF 需要 rasterio", "engine_unavailable") from exc
    try:
        with rasterio.open(path) as src:
            values = np.asarray(src.read(1), dtype=float)
            affine: Affine = src.transform
            nodata = src.nodata
            src_crs = src.crs
    except Exception as exc:  # noqa: BLE001 — 栅格驱动错误统一成执行器不可用
        raise EngineUnavailableError(f"无法读取地形栅格: {path}", "engine_unavailable") from exc
    to_raster: Callable[[float, float], tuple[float, float]] | None = None
    crs_label = _CRS_WGS84
    if src_crs is not None and not src_crs.is_geographic:
        transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)

        def _project(lon: float, lat: float) -> tuple[float, float]:
            east, north = transformer.transform(lon, lat)
            return float(east), float(north)

        to_raster = _project
        crs_label = str(src_crs)
    elif src_crs is not None:
        crs_label = str(src_crs)
    return GridTerrainSampler(
        values=values,
        transform=(float(affine.a), float(affine.b), float(affine.c), float(affine.d), float(affine.e), float(affine.f)),
        kind=_normalize_kind(kind),
        crs=crs_label.lower() if crs_label else _CRS_WGS84,
        nodata=float(nodata) if nodata is not None else None,
        to_raster=to_raster,
    )


def _looks_like_grid(raw: dict[str, Any]) -> bool:
    values = raw.get("values", raw.get("grid", raw.get("elevations")))
    if values is None:
        return False
    return any(
        key in raw
        for key in ("transform", "origin", "pixel_size", "bounds", "west", "south", "east", "north", "bbox")
    )


def _grid_from_mapping(raw: dict[str, Any]) -> GridTerrainSampler:
    values_raw = raw.get("values", raw.get("grid", raw.get("elevations", raw.get("data"))))
    try:
        values = np.asarray(values_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SightlineInputError("terrain.values 必须是二维数值网格", "invalid_input") from exc
    if values.ndim != 2 or values.size == 0:
        raise SightlineInputError("terrain.values 必须是非空二维网格", "invalid_input")
    transform = _parse_transform(raw, rows=int(values.shape[0]), cols=int(values.shape[1]))
    kind = _normalize_kind(raw.get("kind") or raw.get("type") or raw.get("terrain_kind"))
    crs = _crs_of(raw) or _CRS_WGS84
    nodata = _as_float(raw.get("nodata"))
    return GridTerrainSampler(values=values, transform=transform, kind=kind, crs=crs, nodata=nodata)


def _parse_transform(
    raw: dict[str, Any],
    *,
    rows: int,
    cols: int,
) -> tuple[float, float, float, float, float, float]:
    transform = raw.get("transform")
    if isinstance(transform, (list, tuple)) and len(transform) >= 6:
        nums = [_as_float(item) for item in transform[:6]]
        if all(item is not None for item in nums):
            return (nums[0], nums[1], nums[2], nums[3], nums[4], nums[5])  # type: ignore[return-value]
    bounds = raw.get("bounds") or raw.get("bbox")
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        west, south, east, north = (_as_float(item) for item in bounds)
        if None not in (west, south, east, north):
            return _transform_from_bounds(west, south, east, north, cols=cols, rows=rows)  # type: ignore[arg-type]
    west = _as_float(raw.get("west"))
    south = _as_float(raw.get("south"))
    east = _as_float(raw.get("east"))
    north = _as_float(raw.get("north"))
    if None not in (west, south, east, north):
        return _transform_from_bounds(west, south, east, north, cols=cols, rows=rows)  # type: ignore[arg-type]
    origin = raw.get("origin")
    pixel = raw.get("pixel_size") or raw.get("resolution")
    if isinstance(origin, (list, tuple)) and isinstance(pixel, (list, tuple)) and len(origin) >= 2 and len(pixel) >= 2:
        ox = _as_float(origin[0])
        oy = _as_float(origin[1])
        dx = _as_float(pixel[0])
        dy = _as_float(pixel[1])
        if None not in (ox, oy, dx, dy):
            return (dx, 0.0, ox, 0.0, -abs(dy) if oy >= 0 else dy, oy)  # type: ignore[return-value]
    raise SightlineInputError("terrain 缺少 transform / bounds / origin", "invalid_input")


def _transform_from_bounds(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    cols: int,
    rows: int,
) -> tuple[float, float, float, float, float, float]:
    if cols <= 0 or rows <= 0 or east == west or north == south:
        raise SightlineInputError("terrain 网格范围无效", "invalid_input")
    dx = (east - west) / cols
    dy = (south - north) / rows
    return (dx, 0.0, west, 0.0, dy, north)


def _world_to_pixel(
    transform: tuple[float, float, float, float, float, float],
    x: float,
    y: float,
) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    det = a * e - b * d
    if abs(det) < 1e-18:
        raise EngineUnavailableError("地形仿射变换不可逆", "engine_unavailable")
    dx = x - c
    dy = y - f
    col = (e * dx - b * dy) / det
    row = (-d * dx + a * dy) / det
    return col, row


def _parse_curvature() -> tuple[bool, float]:
    raw = os.environ.get("VISIBILITY_CURVATURE", "true").strip().lower()
    enabled = raw not in {"0", "false", "no", "off"}
    k_raw = os.environ.get("VISIBILITY_CURVATURE_K", "").strip()
    k = _DEFAULT_K
    if k_raw:
        parsed = _as_float(k_raw)
        if parsed is not None and parsed > 0.0:
            k = parsed
    return enabled, k


def _normalize_kind(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return _KIND_UNKNOWN
    value = raw.strip().lower()
    if value in {_KIND_DEM, "dtm", "bare_earth", "裸地"}:
        return _KIND_DEM
    if value in {_KIND_DSM, "surface", "nsm"}:
        return _KIND_DSM
    return _KIND_UNKNOWN


def _crs_is_geographic(crs: str) -> bool:
    text = crs.strip().lower()
    return text in {_CRS_WGS84, "epsg:4326", "4326", "crs84"} or "4326" in text


def _assumptions(*, kind: str, curvature: bool) -> list[str]:
    items = [
        _ASSUMPTION_METHOD,
        _ASSUMPTION_DEM if kind != _KIND_DSM else _ASSUMPTION_DSM,
        _ASSUMPTION_NO_VIEWSHED,
        _ASSUMPTION_COORD,
        _ASSUMPTION_CURVATURE if curvature else _ASSUMPTION_NO_CURVATURE,
        _ASSUMPTION_NO_DOWNLOAD,
    ]
    return items


def _point_applied(point: SightPoint, sampler: TerrainSampler) -> dict[str, Any]:
    payload: dict[str, Any] = {"lon": point.lon, "lat": point.lat, "crs": point.crs}
    if point.height_m is not None:
        payload["height_m"] = point.height_m
    if point.elevation_m is not None:
        payload["elevation_m"] = point.elevation_m
    ground = sampler.sample(point.lon, point.lat)
    if ground is not None:
        payload["ground_m"] = ground
        payload["z_m"] = point.elevation_m if point.elevation_m is not None else ground + (point.height_m or 0.0)
    return payload


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


def _resolve_ref(raw: Any, ctx: RuntimeContext | None, *, field: str) -> Any:
    if raw != _PREVIOUS:
        return raw
    previous = ctx.previous_tool_result if ctx is not None else None
    if previous is None:
        raise SightlineInputError(f"无法解析 {field} 的 $previous_tool_result", "missing_input")
    return _unwrap_previous(previous)


def _unwrap_previous(raw: Any) -> Any:
    if isinstance(raw, dict) and isinstance(raw.get("result"), (dict, list)) and (
        "ok" in raw or "error_code" in raw or "artifacts" in raw
    ):
        return raw["result"]
    return raw


def _finite_elev(value: Any, nodata: float | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if nodata is not None and math.isclose(number, nodata, rel_tol=0.0, abs_tol=1e-6):
        return None
    return number


def _first_float(raw: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _as_float(raw.get(key))
        if value is not None:
            return value
    return None


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
    if _looks_like_raster_path(text) or text.lower().startswith(("http://", "https://")):
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
