"""terrain_analysis 本地 DEM 分析：Horn 坡度/坡向、起伏统计与测地剖面。"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from pyproj import Geod, Transformer

from tool._gate import allow_real_api as _allow_real_api, load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext, declared_inputs

_OP = "terrain"
_METHOD = "horn_dem"
_ELLIPSOID = "WGS84"
_CRS_WGS84 = "wgs84"
_GEOD = Geod(ellps=_ELLIPSOID)
_PREVIOUS = "$previous_tool_result"
_PROFILE_MAX = 50
_MAX_SAMPLES = 50_000
_COARSE_RES_M = 30.0
_MAX_BBOX_DEG = 2.0
_METERS_PER_DEG_LAT = 111_320.0
_FLAT_SLOPE_DEG = 1e-3
_DEFAULT_DEM_TYPE = "COP30"
_DEFAULT_OT_ENDPOINT = "https://portal.opentopography.org/API/globaldem"
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_UA = "geoagent-dataset/1.0 (terrain_analysis; local)"
_COORD_SPLIT_RE = re.compile(r"[,;，、\s]+")
_RASTER_SUFFIXES = frozenset({".tif", ".tiff", ".geotiff", ".img", ".asc", ".vrt"})
_TIFF_MAGICS = (b"II*\x00", b"MM\x00*")
_KIND_DEM = "dem"
_KIND_DSM = "dsm"
_KIND_UNKNOWN = "unknown"

_METRIC_ELEVATION = "elevation"
_METRIC_SLOPE = "slope"
_METRIC_ASPECT = "aspect"
_METRIC_PROFILE = "profile"
_METRIC_RELIEF = "relief"
_METRIC_ALIASES: dict[str, str] = {
    "elevation": _METRIC_ELEVATION,
    "elev": _METRIC_ELEVATION,
    "altitude": _METRIC_ELEVATION,
    "height": _METRIC_ELEVATION,
    "dem": _METRIC_ELEVATION,
    "高程": _METRIC_ELEVATION,
    "海拔": _METRIC_ELEVATION,
    "slope": _METRIC_SLOPE,
    "gradient": _METRIC_SLOPE,
    "坡度": _METRIC_SLOPE,
    "aspect": _METRIC_ASPECT,
    "坡向": _METRIC_ASPECT,
    "profile": _METRIC_PROFILE,
    "elevation_profile": _METRIC_PROFILE,
    "terrain_profile": _METRIC_PROFILE,
    "剖面": _METRIC_PROFILE,
    "高程剖面": _METRIC_PROFILE,
    "relief": _METRIC_RELIEF,
    "ruggedness": _METRIC_RELIEF,
    "tri": _METRIC_RELIEF,
    "起伏": _METRIC_RELIEF,
    "起伏度": _METRIC_RELIEF,
}

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
_ASSUMPTION_METHOD = "坡度与坡向按 Horn 1981 在 DEM 上计算，等同 GDAL gdaldem，未使用 QGIS Processing"
_ASSUMPTION_STATS = "统计值来自栅格像元，不是野外实测局部高差"
_ASSUMPTION_COORD = "无法唯一判定时，坐标串按 lon,lat 解析"
_ASSUMPTION_NO_GEOCODE = "本执行器不解析纯地名，需要 bbox 或中心点加半径"
_ASSUMPTION_LICENSE = "OpenTopography / Copernicus DEM 的使用受账户、配额与数据许可约束"
_ASSUMPTION_COARSE = "地形分辨率不足以支持不合理精细的局部高差结论，仅支持山坡/平原等定性判断"

class TerrainInputError(Exception):
    """area / path / metrics / terrain 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """地形栅格无法读取、越界或外部 DEM 服务不可用。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code

@dataclass(frozen=True)
class GeoPoint:
    """WGS84 点。"""

    lon: float
    lat: float
    crs: str = _CRS_WGS84

@dataclass(frozen=True)
class BBox:
    """轴对齐矩形，顺序 west, south, east, north。"""

    west: float
    south: float
    east: float
    north: float

@dataclass(frozen=True)
class ParsedArea:
    """解析后的查询范围。"""

    text: str | None = None
    center: GeoPoint | None = None
    radius_m: int | None = None
    bbox: BBox | None = None
    polygon: tuple[tuple[float, float], ...] | None = None
    applied: Any = None

@dataclass
class DemGrid:
    """内存 DEM；仿射为 GDAL/rasterio 的 (a,b,c,d,e,f)。"""

    values: np.ndarray
    transform: tuple[float, float, float, float, float, float]
    kind: str = _KIND_UNKNOWN
    crs: str = _CRS_WGS84
    nodata: float | None = None
    dataset: str = "local_grid"
    to_raster: Callable[[float, float], tuple[float, float]] | None = None

    def sample(self, lon: float, lat: float) -> float | None:
        """双线性采样高程；越界或 nodata 返回 None。"""

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
            return max(native * _METERS_PER_DEG_LAT * max(math.cos(math.radians(mid_lat)), 0.2), 1.0)
        return max(native, 1.0)

    def bbox(self) -> BBox:
        """像素角点围成的轴对齐范围。"""

        a, b, c, d, e, f = self.transform
        rows, cols = self.values.shape
        xs = [c, c + a * cols, c + b * rows, c + a * cols + b * rows]
        ys = [f, f + d * cols, f + e * rows, f + d * cols + e * rows]
        return BBox(west=min(xs), south=min(ys), east=max(xs), north=max(ys))

@dataclass(frozen=True)
class ProfilePoint:
    """剖面上的一个采样点。"""

    lon: float
    lat: float
    distance_m: float
    elevation_m: float
    slope_deg: float | None = None

@runtime_checkable
class DemProvider(Protocol):
    """按 bbox 拉取 DEM；测试用 extras['terrain_analysis_provider'] 替换。"""

    def fetch(self, bbox: BBox, *, demtype: str) -> DemGrid:
        """返回覆盖 bbox 的高程网格。"""

class OpenTopographyProvider:
    """OpenTopography GlobalDEM GeoTIFF 适配器。"""

    name = "opentopography"

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = _DEFAULT_OT_ENDPOINT,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
        user_agent: str = _DEFAULT_UA,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent

    def fetch(self, bbox: BBox, *, demtype: str) -> DemGrid:
        """下载 bbox 内的 GeoTIFF 并读入内存网格。"""

        query = urllib.parse.urlencode(
            {
                "demtype": demtype,
                "south": f"{bbox.south:.7f}",
                "north": f"{bbox.north:.7f}",
                "west": f"{bbox.west:.7f}",
                "east": f"{bbox.east:.7f}",
                "outputFormat": "GTiff",
                "API_Key": self._api_key,
            }
        )
        url = f"{self._endpoint}?{query}"
        request = urllib.request.Request(url, method="GET")
        request.add_header("User-Agent", self._user_agent)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_sec) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise EngineUnavailableError(
                f"OpenTopography HTTP {exc.code}: {detail[:200]}",
            ) from exc
        except urllib.error.URLError as exc:
            raise EngineUnavailableError(f"OpenTopography 网络失败: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise EngineUnavailableError(f"OpenTopography 调用失败: {exc}") from exc
        if not _looks_like_tiff(payload):
            preview = payload[:200].decode("utf-8", errors="replace")
            raise EngineUnavailableError(f"OpenTopography 未返回 GeoTIFF: {preview}")
        return _load_geotiff_bytes(payload, dataset=demtype, kind=_KIND_DEM)

def execute_terrain(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按 metrics 计算区域高程/坡度/坡向/起伏或路径剖面。"""

    del purpose
    try:
        return _run(inputs, ctx)
    except TerrainInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def _run(inputs: dict[str, Any], ctx: RuntimeContext | None) -> Observation:
    raw = _merge_extensions(dict(inputs))
    declared = declared_inputs(raw, "area", "path", "metrics")
    metrics = _parse_metrics(declared.get("metrics"))
    area = _parse_area(_resolve_area(declared.get("area"), ctx))
    path = _parse_path(declared.get("path"), ctx)
    if area is None and not path:
        raise TerrainInputError("缺少必填输入 area 或 path", "missing_input")
    if _METRIC_PROFILE in metrics and not path:
        raise TerrainInputError("metrics 含 profile 时必须提供 path", "missing_input")
    dem, dem_source = _resolve_dem(raw, ctx, area=area, path=path)
    clip = _analysis_bbox(area, path, dem)
    elev, slope, aspect, tri = _horn_fields(dem)
    mask = _clip_mask(dem, clip)
    valid = mask & np.isfinite(elev)
    if not np.any(valid):
        raise EngineUnavailableError("区域不在地形覆盖范围内或为 nodata")
    coarse = dem.resolution_m() >= _COARSE_RES_M
    statistics = _statistics(
        metrics,
        elev=elev,
        slope=slope,
        aspect=aspect,
        tri=tri,
        valid=valid,
        coarse=coarse,
    )
    profile: list[dict[str, Any]] | None = None
    if path and (_METRIC_PROFILE in metrics or not area):
        profile = _profile_payload(path, dem, slope=slope, coarse=coarse)
    coverage_bbox = clip if clip is not None else dem.bbox()
    result: dict[str, Any] = {
        "operation": _OP,
        "method": _METHOD,
        "metrics": metrics,
        "dataset": dem.dataset,
        "resolution_m": _quantize(dem.resolution_m(), coarse=True, digits=1),
        "crs": dem.crs,
        "coverage": {
            "bbox": [coverage_bbox.west, coverage_bbox.south, coverage_bbox.east, coverage_bbox.north],
            "kind": dem.kind,
        },
        "statistics": statistics,
        "ellipsoid": _ELLIPSOID,
        "applied": {
            "area": None if area is None else area.applied,
            "path": _path_applied(path),
            "dem_source": dem_source,
            "sample_count": int(np.count_nonzero(valid)),
        },
        "assumptions": _assumptions(dem=dem, dem_source=dem_source, coarse=coarse),
    }
    if profile is not None:
        result["profile"] = profile
    return _ok(result)

def _merge_extensions(raw: dict[str, Any]) -> dict[str, Any]:
    extensions = raw.get("extensions")
    if not isinstance(extensions, dict):
        return raw
    merged = dict(raw)
    for key in ("terrain", "dem"):
        if key not in merged and key in extensions:
            merged[key] = extensions[key]
    return merged

def _parse_metrics(raw: Any) -> list[str]:
    if raw is None or raw == "":
        raise TerrainInputError("缺少必填输入 metrics", "missing_input")
    items: list[str] = []
    if isinstance(raw, str):
        parts = [part for part in re.split(r"[,，、/;|]+", raw) if part.strip()]
        items.extend(parts if parts else [raw])
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if item is None or item == "":
                continue
            items.append(str(item))
    else:
        raise TerrainInputError("metrics 必须是字符串或字符串数组", "invalid_input")
    if not items:
        raise TerrainInputError("缺少必填输入 metrics", "missing_input")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.strip().lower()
        metric = _METRIC_ALIASES.get(key) or _METRIC_ALIASES.get(item.strip())
        if metric is None:
            raise TerrainInputError(f"未知 metrics: {item}", "invalid_input")
        if metric not in seen:
            seen.add(metric)
            normalized.append(metric)
    return normalized

def _resolve_area(raw: Any, ctx: RuntimeContext | None) -> Any:
    if raw is None or raw == "" or raw == "$active_area":
        return ctx.active_area if ctx is not None else None
    return raw

def _parse_area(raw: Any) -> ParsedArea | None:
    if raw is None or raw == "":
        return None
    if raw == _PREVIOUS:
        raise TerrainInputError("area 的 $previous_tool_result 缺少可解析范围", "missing_input")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        point = _parse_coordinate_text(text)
        if point is not None:
            return ParsedArea(center=point, applied={"lon": point.lon, "lat": point.lat})
        bbox = _bbox_from_text(text)
        if bbox is not None:
            return ParsedArea(bbox=bbox, applied=_bbox_applied(bbox))
        return ParsedArea(text=text, applied=text)
    if isinstance(raw, (list, tuple)):
        polygon = _polygon_from_points(raw)
        if polygon is not None:
            return ParsedArea(polygon=polygon, applied=_polygon_applied(polygon))
        if len(raw) == 4:
            bbox = _bbox_from_values(list(raw))
            if bbox is not None:
                return ParsedArea(bbox=bbox, applied=_bbox_applied(bbox))
        if len(raw) == 2:
            point = _point_from_pair(raw[0], raw[1])
            if point is not None:
                return ParsedArea(center=point, applied={"lon": point.lon, "lat": point.lat})
        raise TerrainInputError("area 无法解析为 bbox 或多边形", "invalid_input")
    if isinstance(raw, dict):
        polygon_raw = raw.get("polygon")
        if polygon_raw is not None:
            polygon = _polygon_from_points(polygon_raw)
            if polygon is not None:
                return ParsedArea(polygon=polygon, applied=_polygon_applied(polygon))
            raise TerrainInputError("area.polygon 无法解析", "invalid_input")
        bbox = _bbox_from_mapping(raw)
        if bbox is not None:
            return ParsedArea(bbox=bbox, applied=_bbox_applied(bbox))
        center = _center_from_mapping(raw)
        if center is not None:
            radius_raw = raw.get("radius_m", raw.get("radius"))
            radius: int | None = None
            if radius_raw not in (None, ""):
                parsed = _as_float(radius_raw)
                if parsed is None or parsed < 0:
                    raise TerrainInputError("radius_m 必须是非负数", "invalid_input")
                radius = int(round(parsed))
            applied: dict[str, Any] = {"lon": center.lon, "lat": center.lat}
            if radius is not None:
                applied["radius_m"] = radius
            return ParsedArea(center=center, radius_m=radius, applied=applied)
        text = _optional_str(raw.get("city", raw.get("text", raw.get("name", raw.get("area")))))
        if text:
            return ParsedArea(text=text, applied=text)
        raise TerrainInputError("area 无法解析为地理范围", "invalid_input")
    raise TerrainInputError("area 无法解析为地理范围", "invalid_input")

def _parse_path(raw: Any, ctx: RuntimeContext | None) -> list[GeoPoint]:
    if raw is None or raw == "":
        return []
    resolved = _resolve_ref(raw, ctx, field="path")
    if isinstance(resolved, str):
        resolved = _maybe_json(resolved.strip())
        if isinstance(resolved, str):
            points = _points_from_text(resolved)
            if points:
                return points
            if _looks_like_place_name(resolved):
                raise TerrainInputError("path 必须是真实坐标，不能用地名替代", "invalid_input")
            raise TerrainInputError("path 无法解析为剖面线", "invalid_input")
    if isinstance(resolved, dict):
        geom = resolved.get("geometry") if resolved.get("type") == "Feature" else resolved
        if isinstance(geom, dict):
            geom_type = geom.get("type")
            coords = geom.get("coordinates")
            if geom_type == "LineString" and isinstance(coords, list):
                return [_require_point(item, field="path") for item in coords]
            if geom_type == "MultiLineString" and isinstance(coords, list) and coords:
                first = coords[0]
                if isinstance(first, list):
                    return [_require_point(item, field="path") for item in first]
            if geom_type == "Point":
                return [_require_point(coords, field="path")]
        nested = resolved.get("path") or resolved.get("coordinates") or resolved.get("points")
        if nested is not None and nested is not resolved:
            return _parse_path(nested, ctx)
        point = _parse_point(resolved)
        if point is not None:
            return [point]
        raise TerrainInputError("path 无法解析为剖面线", "invalid_input")
    if isinstance(resolved, (list, tuple)):
        if resolved and all(isinstance(item, (int, float, str)) or item is None for item in resolved[:2]) and _is_coord_pair(resolved):
            return [_require_point(resolved, field="path")]
        points = [_require_point(item, field="path") for item in resolved]
        if len(points) < 1:
            raise TerrainInputError("path 至少需要一个坐标点", "invalid_input")
        return points
    if _looks_like_place_name(raw) or _looks_like_place_name(resolved):
        raise TerrainInputError("path 必须是真实坐标，不能用地名替代", "invalid_input")
    raise TerrainInputError("path 无法解析为剖面线", "invalid_input")

def _resolve_dem(
    raw: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    area: ParsedArea | None,
    path: list[GeoPoint],
) -> tuple[DemGrid, str]:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("terrain_analysis_dem")
    if injected is not None:
        return _grid_from_injected(injected), "injected"
    terrain_ref = raw.get("terrain", raw.get("dem"))
    if terrain_ref not in (None, ""):
        return _dem_from_ref(terrain_ref, ctx)
    previous = ctx.previous_tool_result if ctx is not None else None
    if previous not in (None, "") and _looks_like_grid_payload(previous):
        return _grid_from_mapping(_unwrap_previous(previous)), "grid"
    bbox = _bbox_from_area(area) or _bbox_from_path(path)
    if bbox is None:
        if area is not None and area.text:
            raise TerrainInputError(
                "需要 bbox 或中心点加半径；纯地名请先 geocode",
                "missing_input",
            )
        raise TerrainInputError("缺少可解析的 area 或 path 范围", "missing_input")
    _reject_large_bbox(bbox)
    provider = _resolve_provider(ctx)
    demtype = os.environ.get("TERRAIN_DEM_TYPE", _DEFAULT_DEM_TYPE).strip() or _DEFAULT_DEM_TYPE
    grid = provider.fetch(bbox, demtype=demtype)
    source = str(getattr(provider, "name", "provider") or "provider")
    if not grid.dataset or grid.dataset == "local_grid":
        grid.dataset = demtype
    return grid, source

def _dem_from_ref(raw: Any, ctx: RuntimeContext | None) -> tuple[DemGrid, str]:
    resolved = _resolve_ref(raw, ctx, field="terrain")
    if isinstance(resolved, str):
        resolved = _maybe_json(resolved)
        if isinstance(resolved, str):
            return _grid_from_path(resolved)
    if isinstance(resolved, DemGrid):
        return resolved, "injected"
    if isinstance(resolved, dict):
        if "result" in resolved and isinstance(resolved.get("result"), dict) and _looks_like_grid(resolved["result"]):
            return _grid_from_mapping(resolved["result"]), "grid"
        if _looks_like_grid(resolved):
            return _grid_from_mapping(resolved), "grid"
        path = resolved.get("path") or resolved.get("uri") or resolved.get("file")
        if isinstance(path, str) and path.strip():
            grid, source = _grid_from_path(path.strip())
            kind = _normalize_kind(resolved.get("kind") or resolved.get("type"))
            if kind != _KIND_UNKNOWN:
                grid.kind = kind
            return grid, source
    if _looks_like_place_name(raw) or _looks_like_place_name(resolved):
        raise TerrainInputError("terrain 必须是地形网格或本地栅格，不能用地名替代", "invalid_input")
    raise TerrainInputError("terrain 无法解析为地形网格或栅格路径", "invalid_input")

def _grid_from_injected(raw: Any) -> DemGrid:
    if isinstance(raw, DemGrid):
        return raw
    if isinstance(raw, dict):
        return _grid_from_mapping(raw)
    values = getattr(raw, "values", None)
    transform = getattr(raw, "transform", None)
    if values is not None and transform is not None:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2:
            raise TerrainInputError("terrain_analysis_dem.values 必须是二维网格", "invalid_input")
        affine = tuple(float(item) for item in transform[:6])
        return DemGrid(
            values=array,
            transform=(affine[0], affine[1], affine[2], affine[3], affine[4], affine[5]),
            kind=_normalize_kind(getattr(raw, "kind", _KIND_UNKNOWN)),
            crs=str(getattr(raw, "crs", _CRS_WGS84) or _CRS_WGS84),
            nodata=_as_float(getattr(raw, "nodata", None)),
            dataset=str(getattr(raw, "dataset", "injected") or "injected"),
            to_raster=getattr(raw, "to_raster", None),
        )
    raise TerrainInputError("terrain_analysis_dem 需要二维高程网格", "invalid_input")

def _grid_from_path(raw: str) -> tuple[DemGrid, str]:
    text = raw.strip()
    lower = text.lower()
    if lower.startswith(("http://", "https://")):
        raise TerrainInputError("不下载远程 DEM URL，请提供本地栅格、内存网格或由 OpenTopography 获取", "invalid_input")
    if not _looks_like_raster_path(text):
        raise TerrainInputError("terrain 必须是地形网格或本地栅格，不能用地名替代", "invalid_input")
    path = Path(text)
    if not path.is_file():
        raise EngineUnavailableError(f"地形栅格不存在: {text}")
    return _load_geotiff(path), "geotiff"

def _resolve_provider(ctx: RuntimeContext | None) -> DemProvider:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("terrain_analysis_provider")
    if injected is not None:
        if not isinstance(injected, DemProvider):
            raise EngineUnavailableError("terrain_analysis_provider 必须提供 fetch(bbox, demtype=...)")
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_API=false，禁止调用真实地形 API")
    api_key = (
        os.environ.get("OPENTOPOGRAPHY_API_KEY", "").strip()
        or os.environ.get("OPENTOPO_API_KEY", "").strip()
    )
    if not api_key:
        raise EngineUnavailableError("未配置 OPENTOPOGRAPHY_API_KEY")
    timeout_raw = os.environ.get("TERRAIN_OT_TIMEOUT_SEC", "").strip()
    timeout_sec = _parse_timeout(timeout_raw)
    endpoint = os.environ.get("TERRAIN_OT_ENDPOINT", "").strip() or _DEFAULT_OT_ENDPOINT
    user_agent = os.environ.get("TERRAIN_OT_USER_AGENT", "").strip() or _DEFAULT_UA
    return OpenTopographyProvider(
        api_key=api_key,
        endpoint=endpoint,
        timeout_sec=timeout_sec,
        user_agent=user_agent,
    )

def _analysis_bbox(area: ParsedArea | None, path: list[GeoPoint], dem: DemGrid) -> BBox | None:
    bbox = _bbox_from_area(area) or _bbox_from_path(path)
    if bbox is None:
        return None
    cover = dem.bbox()
    west = max(bbox.west, cover.west)
    south = max(bbox.south, cover.south)
    east = min(bbox.east, cover.east)
    north = min(bbox.north, cover.north)
    if west >= east or south >= north:
        raise EngineUnavailableError("区域不在地形覆盖范围内")
    return BBox(west=west, south=south, east=east, north=north)

def _horn_fields(dem: DemGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回与 DEM 同形的高程、坡度(度)、坡向(度)、TRI；无效为 nan。"""

    elev = np.asarray(dem.values, dtype=float)
    if dem.nodata is not None:
        elev = np.where(np.isclose(elev, dem.nodata, rtol=0.0, atol=1e-6), np.nan, elev)
    elev = np.where(np.isfinite(elev), elev, np.nan)
    rows, cols = elev.shape
    slope = np.full((rows, cols), np.nan, dtype=float)
    aspect = np.full((rows, cols), np.nan, dtype=float)
    tri = np.full((rows, cols), np.nan, dtype=float)
    if rows < 3 or cols < 3:
        return elev, slope, aspect, tri
    a = elev[:-2, :-2]
    b = elev[:-2, 1:-1]
    c = elev[:-2, 2:]
    d = elev[1:-1, :-2]
    f = elev[1:-1, 2:]
    g = elev[2:, :-2]
    h = elev[2:, 1:-1]
    i = elev[2:, 2:]
    xres, yres = _cellsize_meters(dem, rows=rows, cols=cols)
    xres_i = xres[1:-1, 1:-1] if xres.ndim == 2 else xres
    yres_i = yres[1:-1, 1:-1] if isinstance(yres, np.ndarray) and yres.ndim == 2 else yres
    dzdx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * xres_i)
    # 北为正：北行 a,b,c 减南行 g,h,i
    dzdy = ((a + 2.0 * b + c) - (g + 2.0 * h + i)) / (8.0 * yres_i)
    slope_rad = np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))
    slope_deg = np.degrees(slope_rad)
    aspect_deg = (np.degrees(np.arctan2(-dzdx, -dzdy)) + 360.0) % 360.0
    flat = slope_deg < _FLAT_SLOPE_DEG
    aspect_deg = np.where(flat, np.nan, aspect_deg)
    slope[1:-1, 1:-1] = slope_deg
    aspect[1:-1, 1:-1] = aspect_deg
    windows = np.lib.stride_tricks.sliding_window_view(elev, (3, 3))
    center = windows[:, :, 1, 1]
    neighbor_mask = np.ones((3, 3), dtype=bool)
    neighbor_mask[1, 1] = False
    neighbors = windows[:, :, neighbor_mask]
    diff = neighbors - center[:, :, None]
    tri_interior = np.sqrt(np.nansum(diff * diff, axis=-1))
    tri[1:-1, 1:-1] = np.where(np.isfinite(center), tri_interior, np.nan)
    return elev, slope, aspect, tri

def _cellsize_meters(dem: DemGrid, *, rows: int, cols: int) -> tuple[np.ndarray | float, np.ndarray | float]:
    a, b, _c, d, e, f = dem.transform
    px = math.hypot(a, d) if (a or d) else abs(a) or 1.0
    py = math.hypot(b, e) if (b or e) else abs(e) or 1.0
    if not _crs_is_geographic(dem.crs):
        return max(px, 1e-6), max(py, 1e-6)
    row_idx = np.arange(rows, dtype=float)
    col_mid = (cols - 1) / 2.0
    lat = f + e * (row_idx + 0.5) + d * (col_mid + 0.5)
    meters_lon = _METERS_PER_DEG_LAT * np.clip(np.cos(np.radians(lat)), 0.2, None)
    xres = np.broadcast_to((px * meters_lon)[:, None], (rows, cols)).astype(float)
    yres = np.full((rows, cols), max(py * _METERS_PER_DEG_LAT, 1e-6), dtype=float)
    return np.maximum(xres, 1e-6), yres

def _clip_mask(dem: DemGrid, clip: BBox | None) -> np.ndarray:
    rows, cols = dem.values.shape
    mask = np.ones((rows, cols), dtype=bool)
    if clip is None:
        return mask
    a, b, c, d, e, f = dem.transform
    row_idx = np.arange(rows, dtype=float)[:, None]
    col_idx = np.arange(cols, dtype=float)[None, :]
    xs = c + a * (col_idx + 0.5) + b * (row_idx + 0.5)
    ys = f + d * (col_idx + 0.5) + e * (row_idx + 0.5)
    if dem.to_raster is not None:
        # 覆盖范围已按栅格 CRS 的 bbox 裁切；像元中心仍按仿射判断。
        pass
    return (xs >= clip.west) & (xs <= clip.east) & (ys >= clip.south) & (ys <= clip.north)

def _statistics(
    metrics: list[str],
    *,
    elev: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    tri: np.ndarray,
    valid: np.ndarray,
    coarse: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    elev_vals = elev[valid]
    if _METRIC_ELEVATION in metrics:
        payload["elevation"] = {
            "min_m": _quantize(float(np.nanmin(elev_vals)), coarse=coarse),
            "max_m": _quantize(float(np.nanmax(elev_vals)), coarse=coarse),
            "mean_m": _quantize(float(np.nanmean(elev_vals)), coarse=coarse),
            "p10_m": _quantize(float(np.nanpercentile(elev_vals, 10)), coarse=coarse),
            "p90_m": _quantize(float(np.nanpercentile(elev_vals, 90)), coarse=coarse),
        }
    if _METRIC_SLOPE in metrics:
        slope_vals = slope[valid & np.isfinite(slope)]
        if slope_vals.size == 0:
            payload["slope"] = {"min_deg": None, "max_deg": None, "mean_deg": None}
        else:
            payload["slope"] = {
                "min_deg": _quantize(float(np.nanmin(slope_vals)), coarse=False, digits=1),
                "max_deg": _quantize(float(np.nanmax(slope_vals)), coarse=False, digits=1),
                "mean_deg": _quantize(float(np.nanmean(slope_vals)), coarse=False, digits=1),
            }
    if _METRIC_ASPECT in metrics:
        aspect_vals = aspect[valid & np.isfinite(aspect)]
        payload["aspect"] = {
            "mean_deg": _circular_mean_deg(aspect_vals),
            "flat_fraction": _quantize(
                float(np.count_nonzero(valid & ~np.isfinite(aspect)) / max(np.count_nonzero(valid), 1)),
                coarse=False,
                digits=3,
            ),
        }
    if _METRIC_RELIEF in metrics:
        relief_m = float(np.nanmax(elev_vals) - np.nanmin(elev_vals))
        slope_vals = slope[valid & np.isfinite(slope)]
        mean_slope = float(np.nanmean(slope_vals)) if slope_vals.size else None
        tri_vals = tri[valid & np.isfinite(tri)]
        payload["relief"] = {
            "relief_m": _quantize(relief_m, coarse=coarse),
            "relief_class": _relief_class(relief_m, mean_slope),
            "tri_m": _quantize(float(np.nanmean(tri_vals)), coarse=coarse) if tri_vals.size else None,
        }
    return payload

def _profile_payload(
    path: list[GeoPoint],
    dem: DemGrid,
    *,
    slope: np.ndarray,
    coarse: bool,
) -> list[dict[str, Any]]:
    samples = _sample_path(path, dem, slope=slope)
    if len(samples) > _PROFILE_MAX:
        last = len(samples) - 1
        chosen = {0, last}
        for index in range(1, _PROFILE_MAX - 1):
            chosen.add(round(index * last / (_PROFILE_MAX - 1)))
        samples = [samples[index] for index in sorted(chosen)]
    payload: list[dict[str, Any]] = []
    for item in samples:
        point: dict[str, Any] = {
            "lon": item.lon,
            "lat": item.lat,
            "distance_m": _quantize(item.distance_m, coarse=False, digits=1),
            "elevation_m": _quantize(item.elevation_m, coarse=coarse),
        }
        if item.slope_deg is not None:
            point["slope_deg"] = _quantize(item.slope_deg, coarse=False, digits=1)
        payload.append(point)
    return payload

def _sample_path(path: list[GeoPoint], dem: DemGrid, *, slope: np.ndarray) -> list[ProfilePoint]:
    if not path:
        return []
    if len(path) == 1:
        point = path[0]
        elev = dem.sample(point.lon, point.lat)
        if elev is None:
            raise EngineUnavailableError("path 不在地形覆盖范围内或为 nodata")
        return [
            ProfilePoint(
                lon=point.lon,
                lat=point.lat,
                distance_m=0.0,
                elevation_m=elev,
                slope_deg=_sample_grid(slope, dem, point.lon, point.lat),
            )
        ]
    step_m = max(0.5 * dem.resolution_m(), 1.0)
    samples: list[ProfilePoint] = []
    offset = 0.0
    for start, end in zip(path, path[1:], strict=False):
        _az, _back, dist_m = _GEOD.inv(start.lon, start.lat, end.lon, end.lat)
        dist_m = float(dist_m)
        if dist_m <= 1e-6:
            coords = [(start.lon, start.lat)]
        else:
            n_seg = max(2, int(math.ceil(dist_m / step_m)))
            n_pts = min(n_seg + 1, _MAX_SAMPLES)
            intermediates = max(n_pts - 2, 0)
            coords = [(start.lon, start.lat)]
            if intermediates > 0:
                mids = _GEOD.npts(start.lon, start.lat, end.lon, end.lat, intermediates)
                coords.extend((float(lon), float(lat)) for lon, lat in mids)
            coords.append((end.lon, end.lat))
        for index, (lon, lat) in enumerate(coords):
            if samples and index == 0:
                continue
            if index == 0:
                d_seg = 0.0
            elif index == len(coords) - 1:
                d_seg = dist_m
            else:
                _fwd, _rev, d_seg = _GEOD.inv(start.lon, start.lat, lon, lat)
                d_seg = float(d_seg)
            elev = dem.sample(lon, lat)
            if elev is None:
                raise EngineUnavailableError("path 不在地形覆盖范围内或为 nodata")
            samples.append(
                ProfilePoint(
                    lon=lon,
                    lat=lat,
                    distance_m=offset + d_seg,
                    elevation_m=elev,
                    slope_deg=_sample_grid(slope, dem, lon, lat),
                )
            )
        offset += dist_m
    return samples

def _sample_grid(grid: np.ndarray, dem: DemGrid, lon: float, lat: float) -> float | None:
    proxy = DemGrid(
        values=grid,
        transform=dem.transform,
        kind=dem.kind,
        crs=dem.crs,
        nodata=None,
        to_raster=dem.to_raster,
    )
    return proxy.sample(lon, lat)

def _relief_class(relief_m: float, mean_slope: float | None) -> str:
    if relief_m < 80.0 and (mean_slope is None or mean_slope < 5.0):
        return "plain"
    if relief_m < 600.0:
        return "hill"
    return "mountain"

def _circular_mean_deg(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    radians = np.radians(values)
    mean_sin = float(np.nanmean(np.sin(radians)))
    mean_cos = float(np.nanmean(np.cos(radians)))
    if not math.isfinite(mean_sin) or not math.isfinite(mean_cos):
        return None
    if abs(mean_sin) < 1e-12 and abs(mean_cos) < 1e-12:
        return None
    deg = math.degrees(math.atan2(mean_sin, mean_cos)) % 360.0
    return _quantize(deg, coarse=False, digits=1)

def _assumptions(*, dem: DemGrid, dem_source: str, coarse: bool) -> list[str]:
    items = [
        _ASSUMPTION_METHOD,
        f"数据集 {dem.dataset}，像元约 {dem.resolution_m():.0f} m",
        _ASSUMPTION_STATS,
        _ASSUMPTION_COORD,
        _ASSUMPTION_NO_GEOCODE,
    ]
    if coarse:
        items.append(_ASSUMPTION_COARSE)
    if dem_source in {"opentopography", "provider"} or dem.dataset.upper().startswith(("COP", "SRTM", "NASADEM", "AW3D")):
        items.append(_ASSUMPTION_LICENSE)
    if dem.kind == _KIND_DSM:
        items.append("DSM 可能含部分地表附着物，不能声称已覆盖全部建筑和树木")
    else:
        items.append("裸地 DEM 未计入建筑和树木")
    return items

def _bbox_from_area(area: ParsedArea | None) -> BBox | None:
    if area is None:
        return None
    if area.bbox is not None:
        return area.bbox
    if area.polygon is not None:
        lons = [point[0] for point in area.polygon]
        lats = [point[1] for point in area.polygon]
        return BBox(west=min(lons), south=min(lats), east=max(lons), north=max(lats))
    if area.center is not None and area.radius_m is not None:
        return _bbox_from_center(area.center, area.radius_m)
    return None

def _bbox_from_path(path: list[GeoPoint]) -> BBox | None:
    if not path:
        return None
    lons = [point.lon for point in path]
    lats = [point.lat for point in path]
    pad = 1e-4
    return BBox(
        west=max(-180.0, min(lons) - pad),
        south=max(-90.0, min(lats) - pad),
        east=min(180.0, max(lons) + pad),
        north=min(90.0, max(lats) + pad),
    )

def _bbox_from_center(center: GeoPoint, radius_m: int) -> BBox:
    lat_delta = radius_m / _METERS_PER_DEG_LAT
    cos_lat = math.cos(math.radians(center.lat))
    lon_delta = radius_m / (_METERS_PER_DEG_LAT * max(abs(cos_lat), 1e-6))
    return BBox(
        west=max(-180.0, center.lon - lon_delta),
        south=max(-90.0, center.lat - lat_delta),
        east=min(180.0, center.lon + lon_delta),
        north=min(90.0, center.lat + lat_delta),
    )

def _reject_large_bbox(bbox: BBox) -> None:
    raw = os.environ.get("TERRAIN_MAX_BBOX_DEG", "").strip()
    limit = _as_float(raw) if raw else _MAX_BBOX_DEG
    if limit is None or limit <= 0:
        limit = _MAX_BBOX_DEG
    if (bbox.east - bbox.west) > limit or (bbox.north - bbox.south) > limit:
        raise EngineUnavailableError(f"查询范围超过 {limit} 度，请缩小 area 或 path")

def _path_applied(path: list[GeoPoint]) -> list[dict[str, float]] | None:
    if not path:
        return None
    return [{"lon": item.lon, "lat": item.lat} for item in path]

def _looks_like_grid_payload(raw: Any) -> bool:
    payload = _unwrap_previous(raw)
    return isinstance(payload, dict) and _looks_like_grid(payload)

def _looks_like_grid(raw: dict[str, Any]) -> bool:
    values = raw.get("values", raw.get("grid", raw.get("elevations")))
    if values is None:
        return False
    return any(
        key in raw
        for key in ("transform", "origin", "pixel_size", "bounds", "west", "south", "east", "north", "bbox")
    )

def _grid_from_mapping(raw: dict[str, Any]) -> DemGrid:
    values_raw = raw.get("values", raw.get("grid", raw.get("elevations", raw.get("data"))))
    try:
        values = np.asarray(values_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TerrainInputError("terrain.values 必须是二维数值网格", "invalid_input") from exc
    if values.ndim != 2 or values.size == 0:
        raise TerrainInputError("terrain.values 必须是非空二维网格", "invalid_input")
    transform = _parse_transform(raw, rows=int(values.shape[0]), cols=int(values.shape[1]))
    kind = _normalize_kind(raw.get("kind") or raw.get("type") or raw.get("terrain_kind"))
    crs = _crs_of(raw) or _CRS_WGS84
    nodata = _as_float(raw.get("nodata"))
    dataset = _optional_str(raw.get("dataset", raw.get("demtype", "local_grid"))) or "local_grid"
    return DemGrid(values=values, transform=transform, kind=kind, crs=crs, nodata=nodata, dataset=dataset)

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
    raise TerrainInputError("terrain 缺少 transform / bounds / origin", "invalid_input")

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
        raise TerrainInputError("terrain 网格范围无效", "invalid_input")
    dx = (east - west) / cols
    dy = (south - north) / rows
    return (dx, 0.0, west, 0.0, dy, north)

def _load_geotiff(path: Path) -> DemGrid:
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:
        raise EngineUnavailableError("读取 GeoTIFF 需要 rasterio") from exc
    try:
        with rasterio.open(path) as src:
            values = np.asarray(src.read(1), dtype=float)
            affine: Affine = src.transform
            nodata = src.nodata
            src_crs = src.crs
    except Exception as exc:  # noqa: BLE001 — 栅格驱动错误统一成执行器不可用
        raise EngineUnavailableError(f"无法读取地形栅格: {path}") from exc
    return _dem_from_raster(values, affine, nodata=nodata, src_crs=src_crs, dataset=path.stem)

def _load_geotiff_bytes(payload: bytes, *, dataset: str, kind: str) -> DemGrid:
    try:
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.transform import Affine
    except ImportError as exc:
        raise EngineUnavailableError("读取 GeoTIFF 需要 rasterio") from exc
    try:
        with MemoryFile(payload) as memfile:
            with memfile.open() as src:
                values = np.asarray(src.read(1), dtype=float)
                affine: Affine = src.transform
                nodata = src.nodata
                src_crs = src.crs
    except Exception as exc:  # noqa: BLE001
        raise EngineUnavailableError("无法解析 OpenTopography GeoTIFF") from exc
    grid = _dem_from_raster(values, affine, nodata=nodata, src_crs=src_crs, dataset=dataset)
    grid.kind = kind
    return grid

def _dem_from_raster(
    values: np.ndarray,
    affine: Any,
    *,
    nodata: Any,
    src_crs: Any,
    dataset: str,
) -> DemGrid:
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
    return DemGrid(
        values=values,
        transform=(
            float(affine.a),
            float(affine.b),
            float(affine.c),
            float(affine.d),
            float(affine.e),
            float(affine.f),
        ),
        kind=_KIND_DEM,
        crs=crs_label.lower() if crs_label else _CRS_WGS84,
        nodata=float(nodata) if nodata is not None else None,
        dataset=dataset,
        to_raster=to_raster,
    )

def _world_to_pixel(
    transform: tuple[float, float, float, float, float, float],
    x: float,
    y: float,
) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    det = a * e - b * d
    if abs(det) < 1e-18:
        raise EngineUnavailableError("地形仿射变换不可逆")
    dx = x - c
    dy = y - f
    col = (e * dx - b * dy) / det
    row = (-d * dx + a * dy) / det
    return col, row

def _center_from_mapping(raw: dict[str, Any]) -> GeoPoint | None:
    lat = _as_float(raw.get("lat", raw.get("latitude")))
    lon = _as_float(raw.get("lon", raw.get("lng", raw.get("longitude"))))
    nested = raw.get("center")
    if isinstance(nested, dict) and (lat is None or lon is None):
        lat = _as_float(nested.get("lat", nested.get("latitude")))
        lon = _as_float(nested.get("lon", nested.get("lng", nested.get("longitude"))))
    if lat is None or lon is None:
        return None
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return GeoPoint(lon=lon, lat=lat)

def _bbox_from_mapping(raw: dict[str, Any]) -> BBox | None:
    bbox = raw.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return _bbox_from_values(list(bbox))
    west = _as_float(raw.get("west", raw.get("min_lon")))
    south = _as_float(raw.get("south", raw.get("min_lat")))
    east = _as_float(raw.get("east", raw.get("max_lon")))
    north = _as_float(raw.get("north", raw.get("max_lat")))
    if None in (west, south, east, north):
        return None
    return _validated_bbox(west, south, east, north)

def _bbox_from_values(values: list[Any]) -> BBox | None:
    nums = [_as_float(item) for item in values]
    if any(item is None for item in nums):
        return None
    west, south, east, north = nums  # type: ignore[misc]
    if abs(west) <= 90.0 and abs(east) <= 90.0 and abs(south) > 90.0:
        south, west, north, east = west, south, east, north
    return _validated_bbox(west, south, east, north)

def _bbox_from_text(raw: str) -> BBox | None:
    parts = [part for part in _COORD_SPLIT_RE.split(raw) if part]
    if len(parts) != 4:
        return None
    return _bbox_from_values(parts)

def _validated_bbox(west: float, south: float, east: float, north: float) -> BBox | None:
    if not -180.0 <= west <= 180.0 or not -180.0 <= east <= 180.0:
        return None
    if not -90.0 <= south <= 90.0 or not -90.0 <= north <= 90.0:
        return None
    if east == west or north == south:
        return None
    if west > east:
        west, east = east, west
    if south > north:
        south, north = north, south
    return BBox(west=west, south=south, east=east, north=north)

def _polygon_from_points(raw: Any) -> tuple[tuple[float, float], ...] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    if raw and isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 1 and isinstance(raw[0][0], (list, tuple)):
        raw = raw[0]
    points: list[tuple[float, float]] = []
    for item in raw:
        point = _parse_point(item)
        if point is None:
            return None
        points.append((point.lon, point.lat))
    if len(points) < 3:
        return None
    return tuple(points)

def _points_from_text(raw: str) -> list[GeoPoint]:
    text = raw.strip()
    if not text:
        return []
    if ";" in text or "|" in text:
        chunks = [part for part in re.split(r"[;|]+", text) if part.strip()]
        points = [_parse_coordinate_text(part) for part in chunks]
        if all(item is not None for item in points):
            return [item for item in points if item is not None]
    parts = [part for part in _COORD_SPLIT_RE.split(text) if part]
    if len(parts) >= 4 and len(parts) % 2 == 0:
        points: list[GeoPoint] = []
        for index in range(0, len(parts), 2):
            point = _point_from_pair(parts[index], parts[index + 1])
            if point is None:
                return []
            points.append(point)
        return points
    point = _parse_coordinate_text(text)
    return [point] if point is not None else []

def _require_point(raw: Any, *, field: str) -> GeoPoint:
    point = _parse_point(raw)
    if point is None:
        raise TerrainInputError(f"{field} 必须是真实坐标点", "invalid_input")
    return point

def _parse_point(raw: Any) -> GeoPoint | None:
    if isinstance(raw, GeoPoint):
        return raw
    if isinstance(raw, str):
        return _parse_coordinate_text(raw)
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return _point_from_pair(raw[0], raw[1])
    if isinstance(raw, dict):
        return _center_from_mapping(raw)
    return None

def _parse_coordinate_text(raw: str) -> GeoPoint | None:
    parts = [part for part in _COORD_SPLIT_RE.split(raw.strip()) if part]
    if len(parts) != 2:
        return None
    return _point_from_pair(parts[0], parts[1])

def _point_from_pair(first: Any, second: Any) -> GeoPoint | None:
    a = _as_float(first)
    b = _as_float(second)
    if a is None or b is None:
        return None
    if abs(a) > 90.0 and abs(b) <= 90.0:
        lon, lat = a, b
    elif abs(b) > 90.0 and abs(a) <= 90.0:
        lat, lon = a, b
    else:
        lon, lat = a, b
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return GeoPoint(lon=lon, lat=lat)

def _bbox_applied(bbox: BBox) -> dict[str, float]:
    return {"west": bbox.west, "south": bbox.south, "east": bbox.east, "north": bbox.north}

def _polygon_applied(polygon: tuple[tuple[float, float], ...]) -> list[list[float]]:
    return [[lon, lat] for lon, lat in polygon]

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

def _crs_of(raw: dict[str, Any]) -> str | None:
    value = raw.get("crs", raw.get("datum"))
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None

def _resolve_ref(raw: Any, ctx: RuntimeContext | None, *, field: str) -> Any:
    if raw != _PREVIOUS:
        return raw
    previous = ctx.previous_tool_result if ctx is not None else None
    if previous is None:
        raise TerrainInputError(f"无法解析 {field} 的 $previous_tool_result", "missing_input")
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

def _looks_like_raster_path(raw: str) -> bool:
    suffix = Path(raw).suffix.lower()
    if suffix in _RASTER_SUFFIXES:
        return True
    return ("/" in raw or "\\" in raw) and bool(suffix)

def _looks_like_tiff(payload: bytes) -> bool:
    return payload[:4] in _TIFF_MAGICS

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
    return _parse_coordinate_text(text) is None and _bbox_from_text(text) is None

def _is_coord_pair(raw: Any) -> bool:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return False
    return _as_float(raw[0]) is not None and _as_float(raw[1]) is not None

def _optional_str(raw: Any) -> str | None:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None

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

def _quantize(value: float, *, coarse: bool, digits: int = 1) -> float:
    if not math.isfinite(value):
        return value
    if coarse:
        return float(round(value))
    return float(round(value, digits))

def _parse_timeout(raw: str) -> float:
    parsed = _as_float(raw)
    if parsed is None or parsed <= 0:
        return _DEFAULT_TIMEOUT_SEC
    return parsed

def _maybe_json(raw: str) -> Any:
    text = raw.strip()
    if not text or text[0] not in "{[":
        return raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw

def _ok(result: dict[str, Any]) -> Observation:
    return Observation(ok=True, result=_strip_forbidden(result))

def _fail(message: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=message, error_code=error_code)

def _strip_forbidden(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_forbidden(item) for key, item in value.items() if key not in _FORBIDDEN_KEYS}
    if isinstance(value, list):
        return [_strip_forbidden(item) for item in value]
    return value
