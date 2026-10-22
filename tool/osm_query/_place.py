"""near 的分块与逐点筛选。只检查当前地点，不做全省组合。"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any

from tool.contract import RuntimeContext
from tool.osm_query._query import (
    BBox,
    EngineUnavailableError,
    OsmQueryInputError,
    OsmQueryRequest,
    QueryScope,
    _KIND_AREA_ID,
    _KIND_AROUND_POINT,
    _KIND_BBOX,
    _RELATION_AREA_OFFSET,
    _fmt_num,
    _map_feature_type,
    _tag_filters,
)

_TILE_EDGE_M = 25_000
_MAX_TILE_QUERIES = 32
_MIN_TILE_M = 3_000
_MAX_SPLIT_DEPTH = 6
_BEARING_TOLERANCE_DEG = 20.0
_RISE_M = 30.0
_SAMPLE_M = 800.0
_COARSE_ADMIN_LEVEL = 4
_M_PER_DEG_LAT = 110_574.0
_MOUNTAIN = frozenset({"山地", "山", "山区"})
_URBAN = frozenset({"城区", "市区", "建成区"})
_AXIS_DEG = {
    "北": 0.0,
    "南": 0.0,
    "东北": 45.0,
    "西南": 45.0,
    "东": 90.0,
    "西": 90.0,
    "东南": 135.0,
    "西北": 135.0,
}
_DIRECTIONS = ("北", "东北", "东", "东南", "南", "西南", "西", "西北")
_URBAN_FILTERS = ('["building"]', '["landuse"="residential"]')
_LANDUSE_NOTE = "缺少建成区面，不能据此判断没有城区"
_PARTIAL_NOTE = "还有未扫描范围，见 unscanned_bbox"
_DOWNLOAD_FALLBACK_CODES = frozenset(
    {"provider_error", "rate_limited", "network_error", "timeout"}
)


@dataclass(frozen=True)
class BearingRule:
    """一条线的走向约束。"""

    feature: str
    axis_deg: float


@dataclass(frozen=True)
class DirectionRule:
    """从参照物指向主体的八方位。"""

    subject: str
    reference: str
    direction: str


@dataclass(frozen=True)
class Layout:
    """当前地点要当场检查的走向与方位。"""

    bearings: tuple[BearingRule, ...] = ()
    directions: tuple[DirectionRule, ...] = ()
    extra_filters: dict[str, str] = field(default_factory=dict)
    want_urban: bool = False
    want_mountain: bool = False


@dataclass
class PlaceSearch:
    """一次 near 的通过地点与扫描回执。"""

    places: list[dict[str, Any]]
    partial: bool
    unscanned_bbox: list[float] | None
    tile_queries: int
    landuse_missing: int
    last_ql: str


def parse_layout(raw: Any, *, subject: str, reference: str, also: str) -> Layout:
    """解析可选 layout。山地和城区不是 OSM 类型。"""

    if raw is None or raw == "" or raw == []:
        return Layout()
    if not isinstance(raw, list):
        raise OsmQueryInputError("layout 必须是数组", "invalid_input")
    roles = {name for name in (subject, reference, also) if name}
    bearings: list[BearingRule] = []
    directions: list[DirectionRule] = []
    extra: dict[str, str] = {}
    want_urban = False
    want_mountain = False
    for item in raw:
        if not isinstance(item, dict):
            raise OsmQueryInputError("layout 的每一项必须是对象", "invalid_input")
        if item.get("bearing") not in (None, ""):
            feature = _required_name(item.get("feature"), field="layout.feature")
            if feature in _MOUNTAIN or feature in _URBAN:
                raise OsmQueryInputError(f"{feature} 没有线走向", "invalid_input")
            bearings.append(BearingRule(feature=feature, axis_deg=_axis_deg(item.get("bearing"))))
            _collect_feature(feature, roles=roles, extra=extra, urban=False, mountain=False)
        elif item.get("direction") not in (None, ""):
            origin = _required_name(item.get("reference"), field="layout.reference")
            target = _required_name(item.get("subject"), field="layout.subject")
            direction = _direction_name(item.get("direction"))
            if origin in _MOUNTAIN or origin in _URBAN:
                raise OsmQueryInputError(f"{origin} 不能作为方位参照", "invalid_input")
            if target in _MOUNTAIN:
                want_mountain = True
            elif target in _URBAN:
                want_urban = True
            else:
                _collect_feature(target, roles=roles, extra=extra, urban=False, mountain=False)
            _collect_feature(origin, roles=roles, extra=extra, urban=False, mountain=False)
            directions.append(DirectionRule(subject=target, reference=origin, direction=direction))
        else:
            raise OsmQueryInputError("layout 需要 bearing 或 direction", "invalid_input")
    return Layout(
        bearings=tuple(bearings),
        directions=tuple(directions),
        extra_filters=extra,
        want_urban=want_urban,
        want_mountain=want_mountain,
    )


_DEM_MAX_DEG = 2.0


def _resolve_dem_provider(ctx: RuntimeContext | None) -> Any:
    """复用 terrain_analysis 的高程提供者。注入优先，否则走 OpenTopography。"""

    from tool.terrain_analysis._terrain import (
        EngineUnavailableError as TerrainUnavailable,
    )
    from tool.terrain_analysis._terrain import (
        _resolve_provider as resolve_terrain_provider,
    )

    try:
        return resolve_terrain_provider(ctx)
    except TerrainUnavailable as exc:
        raise EngineUnavailableError(str(exc)) from exc


class _DemSession:
    """按范围取高程网格。范围不超过地形服务上限时一次取齐，否则按块取。"""

    def __init__(self, ctx: RuntimeContext | None) -> None:
        self._ctx = ctx
        self._provider: Any = None
        self._grids: list[tuple[BBox, Any]] = []

    def open(self) -> None:
        """立刻解析提供者。没有 DEM 时在查地物之前失败。"""

        if self._provider is None:
            self._provider = _resolve_dem_provider(self._ctx)

    def cover(self, bbox: BBox) -> None:
        """保证 bbox 内可以采样。过大的外接矩形留给各小块。"""

        padded = _pad_bbox(bbox, _SAMPLE_M)
        if (padded.east - padded.west) > _DEM_MAX_DEG or (padded.north - padded.south) > _DEM_MAX_DEG:
            return
        if any(_bbox_contains(ready, padded) for ready, _grid in self._grids):
            return
        self.open()
        self._grids.append((padded, _fetch_grid(self._provider, padded)))

    def sample(self, lon: float, lat: float) -> float | None:
        for bbox, grid in self._grids:
            if bbox.west <= lon <= bbox.east and bbox.south <= lat <= bbox.north:
                value = grid.sample(lon, lat)
                if value is None:
                    return None
                return float(value)
        return None


def _fetch_grid(provider: Any, bbox: BBox) -> Any:
    from tool.terrain_analysis._terrain import BBox as TerrainBBox
    from tool.terrain_analysis._terrain import EngineUnavailableError as TerrainUnavailable
    from tool.terrain_analysis._terrain import _DEFAULT_DEM_TYPE

    demtype = os.environ.get("TERRAIN_DEM_TYPE", _DEFAULT_DEM_TYPE).strip() or _DEFAULT_DEM_TYPE
    terrain_bbox = TerrainBBox(west=bbox.west, south=bbox.south, east=bbox.east, north=bbox.north)
    try:
        grid = provider.fetch(terrain_bbox, demtype=demtype)
    except EngineUnavailableError:
        raise
    except TerrainUnavailable as exc:
        raise EngineUnavailableError(str(exc)) from exc
    except Exception as exc:
        raise EngineUnavailableError(f"高程数据不可用: {exc}") from exc
    if grid is None or not hasattr(grid, "sample"):
        raise EngineUnavailableError("山地筛选需要高程数据，当前没有 DEM")
    return grid


def _pad_bbox(bbox: BBox, meters: float) -> BBox:
    mid = (bbox.south + bbox.north) / 2.0
    dlat = meters / _M_PER_DEG_LAT
    dlon = meters / _m_per_deg_lon(mid)
    return BBox(west=bbox.west - dlon, south=bbox.south - dlat, east=bbox.east + dlon, north=bbox.north + dlat)


def _bbox_contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer.west <= inner.west
        and outer.south <= inner.south
        and outer.east >= inner.east
        and outer.north >= inner.north
    )


def search_places(
    *,
    provider: Any,
    scope: QueryScope,
    timeout_sec: int,
    limit: int,
    subject_name: str,
    reference_name: str,
    also_name: str,
    subject_filter: str,
    reference_filter: str,
    also_filter: str,
    within_m: int,
    within_also_m: int,
    layout: Layout,
    ctx: RuntimeContext | None,
) -> PlaceSearch:
    """按块取齐线条，每形成一个地点就筛选，只保留通过的地点。"""

    bounds = _bounds(scope, provider, timeout_sec)
    dem: _DemSession | None = None
    if layout.want_mountain:
        if bounds is None:
            raise EngineUnavailableError("山地筛选需要高程数据，当前没有可采样的范围")
        dem = _DemSession(ctx)
        dem.open()
        dem.cover(bounds)
    level = _admin_level(scope)
    if bounds is None and level is not None and level <= _COARSE_ADMIN_LEVEL:
        raise OsmQueryInputError("范围过大且没有外接矩形，请改传 bbox", "missing_input")
    filters = [subject_filter, reference_filter, *layout.extra_filters.values()]
    if also_filter:
        filters.append(also_filter)
    if layout.want_urban:
        filters.extend(_URBAN_FILTERS)
    budget = _Budget()
    found: list[dict[str, Any]] = []
    missing = 0
    if bounds is None:
        prefix, spatial = _area_clause(scope)
        ql = _bundle_ql(filters, spatial, timeout_sec=timeout_sec, limit=limit, prefix=prefix)
        if budget.take():
            elements, ql = _load_elements(
                provider,
                ql,
                filters=filters,
                spatial=spatial,
                timeout_sec=timeout_sec,
                limit=limit,
                prefix=prefix,
            )
            budget.last_ql = ql
            if len(elements) >= limit:
                budget.partial = True
            else:
                places, absent = _places_from(
                    elements,
                    subject_name=subject_name,
                    reference_name=reference_name,
                    also_name=also_name,
                    subject_filter=subject_filter,
                    reference_filter=reference_filter,
                    also_filter=also_filter,
                    extra_filters=layout.extra_filters,
                    within_m=within_m,
                    within_also_m=within_also_m,
                    layout=layout,
                    dem=dem,
                )
                found.extend(places)
                missing += absent
        else:
            budget.partial = True
    else:
        tiles = _cover(bounds, within_m) if _must_tile(bounds, level) else [bounds]
        for tile in tiles:
            got, absent = _scan(
                tile,
                provider=provider,
                timeout_sec=timeout_sec,
                limit=limit,
                filters=filters,
                budget=budget,
                subject_name=subject_name,
                reference_name=reference_name,
                also_name=also_name,
                subject_filter=subject_filter,
                reference_filter=reference_filter,
                also_filter=also_filter,
                extra_filters=layout.extra_filters,
                within_m=within_m,
                within_also_m=within_also_m,
                layout=layout,
                dem=dem,
                depth=0,
            )
            found.extend(got)
            missing += absent
    places = _dedupe(found)
    for index, place in enumerate(places, start=1):
        place["result_id"] = f"place_{index}"
    return PlaceSearch(
        places=places,
        partial=budget.partial or bool(budget.unscanned),
        unscanned_bbox=_merge_bbox(budget.unscanned),
        tile_queries=budget.used,
        landuse_missing=missing,
        last_ql=budget.last_ql,
    )


def _collect_feature(
    name: str,
    *,
    roles: set[str],
    extra: dict[str, str],
    urban: bool,
    mountain: bool,
) -> None:
    del urban, mountain
    if name in roles or name in extra:
        return
    extra[name] = _tag_filters(_map_feature_type(name))


def _required_name(raw: Any, *, field: str) -> str:
    text = str(raw or "").strip()
    if not text:
        raise OsmQueryInputError(f"缺少 {field}", "missing_input")
    return text


def _axis_deg(raw: Any) -> float:
    text = str(raw or "").strip()
    head = text.split("-", 1)[0].strip()
    if head not in _AXIS_DEG:
        raise OsmQueryInputError(f"无法识别走向 {text!r}", "invalid_input")
    return _AXIS_DEG[head]


def _direction_name(raw: Any) -> str:
    text = str(raw or "").strip()
    if text not in _DIRECTIONS:
        raise OsmQueryInputError(f"无法识别方位 {text!r}", "invalid_input")
    return text


def _bounds(scope: QueryScope, provider: Any, timeout_sec: int) -> BBox | None:
    if scope.kind == _KIND_BBOX and scope.bbox is not None:
        return scope.bbox
    if scope.kind == _KIND_AROUND_POINT and scope.center is not None and scope.radius_m:
        return _box_around(scope.center.lat, scope.center.lon, scope.radius_m)
    if scope.kind == _KIND_AREA_ID and scope.area_id is not None and scope.area_id >= _RELATION_AREA_OFFSET:
        relation_id = scope.area_id - _RELATION_AREA_OFFSET
        ql = f"[out:json][timeout:{timeout_sec}];\nrel({relation_id});\nout bb;"
        payload = _query(provider, ql, timeout_sec)
        return _bounds_from_payload(payload)
    return None


def _bounds_from_payload(payload: dict[str, Any]) -> BBox | None:
    for item in _elements(payload):
        raw = item.get("bounds")
        if not isinstance(raw, dict):
            continue
        try:
            return BBox(
                west=float(raw["minlon"]),
                south=float(raw["minlat"]),
                east=float(raw["maxlon"]),
                north=float(raw["maxlat"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _box_around(lat: float, lon: float, radius_m: int) -> BBox:
    dlat = radius_m / _M_PER_DEG_LAT
    dlon = radius_m / _m_per_deg_lon(lat)
    return BBox(west=lon - dlon, south=lat - dlat, east=lon + dlon, north=lat + dlat)


def _admin_level(scope: QueryScope) -> int | None:
    applied = scope.applied
    if not isinstance(applied, dict):
        return None
    try:
        return int(applied.get("admin_level"))
    except (TypeError, ValueError):
        return None


def _must_tile(bounds: BBox, level: int | None) -> bool:
    del level
    return _span_m(bounds) > _TILE_EDGE_M


def _area_clause(scope: QueryScope) -> tuple[str, str]:
    if scope.kind == _KIND_AREA_ID and scope.area_id is not None:
        return f"area({scope.area_id})->.searchArea;", "(area.searchArea)"
    raise OsmQueryInputError("范围过大且没有外接矩形，请改传 bbox", "missing_input")


def _cover(bounds: BBox, within_m: int) -> list[BBox]:
    if _span_m(bounds) <= _TILE_EDGE_M:
        return [bounds]
    mid = (bounds.south + bounds.north) / 2.0
    height = (bounds.north - bounds.south) * _M_PER_DEG_LAT
    width = (bounds.east - bounds.west) * _m_per_deg_lon(mid)
    n_lat = max(1, math.ceil(height / _TILE_EDGE_M - 1e-9))
    n_lon = max(1, math.ceil(width / _TILE_EDGE_M - 1e-9))
    dlat = (bounds.north - bounds.south) / n_lat
    dlon = (bounds.east - bounds.west) / n_lon
    olat = within_m / _M_PER_DEG_LAT
    olon = within_m / _m_per_deg_lon(mid)
    tiles: list[BBox] = []
    for i in range(n_lat):
        for j in range(n_lon):
            south = bounds.south + i * dlat
            north = south + dlat
            west = bounds.west + j * dlon
            east = west + dlon
            if i:
                south -= olat
            if i + 1 < n_lat:
                north += olat
            if j:
                west -= olon
            if j + 1 < n_lon:
                east += olon
            tiles.append(BBox(west=west, south=south, east=east, north=north))
    return tiles


def _split4(bounds: BBox, within_m: int) -> list[BBox] | None:
    if _span_m(bounds) <= _MIN_TILE_M:
        return None
    mid_lat = (bounds.south + bounds.north) / 2.0
    mid_lon = (bounds.west + bounds.east) / 2.0
    olat = within_m / _M_PER_DEG_LAT
    olon = within_m / _m_per_deg_lon(mid_lat)
    boxes: list[BBox] = []
    for south, north in ((bounds.south, mid_lat), (mid_lat, bounds.north)):
        for west, east in ((bounds.west, mid_lon), (mid_lon, bounds.east)):
            boxes.append(
                BBox(
                    west=west - olon,
                    south=south - olat,
                    east=east + olon,
                    north=north + olat,
                )
            )
    return boxes


def _scan(
    tile: BBox,
    *,
    provider: Any,
    timeout_sec: int,
    limit: int,
    filters: list[str],
    budget: _Budget,
    subject_name: str,
    reference_name: str,
    also_name: str,
    subject_filter: str,
    reference_filter: str,
    also_filter: str,
    extra_filters: dict[str, str],
    within_m: int,
    within_also_m: int,
    layout: Layout,
    dem: Any | None,
    depth: int,
) -> tuple[list[dict[str, Any]], int]:
    if not budget.take():
        budget.unscanned.append(tile)
        budget.partial = True
        return [], 0
    spatial = _bbox_filter(tile)
    ql = _bundle_ql(filters, spatial, timeout_sec=timeout_sec, limit=limit)
    elements, ql = _load_elements(
        provider,
        ql,
        filters=filters,
        spatial=spatial,
        timeout_sec=timeout_sec,
        limit=limit,
    )
    budget.last_ql = ql
    if len(elements) >= limit:
        children = None if depth >= _MAX_SPLIT_DEPTH else _split4(tile, within_m)
        if not children:
            budget.unscanned.append(tile)
            budget.partial = True
            return [], 0
        found: list[dict[str, Any]] = []
        missing = 0
        for child in children:
            got, absent = _scan(
                child,
                provider=provider,
                timeout_sec=timeout_sec,
                limit=limit,
                filters=filters,
                budget=budget,
                subject_name=subject_name,
                reference_name=reference_name,
                also_name=also_name,
                subject_filter=subject_filter,
                reference_filter=reference_filter,
                also_filter=also_filter,
                extra_filters=extra_filters,
                within_m=within_m,
                within_also_m=within_also_m,
                layout=layout,
                dem=dem,
                depth=depth + 1,
            )
            found.extend(got)
            missing += absent
        return found, missing
    if dem is not None:
        dem.cover(tile)
    return _places_from(
        elements,
        subject_name=subject_name,
        reference_name=reference_name,
        also_name=also_name,
        subject_filter=subject_filter,
        reference_filter=reference_filter,
        also_filter=also_filter,
        extra_filters=extra_filters,
        within_m=within_m,
        within_also_m=within_also_m,
        layout=layout,
        dem=dem,
    )


def _places_from(
    elements: list[dict[str, Any]],
    *,
    subject_name: str,
    reference_name: str,
    also_name: str,
    subject_filter: str,
    reference_filter: str,
    also_filter: str,
    extra_filters: dict[str, str],
    within_m: int,
    within_also_m: int,
    layout: Layout,
    dem: Any | None,
) -> tuple[list[dict[str, Any]], int]:
    subjects = [item for item in elements if _matches(item, subject_filter) and len(_points(item)) >= 2]
    references = [item for item in elements if _matches(item, reference_filter) and len(_points(item)) >= 2]
    others = [item for item in elements if _matches(item, also_filter) and len(_points(item)) >= 2] if also_filter else []
    extras = {
        name: [item for item in elements if _matches(item, filt) and len(_points(item)) >= 2]
        for name, filt in extra_filters.items()
    }
    urban = [item for item in elements if _is_urban(item) and _points(item)]
    urban_point = _mean_point([point for item in urban for point in _points(item)])
    places: list[dict[str, Any]] = []
    missing = 0
    for subject in subjects:
        for reference in references:
            if _same(subject, reference):
                continue
            if _distance_m(_points(subject), _points(reference)) > within_m:
                continue
            if also_filter and not _near_any(_points(subject), others, within_also_m):
                continue
            status = _screen(
                subject=subject,
                reference=reference,
                others=others,
                extras=extras,
                urban_point=urban_point,
                subject_name=subject_name,
                reference_name=reference_name,
                also_name=also_name,
                layout=layout,
                dem=dem,
            )
            if status == "landuse_missing":
                missing += 1
                continue
            if status != "ok":
                continue
            places.append(_place(subject, reference))
    return places, missing


def _screen(
    *,
    subject: dict[str, Any],
    reference: dict[str, Any],
    others: list[dict[str, Any]],
    extras: dict[str, list[dict[str, Any]]],
    urban_point: tuple[float, float] | None,
    subject_name: str,
    reference_name: str,
    also_name: str,
    layout: Layout,
    dem: Any | None,
) -> str:
    lines = {
        subject_name: subject,
        reference_name: reference,
    }
    if also_name:
        nearest = _nearest(reference, others)
        if nearest is not None:
            lines[also_name] = nearest
    for name, pool in extras.items():
        nearest = _nearest(subject, pool)
        if nearest is not None:
            lines[name] = nearest
    for rule in layout.bearings:
        way = lines.get(rule.feature)
        axis = _line_axis(_points(way)) if way is not None else None
        if axis is None or not _axis_ok(axis, rule.axis_deg):
            return "fail"
    for rule in layout.directions:
        origin = _named_point(
            rule.reference,
            lines=lines,
            urban_point=urban_point,
            dem=dem,
            direction=rule.direction,
            as_target=False,
        )
        if rule.subject in _MOUNTAIN:
            if origin is None or dem is None or not _rises(dem, origin, rule.direction):
                return "fail"
            continue
        if rule.subject in _URBAN:
            if origin is None or urban_point is None:
                return "landuse_missing"
            if _direction_of(origin, urban_point) != rule.direction:
                return "fail"
            continue
        target = _named_point(
            rule.subject,
            lines=lines,
            urban_point=urban_point,
            dem=dem,
            direction=rule.direction,
            as_target=True,
        )
        if origin is None or target is None:
            return "fail"
        if _direction_of(origin, target) != rule.direction:
            return "fail"
    return "ok"


def _named_point(
    name: str,
    *,
    lines: dict[str, dict[str, Any]],
    urban_point: tuple[float, float] | None,
    dem: Any | None,
    direction: str,
    as_target: bool,
) -> tuple[float, float] | None:
    del dem, direction, as_target
    if name in _URBAN:
        return urban_point
    way = lines.get(name)
    if way is None:
        return None
    return _centroid(_points(way))


def _place(subject: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    point = _centroid(_points(subject))
    assert point is not None
    lat, lon = point
    return {
        "result_id": "place_0",
        "osm_type": subject.get("type") or "way",
        "osm_id": subject.get("id"),
        "tags": subject.get("tags") if isinstance(subject.get("tags"), dict) else {},
        "latitude": lat,
        "longitude": lon,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "subject": _brief(subject),
        "reference": _brief(reference),
    }


def _brief(element: dict[str, Any]) -> dict[str, Any]:
    tags = element.get("tags") if isinstance(element.get("tags"), dict) else {}
    return {
        "osm_type": element.get("type") or "way",
        "osm_id": element.get("id"),
        "name": tags.get("name"),
    }


def _dedupe(places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any]] = set()
    chosen: list[dict[str, Any]] = []
    ordered = sorted(places, key=lambda item: (str(item.get("osm_id")), str(item["reference"].get("osm_id"))))
    for place in ordered:
        key = (place.get("osm_id"), place["reference"].get("osm_id"))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(place)
    return chosen


def _rises(dem: Any, origin: tuple[float, float], direction: str) -> bool:
    lat, lon = origin
    base = dem.sample(lon, lat)
    far_lat, far_lon = _offset(lat, lon, direction, _SAMPLE_M)
    far = dem.sample(far_lon, far_lat)
    if base is None or far is None:
        return False
    return float(far) - float(base) >= _RISE_M


def _offset(lat: float, lon: float, direction: str, meters: float) -> tuple[float, float]:
    deg = _DIRECTIONS.index(direction) * 45.0
    north = math.cos(math.radians(deg)) * meters
    east = math.sin(math.radians(deg)) * meters
    return lat + north / _M_PER_DEG_LAT, lon + east / _m_per_deg_lon(lat)


def _line_axis(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    return _bearing_deg(points[0], points[-1]) % 180.0


def _axis_ok(actual: float, expected: float) -> bool:
    delta = abs(actual - expected) % 180.0
    delta = min(delta, 180.0 - delta)
    return delta <= _BEARING_TOLERANCE_DEG


def _direction_of(origin: tuple[float, float], target: tuple[float, float]) -> str:
    deg = _bearing_deg(origin, target) % 360.0
    index = int((deg + 22.5) // 45.0) % 8
    return _DIRECTIONS[index]


def _bearing_deg(origin: tuple[float, float], target: tuple[float, float]) -> float:
    north = target[0] - origin[0]
    east = (target[1] - origin[1]) * math.cos(math.radians((origin[0] + target[0]) / 2.0))
    return math.degrees(math.atan2(east, north))


def _is_urban(element: dict[str, Any]) -> bool:
    tags = element.get("tags") if isinstance(element.get("tags"), dict) else {}
    building = tags.get("building")
    if building not in (None, "", "no", "false"):
        return True
    return tags.get("landuse") == "residential"


def _matches(element: dict[str, Any], filters: str) -> bool:
    if not filters:
        return False
    tags = element.get("tags") if isinstance(element.get("tags"), dict) else {}
    for key, value in re.findall(r'\["([^"]+)"="([^"]+)"\]', filters):
        if str(tags.get(key)) != value:
            return False
    for key in re.findall(r'\["([^"=\]]+)"\]', filters):
        if key not in tags:
            return False
    return True


def _same(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("type") == right.get("type") and left.get("id") == right.get("id")


def _near_any(points: list[tuple[float, float]], ways: list[dict[str, Any]], meters: int) -> bool:
    return any(_distance_m(points, _points(item)) <= meters for item in ways)


def _nearest(anchor: dict[str, Any], ways: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not ways:
        return None
    origin = _points(anchor)
    return min(ways, key=lambda item: _distance_m(origin, _points(item)))


def _distance_m(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> float:
    if len(left) < 2 or len(right) < 2:
        return math.inf
    best = math.inf
    for poly, other in ((left, right), (right, left)):
        for index in range(len(other) - 1):
            for point in poly:
                best = min(best, _point_segment_m(point, other[index], other[index + 1]))
    return best


def _point_segment_m(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    lat0 = point[0]
    px, py = _xy(point, lat0)
    sx, sy = _xy(start, lat0)
    ex, ey = _xy(end, lat0)
    dx, dy = ex - sx, ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - sx, py - sy)
    ratio = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    return math.hypot(px - (sx + ratio * dx), py - (sy + ratio * dy))


def _xy(point: tuple[float, float], lat0: float) -> tuple[float, float]:
    return (
        math.radians(point[1]) * 6_371_000.0 * math.cos(math.radians(lat0)),
        math.radians(point[0]) * 6_371_000.0,
    )


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return (sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))


def _mean_point(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    return _centroid(points)


def _points(element: dict[str, Any]) -> list[tuple[float, float]]:
    raw = element.get("geometry")
    if not isinstance(raw, list):
        return []
    points: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, dict) or "lat" not in item or "lon" not in item:
            continue
        try:
            points.append((float(item["lat"]), float(item["lon"])))
        except (TypeError, ValueError):
            continue
    return points


def _load_elements(
    provider: Any,
    ql: str,
    *,
    filters: list[str],
    spatial: str,
    timeout_sec: int,
    limit: int,
    prefix: str = "",
) -> tuple[list[dict[str, Any]], str]:
    """先一次下载。超时或服务器拒绝时，按类型分别下载，再交给本地筛选。"""

    try:
        payload = _query(provider, ql, timeout_sec)
    except EngineUnavailableError as exc:
        if exc.error_code not in _DOWNLOAD_FALLBACK_CODES:
            raise
        return _load_elements_separately(
            provider,
            filters=filters,
            spatial=spatial,
            timeout_sec=timeout_sec,
            limit=limit,
            prefix=prefix,
        )
    return _elements(payload), ql


def _load_elements_separately(
    provider: Any,
    *,
    filters: list[str],
    spatial: str,
    timeout_sec: int,
    limit: int,
    prefix: str,
) -> tuple[list[dict[str, Any]], str]:
    elements: list[dict[str, Any]] = []
    last_ql = ""
    for item in filters:
        if not item:
            continue
        last_ql = _bundle_ql([item], spatial, timeout_sec=timeout_sec, limit=limit, prefix=prefix)
        payload = _query(provider, last_ql, timeout_sec)
        got = _elements(payload)
        if len(got) >= limit:
            return got, last_ql
        elements.extend(got)
    return elements, last_ql


def _bundle_ql(
    filters: list[str],
    spatial: str,
    *,
    timeout_sec: int,
    limit: int,
    prefix: str = "",
) -> str:
    parts = [f"way{item}{spatial}" for item in filters if item]
    inner = "\n  ".join(f"{item};" for item in parts)
    lines = [f"[out:json][timeout:{timeout_sec}];"]
    if prefix:
        lines.append(prefix)
    lines.append(f"(\n  {inner}\n);")
    lines.append(f"out geom {limit};")
    return "\n".join(lines)


def _bbox_filter(bounds: BBox) -> str:
    return (
        f"({_fmt_num(bounds.south)},{_fmt_num(bounds.west)},"
        f"{_fmt_num(bounds.north)},{_fmt_num(bounds.east)})"
    )


def _query(provider: Any, ql: str, timeout_sec: int) -> dict[str, Any]:
    payload = provider.query(OsmQueryRequest(ql=ql, timeout_sec=float(timeout_sec), operation="near"))
    if not isinstance(payload, dict):
        raise EngineUnavailableError("Overpass 回执不是 JSON 对象")
    return payload


def _elements(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("elements")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _span_m(bounds: BBox) -> float:
    mid = (bounds.south + bounds.north) / 2.0
    height = abs(bounds.north - bounds.south) * _M_PER_DEG_LAT
    width = abs(bounds.east - bounds.west) * _m_per_deg_lon(mid)
    return max(height, width)


def _m_per_deg_lon(lat: float) -> float:
    return max(111_320.0 * math.cos(math.radians(lat)), 1.0)


def _merge_bbox(boxes: list[BBox]) -> list[float] | None:
    if not boxes:
        return None
    return [
        min(item.west for item in boxes),
        min(item.south for item in boxes),
        max(item.east for item in boxes),
        max(item.north for item in boxes),
    ]


class _Budget:
    def __init__(self) -> None:
        self.used = 0
        self.partial = False
        self.unscanned: list[BBox] = []
        self.last_ql = ""

    def take(self) -> bool:
        if self.used >= _MAX_TILE_QUERIES:
            return False
        self.used += 1
        return True

