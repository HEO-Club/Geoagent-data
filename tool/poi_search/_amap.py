"""poi_search 共享执行器：高德 Web 服务 POI REST，返回地点名、地址与 GCJ-02 坐标。"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tool._crs import (
    CRS_WGS84,
    is_gcj02,
    location_to_wgs84,
    shift_amap_coord_string,
    to_wgs84,
    transform_bbox,
)
from tool._gate import allow_real_api as _allow_real_api, load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext, declared_inputs

_DEFAULT_TOP_K = 10
_TOP_K_MIN = 1
_TOP_K_MAX = 500
_AMAP_PAGE_SIZE = 25
_AMAP_MAX_PAGES = 20
_DEFAULT_AROUND_RADIUS_M = 3000
_AROUND_RADIUS_MIN = 0
_AROUND_RADIUS_MAX = 50_000
_PROVIDER_AMAP = "amap"
_CRS_GCJ02 = "gcj02"
_MODE_TEXT = "text"
_MODE_AROUND = "around"
_MODE_POLYGON = "polygon"
_MODE_DETAIL = "detail"
_MODE_LOCAL = "local_browse"
_DEFAULT_ENDPOINT = "https://restapi.amap.com/v3/place"
_DEFAULT_TIMEOUT_SEC = 30.0
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
_ASSUMPTIONS = [
    "当前仅接入高德国内 POI，未覆盖 OSM 或其他地图源",
    "坐标为高德 GCJ-02，已近似转换为 WGS84",
    "检索命中不授予再分发或训练使用权",
    "本步只返回地点列表，未打开街景或地图会话",
]

class SearchInputError(Exception):
    """area / query / categories / radius_m / filters 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """真实 POI 服务未配置、被闸门拒绝或调用失败。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code

@dataclass(frozen=True)
class GeoPoint:
    """Agent 可见 WGS84 点；高德线上坐标在适配器边界转换。"""

    lat: float
    lon: float

@dataclass(frozen=True)
class BBox:
    """轴对齐矩形，顺序为 west, south, east, north。"""

    west: float
    south: float
    east: float
    north: float

@dataclass(frozen=True)
class ParsedArea:
    """解析后的检索范围。"""

    text: str | None = None
    center: GeoPoint | None = None
    radius_m: int | None = None
    bbox: BBox | None = None
    polygon: tuple[tuple[float, float], ...] | None = None
    unsupported: Any = None
    applied: Any = None

@dataclass(frozen=True)
class PoiSearchRequest:
    """组装后的 POI 检索请求。"""

    mode: str
    top_k: int
    keywords: str | None = None
    types: str | None = None
    city: str | None = None
    location: str | None = None
    radius_m: int | None = None
    polygon: str | None = None
    poi_ids: str | None = None

@runtime_checkable
class PoiSearchProvider(Protocol):
    """可注入的 POI 后端；测试用 extras['poi_search_provider'] 替换。"""

    def search(self, request: PoiSearchRequest) -> dict[str, Any]:
        """提交检索，返回高德风格 `{pois: [...]}` 或 `{results: [...]}`。"""

class AmapPoiSearchProvider:
    """高德 Web 服务 POI 适配器；密钥与端点只读环境变量。"""

    name = _PROVIDER_AMAP
    max_top_k = _TOP_K_MAX

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        timeout_sec: float,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self._timeout_sec = timeout_sec

    def search(self, request: PoiSearchRequest) -> dict[str, Any]:
        if request.mode == _MODE_DETAIL:
            return self._get(request, page=1, offset=min(_AMAP_PAGE_SIZE, max(1, request.top_k)))

        collected: list[dict[str, Any]] = []
        page = 1
        remaining = request.top_k
        while remaining > 0 and page <= _AMAP_MAX_PAGES:
            offset = min(_AMAP_PAGE_SIZE, remaining)
            payload = self._get(request, page=page, offset=offset)
            pois = payload.get("pois")
            if not isinstance(pois, list) or not pois:
                break
            for item in pois:
                if isinstance(item, dict):
                    collected.append(item)
            if len(pois) < offset:
                break
            remaining = request.top_k - len(collected)
            page += 1
        return {"pois": collected[: request.top_k], "count": str(len(collected))}

    def _get(self, request: PoiSearchRequest, *, page: int, offset: int) -> dict[str, Any]:
        params: dict[str, str] = {
            "key": self._api_key,
            "output": "json",
            "extensions": "base",
            "offset": str(offset),
            "page": str(page),
        }
        if request.keywords:
            params["keywords"] = request.keywords
        if request.types:
            params["types"] = request.types
        if request.mode == _MODE_TEXT:
            if request.city:
                params["city"] = request.city
                params["citylimit"] = "true"
        elif request.mode == _MODE_AROUND:
            if not request.location:
                raise EngineUnavailableError("周边检索缺少 location")
            params["location"] = shift_amap_coord_string(request.location)
            params["radius"] = str(
                request.radius_m if request.radius_m is not None else _DEFAULT_AROUND_RADIUS_M
            )
        elif request.mode == _MODE_POLYGON:
            if not request.polygon:
                raise EngineUnavailableError("多边形检索缺少 polygon")
            params["polygon"] = shift_amap_coord_string(request.polygon)
        elif request.mode == _MODE_DETAIL:
            if not request.poi_ids:
                raise EngineUnavailableError("详情查询缺少 id")
            params["id"] = request.poi_ids
        else:
            raise EngineUnavailableError(f"未知检索模式: {request.mode}")

        url = _append_query(f"{self._endpoint}/{request.mode}", urllib.parse.urlencode(params))
        raw = _http_json(
            url,
            headers={"Accept": "application/json"},
            timeout_sec=self._timeout_sec,
            error_prefix="高德 POI",
        )
        if not isinstance(raw, dict):
            raise EngineUnavailableError("高德 POI 回执不是 JSON 对象")
        status = raw.get("status")
        if str(status) != "1":
            info = raw.get("info") or raw.get("infocode") or "unknown"
            raise EngineUnavailableError(f"高德 POI 失败: {info}")
        return raw

def execute_poi_search(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按名称、类别和区域搜索兴趣点。"""

    del purpose
    return _run("poi_search", inputs, ctx)

def execute_browse(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """对已有候选分页、筛选或展开详情；必要时在区域内重新检索。"""

    del purpose
    return _run("browse", inputs, ctx)

def _run(
    operation: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    try:
        if operation == "browse":
            inputs = declared_inputs(
                inputs,
                "area",
                "bbox",
                "center",
                "query",
                "categories",
                "filters",
                "radius_m",
                "top_k",
            )
        else:
            inputs = declared_inputs(
                inputs,
                "area",
                "query",
                "bbox",
                "center",
                "categories",
                "filters",
                "radius_m",
                "language",
                "top_k",
            )
        query = _parse_query(inputs.get("query"))
        area = _parse_area(_resolve_area(inputs.get("area"), ctx))
        requested_top_k = _parse_top_k(inputs.get("top_k"))
        previous = _previous_results(ctx)
        if operation == "browse":
            categories = ""
            radius_m = None
            filters = _parse_filters(inputs.get("filters"))
            detail_ids = _detail_ids(filters, previous)
        else:
            categories = _parse_categories(inputs.get("categories"))
            radius_m, radius_error = _parse_radius_m(inputs.get("radius_m"))
            if radius_error is not None:
                raise radius_error
            filters = {}
            detail_ids: list[str] = []

        # 显式展开详情优先于本地翻页，避免带着上一页列表却无法看单点。
        if operation == "browse" and detail_ids:
            provider = _resolve_provider(ctx)
            top_k = _capped_top_k(requested_top_k, provider)
            request = PoiSearchRequest(
                mode=_MODE_DETAIL,
                top_k=top_k,
                poi_ids="|".join(detail_ids[:20]),
            )
            payload = provider.search(request)
            hits = _normalize_results(payload, provider_name=_provider_name(provider))
            return _ok(
                operation,
                hits[:top_k],
                provider_name=_provider_name(provider),
                mode=_MODE_DETAIL,
                query=query,
                categories=categories,
                area=area,
                radius_m=None,
                top_k=top_k,
                extra_applied={"poi_ids": list(detail_ids[:20])},
                unsupported=_unsupported(area, radius_m, has_coords=True),
            )

        # 已有候选时只做本地分页/筛选，不模拟拖动地图，也不重新打搜索。
        if operation == "browse" and previous:
            page_hits = _local_browse(previous, query=query, filters=filters, top_k=requested_top_k)
            extra: dict[str, Any] = {"source": "previous_results"}
            if filters:
                extra["filters"] = dict(filters)
            return _ok(
                operation,
                page_hits,
                provider_name="previous_results",
                mode=_MODE_LOCAL,
                query=query,
                categories=categories,
                area=area,
                radius_m=radius_m,
                top_k=requested_top_k,
                extra_applied=extra,
                unsupported=_unsupported(
                    area,
                    radius_m,
                    has_coords=_has_coords(area),
                ),
            )

        if operation == "poi_search":
            if not query and not categories and area.text is None and not _has_coords(area):
                if area.unsupported is not None:
                    raise SearchInputError("area 无法解析且缺少 query/categories", "invalid_area")
                raise SearchInputError("缺少必填输入 area、query 或 categories", "missing_input")
        else:
            if area.text is None and not _has_coords(area):
                if area.unsupported is not None:
                    raise SearchInputError("area 无法解析", "invalid_area")
                raise SearchInputError("缺少必填输入 area", "missing_input")

        mode, request_fields, used_radius = _route_search(area, query, categories, radius_m)
        provider = _resolve_provider(ctx)
        top_k = _capped_top_k(requested_top_k, provider)
        request = PoiSearchRequest(mode=mode, top_k=top_k, **request_fields)
        payload = provider.search(request)
        hits = _normalize_results(payload, provider_name=_provider_name(provider))
        return _ok(
            operation,
            hits[:top_k],
            provider_name=_provider_name(provider),
            mode=mode,
            query=query,
            categories=categories,
            area=area,
            radius_m=used_radius,
            top_k=top_k,
            extra_applied=_request_applied(request),
            unsupported=_unsupported(area, radius_m, has_coords=used_radius is not None),
        )
    except SearchInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def _ok(
    operation: str,
    results: list[dict[str, Any]],
    *,
    provider_name: str,
    mode: str,
    query: str,
    categories: str,
    area: ParsedArea,
    radius_m: int | None,
    top_k: int,
    extra_applied: dict[str, Any],
    unsupported: dict[str, Any],
) -> Observation:
    numbered: list[dict[str, Any]] = []
    for item in results:
        row = dict(item)
        row["result_id"] = f"poi_{len(numbered) + 1}"
        numbered.append(row)
    applied: dict[str, Any] = {
        "provider": provider_name,
        "search_mode": mode,
        "top_k": top_k,
        "crs": CRS_WGS84,
    }
    if query:
        applied["query"] = query
    if categories:
        applied["categories"] = categories
    if area.applied is not None:
        applied["area"] = area.applied
    if radius_m is not None:
        applied["radius_m"] = radius_m
    applied.update(extra_applied)
    if unsupported:
        applied["unsupported"] = unsupported
    result: dict[str, Any] = {
        "operation": operation,
        "results": numbered,
        "applied": applied,
        "assumptions": list(_ASSUMPTIONS),
    }
    return Observation(ok=True, result=_strip_forbidden(result))

def _request_applied(request: PoiSearchRequest) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    if request.city:
        applied["city"] = request.city
    if request.location:
        applied["location"] = request.location
    if request.polygon:
        applied["polygon"] = request.polygon
    return applied

def _unsupported(
    area: ParsedArea,
    radius_m: int | None,
    *,
    has_coords: bool,
) -> dict[str, Any]:
    unsupported: dict[str, Any] = {}
    if area.unsupported is not None:
        unsupported["area"] = area.unsupported
    # 有半径但没有坐标时不静默 geocode，半径不发送。
    if radius_m is not None and not has_coords:
        unsupported["radius_m"] = radius_m
    return unsupported

def _has_coords(area: ParsedArea) -> bool:
    return area.center is not None or area.bbox is not None or area.polygon is not None

def _route_search(
    area: ParsedArea,
    query: str,
    categories: str,
    radius_m: int | None,
) -> tuple[str, dict[str, Any], int | None]:
    """按 area 形态选择 text/around/polygon；禁止为 around 去猜坐标。"""

    keywords = query or None
    types = categories or None
    fields: dict[str, Any] = {"keywords": keywords, "types": types}
    if area.polygon:
        fields["polygon"] = _format_polygon(area.polygon)
        return _MODE_POLYGON, fields, None
    if area.bbox is not None:
        fields["polygon"] = _format_bbox(area.bbox)
        return _MODE_POLYGON, fields, None
    if area.center is not None:
        used_radius = _clamp_radius(
            radius_m if radius_m is not None else area.radius_m,
            default=_DEFAULT_AROUND_RADIUS_M,
        )
        fields["location"] = _format_location(area.center)
        fields["radius_m"] = used_radius
        return _MODE_AROUND, fields, used_radius
    if area.text:
        fields["city"] = area.text
    return _MODE_TEXT, fields, None

def _local_browse(
    previous: list[dict[str, Any]],
    *,
    query: str,
    filters: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    name_filter = _optional_str(filters.get("name")) or query
    category_filter = _join_str_or_list(
        filters.get("category", filters.get("categories", filters.get("type"))),
        joiner="|",
    )
    center, filter_radius = _filter_geo(filters)
    matched: list[dict[str, Any]] = []
    for item in previous:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        category = str(item.get("category") or "")
        if name_filter and name_filter.lower() not in name.lower():
            continue
        if category_filter:
            tokens = [part.strip().lower() for part in category_filter.split("|") if part.strip()]
            if tokens and not any(token in category.lower() for token in tokens):
                continue
        if center is not None and filter_radius is not None:
            loc = item.get("location")
            if not isinstance(loc, dict):
                continue
            lat = _as_float(loc.get("lat"))
            lon = _as_float(loc.get("lon", loc.get("lng")))
            if lat is None or lon is None:
                continue
            if _haversine_m(center.lat, center.lon, lat, lon) > filter_radius:
                continue
        row = dict(item)
        loc = row.get("location")
        if isinstance(loc, dict):
            lat = _as_float(loc.get("lat"))
            lon = _as_float(loc.get("lon", loc.get("lng")))
            if lat is not None and lon is not None:
                row["location"] = location_to_wgs84(lon, lat, loc.get("crs") or _CRS_GCJ02)
        matched.append(row)
    skip = _browse_skip(filters, top_k)
    return matched[skip : skip + top_k]

def _browse_skip(filters: dict[str, Any], top_k: int) -> int:
    offset = filters.get("offset")
    if offset is not None and offset != "":
        value = _as_int(offset)
        if value is None or value < 0:
            raise SearchInputError("filters.offset 必须是非负整数", "invalid_filters")
        return value
    page = filters.get("page")
    if page is None or page == "":
        return 0
    value = _as_int(page)
    if value is None or value < 1:
        raise SearchInputError("filters.page 必须是从 1 起的整数", "invalid_filters")
    return (value - 1) * top_k

def _filter_geo(filters: dict[str, Any]) -> tuple[GeoPoint | None, int | None]:
    radius_raw = filters.get("radius_m", filters.get("radius"))
    lat = _as_float(filters.get("lat", filters.get("latitude")))
    lon = _as_float(filters.get("lon", filters.get("lng", filters.get("longitude"))))
    center_raw = filters.get("center")
    if isinstance(center_raw, dict) and (lat is None or lon is None):
        lat = _as_float(center_raw.get("lat", center_raw.get("latitude")))
        lon = _as_float(center_raw.get("lon", center_raw.get("lng", center_raw.get("longitude"))))
    if lat is None or lon is None:
        return None, None
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None, None
    radius = _DEFAULT_AROUND_RADIUS_M
    if radius_raw is not None and radius_raw != "":
        parsed, error = _parse_radius_m(radius_raw)
        if error is not None:
            raise error
        if parsed is not None:
            radius = parsed
    return GeoPoint(lat=lat, lon=lon), radius

def _detail_ids(
    filters: dict[str, Any],
    previous: list[dict[str, Any]],
) -> list[str]:
    raw_ids = filters.get("poi_id", filters.get("poi_ids", filters.get("id")))
    ids = _as_id_list(raw_ids)
    expand = filters.get("expand")
    if _is_truthy(expand) and not ids:
        for item in previous:
            if not isinstance(item, dict):
                continue
            provider_id = item.get("provider_id")
            if isinstance(provider_id, str) and provider_id.strip():
                ids.append(provider_id.strip())
    if _is_truthy(expand) or ids:
        if not ids:
            raise SearchInputError("展开详情缺少 poi_id", "missing_input")
        return ids
    return []

def _as_id_list(raw: Any) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.replace(",", "|").split("|")]
        return [part for part in parts if part]
    if isinstance(raw, list):
        ids: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise SearchInputError("poi_id 必须是字符串或字符串列表", "invalid_filters")
            ids.append(item.strip())
        return ids
    raise SearchInputError("poi_id 必须是字符串或字符串列表", "invalid_filters")

def _is_truthy(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return raw != 0
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on", "all"}
    return False

def _previous_results(ctx: RuntimeContext | None) -> list[dict[str, Any]]:
    if ctx is None or not isinstance(ctx.previous_tool_result, dict):
        return []
    raw = ctx.previous_tool_result
    rows = raw.get("results")
    if isinstance(rows, list):
        return [item for item in rows if isinstance(item, dict)]
    nested = raw.get("result")
    if isinstance(nested, dict):
        inner = nested.get("results")
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]
    return []

def _parse_query(raw: Any) -> str:
    return _join_str_or_list(raw, joiner="|", error_code="invalid_query", field="query")

def _parse_categories(raw: Any) -> str:
    return _join_str_or_list(raw, joiner="|", error_code="invalid_query", field="categories")

def _join_str_or_list(
    raw: Any,
    *,
    joiner: str,
    error_code: str = "invalid_query",
    field: str = "value",
) -> str:
    if raw is None or raw == "":
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise SearchInputError(f"{field} 必须是字符串或字符串列表", error_code)
            stripped = item.strip()
            if stripped:
                parts.append(stripped)
        return joiner.join(parts)
    raise SearchInputError(f"{field} 必须是字符串或字符串列表", error_code)

def _parse_filters(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if not isinstance(raw, dict):
        raise SearchInputError("filters 必须是对象", "invalid_filters")
    return dict(raw)

def _resolve_area(raw: Any, ctx: RuntimeContext | None) -> Any:
    if raw == "$active_area" and ctx is not None:
        return ctx.active_area
    return raw

def _parse_area(raw: Any) -> ParsedArea:
    if raw is None or raw == "":
        return ParsedArea()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ParsedArea()
        return ParsedArea(text=text, applied=text)
    if isinstance(raw, (list, tuple)):
        polygon = _polygon_from_points(raw)
        if polygon is not None:
            return ParsedArea(polygon=polygon, applied=_polygon_applied(polygon))
        if len(raw) == 4:
            bbox = _bbox_from_values(list(raw))
            if bbox is not None:
                return ParsedArea(bbox=bbox, applied=_bbox_applied(bbox))
        return ParsedArea(unsupported=raw)
    if isinstance(raw, dict):
        polygon_raw = raw.get("polygon")
        if polygon_raw is not None:
            polygon = _polygon_from_points(polygon_raw)
            if polygon is not None:
                return ParsedArea(polygon=polygon, applied=_polygon_applied(polygon))
            return ParsedArea(unsupported=raw)
        bbox = _bbox_from_mapping(raw)
        if bbox is not None:
            return ParsedArea(bbox=bbox, applied=_bbox_applied(bbox))
        center = _center_from_mapping(raw)
        if center is not None:
            radius_raw = raw.get("radius_m", raw.get("radius"))
            radius: int | None = None
            if radius_raw is not None and radius_raw != "":
                parsed, error = _parse_radius_m(radius_raw)
                if error is None:
                    radius = parsed
            applied: dict[str, Any] = {"lat": center.lat, "lon": center.lon}
            if radius is not None:
                applied["radius_m"] = radius
            return ParsedArea(center=center, radius_m=radius, applied=applied)
        text = _optional_str(raw.get("city", raw.get("text", raw.get("name", raw.get("area")))))
        if text:
            return ParsedArea(text=text, applied=text)
        return ParsedArea(unsupported=raw)
    return ParsedArea(unsupported=raw)

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
    crs = _crs_of_mapping(raw)
    if isinstance(nested, dict) and crs is None:
        crs = _crs_of_mapping(nested)
    lon, lat = to_wgs84(lon, lat, crs)
    return GeoPoint(lat=lat, lon=lon)

def _crs_of_mapping(raw: dict[str, Any]) -> str | None:
    value = raw.get("crs", raw.get("datum"))
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None

def _bbox_to_wgs84(bbox: BBox | None, crs: str | None) -> BBox | None:
    if bbox is None:
        return None
    if not is_gcj02(crs):
        return bbox
    west, south, east, north = transform_bbox(
        bbox.west,
        bbox.south,
        bbox.east,
        bbox.north,
        from_crs=_CRS_GCJ02,
        to_crs=CRS_WGS84,
    )
    return BBox(west=west, south=south, east=east, north=north)

def _bbox_from_mapping(raw: dict[str, Any]) -> BBox | None:
    bbox = raw.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        parsed = _bbox_from_values(list(bbox))
        return _bbox_to_wgs84(parsed, _crs_of_mapping(raw))
    west = _as_float(raw.get("west", raw.get("min_lon")))
    south = _as_float(raw.get("south", raw.get("min_lat")))
    east = _as_float(raw.get("east", raw.get("max_lon")))
    north = _as_float(raw.get("north", raw.get("max_lat")))
    if None in (west, south, east, north):
        return None
    return _bbox_to_wgs84(_validated_bbox(west, south, east, north), _crs_of_mapping(raw))

def _bbox_from_values(values: list[Any]) -> BBox | None:
    nums = [_as_float(item) for item in values]
    if any(item is None for item in nums):
        return None
    first, second, third, fourth = nums[0], nums[1], nums[2], nums[3]
    # GeoJSON [min_lon, min_lat, max_lon, max_lat]：经度绝对值常大于纬度。
    if abs(first) <= 90 and abs(third) <= 90 and (abs(second) > 90 or abs(fourth) > 90):
        return _validated_bbox(second, first, fourth, third)
    return _validated_bbox(first, second, third, fourth)

def _validated_bbox(west: float, south: float, east: float, north: float) -> BBox | None:
    if not -90.0 <= south <= 90.0 or not -90.0 <= north <= 90.0:
        return None
    if not -180.0 <= west <= 180.0 or not -180.0 <= east <= 180.0:
        return None
    return BBox(west=west, south=south, east=east, north=north)

def _polygon_from_points(raw: Any) -> tuple[tuple[float, float], ...] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    if len(raw) == 4 and all(_as_float(item) is not None for item in raw):
        return None
    points: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None
        lon = _as_float(item[0])
        lat = _as_float(item[1])
        if lon is None or lat is None:
            return None
        if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
            return None
        points.append((lon, lat))
    if len(points) < 2:
        return None
    return tuple(points)

def _parse_radius_m(raw: Any) -> tuple[int | None, SearchInputError | None]:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, None
    value = _as_float(raw)
    if value is None or value < 0:
        return None, SearchInputError("radius_m 必须是非负数", "invalid_radius")
    return _clamp_radius(int(round(value)), default=0), None

def _clamp_radius(raw: int | None, *, default: int) -> int:
    if raw is None:
        return default
    return max(_AROUND_RADIUS_MIN, min(_AROUND_RADIUS_MAX, raw))

def _parse_top_k(raw: Any) -> int:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DEFAULT_TOP_K
    value = _as_int(raw)
    if value is None:
        raise SearchInputError("top_k 必须是整数", "invalid_top_k")
    return max(_TOP_K_MIN, min(_TOP_K_MAX, value))

def _capped_top_k(requested: int, provider: PoiSearchProvider) -> int:
    raw = getattr(provider, "max_top_k", _TOP_K_MAX)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = _TOP_K_MAX
    if value < _TOP_K_MIN:
        value = _TOP_K_MAX
    return min(requested, min(value, _TOP_K_MAX))

def _resolve_provider(ctx: RuntimeContext | None) -> PoiSearchProvider:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("poi_search_provider")
    if injected is not None:
        if not isinstance(injected, PoiSearchProvider):
            raise EngineUnavailableError("poi_search_provider 必须提供 search(request)")
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_API=false，禁止调用真实 POI API")
    api_key = _amap_api_key()
    if not api_key:
        raise EngineUnavailableError("未配置 AMAP_WEB_KEY 或 AMAP_API_KEY")
    return AmapPoiSearchProvider(
        api_key=api_key,
        endpoint=_env_value("AMAP_POI_ENDPOINT", _DEFAULT_ENDPOINT),
        timeout_sec=_env_timeout(),
    )

def _provider_name(provider: PoiSearchProvider) -> str:
    return str(getattr(provider, "name", "injected"))

def _amap_api_key() -> str:
    return (
        os.environ.get("AMAP_WEB_KEY", "").strip()
        or os.environ.get("AMAP_API_KEY", "").strip()
    )

def _env_value(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default

def _env_timeout() -> float:
    raw = os.environ.get("AMAP_TIMEOUT_SEC", "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    return _DEFAULT_TIMEOUT_SEC

def _http_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout_sec: float,
    error_prefix: str,
) -> Any:
    request = urllib.request.Request(url, method="GET")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise EngineUnavailableError(
            f"{error_prefix} HTTP {exc.code}: {detail[:200]}",
        ) from exc
    except urllib.error.URLError as exc:
        raise EngineUnavailableError(
            f"{error_prefix} 网络失败: {exc.reason}",
        ) from exc
    except (json.JSONDecodeError, TimeoutError, OSError) as exc:
        raise EngineUnavailableError(f"{error_prefix} 调用失败: {exc}") from exc
    if isinstance(raw, dict) and raw.get("error"):
        raise EngineUnavailableError(f"{error_prefix} 失败: {raw['error']}")
    return raw

def _append_query(endpoint: str, query: str) -> str:
    separator = "&" if urllib.parse.urlparse(endpoint).query else "?"
    return f"{endpoint}{separator}{query}"

def _normalize_results(payload: dict[str, Any], *, provider_name: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        hits: list[dict[str, Any]] = []
        for item in payload["results"]:
            if not isinstance(item, dict):
                continue
            hit = _normalize_hit(item, provider_name=provider_name)
            if hit is not None:
                hits.append(hit)
        return hits
    pois = payload.get("pois") if isinstance(payload, dict) else None
    if not isinstance(pois, list):
        return []
    hits = []
    for item in pois:
        if not isinstance(item, dict):
            continue
        hit = _normalize_amap_poi(item, provider_name=provider_name)
        if hit is not None:
            hits.append(hit)
    return hits

def _normalize_hit(item: dict[str, Any], *, provider_name: str) -> dict[str, Any] | None:
    name = _amap_text(item.get("name"))
    provider_id = _amap_text(item.get("provider_id", item.get("id")))
    if not name and not provider_id:
        return None
    hit: dict[str, Any] = {
        "name": name or "",
        "source": _amap_text(item.get("source")) or provider_name,
    }
    address = _amap_text(item.get("address"))
    if address:
        hit["address"] = address
    category = _amap_text(item.get("category", item.get("type")))
    if category:
        hit["category"] = category
    location = item.get("location")
    parsed = None
    if isinstance(location, dict):
        lat = _as_float(location.get("lat"))
        lon = _as_float(location.get("lon", location.get("lng")))
        if lat is not None and lon is not None:
            parsed = location_to_wgs84(lon, lat, location.get("crs") or _CRS_GCJ02)
    elif isinstance(location, str):
        parsed = _parse_location_string(location)
    if parsed is not None:
        hit["location"] = parsed
    if provider_id:
        hit["provider_id"] = provider_id
    return hit

def _normalize_amap_poi(item: dict[str, Any], *, provider_name: str) -> dict[str, Any] | None:
    name = _amap_text(item.get("name"))
    provider_id = _amap_text(item.get("id"))
    if not name and not provider_id:
        return None
    hit: dict[str, Any] = {
        "name": name or "",
        "source": provider_name,
    }
    address = _amap_text(item.get("address"))
    if address:
        hit["address"] = address
    category = _amap_text(item.get("type"))
    if category:
        hit["category"] = category
    parsed = _parse_location_string(item.get("location"))
    if parsed is not None:
        hit["location"] = parsed
    if provider_id:
        hit["provider_id"] = provider_id
    return hit

def _amap_text(raw: Any) -> str:
    if raw is None or isinstance(raw, bool):
        return ""
    if isinstance(raw, list):
        return ""
    if isinstance(raw, str):
        return raw.strip()
    return str(raw).strip()

def _parse_location_string(raw: Any) -> dict[str, float | str] | None:
    if not isinstance(raw, str) or not raw.strip() or raw.strip() == "[]":
        return None
    parts = raw.split(",")
    if len(parts) != 2:
        return None
    lon = _as_float(parts[0])
    lat = _as_float(parts[1])
    if lon is None or lat is None:
        return None
    lon, lat = to_wgs84(lon, lat, _CRS_GCJ02)
    return {"lon": lon, "lat": lat, "crs": CRS_WGS84}

def _format_location(point: GeoPoint) -> str:
    return f"{_fmt_coord(point.lon)},{_fmt_coord(point.lat)}"

def _format_bbox(bbox: BBox) -> str:
    # 高德矩形：左上 | 右下。
    return (
        f"{_fmt_coord(bbox.west)},{_fmt_coord(bbox.north)}|"
        f"{_fmt_coord(bbox.east)},{_fmt_coord(bbox.south)}"
    )

def _format_polygon(points: tuple[tuple[float, float], ...]) -> str:
    coords = list(points)
    if len(coords) > 2 and coords[0] != coords[-1]:
        coords.append(coords[0])
    return "|".join(f"{_fmt_coord(lon)},{_fmt_coord(lat)}" for lon, lat in coords)

def _bbox_applied(bbox: BBox) -> dict[str, float]:
    return {"west": bbox.west, "south": bbox.south, "east": bbox.east, "north": bbox.north}

def _polygon_applied(points: tuple[tuple[float, float], ...]) -> list[list[float]]:
    return [[lon, lat] for lon, lat in points]

def _fmt_coord(value: float) -> str:
    return f"{value:.6f}"

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))

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

def _as_int(raw: Any) -> int | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            return int(raw.strip())
        except ValueError:
            return None
    return None

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

def _fail(message: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=message, error_code=error_code)
