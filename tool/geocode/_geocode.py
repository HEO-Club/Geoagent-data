"""geocode 共享执行器：高德正/逆地理编码，可选自建 Nominatim。"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tool._crs import (
    format_gcj02_lonlat,
    is_gcj02,
    location_to_wgs84,
    to_wgs84,
    transform_bbox,
    wgs84_location,
)
from tool._gate import allow_real_api as _allow_real_api, load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext, declared_inputs

_DEFAULT_TOP_K = 10
_TOP_K_MIN = 1
_TOP_K_MAX = 50
_PROVIDER_AMAP = "amap"
_PROVIDER_NOMINATIM = "nominatim"
_CRS_GCJ02 = "gcj02"
_CRS_WGS84 = "wgs84"
_DIRECTION_FORWARD = "forward"
_DIRECTION_REVERSE = "reverse"
_DIRECTION_AUTO = "auto"
_DIRECTIONS = frozenset({_DIRECTION_FORWARD, _DIRECTION_REVERSE, _DIRECTION_AUTO})
_DEFAULT_AMAP_ENDPOINT = "https://restapi.amap.com/v3/geocode"
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_NOMINATIM_UA = "geoagent-dataset/1.0 (geocode; local)"
_PUBLIC_NOMINATIM_HOSTS = frozenset(
    {
        "nominatim.openstreetmap.org",
        "www.nominatim.openstreetmap.org",
        "nominatim.osm.org",
        "www.nominatim.osm.org",
    }
)
_COORD_SPLIT_RE = re.compile(r"[,;，、\s]+")
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
_NOMINATIM_LOCK = threading.Lock()
_NOMINATIM_LAST_REQUEST_MONOTONIC = 0.0

_PHOTO_POINT_ASSUMPTION = (
    "候选坐标是地名或地址的匹配点，不是照片拍摄点；村、镇、区等行政区中心不得当作拍摄点"
)
_COORD_ORDER_ASSUMPTION = "无法唯一判定时，坐标串按 lon,lat 解析"
_NO_SESSION_ASSUMPTION = "本步只返回地理编码候选，未打开街景或地图会话"
_AMAP_LICENSE_ASSUMPTION = "高德结果不授予再分发或训练使用权"

class GeocodeInputError(Exception):
    """query / area / direction / top_k 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """真实地理编码服务未配置、被闸门拒绝或调用失败。"""

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
    """解析后的可选范围；几何不会被当成拍摄点。"""

    text: str | None = None
    center: GeoPoint | None = None
    radius_m: int | None = None
    bbox: BBox | None = None
    polygon: tuple[tuple[float, float], ...] | None = None
    unsupported: Any = None
    applied: Any = None

@dataclass(frozen=True)
class GeocodeRequest:
    """组装后的正/逆地理编码请求。"""

    direction: str
    query: str
    top_k: int
    city: str | None = None
    bbox: BBox | None = None
    location: GeoPoint | None = None

@runtime_checkable
class GeocodeProvider(Protocol):
    """可注入的地理编码后端；测试用 extras['geocode_provider'] 替换。"""

    def lookup(self, request: GeocodeRequest) -> dict[str, Any]:
        """提交正/逆地理编码，返回可归一化的 JSON 对象。"""

class AmapGeocodeProvider:
    """高德 Web 服务地理编码适配器；密钥与端点只读环境变量。"""

    name = _PROVIDER_AMAP
    crs = _CRS_GCJ02

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

    def lookup(self, request: GeocodeRequest) -> dict[str, Any]:
        if request.direction == _DIRECTION_REVERSE:
            return self._regeo(request)
        return self._geo(request)

    def _geo(self, request: GeocodeRequest) -> dict[str, Any]:
        params: dict[str, str] = {
            "key": self._api_key,
            "output": "json",
            "address": request.query,
        }
        if request.city:
            params["city"] = request.city
        return self._get("geo", params, error_prefix="高德地理编码")

    def _regeo(self, request: GeocodeRequest) -> dict[str, Any]:
        if request.location is None:
            raise EngineUnavailableError("逆地理编码缺少 location")
        params = {
            "key": self._api_key,
            "output": "json",
            "location": format_gcj02_lonlat(request.location.lon, request.location.lat),
            "extensions": "base",
        }
        return self._get("regeo", params, error_prefix="高德逆地理编码")

    def _get(self, path: str, params: dict[str, str], *, error_prefix: str) -> dict[str, Any]:
        url = _append_query(f"{self._endpoint}/{path}", urllib.parse.urlencode(params))
        raw = _http_json(
            url,
            headers={"Accept": "application/json"},
            timeout_sec=self._timeout_sec,
            error_prefix=error_prefix,
        )
        if not isinstance(raw, dict):
            raise EngineUnavailableError(f"{error_prefix} 回执不是 JSON 对象")
        status = raw.get("status")
        if str(status) != "1":
            info = raw.get("info") or raw.get("infocode") or "unknown"
            raise EngineUnavailableError(f"{error_prefix} 失败: {info}")
        return raw

class NominatimGeocodeProvider:
    """Nominatim 适配器；默认拒绝公共 OSM 实例，坐标为 WGS84。"""

    name = _PROVIDER_NOMINATIM
    crs = _CRS_WGS84

    def __init__(
        self,
        *,
        endpoint: str,
        user_agent: str,
        timeout_sec: float,
        throttle_public: bool,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._user_agent = user_agent
        self._timeout_sec = timeout_sec
        self._throttle_public = throttle_public

    def lookup(self, request: GeocodeRequest) -> dict[str, Any]:
        if self._throttle_public:
            _nominatim_throttle()
        if request.direction == _DIRECTION_REVERSE:
            if request.location is None:
                raise EngineUnavailableError("逆地理编码缺少 location")
            params = {
                "lat": _fmt_coord(request.location.lat),
                "lon": _fmt_coord(request.location.lon),
                "format": "jsonv2",
                "addressdetails": "1",
            }
            path = "reverse"
        else:
            params = {
                "q": request.query,
                "format": "jsonv2",
                "limit": str(request.top_k),
                "addressdetails": "1",
            }
            if request.bbox is not None:
                params["viewbox"] = (
                    f"{_fmt_coord(request.bbox.west)},{_fmt_coord(request.bbox.north)},"
                    f"{_fmt_coord(request.bbox.east)},{_fmt_coord(request.bbox.south)}"
                )
                params["bounded"] = "1"
            path = "search"
        url = _append_query(f"{self._endpoint}/{path}", urllib.parse.urlencode(params))
        raw = _http_json(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self._user_agent,
            },
            timeout_sec=self._timeout_sec,
            error_prefix="Nominatim",
        )
        if isinstance(raw, list):
            return {"nominatim": [item for item in raw if isinstance(item, dict)]}
        if isinstance(raw, dict):
            if raw.get("error"):
                return {"nominatim": []}
            return {"nominatim": [raw]}
        raise EngineUnavailableError("Nominatim 回执不是 JSON 对象或数组")

def execute_geocode(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """在地名、地址和坐标表达之间查询映射。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "query", "area", "direction", "language", "top_k")
        query = _parse_query(inputs.get("query"))
        if not query:
            raise GeocodeInputError("缺少必填输入 query", "missing_input")
        direction_in = _parse_direction(inputs.get("direction"))
        area = _parse_area(_resolve_area(inputs.get("area"), ctx))
        top_k = _parse_top_k(None)
        direction, location = _resolve_lookup(direction_in, query)
        provider = _resolve_provider(ctx)
        provider_name = _provider_name(provider)
        native_crs = _provider_crs(provider)
        city, bbox, unused = _route_area(area, direction=direction, provider_name=provider_name)
        request = GeocodeRequest(
            direction=direction,
            query=query,
            top_k=top_k,
            city=city,
            bbox=bbox,
            location=location,
        )
        payload = provider.lookup(request)
        hits = _normalize_results(
            payload,
            provider_name=provider_name,
            crs=native_crs,
            reverse_point=location,
        )
        return _ok(
            hits[:top_k],
            provider_name=provider_name,
            crs=_CRS_WGS84,
            native_crs=native_crs,
            direction=direction,
            query=query,
            top_k=top_k,
            area=area,
            city=city,
            location=location,
            unused=unused,
        )
    except GeocodeInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def _ok(
    results: list[dict[str, Any]],
    *,
    provider_name: str,
    crs: str,
    native_crs: str,
    direction: str,
    query: str,
    top_k: int,
    area: ParsedArea,
    city: str | None,
    location: GeoPoint | None,
    unused: dict[str, Any],
) -> Observation:
    numbered: list[dict[str, Any]] = []
    for item in results:
        row = dict(item)
        row["result_id"] = f"geocode_{len(numbered) + 1}"
        numbered.append(row)
    applied: dict[str, Any] = {
        "provider": provider_name,
        "direction": direction,
        "top_k": top_k,
        "crs": crs,
        "query": query,
    }
    if city:
        applied["city"] = city
    if area.applied is not None:
        applied["area"] = area.applied
    if location is not None:
        applied["location"] = {"lon": location.lon, "lat": location.lat, "crs": crs}
    if unused:
        applied["unsupported"] = unused
    result: dict[str, Any] = {
        "operation": "geocode",
        "results": numbered,
        "applied": applied,
        "assumptions": _assumptions(provider_name, native_crs),
    }
    return Observation(ok=True, result=_strip_forbidden(result))

def _assumptions(provider_name: str, native_crs: str) -> list[str]:
    if native_crs == _CRS_GCJ02 or provider_name == _PROVIDER_AMAP:
        crs_note = "高德接口为 GCJ-02，已近似转换为 WGS84"
    else:
        crs_note = f"当前后端为 {provider_name}，坐标为 WGS84，未做坐标系转换"
    items = [
        _PHOTO_POINT_ASSUMPTION,
        _COORD_ORDER_ASSUMPTION,
        crs_note,
        _NO_SESSION_ASSUMPTION,
    ]
    if provider_name == _PROVIDER_AMAP:
        items.insert(3, _AMAP_LICENSE_ASSUMPTION)
    return items

def _route_area(
    area: ParsedArea,
    *,
    direction: str,
    provider_name: str,
) -> tuple[str | None, BBox | None, dict[str, Any]]:
    """按后端能力选用 city/viewbox；其余几何记入 unsupported，不静默 geocode。"""

    unused: dict[str, Any] = {}
    if area.unsupported is not None:
        unused["area"] = area.unsupported
    if direction == _DIRECTION_REVERSE:
        if area.text:
            unused["area_text"] = area.text
        if area.bbox is not None:
            unused["bbox"] = _bbox_applied(area.bbox)
        if area.center is not None:
            unused["center"] = {"lat": area.center.lat, "lon": area.center.lon}
        if area.radius_m is not None:
            unused["radius_m"] = area.radius_m
        if area.polygon is not None:
            unused["polygon"] = _polygon_applied(area.polygon)
        return None, None, unused

    city: str | None = None
    bbox: BBox | None = None
    if provider_name == _PROVIDER_NOMINATIM:
        bbox = area.bbox
        if area.text:
            unused["area_text"] = area.text
    else:
        city = area.text
        if area.bbox is not None:
            unused["bbox"] = _bbox_applied(area.bbox)
    if area.center is not None:
        unused["center"] = {"lat": area.center.lat, "lon": area.center.lon}
    if area.radius_m is not None:
        unused["radius_m"] = area.radius_m
    if area.polygon is not None:
        unused["polygon"] = _polygon_applied(area.polygon)
    return city, bbox, unused

def _resolve_lookup(direction: str, query: str) -> tuple[str, GeoPoint | None]:
    point = _parse_coordinate_query(query)
    if direction == _DIRECTION_AUTO:
        if point is not None:
            return _DIRECTION_REVERSE, point
        return _DIRECTION_FORWARD, None
    if direction == _DIRECTION_REVERSE:
        if point is None:
            raise GeocodeInputError("direction=reverse 时 query 必须是坐标", "invalid_query")
        return _DIRECTION_REVERSE, point
    return _DIRECTION_FORWARD, None

def _parse_direction(raw: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DIRECTION_AUTO
    if not isinstance(raw, str):
        raise GeocodeInputError("direction 必须是 forward、reverse 或 auto", "invalid_direction")
    value = raw.strip().lower()
    if value not in _DIRECTIONS:
        raise GeocodeInputError("direction 必须是 forward、reverse 或 auto", "invalid_direction")
    return value

def _parse_query(raw: Any) -> str:
    if raw is None or raw == "":
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise GeocodeInputError("query 必须是字符串或字符串列表", "invalid_query")
            stripped = item.strip()
            if stripped:
                parts.append(stripped)
        return " ".join(parts)
    raise GeocodeInputError("query 必须是字符串或字符串列表", "invalid_query")

def _parse_coordinate_query(raw: str) -> GeoPoint | None:
    text = raw.strip()
    if not text:
        return None
    parts = [part for part in _COORD_SPLIT_RE.split(text) if part]
    if len(parts) != 2:
        return None
    first = _as_float(parts[0])
    second = _as_float(parts[1])
    if first is None or second is None:
        return None
    if abs(first) > 90.0 and abs(second) <= 90.0:
        lon, lat = first, second
    elif abs(second) > 90.0 and abs(first) <= 90.0:
        lat, lon = first, second
    else:
        lon, lat = first, second
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return GeoPoint(lat=lat, lon=lon)

def _parse_top_k(raw: Any) -> int:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DEFAULT_TOP_K
    value = _as_int(raw)
    if value is None:
        raise GeocodeInputError("top_k 必须是整数", "invalid_top_k")
    return max(_TOP_K_MIN, min(_TOP_K_MAX, value))

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
                parsed = _as_float(radius_raw)
                if parsed is None or parsed < 0:
                    raise GeocodeInputError("radius_m 必须是非负数", "invalid_area")
                radius = int(round(parsed))
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
        to_crs=_CRS_WGS84,
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
    bbox = _validated_bbox(west, south, east, north)
    return _bbox_to_wgs84(bbox, _crs_of_mapping(raw))

def _bbox_from_values(values: list[Any]) -> BBox | None:
    nums = [_as_float(item) for item in values]
    if any(item is None for item in nums):
        return None
    first, second, third, fourth = nums[0], nums[1], nums[2], nums[3]
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

def _resolve_provider(ctx: RuntimeContext | None) -> GeocodeProvider:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("geocode_provider")
    if injected is not None:
        if not isinstance(injected, GeocodeProvider):
            raise EngineUnavailableError("geocode_provider 必须提供 lookup(request)")
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_API=false，禁止调用真实地理编码 API")
    name = _env_value("GEOCODE_PROVIDER", _PROVIDER_AMAP).lower()
    if name == _PROVIDER_NOMINATIM:
        return _build_nominatim_provider()
    if name != _PROVIDER_AMAP:
        raise EngineUnavailableError(f"未知 GEOCODE_PROVIDER: {name}")
    api_key = _amap_api_key()
    if not api_key:
        raise EngineUnavailableError("未配置 AMAP_WEB_KEY 或 AMAP_API_KEY")
    return AmapGeocodeProvider(
        api_key=api_key,
        endpoint=_env_value("AMAP_GEOCODE_ENDPOINT", _DEFAULT_AMAP_ENDPOINT),
        timeout_sec=_env_timeout("AMAP_TIMEOUT_SEC"),
    )

def _build_nominatim_provider() -> NominatimGeocodeProvider:
    endpoint = os.environ.get("GEOCODE_NOMINATIM_ENDPOINT", "").strip()
    if not endpoint:
        raise EngineUnavailableError("未配置 GEOCODE_NOMINATIM_ENDPOINT")
    public = _is_public_nominatim(endpoint)
    if public and not _allow_public_nominatim():
        raise EngineUnavailableError(
            "拒绝公共 Nominatim 实例；请配置自建 GEOCODE_NOMINATIM_ENDPOINT"
        )
    user_agent = (
        os.environ.get("GEOCODE_NOMINATIM_USER_AGENT", "").strip()
        or os.environ.get("NOMINATIM_USER_AGENT", "").strip()
        or _DEFAULT_NOMINATIM_UA
    )
    timeout_raw = os.environ.get("GEOCODE_NOMINATIM_TIMEOUT_SEC", "").strip() or os.environ.get(
        "NOMINATIM_TIMEOUT_SEC", ""
    ).strip()
    timeout_sec = _parse_timeout(timeout_raw, _DEFAULT_TIMEOUT_SEC)
    return NominatimGeocodeProvider(
        endpoint=endpoint,
        user_agent=user_agent,
        timeout_sec=timeout_sec,
        throttle_public=public,
    )

def _is_public_nominatim(endpoint: str) -> bool:
    host = (urllib.parse.urlparse(endpoint).hostname or "").lower()
    return host in _PUBLIC_NOMINATIM_HOSTS

def _allow_public_nominatim() -> bool:
    raw = os.environ.get("GEOCODE_ALLOW_PUBLIC_NOMINATIM", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}

def _nominatim_throttle() -> None:
    global _NOMINATIM_LAST_REQUEST_MONOTONIC
    with _NOMINATIM_LOCK:
        now = time.monotonic()
        wait = 1.0 - (now - _NOMINATIM_LAST_REQUEST_MONOTONIC)
        if wait > 0:
            time.sleep(wait)
        _NOMINATIM_LAST_REQUEST_MONOTONIC = time.monotonic()

def _provider_name(provider: GeocodeProvider) -> str:
    return str(getattr(provider, "name", "injected"))

def _provider_crs(provider: GeocodeProvider) -> str:
    raw = getattr(provider, "crs", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if _provider_name(provider) == _PROVIDER_NOMINATIM:
        return _CRS_WGS84
    return _CRS_GCJ02

def _amap_api_key() -> str:
    return (
        os.environ.get("AMAP_WEB_KEY", "").strip()
        or os.environ.get("AMAP_API_KEY", "").strip()
    )

def _env_value(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default

def _env_timeout(name: str) -> float:
    return _parse_timeout(os.environ.get(name, "").strip(), _DEFAULT_TIMEOUT_SEC)

def _parse_timeout(raw: str, default: float) -> float:
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    return default

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
    if isinstance(raw, dict) and raw.get("error") and "display_name" not in raw:
        error = raw.get("error")
        if error and error != "Unable to geocode":
            raise EngineUnavailableError(f"{error_prefix} 失败: {error}")
    return raw

def _append_query(endpoint: str, query: str) -> str:
    separator = "&" if urllib.parse.urlparse(endpoint).query else "?"
    return f"{endpoint}{separator}{query}"

def _normalize_results(
    payload: dict[str, Any],
    *,
    provider_name: str,
    crs: str,
    reverse_point: GeoPoint | None,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("results"), list):
        hits: list[dict[str, Any]] = []
        for item in payload["results"]:
            if not isinstance(item, dict):
                continue
            hit = _normalize_hit(item, provider_name=provider_name, crs=crs)
            if hit is not None:
                hits.append(hit)
        return hits
    geocodes = payload.get("geocodes")
    if isinstance(geocodes, list):
        hits = []
        for item in geocodes:
            if not isinstance(item, dict):
                continue
            hit = _normalize_amap_geocode(item, provider_name=provider_name, crs=crs)
            if hit is not None:
                hits.append(hit)
        return hits
    regeocode = payload.get("regeocode")
    if isinstance(regeocode, dict):
        hit = _normalize_amap_regeocode(
            regeocode,
            provider_name=provider_name,
            crs=crs,
            reverse_point=reverse_point,
        )
        return [hit] if hit is not None else []
    nominatim = payload.get("nominatim")
    if isinstance(nominatim, list):
        hits = []
        for item in nominatim:
            if not isinstance(item, dict):
                continue
            hit = _normalize_nominatim(item, provider_name=provider_name, crs=crs)
            if hit is not None:
                hits.append(hit)
        return hits
    return []

def _normalize_hit(
    item: dict[str, Any],
    *,
    provider_name: str,
    crs: str,
) -> dict[str, Any] | None:
    formatted = _amap_text(
        item.get("formatted_address", item.get("display_name", item.get("address")))
    )
    name = _amap_text(item.get("name"))
    if not formatted and not name:
        return None
    hit: dict[str, Any] = {"source": _amap_text(item.get("source")) or provider_name}
    if formatted:
        hit["formatted_address"] = formatted
    if name:
        hit["name"] = name
    match_level = _amap_text(item.get("match_level", item.get("level", item.get("addresstype"))))
    if match_level:
        hit["match_level"] = match_level
    provider_id = _amap_text(item.get("provider_id", item.get("id")))
    if provider_id:
        hit["provider_id"] = provider_id
    location = item.get("location")
    parsed = None
    if isinstance(location, dict):
        lat = _as_float(location.get("lat"))
        lon = _as_float(location.get("lon", location.get("lng")))
        if lat is not None and lon is not None:
            parsed = location_to_wgs84(lon, lat, _amap_text(location.get("crs")) or crs)
    elif isinstance(location, str):
        parsed = _parse_location_string(location, crs=crs)
    if parsed is not None:
        hit["location"] = parsed
    return hit

def _normalize_amap_geocode(
    item: dict[str, Any],
    *,
    provider_name: str,
    crs: str,
) -> dict[str, Any] | None:
    del crs
    formatted = _amap_text(item.get("formatted_address"))
    if not formatted:
        return None
    hit: dict[str, Any] = {
        "formatted_address": formatted,
        "source": provider_name,
    }
    name = _amap_text(item.get("name")) or _amap_text(item.get("district"))
    if name:
        hit["name"] = name
    match_level = _amap_text(item.get("level"))
    if match_level:
        hit["match_level"] = match_level
    parsed = _parse_location_string(item.get("location"), crs=_CRS_GCJ02)
    if parsed is not None:
        hit["location"] = parsed
    adcode = _amap_text(item.get("adcode"))
    if adcode:
        hit["provider_id"] = adcode
    return hit

def _normalize_amap_regeocode(
    item: dict[str, Any],
    *,
    provider_name: str,
    crs: str,
    reverse_point: GeoPoint | None,
) -> dict[str, Any] | None:
    del crs
    formatted = _amap_text(item.get("formatted_address"))
    if not formatted:
        return None
    hit: dict[str, Any] = {
        "formatted_address": formatted,
        "source": provider_name,
    }
    component = item.get("addressComponent")
    if isinstance(component, dict):
        name = _amap_text(component.get("township")) or _amap_text(component.get("district"))
        if name:
            hit["name"] = name
        match_level = _regeo_match_level(component)
        if match_level:
            hit["match_level"] = match_level
        adcode = _amap_text(component.get("adcode"))
        if adcode:
            hit["provider_id"] = adcode
    if reverse_point is not None:
        hit["location"] = wgs84_location(reverse_point.lon, reverse_point.lat)
    return hit

def _regeo_match_level(component: dict[str, Any]) -> str:
    street_number = component.get("streetNumber")
    if isinstance(street_number, dict) and _amap_text(street_number.get("number")):
        return "门牌号"
    if _amap_text(component.get("street")) or (
        isinstance(street_number, dict) and _amap_text(street_number.get("street"))
    ):
        return "道路"
    if _amap_text(component.get("township")):
        return "乡镇"
    if _amap_text(component.get("district")):
        return "区县"
    if _amap_text(component.get("city")):
        return "市"
    if _amap_text(component.get("province")):
        return "省"
    if _amap_text(component.get("country")):
        return "国家"
    return ""

def _normalize_nominatim(
    item: dict[str, Any],
    *,
    provider_name: str,
    crs: str,
) -> dict[str, Any] | None:
    formatted = _amap_text(item.get("display_name", item.get("formatted_address")))
    if not formatted:
        return None
    hit: dict[str, Any] = {
        "formatted_address": formatted,
        "source": provider_name,
    }
    name = _amap_text(item.get("name"))
    if name:
        hit["name"] = name
    match_level = _amap_text(item.get("addresstype")) or _amap_text(item.get("type"))
    if match_level:
        hit["match_level"] = match_level
    lat = _as_float(item.get("lat"))
    lon = _as_float(item.get("lon"))
    if lat is not None and lon is not None:
        hit["location"] = wgs84_location(lon, lat)
    osm_type = _amap_text(item.get("osm_type"))
    osm_id = _amap_text(item.get("osm_id"))
    if osm_type and osm_id:
        hit["provider_id"] = f"{osm_type}/{osm_id}"
    elif osm_id:
        hit["provider_id"] = osm_id
    return hit

def _amap_text(raw: Any) -> str:
    if raw is None or isinstance(raw, bool):
        return ""
    if isinstance(raw, list):
        return ""
    if isinstance(raw, str):
        return raw.strip()
    return str(raw).strip()

def _parse_location_string(raw: Any, *, crs: str) -> dict[str, float | str] | None:
    if not isinstance(raw, str) or not raw.strip() or raw.strip() == "[]":
        return None
    parts = raw.split(",")
    if len(parts) != 2:
        return None
    lon = _as_float(parts[0])
    lat = _as_float(parts[1])
    if lon is None or lat is None:
        return None
    lon, lat = to_wgs84(lon, lat, crs)
    return wgs84_location(lon, lat)

def _bbox_applied(bbox: BBox) -> dict[str, float]:
    return {"west": bbox.west, "south": bbox.south, "east": bbox.east, "north": bbox.north}

def _polygon_applied(points: tuple[tuple[float, float], ...]) -> list[list[float]]:
    return [[lon, lat] for lon, lat in points]

def _fmt_coord(value: float) -> str:
    return f"{value:.6f}"

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
