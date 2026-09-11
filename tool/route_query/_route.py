"""route_query 共享执行器：高德路径规划，可选自建 OSRM。"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tool._crs import (
    format_gcj02_lonlat,
    is_gcj02,
    to_wgs84,
    transform_coordinates,
)
from tool._gate import allow_real_api as _allow_real_api, load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext, declared_inputs

_PROVIDER_AMAP = "amap"
_PROVIDER_OSRM = "osrm"
_CRS_GCJ02 = "gcj02"
_CRS_WGS84 = "wgs84"
_MODE_DRIVE = "drive"
_MODE_WALK = "walk"
_MODE_BIKE = "bike"
_MODE_TRANSIT = "transit"
_ROUTE_FIELDS = ("origin", "destination", "waypoints", "travel_mode", "city", "cityd")
_MODE_ANY = "any"
_TRAVEL_MODES = frozenset({_MODE_DRIVE, _MODE_WALK, _MODE_BIKE, _MODE_TRANSIT, _MODE_ANY})
_AMAP_NO_WAYPOINT_MODES = frozenset({_MODE_WALK, _MODE_BIKE, _MODE_TRANSIT})
_OSRM_PROFILES = {
    _MODE_DRIVE: "driving",
    _MODE_WALK: "walking",
    _MODE_BIKE: "cycling",
}
_DEFAULT_AMAP_DIRECTION_ENDPOINT = "https://restapi.amap.com/v3/direction"
_DEFAULT_AMAP_BICYCLING_ENDPOINT = "https://restapi.amap.com/v4/direction/bicycling"
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_OSRM_UA = "geoagent-dataset/1.0 (route_query; local)"
_MAX_GEOMETRY_POINTS = 200
_EARTH_RADIUS_M = 6_371_000.0
_COORD_SPLIT_RE = re.compile(r"[,;，、\s]+")
_PUBLIC_OSRM_HOSTS = frozenset(
    {
        "router.project-osrm.org",
        "www.router.project-osrm.org",
        "routing.openstreetmap.de",
        "www.routing.openstreetmap.de",
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
_DISTANCE_ASSUMPTION = "distance_m 是路网路径长度，straight_line_distance_m 是途经点折线的直线距离，二者不得混用"
_HISTORICAL_ASSUMPTION = "当前路网不能用来证明几十年前存在同样的道路"
_NO_SESSION_ASSUMPTION = "本步只返回路线规划结果，未打开街景或地图会话"
_AMAP_LICENSE_ASSUMPTION = "高德结果不授予再分发或训练使用权"
_COORD_ORDER_ASSUMPTION = "无法唯一判定时，坐标串按 lon,lat 解析"
_ANY_MODE_ASSUMPTION = "travel_mode=any 按 drive 执行"
_DEFAULT_MODE_ASSUMPTION = "未指定 travel_mode，按 drive 执行"

class RouteInputError(Exception):
    """origin / destination / waypoints / travel_mode 无法按合同解析或后端不支持。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """真实路线规划服务未配置、被闸门拒绝或调用失败。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code

@dataclass(frozen=True)
class GeoPoint:
    """Agent 可见 WGS84 点；高德线上坐标在适配器边界转换。"""

    lat: float
    lon: float

@dataclass(frozen=True)
class RouteRequest:
    """组装后的路径规划请求。"""

    travel_mode: str
    origin: GeoPoint
    destination: GeoPoint
    waypoints: tuple[GeoPoint, ...] = ()
    city: str | None = None
    cityd: str | None = None
    requested_travel_mode: str = _MODE_DRIVE

@runtime_checkable
class RouteProvider(Protocol):
    """可注入的路线规划后端；测试用 extras['route_query_provider'] 替换。"""

    def route(self, request: RouteRequest) -> dict[str, Any]:
        """提交路径规划，返回可归一化的 JSON 对象。"""

class AmapDirectionProvider:
    """高德 Web 服务路径规划适配器；密钥与端点只读环境变量。"""

    name = _PROVIDER_AMAP
    crs = _CRS_GCJ02

    def __init__(
        self,
        *,
        api_key: str,
        direction_endpoint: str,
        bicycling_endpoint: str,
        timeout_sec: float,
    ) -> None:
        self._api_key = api_key
        self._direction_endpoint = direction_endpoint.rstrip("/")
        self._bicycling_endpoint = bicycling_endpoint.rstrip("/")
        self._timeout_sec = timeout_sec

    def route(self, request: RouteRequest) -> dict[str, Any]:
        _check_amap_capabilities(request)
        if request.travel_mode == _MODE_TRANSIT:
            return self._transit(request)
        if request.travel_mode == _MODE_BIKE:
            return self._bicycling(request)
        path = "walking" if request.travel_mode == _MODE_WALK else "driving"
        params: dict[str, str] = {
            "key": self._api_key,
            "output": "json",
            "origin": _format_amap_location(request.origin),
            "destination": _format_amap_location(request.destination),
        }
        if request.waypoints:
            params["waypoints"] = ";".join(_format_amap_location(point) for point in request.waypoints)
        return self._get(f"{self._direction_endpoint}/{path}", params, error_prefix="高德路径规划")

    def _transit(self, request: RouteRequest) -> dict[str, Any]:
        params: dict[str, str] = {
            "key": self._api_key,
            "output": "json",
            "origin": _format_amap_location(request.origin),
            "destination": _format_amap_location(request.destination),
            "city": request.city or "",
        }
        if request.cityd:
            params["cityd"] = request.cityd
        return self._get(
            f"{self._direction_endpoint}/transit/integrated",
            params,
            error_prefix="高德公交路径规划",
        )

    def _bicycling(self, request: RouteRequest) -> dict[str, Any]:
        params = {
            "key": self._api_key,
            "origin": _format_amap_location(request.origin),
            "destination": _format_amap_location(request.destination),
        }
        return self._get(
            self._bicycling_endpoint,
            params,
            error_prefix="高德骑行路径规划",
            v4=True,
        )

    def _get(
        self,
        url: str,
        params: dict[str, str],
        *,
        error_prefix: str,
        v4: bool = False,
    ) -> dict[str, Any]:
        raw = _http_json(
            _append_query(url, urllib.parse.urlencode(params)),
            headers={"Accept": "application/json"},
            timeout_sec=self._timeout_sec,
            error_prefix=error_prefix,
        )
        if not isinstance(raw, dict):
            raise EngineUnavailableError(f"{error_prefix} 回执不是 JSON 对象")
        if v4:
            errcode = raw.get("errcode")
            if errcode not in (0, "0", None):
                detail = raw.get("errdetail") or raw.get("errmsg") or errcode
                raise EngineUnavailableError(f"{error_prefix} 失败: {detail}")
            return raw
        status = raw.get("status")
        if str(status) != "1":
            info = raw.get("info") or raw.get("infocode") or "unknown"
            raise EngineUnavailableError(f"{error_prefix} 失败: {info}")
        return raw

class OsrmRouteProvider:
    """OSRM 适配器；默认拒绝公共演示实例，坐标为 WGS84。"""

    name = _PROVIDER_OSRM
    crs = _CRS_WGS84

    def __init__(
        self,
        *,
        endpoint: str,
        user_agent: str,
        timeout_sec: float,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._user_agent = user_agent
        self._timeout_sec = timeout_sec

    def route(self, request: RouteRequest) -> dict[str, Any]:
        if request.travel_mode == _MODE_TRANSIT:
            raise RouteInputError("OSRM 不支持 transit，未改成驾车", "unsupported_travel_mode")
        profile = _OSRM_PROFILES.get(request.travel_mode)
        if profile is None:
            raise RouteInputError(
                f"OSRM 不支持 travel_mode={request.travel_mode}",
                "unsupported_travel_mode",
            )
        points = (request.origin, *request.waypoints, request.destination)
        coords = ";".join(_format_location(point) for point in points)
        params = {
            "geometries": "geojson",
            "overview": "simplified",
            "steps": "true",
        }
        url = _append_query(
            f"{self._endpoint}/route/v1/{profile}/{coords}",
            urllib.parse.urlencode(params),
        )
        raw = _http_json(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self._user_agent,
            },
            timeout_sec=self._timeout_sec,
            error_prefix="OSRM",
        )
        if not isinstance(raw, dict):
            raise EngineUnavailableError("OSRM 回执不是 JSON 对象")
        code = str(raw.get("code") or "")
        if code.lower() == "noroute":
            return {"osrm": raw, "routes": []}
        if code.lower() not in {"ok", ""}:
            raise EngineUnavailableError(f"OSRM 失败: {code}")
        return raw

def execute_route(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """查询两点或多点之间的道路与路线关系。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, *_ROUTE_FIELDS)
        origin = _require_point(inputs.get("origin"), field="origin")
        destination = _require_point(inputs.get("destination"), field="destination")
        waypoints = _parse_waypoints(inputs.get("waypoints"))
        requested_mode, travel_mode = _parse_travel_mode(inputs.get("travel_mode"))
        request = RouteRequest(
            travel_mode=travel_mode,
            origin=origin,
            destination=destination,
            waypoints=waypoints,
            city=_optional_city(inputs.get("city"), field="city"),
            cityd=_optional_city(inputs.get("cityd"), field="cityd"),
            requested_travel_mode=requested_mode,
        )
        provider = _resolve_provider(ctx)
        _check_provider_capabilities(provider, request)
        payload = provider.route(request)
        provider_name = _provider_name(provider)
        native_crs = _provider_crs(provider)
        routes = _normalize_results(
            payload,
            provider_name=provider_name,
            crs=native_crs,
            travel_mode=travel_mode,
        )
        straight_line_m = _straight_line_distance_m(origin, destination, waypoints)
        simplified = False
        numbered: list[dict[str, Any]] = []
        for item in routes:
            row = dict(item)
            geometry, was_simplified = _simplify_geometry(row.get("geometry"))
            if geometry is not None:
                row["geometry"] = geometry
            simplified = simplified or was_simplified
            row["straight_line_distance_m"] = straight_line_m
            row["travel_mode"] = travel_mode
            row["route_id"] = f"route_{len(numbered) + 1}"
            numbered.append(row)
        return _ok(
            numbered,
            provider_name=provider_name,
            crs=_CRS_WGS84,
            native_crs=native_crs,
            request=request,
            geometry_simplified=simplified,
            straight_line_m=straight_line_m,
        )
    except RouteInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def _ok(
    routes: list[dict[str, Any]],
    *,
    provider_name: str,
    crs: str,
    native_crs: str,
    request: RouteRequest,
    geometry_simplified: bool,
    straight_line_m: float,
) -> Observation:
    applied: dict[str, Any] = {
        "provider": provider_name,
        "crs": crs,
        "travel_mode": request.travel_mode,
        "requested_travel_mode": request.requested_travel_mode,
        "origin": _point_applied(request.origin, crs),
        "destination": _point_applied(request.destination, crs),
        "straight_line_distance_m": straight_line_m,
    }
    if request.waypoints:
        applied["waypoints"] = [_point_applied(point, crs) for point in request.waypoints]
    if request.city:
        applied["city"] = request.city
    if request.cityd:
        applied["cityd"] = request.cityd
    if geometry_simplified:
        applied["geometry_simplified"] = True
    result: dict[str, Any] = {
        "operation": "route",
        "routes": routes,
        "applied": applied,
        "assumptions": _assumptions(provider_name, native_crs, request.requested_travel_mode),
    }
    return Observation(ok=True, result=_strip_forbidden(result))

def _assumptions(provider_name: str, native_crs: str, requested_travel_mode: str) -> list[str]:
    if native_crs == _CRS_GCJ02 or provider_name == _PROVIDER_AMAP:
        crs_note = "高德接口为 GCJ-02，已近似转换为 WGS84"
    else:
        crs_note = f"当前后端为 {provider_name}，坐标为 WGS84，未做坐标系转换"
    items = [
        _DISTANCE_ASSUMPTION,
        _HISTORICAL_ASSUMPTION,
        _COORD_ORDER_ASSUMPTION,
        crs_note,
        _NO_SESSION_ASSUMPTION,
    ]
    if provider_name == _PROVIDER_AMAP:
        items.insert(3, _AMAP_LICENSE_ASSUMPTION)
    if requested_travel_mode == _MODE_ANY:
        items.append(_ANY_MODE_ASSUMPTION)
    elif requested_travel_mode == "":
        items.append(_DEFAULT_MODE_ASSUMPTION)
    return items

def _check_provider_capabilities(provider: RouteProvider, request: RouteRequest) -> None:
    name = _provider_name(provider)
    if name == _PROVIDER_OSRM and request.travel_mode == _MODE_TRANSIT:
        raise RouteInputError("OSRM 不支持 transit，未改成驾车", "unsupported_travel_mode")
    if name == _PROVIDER_AMAP:
        _check_amap_capabilities(request)

def _check_amap_capabilities(request: RouteRequest) -> None:
    if request.waypoints and request.travel_mode in _AMAP_NO_WAYPOINT_MODES:
        raise RouteInputError(
            "高德步行/骑行/公交不支持途经点，未丢弃途经点继续计算",
            "unsupported_waypoints",
        )
    if request.travel_mode == _MODE_TRANSIT and not request.city:
        raise RouteInputError("高德公交需要城市名 city", "missing_input")

def _optional_city(raw: Any, *, field: str) -> str | None:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise RouteInputError(f"{field} 必须是城市名字符串", f"invalid_{field}")
    return raw.strip()

def _require_point(raw: Any, *, field: str) -> GeoPoint:
    if raw is None or raw == "":
        raise RouteInputError(f"缺少必填输入 {field}", "missing_input")
    point = _parse_point(raw)
    if point is None:
        raise RouteInputError(
            f"{field} 必须是坐标，请先使用 geocode",
            f"invalid_{field}",
        )
    return point

def _parse_waypoints(raw: Any) -> tuple[GeoPoint, ...]:
    if raw is None or raw == "":
        return ()
    if not isinstance(raw, (list, tuple)):
        raise RouteInputError("waypoints 必须是坐标数组", "invalid_waypoints")
    points: list[GeoPoint] = []
    for item in raw:
        point = _parse_point(item)
        if point is None:
            raise RouteInputError("waypoints 必须是坐标数组", "invalid_waypoints")
        points.append(point)
    return tuple(points)

def _parse_travel_mode(raw: Any) -> tuple[str, str]:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return "", _MODE_DRIVE
    if not isinstance(raw, str):
        raise RouteInputError(
            "travel_mode 必须是 drive、walk、bike、transit 或 any",
            "invalid_travel_mode",
        )
    value = raw.strip().lower()
    if value not in _TRAVEL_MODES:
        raise RouteInputError(
            "travel_mode 必须是 drive、walk、bike、transit 或 any",
            "invalid_travel_mode",
        )
    if value == _MODE_ANY:
        return _MODE_ANY, _MODE_DRIVE
    return value, value

def _parse_point(raw: Any) -> GeoPoint | None:
    if isinstance(raw, str):
        return _parse_coordinate_query(raw)
    if isinstance(raw, (list, tuple)):
        if len(raw) < 2:
            return None
        first = _as_float(raw[0])
        second = _as_float(raw[1])
        if first is None or second is None:
            return None
        return _point_from_pair(first, second)
    if isinstance(raw, dict):
        lat = _as_float(raw.get("lat", raw.get("latitude")))
        lon = _as_float(raw.get("lon", raw.get("lng", raw.get("longitude"))))
        nested = raw.get("location")
        if isinstance(nested, dict) and (lat is None or lon is None):
            lat = _as_float(nested.get("lat", nested.get("latitude")))
            lon = _as_float(nested.get("lon", nested.get("lng", nested.get("longitude"))))
        elif isinstance(nested, str) and (lat is None or lon is None):
            return _parse_coordinate_query(nested)
        elif isinstance(nested, (list, tuple)) and (lat is None or lon is None):
            return _parse_point(nested)
        if lat is None or lon is None:
            return None
        if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
            return None
        crs = None
        value = raw.get("crs", raw.get("datum"))
        if isinstance(value, str) and value.strip():
            crs = value.strip()
        elif isinstance(nested, dict):
            nested_crs = nested.get("crs", nested.get("datum"))
            if isinstance(nested_crs, str) and nested_crs.strip():
                crs = nested_crs.strip()
        lon, lat = to_wgs84(lon, lat, crs)
        return GeoPoint(lat=lat, lon=lon)
    return None

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
    return _point_from_pair(first, second)

def _point_from_pair(first: float, second: float) -> GeoPoint | None:
    if abs(first) > 90.0 and abs(second) <= 90.0:
        lon, lat = first, second
    elif abs(second) > 90.0 and abs(first) <= 90.0:
        lat, lon = first, second
    else:
        lon, lat = first, second
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return GeoPoint(lat=lat, lon=lon)

def _resolve_provider(ctx: RuntimeContext | None) -> RouteProvider:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("route_query_provider")
    if injected is not None:
        if not isinstance(injected, RouteProvider):
            raise EngineUnavailableError("route_query_provider 必须提供 route(request)")
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_API=false，禁止调用真实路线规划 API")
    name = _env_value("ROUTE_QUERY_PROVIDER", _PROVIDER_AMAP).lower()
    if name == _PROVIDER_OSRM:
        return _build_osrm_provider()
    if name != _PROVIDER_AMAP:
        raise EngineUnavailableError(f"未知 ROUTE_QUERY_PROVIDER: {name}")
    api_key = _amap_api_key()
    if not api_key:
        raise EngineUnavailableError("未配置 AMAP_WEB_KEY 或 AMAP_API_KEY")
    return AmapDirectionProvider(
        api_key=api_key,
        direction_endpoint=_env_value("AMAP_DIRECTION_ENDPOINT", _DEFAULT_AMAP_DIRECTION_ENDPOINT),
        bicycling_endpoint=_env_value(
            "AMAP_BICYCLING_ENDPOINT",
            _DEFAULT_AMAP_BICYCLING_ENDPOINT,
        ),
        timeout_sec=_env_timeout("AMAP_TIMEOUT_SEC"),
    )

def _build_osrm_provider() -> OsrmRouteProvider:
    endpoint = os.environ.get("ROUTE_OSRM_ENDPOINT", "").strip()
    if not endpoint:
        raise EngineUnavailableError("未配置 ROUTE_OSRM_ENDPOINT")
    if _is_public_osrm(endpoint) and not _allow_public_osrm():
        raise EngineUnavailableError("拒绝公共 OSRM 实例；请配置自建 ROUTE_OSRM_ENDPOINT")
    user_agent = (
        os.environ.get("ROUTE_OSRM_USER_AGENT", "").strip() or _DEFAULT_OSRM_UA
    )
    timeout_sec = _parse_timeout(
        os.environ.get("ROUTE_OSRM_TIMEOUT_SEC", "").strip(),
        _DEFAULT_TIMEOUT_SEC,
    )
    return OsrmRouteProvider(
        endpoint=endpoint,
        user_agent=user_agent,
        timeout_sec=timeout_sec,
    )

def _is_public_osrm(endpoint: str) -> bool:
    host = (urllib.parse.urlparse(endpoint).hostname or "").lower()
    return host in _PUBLIC_OSRM_HOSTS

def _allow_public_osrm() -> bool:
    raw = os.environ.get("ROUTE_ALLOW_PUBLIC_OSRM", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}

def _provider_name(provider: RouteProvider) -> str:
    return str(getattr(provider, "name", "injected"))

def _provider_crs(provider: RouteProvider) -> str:
    raw = getattr(provider, "crs", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if _provider_name(provider) == _PROVIDER_OSRM:
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
    return raw

def _append_query(endpoint: str, query: str) -> str:
    separator = "&" if urllib.parse.urlparse(endpoint).query else "?"
    return f"{endpoint}{separator}{query}"

def _normalize_results(
    payload: dict[str, Any],
    *,
    provider_name: str,
    crs: str,
    travel_mode: str,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    if _looks_injected(payload):
        hits: list[dict[str, Any]] = []
        for item in payload.get("routes") or []:
            if not isinstance(item, dict):
                continue
            hit = _normalize_injected(item, provider_name=provider_name, crs=crs)
            if hit is not None:
                hits.append(hit)
        return hits
    transits = _amap_transits(payload)
    if transits is not None:
        hits = []
        for item in transits:
            if not isinstance(item, dict):
                continue
            hit = _normalize_amap_transit(item, provider_name=provider_name, crs=crs)
            if hit is not None:
                hits.append(hit)
        return hits
    paths = _amap_paths(payload)
    if paths is not None:
        hits = []
        for item in paths:
            if not isinstance(item, dict):
                continue
            hit = _normalize_amap_path(
                item,
                provider_name=provider_name,
                crs=crs,
                travel_mode=travel_mode,
            )
            if hit is not None:
                hits.append(hit)
        return hits
    osrm_routes = payload.get("routes")
    if isinstance(osrm_routes, list) and str(payload.get("code") or "").lower() in {
        "ok",
        "noroute",
        "",
    }:
        hits = []
        for item in osrm_routes:
            if not isinstance(item, dict):
                continue
            hit = _normalize_osrm(item, provider_name=provider_name, crs=crs)
            if hit is not None:
                hits.append(hit)
        return hits
    return []

def _looks_injected(payload: dict[str, Any]) -> bool:
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes:
        return False
    first = routes[0]
    if not isinstance(first, dict):
        return False
    return "distance_m" in first or "source" in first

def _amap_paths(payload: dict[str, Any]) -> list[Any] | None:
    route = payload.get("route")
    if isinstance(route, dict) and isinstance(route.get("paths"), list):
        return route["paths"]
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("paths"), list):
        return data["paths"]
    return None

def _amap_transits(payload: dict[str, Any]) -> list[Any] | None:
    route = payload.get("route")
    if isinstance(route, dict) and isinstance(route.get("transits"), list):
        return route["transits"]
    return None

def _normalize_injected(
    item: dict[str, Any],
    *,
    provider_name: str,
    crs: str,
) -> dict[str, Any] | None:
    distance_m = _as_float(item.get("distance_m", item.get("distance")))
    if distance_m is None:
        return None
    hit: dict[str, Any] = {
        "distance_m": distance_m,
        "source": _amap_text(item.get("source")) or provider_name,
    }
    duration_s = _as_float(item.get("duration_s", item.get("duration")))
    if duration_s is not None:
        hit["duration_s"] = duration_s
    geometry = _geometry_from_value(item.get("geometry"), crs=crs)
    if geometry is not None:
        hit["geometry"] = geometry
    legs = _normalize_legs(item.get("legs"), crs=crs)
    if legs:
        hit["legs"] = legs
    return hit

def _normalize_amap_path(
    item: dict[str, Any],
    *,
    provider_name: str,
    crs: str,
    travel_mode: str,
) -> dict[str, Any] | None:
    distance_m = _as_float(item.get("distance"))
    if distance_m is None:
        return None
    hit: dict[str, Any] = {
        "distance_m": distance_m,
        "source": provider_name,
        "travel_mode": travel_mode,
    }
    duration_s = _as_float(item.get("duration"))
    if duration_s is not None:
        hit["duration_s"] = duration_s
    coords = _parse_amap_polyline(item.get("polyline"))
    if not coords:
        coords = _coords_from_amap_steps(item.get("steps"))
    if coords:
        hit["geometry"] = _linestring(coords, crs)
    legs = _legs_from_amap_steps(item.get("steps"))
    if legs:
        hit["legs"] = legs
    return hit

def _normalize_amap_transit(
    item: dict[str, Any],
    *,
    provider_name: str,
    crs: str,
) -> dict[str, Any] | None:
    distance_m = _as_float(item.get("distance"))
    if distance_m is None:
        return None
    hit: dict[str, Any] = {
        "distance_m": distance_m,
        "source": provider_name,
        "travel_mode": _MODE_TRANSIT,
    }
    duration_s = _as_float(item.get("duration"))
    if duration_s is not None:
        hit["duration_s"] = duration_s
    coords, legs = _transit_geometry_and_legs(item)
    if coords:
        hit["geometry"] = _linestring(coords, crs)
    if legs:
        hit["legs"] = legs
    return hit

def _normalize_osrm(
    item: dict[str, Any],
    *,
    provider_name: str,
    crs: str,
) -> dict[str, Any] | None:
    distance_m = _as_float(item.get("distance"))
    if distance_m is None:
        return None
    hit: dict[str, Any] = {
        "distance_m": distance_m,
        "source": provider_name,
    }
    duration_s = _as_float(item.get("duration"))
    if duration_s is not None:
        hit["duration_s"] = duration_s
    geometry = _geometry_from_value(item.get("geometry"), crs=crs)
    if geometry is not None:
        hit["geometry"] = geometry
    legs = _normalize_legs(item.get("legs"), crs=crs)
    if legs:
        hit["legs"] = legs
    return hit

def _legs_from_amap_steps(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    legs: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        distance_m = _as_float(item.get("distance"))
        if distance_m is None:
            continue
        leg: dict[str, Any] = {"distance_m": distance_m}
        duration_s = _as_float(item.get("duration"))
        if duration_s is not None:
            leg["duration_s"] = duration_s
        summary = _amap_text(item.get("road")) or _amap_text(item.get("instruction"))
        if summary:
            leg["summary"] = summary
        legs.append(leg)
    return legs

def _coords_from_amap_steps(raw: Any) -> list[list[float]]:
    if not isinstance(raw, list):
        return []
    coords: list[list[float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        part = _parse_amap_polyline(item.get("polyline"))
        coords = _extend_coords(coords, part)
    return coords

def _transit_geometry_and_legs(item: dict[str, Any]) -> tuple[list[list[float]], list[dict[str, Any]]]:
    coords: list[list[float]] = []
    legs: list[dict[str, Any]] = []
    segments = item.get("segments")
    if not isinstance(segments, list):
        return coords, legs
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        walking = segment.get("walking")
        if isinstance(walking, dict):
            coords = _extend_coords(coords, _coords_from_amap_steps(walking.get("steps")))
            distance_m = _as_float(walking.get("distance"))
            if distance_m is not None:
                leg: dict[str, Any] = {"distance_m": distance_m, "summary": "步行"}
                duration_s = _as_float(walking.get("duration"))
                if duration_s is not None:
                    leg["duration_s"] = duration_s
                legs.append(leg)
        bus = segment.get("bus")
        if isinstance(bus, dict):
            buslines = bus.get("buslines")
            if isinstance(buslines, list):
                for line in buslines:
                    if not isinstance(line, dict):
                        continue
                    coords = _extend_coords(coords, _parse_amap_polyline(line.get("polyline")))
                    distance_m = _as_float(line.get("distance"))
                    if distance_m is None:
                        continue
                    summary = _amap_text(line.get("name")) or "公交"
                    leg = {"distance_m": distance_m, "summary": summary}
                    duration_s = _as_float(line.get("duration"))
                    if duration_s is not None:
                        leg["duration_s"] = duration_s
                    legs.append(leg)
    return coords, legs

def _normalize_legs(raw: Any, *, crs: str) -> list[dict[str, Any]]:
    del crs
    if not isinstance(raw, list):
        return []
    legs: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        distance_m = _as_float(item.get("distance_m", item.get("distance")))
        if distance_m is None:
            continue
        leg: dict[str, Any] = {"distance_m": distance_m}
        duration_s = _as_float(item.get("duration_s", item.get("duration")))
        if duration_s is not None:
            leg["duration_s"] = duration_s
        summary = _amap_text(item.get("summary"))
        if summary:
            leg["summary"] = summary
        legs.append(leg)
    return legs

def _parse_amap_polyline(raw: Any) -> list[list[float]]:
    text = _amap_text(raw)
    if not text:
        return []
    coords: list[list[float]] = []
    for part in text.split(";"):
        chunk = part.strip()
        if not chunk:
            continue
        bits = chunk.split(",")
        if len(bits) != 2:
            continue
        lon = _as_float(bits[0])
        lat = _as_float(bits[1])
        if lon is None or lat is None:
            continue
        coords.append([lon, lat])
    return coords

def _geometry_from_value(raw: Any, *, crs: str) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        coords_raw = raw.get("coordinates")
        coords = _coords_from_pairs(coords_raw)
        if coords:
            return _linestring(coords, _amap_text(raw.get("crs")) or crs)
        return None
    if isinstance(raw, str):
        coords = _parse_amap_polyline(raw)
        if coords:
            return _linestring(coords, crs)
    if isinstance(raw, (list, tuple)):
        coords = _coords_from_pairs(raw)
        if coords:
            return _linestring(coords, crs)
    return None

def _coords_from_pairs(raw: Any) -> list[list[float]]:
    if not isinstance(raw, (list, tuple)):
        return []
    coords: list[list[float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        lon = _as_float(item[0])
        lat = _as_float(item[1])
        if lon is None or lat is None:
            continue
        coords.append([lon, lat])
    return coords

def _linestring(coords: list[list[float]], crs: str) -> dict[str, Any]:
    if is_gcj02(crs):
        converted = transform_coordinates(coords, from_crs=_CRS_GCJ02, to_crs=_CRS_WGS84)
        coords = [[float(pair[0]), float(pair[1])] for pair in converted]
    return {"type": "LineString", "coordinates": coords, "crs": _CRS_WGS84}

def _extend_coords(base: list[list[float]], extra: list[list[float]]) -> list[list[float]]:
    if not extra:
        return base
    if not base:
        return list(extra)
    if base[-1] == extra[0]:
        return base + extra[1:]
    return base + extra

def _simplify_geometry(raw: Any) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(raw, dict):
        return None, False
    coords = _coords_from_pairs(raw.get("coordinates"))
    if not coords:
        return None, False
    crs = _amap_text(raw.get("crs")) or _CRS_WGS84
    if len(coords) <= _MAX_GEOMETRY_POINTS:
        return _linestring(coords, crs), False
    last_index = len(coords) - 1
    sampled: list[list[float]] = []
    seen: set[int] = set()
    for i in range(_MAX_GEOMETRY_POINTS):
        idx = int(round(i * last_index / (_MAX_GEOMETRY_POINTS - 1)))
        if idx in seen:
            continue
        seen.add(idx)
        sampled.append(coords[idx])
    if sampled[-1] != coords[-1]:
        sampled.append(coords[-1])
    return _linestring(sampled, crs), True

def _straight_line_distance_m(
    origin: GeoPoint,
    destination: GeoPoint,
    waypoints: tuple[GeoPoint, ...],
) -> float:
    points = (origin, *waypoints, destination)
    total = 0.0
    for start, end in zip(points, points[1:]):
        total += _haversine_m(start, end)
    return total

def _haversine_m(start: GeoPoint, end: GeoPoint) -> float:
    phi1 = math.radians(start.lat)
    phi2 = math.radians(end.lat)
    dphi = math.radians(end.lat - start.lat)
    dlambda = math.radians(end.lon - start.lon)
    hav = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(hav)))

def _point_applied(point: GeoPoint, crs: str) -> dict[str, float | str]:
    return {"lon": point.lon, "lat": point.lat, "crs": crs}

def _format_amap_location(point: GeoPoint) -> str:
    return format_gcj02_lonlat(point.lon, point.lat)

def _format_location(point: GeoPoint) -> str:
    return f"{_fmt_coord(point.lon)},{_fmt_coord(point.lat)}"

def _fmt_coord(value: float) -> str:
    return f"{value:.6f}"

def _amap_text(raw: Any) -> str:
    if raw is None or isinstance(raw, bool):
        return ""
    if isinstance(raw, list):
        return ""
    if isinstance(raw, str):
        return raw.strip()
    return str(raw).strip()

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
