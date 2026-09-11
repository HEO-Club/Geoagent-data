"""flight_data_query 共享执行器：OpenSky REST 实际 ADS-B 档案与航迹。"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timezone
from typing import Any, Protocol, runtime_checkable

from tool._gate import allow_real_api as _allow_real_api, load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext, declared_inputs

_OP_SEARCH = "search"
_OP_TRACK = "track"
_OP_NEARBY = "nearby_traffic"
_PROVIDER_OPENSKY = "opensky"
_DATA_SOURCE = "adsb_reception"
_CRS_WGS84 = "wgs84"
_RECORD_FLIGHT = "actual_flight"
_RECORD_TRACK = "actual_track"
_RECORD_SNAPSHOT = "state_snapshot"
_SESSIONS_KEY = "flight_data_sessions"
_SESSION_COUNTER_KEY = "flight_data_session_counter"
_PROVIDER_EXTRAS_KEY = "flight_data_query_provider"
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_UA = "geoagent-dataset/1.0 (flight_data_query; local)"
_DEFAULT_ENDPOINT = "https://opensky-network.org/api"
_DEFAULT_TOKEN_ENDPOINT = (
    "https://auth.opensky-network.org/auth/realms/opensky-network"
    "/protocol/openid-connect/token"
)
_METERS_PER_DEG_LAT = 111_320.0
_MAX_FLIGHT_WINDOW_SEC = 2 * 24 * 3600
_STATE_LOOKBACK_AUTH_SEC = 3600
_TRACK_LOOKBACK_SEC = 30 * 24 * 3600
_LIVE_SLACK_SEC = 120
_TOKEN_REFRESH_MARGIN_SEC = 30
_SEARCH_FIELDS = ("area", "date", "airports", "route", "flight_number", "icao24")
_TRACK_FIELDS = ("area", "date", "flight_number", "airports", "icao24")
_NEARBY_FIELDS = ("area", "time_range", "radius_km")
_ICAO_RE = re.compile(r"^[A-Z]{4}$")
_ICAO24_RE = re.compile(r"^[0-9a-f]{6}$")
_COORD_SPLIT_RE = re.compile(r"[,;，、\s]+")
_ROUTE_SPLIT_RE = re.compile(r"\s*[-–—/→至到]+\s*")
_CALLSIGN_STRIP_RE = re.compile(r"[\s\-]+")
_TIME_RANGE_SPLIT_RE = re.compile(
    r"\s*(?:to|/|[–—至到]|\s-\s)\s*",
    re.IGNORECASE,
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
_BASE_ASSUMPTIONS = (
    "本步只返回 ADS-B 实际接收，不是计划航班",
    "缺测点未插值，不能当真实记录",
    "气压高度与几何高度不可混用",
    "到拍摄点的距离请另调 distance_bearing_calculator，并注明采样间隔与高度误差",
    "按当前 OpenSky REST 权限：匿名仅当前状态；认证状态向量最多回溯 1 小时；航迹最多 30 天",
)
_TOKEN_LOCK = threading.Lock()
_TOKEN_CACHE: dict[str, tuple[str, float]] = {}

class FlightInputError(Exception):
    """area / date / airports / route / flight_number 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """真实航迹服务未配置、被闸门拒绝、超 REST 权限或调用失败。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code

@dataclass(frozen=True)
class GeoPoint:
    """WGS84 点；本执行器不转换坐标系。"""

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
    """解析后的查询范围。"""

    text: str | None = None
    center: GeoPoint | None = None
    radius_m: int | None = None
    bbox: BBox | None = None
    applied: Any = None
    unsupported: Any = None

@dataclass(frozen=True)
class TimeWindow:
    """UTC 闭区间；点时刻时 start == end。"""

    start: datetime
    end: datetime

@dataclass(frozen=True)
class RouteAirports:
    """已解析的起降 ICAO。"""

    origin: str
    destination: str

@dataclass
class FlightSession:
    """一次查询回执，供会话引用。"""

    session_id: str
    operation: str
    result: dict[str, Any]

@runtime_checkable
class FlightArchiveProvider(Protocol):
    """可注入的航班档案后端；测试用 extras['flight_data_query_provider'] 替换。"""

    name: str

    def fetch_departures(self, airport: str, begin: int, end: int) -> list[dict[str, Any]]:
        """按起飞机场与 Unix 时间窗返回 OpenSky 风格航班列表。"""

    def fetch_arrivals(self, airport: str, begin: int, end: int) -> list[dict[str, Any]]:
        """按到达机场与 Unix 时间窗返回 OpenSky 风格航班列表。"""

    def fetch_states(self, *, bbox: BBox, time_unix: int | None) -> dict[str, Any]:
        """按 bbox 拉取状态向量；time_unix 为 None 表示当前。"""

    def fetch_track(self, *, icao24: str, time_unix: int) -> dict[str, Any]:
        """按 icao24 与时刻拉取航迹；空结果返回空对象。"""

class OpenSkyProvider:
    """OpenSky REST 适配器；密钥与端点只读环境变量。"""

    name = _PROVIDER_OPENSKY

    def __init__(
        self,
        *,
        endpoint: str,
        token_endpoint: str,
        timeout_sec: float,
        user_agent: str,
        client_id: str | None,
        client_secret: str | None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._token_endpoint = token_endpoint
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent
        self._client_id = client_id or None
        self._client_secret = client_secret or None
        self.authenticated = bool(self._client_id and self._client_secret)
        # 匿名只能查当前状态；认证用户状态向量最多 1 小时。
        self.max_state_lookback_sec: int | None = (
            _STATE_LOOKBACK_AUTH_SEC if self.authenticated else 0
        )
        self.max_track_lookback_sec: int | None = _TRACK_LOOKBACK_SEC

    def fetch_departures(self, airport: str, begin: int, end: int) -> list[dict[str, Any]]:
        payload = self._get(
            "/flights/departure",
            {"airport": airport, "begin": str(begin), "end": str(end)},
            empty_on_404="list",
        )
        return payload if isinstance(payload, list) else []

    def fetch_arrivals(self, airport: str, begin: int, end: int) -> list[dict[str, Any]]:
        payload = self._get(
            "/flights/arrival",
            {"airport": airport, "begin": str(begin), "end": str(end)},
            empty_on_404="list",
        )
        return payload if isinstance(payload, list) else []

    def fetch_states(self, *, bbox: BBox, time_unix: int | None) -> dict[str, Any]:
        params: dict[str, str] = {
            "lamin": f"{bbox.south:.6f}",
            "lomin": f"{bbox.west:.6f}",
            "lamax": f"{bbox.north:.6f}",
            "lomax": f"{bbox.east:.6f}",
        }
        if time_unix is not None:
            params["time"] = str(time_unix)
        payload = self._get("/states/all", params, empty_on_404="object")
        return payload if isinstance(payload, dict) else {"time": time_unix, "states": None}

    def fetch_track(self, *, icao24: str, time_unix: int) -> dict[str, Any]:
        payload = self._get(
            "/tracks/all",
            {"icao24": icao24, "time": str(time_unix)},
            empty_on_404="object",
        )
        return payload if isinstance(payload, dict) else {}

    def _get(
        self,
        path: str,
        params: dict[str, str],
        *,
        empty_on_404: str,
    ) -> Any:
        url = _append_query(f"{self._endpoint}{path}", urllib.parse.urlencode(params))
        headers = {
            "Accept": "application/json",
            "User-Agent": self._user_agent,
        }
        token = self._access_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return _http_json(
            url,
            headers=headers,
            timeout_sec=self._timeout_sec,
            error_prefix="OpenSky",
            empty_on_404=empty_on_404,
        )

    def _access_token(self) -> str | None:
        if not self.authenticated or self._client_id is None or self._client_secret is None:
            return None
        now = time.monotonic()
        with _TOKEN_LOCK:
            cached = _TOKEN_CACHE.get(self._client_id)
            if cached is not None and cached[1] > now:
                return cached[0]
        payload = _http_post_form(
            self._token_endpoint,
            {
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
            headers={
                "Accept": "application/json",
                "User-Agent": self._user_agent,
            },
            timeout_sec=self._timeout_sec,
            error_prefix="OpenSky OAuth",
        )
        if not isinstance(payload, dict):
            raise EngineUnavailableError("OpenSky OAuth 回执不是 JSON 对象")
        token = payload.get("access_token")
        if not isinstance(token, str) or not token.strip():
            raise EngineUnavailableError("OpenSky OAuth 未返回 access_token")
        expires = _as_float(payload.get("expires_in")) or 1800.0
        expiry = time.monotonic() + max(expires - _TOKEN_REFRESH_MARGIN_SEC, 1.0)
        with _TOKEN_LOCK:
            _TOKEN_CACHE[self._client_id] = (token, expiry)
        return token

def execute_search(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按日期、区域、机场或航线查询真实航班档案。"""

    return _execute(_OP_SEARCH, purpose=purpose, inputs=inputs, ctx=ctx)

def execute_track(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """查询指定航班或区域内的真实航迹。"""

    return _execute(_OP_TRACK, purpose=purpose, inputs=inputs, ctx=ctx)

def execute_nearby_traffic(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """统计指定时空范围内的航空器活动（状态向量快照）。"""

    return _execute(_OP_NEARBY, purpose=purpose, inputs=inputs, ctx=ctx)

def _execute(
    operation: str,
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    del purpose
    try:
        if operation == _OP_NEARBY:
            return _run_nearby(declared_inputs(inputs, *_NEARBY_FIELDS), ctx)
        if operation == _OP_TRACK:
            return _run_track(declared_inputs(inputs, *_TRACK_FIELDS), ctx)
        return _run_search(declared_inputs(inputs, *_SEARCH_FIELDS), ctx)
    except FlightInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def _run_search(inputs: dict[str, Any], ctx: RuntimeContext | None) -> Observation:
    selector = _parse_selector(inputs, ctx, require_any=True)
    if (selector.flight_number or selector.icao24) and not (
        selector.airports or selector.route or selector.location
    ):
        raise FlightInputError("航班号或 icao24 查询需要机场、航线或区域", "missing_input")
    provider = _resolve_provider(ctx)
    now = _utcnow()
    if selector.airports or selector.route:
        window = selector.window
        if window is None:
            raise FlightInputError("机场或航线查询需要 date", "missing_input")
        flights = _search_airport_flights(provider, selector, window)
        record_kind = _RECORD_FLIGHT
        payload_key = "flights"
        items: list[dict[str, Any]] = flights
    else:
        assert selector.location is not None
        snapshot = _snapshot_time(selector.window, now)
        time_unix = _state_time_param(provider, snapshot, now)
        raw_states = provider.fetch_states(bbox=selector.location.bbox, time_unix=time_unix)
        items = _normalize_states(raw_states, selector.flight_number)
        if selector.icao24:
            items = [item for item in items if item.get("icao24") == selector.icao24]
        record_kind = _RECORD_SNAPSHOT
        payload_key = "flights"
    return _finish(
        operation=_OP_SEARCH,
        record_kind=record_kind,
        body={payload_key: items},
        applied=_applied_selector(selector),
        ctx=ctx,
        extra_assumptions=(
            "est_departure_airport / est_arrival_airport 是 ADS-B 估计，不是计划时刻表",
            "航班号按 callsign 字面匹配，未做航司 IATA/ICAO 对照",
        ),
    )

def _run_track(inputs: dict[str, Any], ctx: RuntimeContext | None) -> Observation:
    selector = _parse_selector(inputs, ctx, require_any=True)
    provider = _resolve_provider(ctx)
    now = _utcnow()
    icao24 = selector.icao24
    tracks: list[dict[str, Any]] = []
    if icao24:
        when = _track_time(selector.window, now)
        _ensure_track_lookback(provider, when, now)
        tracks.append(_normalize_track(provider.fetch_track(icao24=icao24, time_unix=_unix(when))))
    elif selector.flight_number and (selector.airports or selector.route):
        window = selector.window
        if window is None:
            raise FlightInputError("按航班号查航迹需要 date", "missing_input")
        flights = _search_airport_flights(provider, selector, window)
        tracks.extend(_tracks_for_flights(provider, flights, window, now))
    elif selector.flight_number and selector.location is not None:
        snapshot = _snapshot_time(selector.window, now)
        time_unix = _state_time_param(provider, snapshot, now)
        states = _normalize_states(
            provider.fetch_states(bbox=selector.location.bbox, time_unix=time_unix),
            selector.flight_number,
        )
        tracks.extend(_tracks_for_states(provider, states, snapshot, now))
    elif selector.location is not None:
        snapshot = _snapshot_time(selector.window, now)
        time_unix = _state_time_param(provider, snapshot, now)
        aircraft = _normalize_states(
            provider.fetch_states(bbox=selector.location.bbox, time_unix=time_unix),
            None,
        )
        return _finish(
            operation=_OP_TRACK,
            record_kind=_RECORD_SNAPSHOT,
            body={"aircraft": aircraft, "count": len(aircraft)},
            applied=_applied_selector(selector),
            ctx=ctx,
            extra_assumptions=("仅区域、无航班标识时返回状态快照，不是连续航迹",),
        )
    else:
        raise FlightInputError("缺少 area、flight_number、airports 或 icao24", "missing_input")
    return _finish(
        operation=_OP_TRACK,
        record_kind=_RECORD_TRACK,
        body={"tracks": tracks},
        applied=_applied_selector(selector),
        ctx=ctx,
        extra_assumptions=(
            "OpenSky tracks 只有气压高度，geo_altitude_m 为 null",
            "航迹点均为接收记录，interpolated 恒为 false",
        ),
    )

def _run_nearby(inputs: dict[str, Any], ctx: RuntimeContext | None) -> Observation:
    if not _filled(inputs.get("area")):
        raise FlightInputError("缺少必填输入 area", "missing_input")
    if not _filled(inputs.get("time_range")):
        raise FlightInputError("缺少必填输入 time_range", "missing_input")
    location = _require_location(inputs.get("area"), ctx, radius_km=inputs.get("radius_km"))
    window = _require_time_window(inputs.get("time_range"), field="time_range")
    provider = _resolve_provider(ctx)
    now = _utcnow()
    snapshot = _snapshot_time(window, now)
    time_unix = _state_time_param(provider, snapshot, now)
    aircraft = _normalize_states(
        provider.fetch_states(bbox=location.bbox, time_unix=time_unix),
        None,
    )
    applied: dict[str, Any] = {
        "area": location.applied,
        "time_range": _window_applied(window),
        "snapshot_time": snapshot.isoformat().replace("+00:00", "Z"),
    }
    if inputs.get("radius_km") not in (None, ""):
        applied["radius_km"] = _as_float(inputs.get("radius_km"))
    return _finish(
        operation=_OP_NEARBY,
        record_kind=_RECORD_SNAPSHOT,
        body={"count": len(aircraft), "aircraft": aircraft, "time": applied["snapshot_time"]},
        applied=applied,
        ctx=ctx,
        extra_assumptions=("这是指定时刻的状态向量快照，不是时段内流量积分",),
        location_applied=location.applied,
    )

@dataclass
class _Selector:
    """search / track 共用的已解析选择条件。"""

    location: _ResolvedLocation | None = None
    window: TimeWindow | None = None
    airports: tuple[str, ...] = ()
    route: RouteAirports | None = None
    flight_number: str | None = None
    icao24: str | None = None
    applied: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class _ResolvedLocation:
    center: GeoPoint
    bbox: BBox
    applied: dict[str, Any]

def _parse_selector(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    require_any: bool,
) -> _Selector:
    location = None
    if _filled(inputs.get("area")):
        location = _require_location(inputs.get("area"), ctx, radius_km=None)
    airports = _parse_airports(inputs.get("airports"))
    route = _parse_route(inputs.get("route"))
    flight_number = _normalize_callsign(inputs.get("flight_number"))
    icao24 = _parse_icao24(inputs.get("icao24"))
    window = None
    if _filled(inputs.get("date")):
        window = _require_time_window(inputs.get("date"), field="date")
    if require_any and location is None and not airports and route is None and not flight_number and not icao24:
        raise FlightInputError(
            "缺少 area、airports、route、flight_number 或 icao24",
            "missing_input",
        )
    applied: dict[str, Any] = {}
    if location is not None:
        applied["area"] = location.applied
    if window is not None:
        applied["date"] = _window_applied(window)
    if airports:
        applied["airports"] = list(airports)
    if route is not None:
        applied["route"] = {"origin": route.origin, "destination": route.destination}
    if flight_number:
        applied["flight_number"] = flight_number
    if icao24:
        applied["icao24"] = icao24
    return _Selector(
        location=location,
        window=window,
        airports=airports,
        route=route,
        flight_number=flight_number,
        icao24=icao24,
        applied=applied,
    )

def _search_airport_flights(
    provider: FlightArchiveProvider,
    selector: _Selector,
    window: TimeWindow,
) -> list[dict[str, Any]]:
    begin, end = _flight_api_window(window)
    collected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    airports = list(selector.airports)
    if selector.route is not None:
        airports.extend([selector.route.origin, selector.route.destination])
    for airport in _unique(airports):
        for row in provider.fetch_departures(airport, begin, end):
            _append_flight(collected, seen, row, selector)
        for row in provider.fetch_arrivals(airport, begin, end):
            _append_flight(collected, seen, row, selector)
    if selector.route is not None:
        origin = selector.route.origin
        dest = selector.route.destination
        collected = [
            item
            for item in collected
            if item.get("est_departure_airport") == origin
            and item.get("est_arrival_airport") == dest
        ]
    return collected

def _append_flight(
    collected: list[dict[str, Any]],
    seen: set[tuple[str, int]],
    raw: dict[str, Any],
    selector: _Selector,
) -> None:
    item = _normalize_flight(raw)
    if selector.flight_number and _normalize_callsign(item.get("callsign")) != selector.flight_number:
        return
    if selector.icao24 and str(item.get("icao24") or "") != selector.icao24:
        return
    icao24 = str(item.get("icao24") or "")
    first_seen = int(item.get("first_seen_unix") or 0)
    key = (icao24, first_seen)
    if key in seen:
        return
    seen.add(key)
    collected.append(item)

def _tracks_for_flights(
    provider: FlightArchiveProvider,
    flights: list[dict[str, Any]],
    window: TimeWindow,
    now: datetime,
) -> list[dict[str, Any]]:
    when = _track_time(window, now)
    _ensure_track_lookback(provider, when, now)
    tracks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for flight in flights:
        icao24 = str(flight.get("icao24") or "")
        if not icao24 or icao24 in seen:
            continue
        seen.add(icao24)
        stamp = _as_int(flight.get("first_seen_unix")) or _unix(when)
        tracks.append(_normalize_track(provider.fetch_track(icao24=icao24, time_unix=stamp)))
    return tracks

def _tracks_for_states(
    provider: FlightArchiveProvider,
    states: list[dict[str, Any]],
    snapshot: datetime,
    now: datetime,
) -> list[dict[str, Any]]:
    _ensure_track_lookback(provider, snapshot, now)
    tracks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in states:
        icao24 = str(row.get("icao24") or "")
        if not icao24 or icao24 in seen:
            continue
        seen.add(icao24)
        stamp = _as_int(row.get("last_contact_unix")) or _unix(snapshot)
        tracks.append(_normalize_track(provider.fetch_track(icao24=icao24, time_unix=stamp)))
    return tracks

def _normalize_flight(raw: dict[str, Any]) -> dict[str, Any]:
    first_seen = _as_int(raw.get("firstSeen"))
    last_seen = _as_int(raw.get("lastSeen"))
    callsign = _optional_str(raw.get("callsign"))
    return _strip_forbidden(
        {
            "icao24": _optional_str(raw.get("icao24")),
            "callsign": callsign.strip() if callsign else None,
            "flight_number": _normalize_callsign(callsign),
            "first_seen": _iso_unix(first_seen),
            "last_seen": _iso_unix(last_seen),
            "first_seen_unix": first_seen,
            "last_seen_unix": last_seen,
            "est_departure_airport": _optional_str(raw.get("estDepartureAirport")),
            "est_arrival_airport": _optional_str(raw.get("estArrivalAirport")),
            "source": _PROVIDER_OPENSKY,
        }
    )

def _normalize_track(raw: dict[str, Any]) -> dict[str, Any]:
    path = raw.get("path") if isinstance(raw, dict) else None
    waypoints: list[dict[str, Any]] = []
    if isinstance(path, list):
        for point in path:
            parsed = _normalize_waypoint(point)
            if parsed is not None:
                waypoints.append(parsed)
    callsign = _optional_str(raw.get("callsign") or raw.get("calllsign"))
    return _strip_forbidden(
        {
            "icao24": _optional_str(raw.get("icao24")),
            "callsign": callsign.strip() if callsign else None,
            "start_time": _iso_unix(_as_int(raw.get("startTime"))),
            "end_time": _iso_unix(_as_int(raw.get("endTime"))),
            "waypoints": waypoints,
        }
    )

def _normalize_waypoint(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 6:
        return None
    lat = _as_float(raw[1])
    lon = _as_float(raw[2])
    if lat is None or lon is None:
        return None
    return {
        "time": _iso_unix(_as_int(raw[0])),
        "lat": lat,
        "lon": lon,
        "baro_altitude_m": _as_float(raw[3]),
        "geo_altitude_m": None,
        "true_track_deg": _as_float(raw[4]),
        "on_ground": bool(raw[5]),
        "interpolated": False,
    }

def _normalize_states(raw: dict[str, Any], flight_number: str | None) -> list[dict[str, Any]]:
    rows = raw.get("states") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return []
    wanted = _normalize_callsign(flight_number) if flight_number else None
    items: list[dict[str, Any]] = []
    for row in rows:
        item = _normalize_state_row(row)
        if item is None:
            continue
        if wanted and _normalize_callsign(item.get("callsign")) != wanted:
            continue
        items.append(item)
    return items

def _normalize_state_row(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 14:
        return None
    lat = _as_float(raw[6])
    lon = _as_float(raw[5])
    if lat is None or lon is None:
        return None
    callsign = _optional_str(raw[1])
    return _strip_forbidden(
        {
            "icao24": _optional_str(raw[0]),
            "callsign": callsign.strip() if callsign else None,
            "flight_number": _normalize_callsign(callsign),
            "lat": lat,
            "lon": lon,
            "baro_altitude_m": _as_float(raw[7]),
            "geo_altitude_m": _as_float(raw[13]),
            "on_ground": bool(raw[8]),
            "true_track_deg": _as_float(raw[10]) if len(raw) > 10 else None,
            "last_contact": _iso_unix(_as_int(raw[4])),
            "last_contact_unix": _as_int(raw[4]),
            "origin_country": _optional_str(raw[2]),
            "source": _PROVIDER_OPENSKY,
        }
    )

def _finish(
    *,
    operation: str,
    record_kind: str,
    body: dict[str, Any],
    applied: dict[str, Any],
    ctx: RuntimeContext | None,
    extra_assumptions: tuple[str, ...] = (),
    location_applied: dict[str, Any] | None = None,
) -> Observation:
    result_id = _new_session_id(ctx)
    result: dict[str, Any] = {
        "operation": operation,
        "result_id": result_id,
        "record_kind": record_kind,
        "provider": _PROVIDER_OPENSKY,
        "data_source": _DATA_SOURCE,
        "crs": _CRS_WGS84,
        "applied": applied,
        "assumptions": [*_BASE_ASSUMPTIONS, *extra_assumptions],
    }
    result.update(body)
    session = FlightSession(session_id=result_id, operation=operation, result=result)
    _store_session(ctx, session)
    if ctx is not None:
        ctx.previous_tool_result = result
        ctx.active_session = result_id
        if location_applied is not None:
            ctx.active_area = location_applied
        elif isinstance(applied.get("area"), dict):
            ctx.active_area = applied["area"]
    return Observation(ok=True, result=_strip_forbidden(result), session=result_id)

def _require_location(
    raw: Any,
    ctx: RuntimeContext | None,
    *,
    radius_km: Any,
) -> _ResolvedLocation:
    area = _parse_area(_resolve_area(raw, ctx))
    center = area.center
    bbox = area.bbox
    extra_radius = _parse_radius_km(radius_km)
    radius_m = area.radius_m
    if extra_radius is not None:
        radius_m = int(round(extra_radius * 1000.0))
    if center is not None and bbox is None:
        if radius_m is None:
            raise FlightInputError(
                "需要 bbox 或中心点加半径；纯地名请先 geocode",
                "missing_input",
            )
        bbox = _bbox_from_center(center, radius_m)
    if bbox is not None and extra_radius is not None:
        if center is None:
            center = GeoPoint(
                lat=(bbox.south + bbox.north) / 2.0,
                lon=(bbox.west + bbox.east) / 2.0,
            )
        bbox = _bbox_from_center(center, int(round(extra_radius * 1000.0)))
    if bbox is not None and center is None:
        center = GeoPoint(
            lat=(bbox.south + bbox.north) / 2.0,
            lon=(bbox.west + bbox.east) / 2.0,
        )
    if bbox is None or center is None:
        if area.text:
            raise FlightInputError(
                "需要 bbox 或中心点加半径；纯地名请先 geocode",
                "missing_input",
            )
        raise FlightInputError("area 无法解析为范围", "invalid_input")
    applied = {
        "west": bbox.west,
        "south": bbox.south,
        "east": bbox.east,
        "north": bbox.north,
        "crs": _CRS_WGS84,
    }
    if area.applied is not None and not isinstance(area.applied, dict):
        applied["text"] = area.applied
    return _ResolvedLocation(center=center, bbox=bbox, applied=applied)

def _resolve_area(raw: Any, ctx: RuntimeContext | None) -> Any:
    if raw is None or raw == "" or raw == "$active_area":
        return ctx.active_area if ctx is not None else None
    return raw

def _parse_area(raw: Any) -> ParsedArea:
    if raw is None or raw == "":
        return ParsedArea()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ParsedArea()
        point = _parse_coordinate_text(text)
        if point is not None:
            return ParsedArea(center=point, applied={"lat": point.lat, "lon": point.lon})
        return ParsedArea(text=text, applied=text)
    if isinstance(raw, (list, tuple)):
        if len(raw) == 4:
            bbox = _bbox_from_values(list(raw))
            if bbox is not None:
                return ParsedArea(bbox=bbox, applied=_bbox_applied(bbox))
        if len(raw) == 2:
            point = _parse_lonlat_pair(raw[0], raw[1])
            if point is not None:
                return ParsedArea(center=point, applied={"lat": point.lat, "lon": point.lon})
        return ParsedArea(unsupported=raw)
    if isinstance(raw, dict):
        bbox = _bbox_from_mapping(raw)
        if bbox is not None:
            return ParsedArea(bbox=bbox, applied=_bbox_applied(bbox))
        center = _center_from_mapping(raw)
        if center is not None:
            radius_raw = raw.get("radius_m", raw.get("radius", raw.get("radius_km")))
            radius: int | None = None
            if radius_raw not in (None, ""):
                parsed = _as_float(radius_raw)
                if parsed is None or parsed < 0:
                    raise FlightInputError("radius 必须是非负数", "invalid_input")
                if "radius_km" in raw and "radius_m" not in raw and "radius" not in raw:
                    parsed *= 1000.0
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

def _parse_airports(raw: Any) -> tuple[str, ...]:
    if not _filled(raw):
        return ()
    values: list[str] = []
    if isinstance(raw, str):
        values = [part.strip() for part in re.split(r"[,，、;/|]+", raw) if part.strip()] or [raw.strip()]
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise FlightInputError("airports 必须是 ICAO 四字码", "invalid_input")
            values.append(item.strip())
    else:
        raise FlightInputError("airports 必须是字符串或字符串列表", "invalid_input")
    return tuple(_require_icao(item) for item in values)

def _parse_route(raw: Any) -> RouteAirports | None:
    if not _filled(raw):
        return None
    if isinstance(raw, str):
        parts = [part.strip() for part in _ROUTE_SPLIT_RE.split(raw.strip()) if part.strip()]
        if len(parts) != 2:
            raise FlightInputError("route 必须是 origin-destination 的 ICAO 对", "invalid_input")
        return RouteAirports(origin=_require_icao(parts[0]), destination=_require_icao(parts[1]))
    if isinstance(raw, dict):
        origin = raw.get("origin") or raw.get("from") or raw.get("departure")
        destination = raw.get("destination") or raw.get("to") or raw.get("arrival")
        if not isinstance(origin, str) or not isinstance(destination, str):
            raise FlightInputError("route 必须包含 origin 与 destination", "invalid_input")
        return RouteAirports(origin=_require_icao(origin), destination=_require_icao(destination))
    raise FlightInputError("route 必须是字符串或对象", "invalid_input")

def _require_icao(raw: str) -> str:
    token = raw.strip().upper()
    if _ICAO_RE.fullmatch(token):
        return token
    if re.fullmatch(r"[A-Z]{3}", token):
        raise FlightInputError(
            f"机场 {raw} 是三字码；OpenSky 需要 ICAO 四字码，本工具不做 IATA 对照",
            "invalid_input",
        )
    raise FlightInputError(
        f"机场 {raw} 不是 ICAO 四字码；中文名或三字码请先查 ICAO",
        "invalid_input",
    )

def _parse_icao24(raw: Any) -> str | None:
    if not _filled(raw):
        return None
    if not isinstance(raw, str):
        raise FlightInputError("icao24 必须是 6 位十六进制", "invalid_input")
    token = raw.strip().lower()
    if not _ICAO24_RE.fullmatch(token):
        raise FlightInputError("icao24 必须是 6 位十六进制", "invalid_input")
    return token

def _parse_radius_km(raw: Any) -> float | None:
    if not _filled(raw):
        return None
    value = _as_float(raw)
    if value is None or value < 0:
        raise FlightInputError("radius_km 必须是非负数", "invalid_input")
    return value

def _require_time_window(raw: Any, *, field: str) -> TimeWindow:
    if not _filled(raw):
        raise FlightInputError(f"缺少必填输入 {field}", "missing_input")
    window = _parse_time_window(raw)
    if window is None:
        raise FlightInputError(f"{field} 无法解析为日期、时间或起止范围", "invalid_input")
    return window

def _parse_time_window(raw: Any) -> TimeWindow | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        stamp = datetime.fromtimestamp(float(raw), tz=timezone.utc)
        return TimeWindow(stamp, stamp)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        range_parts = _TIME_RANGE_SPLIT_RE.split(text, maxsplit=1)
        if len(range_parts) == 2 and range_parts[0] and range_parts[1]:
            start = _parse_datetime_bound(range_parts[0], start=True)
            end = _parse_datetime_bound(range_parts[1], start=False)
            if start is None or end is None:
                return None
            if end < start:
                start, end = end, start
            return TimeWindow(start, end)
        point = _parse_datetime_bound(text, start=True)
        if point is None:
            return None
        if "T" in text or " " in text:
            return TimeWindow(point, point)
        end = _parse_datetime_bound(text, start=False)
        return TimeWindow(point, end if end is not None else point)
    if isinstance(raw, dict):
        start = _parse_datetime_bound(
            raw.get("start") or raw.get("from") or raw.get("begin"),
            start=True,
        )
        end = _parse_datetime_bound(raw.get("end") or raw.get("to"), start=False)
        if start is None or end is None:
            return None
        if end < start:
            start, end = end, start
        return TimeWindow(start, end)
    return None

def _parse_datetime_bound(raw: Any, *, start: bool) -> datetime | None:
    if raw is None or raw == "" or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    if len(raw.strip()) <= 10:
        if start:
            return parsed.replace(hour=0, minute=0, second=0, microsecond=0)
        return parsed.replace(hour=23, minute=59, second=59, microsecond=0)
    return parsed

def _snapshot_time(window: TimeWindow | None, now: datetime) -> datetime:
    if window is None:
        return now
    end = window.end
    # 仅给到当天日期时，快照用当前时刻，避免把「今天」当成日末历史查询。
    if (
        window.start.date() == now.date()
        and end.date() == now.date()
        and window.start.timetz().replace(tzinfo=None) == dt_time.min
        and end.hour >= 23
    ):
        return now
    return end

def _track_time(window: TimeWindow | None, now: datetime) -> datetime:
    if window is None:
        return now
    return window.start if window.start == window.end else window.start

def _state_time_param(
    provider: FlightArchiveProvider,
    snapshot: datetime,
    now: datetime,
) -> int | None:
    lookback = getattr(provider, "max_state_lookback_sec", None)
    delta = (now - snapshot).total_seconds()
    if delta < -60:
        raise EngineUnavailableError("不能查询未来的航空状态向量")
    if lookback is None:
        return None if delta <= _LIVE_SLACK_SEC else _unix(snapshot)
    if lookback <= 0:
        if delta > _LIVE_SLACK_SEC:
            raise EngineUnavailableError(
                "匿名 OpenSky 只能查当前状态向量，不能改写成 live",
            )
        return None
    if delta > float(lookback):
        raise EngineUnavailableError(
            f"认证 OpenSky 状态向量最多回溯 {int(lookback)} 秒，不能改写成 live",
        )
    return None if delta <= _LIVE_SLACK_SEC else _unix(snapshot)

def _ensure_track_lookback(
    provider: FlightArchiveProvider,
    when: datetime,
    now: datetime,
) -> None:
    lookback = getattr(provider, "max_track_lookback_sec", None)
    if lookback is None:
        return
    delta = (now - when).total_seconds()
    if delta < -60:
        raise EngineUnavailableError("不能查询未来的航迹")
    if delta > float(lookback):
        raise EngineUnavailableError(
            "OpenSky 航迹最多 30 天，不能改写成 live",
        )

def _flight_api_window(window: TimeWindow) -> tuple[int, int]:
    begin = _unix(window.start)
    end = _unix(window.end)
    if end <= begin:
        end = begin + 3600
    if end - begin > _MAX_FLIGHT_WINDOW_SEC:
        raise EngineUnavailableError(
            "OpenSky 机场航班时间窗不能超过 2 天",
        )
    return begin, end

def _applied_selector(selector: _Selector) -> dict[str, Any]:
    return dict(selector.applied)

def _resolve_provider(ctx: RuntimeContext | None) -> FlightArchiveProvider:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get(_PROVIDER_EXTRAS_KEY)
    if injected is not None:
        if not isinstance(injected, FlightArchiveProvider):
            raise EngineUnavailableError(
                "flight_data_query_provider 必须提供 flights/states/track 接口",
            )
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_API=false，禁止调用真实航班档案 API")
    return OpenSkyProvider(
        endpoint=_env_value("OPENSKY_ENDPOINT", _DEFAULT_ENDPOINT),
        token_endpoint=_env_value("OPENSKY_TOKEN_ENDPOINT", _DEFAULT_TOKEN_ENDPOINT),
        timeout_sec=_env_timeout("OPENSKY_TIMEOUT_SEC"),
        user_agent=_env_value("OPENSKY_USER_AGENT", _DEFAULT_UA),
        client_id=_optional_env("OPENSKY_CLIENT_ID"),
        client_secret=_optional_env("OPENSKY_CLIENT_SECRET"),
    )

def _new_session_id(ctx: RuntimeContext | None) -> str:
    extras = _extras(ctx)
    counter = int(extras.get(_SESSION_COUNTER_KEY, 0)) + 1
    extras[_SESSION_COUNTER_KEY] = counter
    return f"flight_{counter}"

def _store_session(ctx: RuntimeContext | None, session: FlightSession) -> None:
    if ctx is None:
        return
    sessions = ctx.extras.setdefault(_SESSIONS_KEY, {})
    sessions[session.session_id] = session

def _extras(ctx: RuntimeContext | None) -> dict[str, Any]:
    if ctx is None:
        return {}
    return ctx.extras

def _bbox_from_center(center: GeoPoint, radius_m: int) -> BBox:
    dlat = radius_m / _METERS_PER_DEG_LAT
    cos_lat = math.cos(math.radians(center.lat))
    dlon = radius_m / (_METERS_PER_DEG_LAT * max(abs(cos_lat), 1e-6))
    return BBox(
        west=max(-180.0, center.lon - dlon),
        south=max(-90.0, center.lat - dlat),
        east=min(180.0, center.lon + dlon),
        north=min(90.0, center.lat + dlat),
    )

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

def _bbox_applied(bbox: BBox) -> dict[str, float | str]:
    return {
        "west": bbox.west,
        "south": bbox.south,
        "east": bbox.east,
        "north": bbox.north,
        "crs": _CRS_WGS84,
    }

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
    return GeoPoint(lat=lat, lon=lon)

def _parse_coordinate_text(text: str) -> GeoPoint | None:
    parts = [part for part in _COORD_SPLIT_RE.split(text.strip()) if part]
    if len(parts) != 2:
        return None
    return _parse_lonlat_pair(parts[0], parts[1])

def _parse_lonlat_pair(first_raw: Any, second_raw: Any) -> GeoPoint | None:
    first = _as_float(first_raw)
    second = _as_float(second_raw)
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

def _normalize_callsign(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    token = _CALLSIGN_STRIP_RE.sub("", raw.strip().upper())
    return token or None

def _window_applied(window: TimeWindow) -> dict[str, str]:
    return {
        "start": window.start.isoformat().replace("+00:00", "Z"),
        "end": window.end.isoformat().replace("+00:00", "Z"),
    }

def _iso_unix(stamp: int | None) -> str | None:
    if stamp is None:
        return None
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")

def _unix(value: datetime) -> int:
    return int(value.timestamp())

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        items.append(value)
    return items

def _filled(raw: Any) -> bool:
    return raw not in (None, "", [], {})

def _as_float(raw: Any) -> float | None:
    if isinstance(raw, bool) or raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except ValueError:
            return None
    return None

def _as_int(raw: Any) -> int | None:
    value = _as_float(raw)
    if value is None:
        return None
    return int(value)

def _optional_str(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    return text or None

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

def _env_value(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default

def _optional_env(name: str) -> str | None:
    raw = os.environ.get(name, "").strip()
    return raw or None

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

def _append_query(endpoint: str, query: str) -> str:
    separator = "&" if urllib.parse.urlparse(endpoint).query else "?"
    return f"{endpoint}{separator}{query}"

def _http_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout_sec: float,
    error_prefix: str,
    empty_on_404: str | None = None,
) -> Any:
    raw = _http_bytes(
        url,
        headers=headers,
        timeout_sec=timeout_sec,
        error_prefix=error_prefix,
        empty_on_404=empty_on_404,
    )
    if raw == b"" and empty_on_404 == "list":
        return []
    if raw == b"" and empty_on_404 == "object":
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EngineUnavailableError(f"{error_prefix} 回执不是 JSON: {exc}") from exc
    return payload

def _http_bytes(
    url: str,
    *,
    headers: dict[str, str],
    timeout_sec: float,
    error_prefix: str,
    empty_on_404: str | None = None,
) -> bytes:
    request = urllib.request.Request(url, method="GET")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        if empty_on_404 is not None and exc.code == 404:
            return b""
        detail = exc.read()[:200] if exc.fp else b""
        text = detail.decode("utf-8", errors="replace")
        raise EngineUnavailableError(f"{error_prefix} HTTP {exc.code}: {text}") from exc
    except urllib.error.URLError as exc:
        raise EngineUnavailableError(f"{error_prefix} 网络失败: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise EngineUnavailableError(f"{error_prefix} 调用失败: {exc}") from exc

def _http_post_form(
    url: str,
    data: dict[str, str],
    *,
    headers: dict[str, str],
    timeout_sec: float,
    error_prefix: str,
) -> Any:
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    request = urllib.request.Request(url, data=encoded, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = bytes(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200] if exc.fp else b""
        text = detail.decode("utf-8", errors="replace")
        raise EngineUnavailableError(f"{error_prefix} HTTP {exc.code}: {text}") from exc
    except urllib.error.URLError as exc:
        raise EngineUnavailableError(f"{error_prefix} 网络失败: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise EngineUnavailableError(f"{error_prefix} 调用失败: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EngineUnavailableError(f"{error_prefix} 回执不是 JSON: {exc}") from exc
