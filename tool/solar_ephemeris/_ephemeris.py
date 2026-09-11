"""solar_ephemeris 共享执行器：pvlib NREL SPA 本地太阳位置与日出日没。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Any, Protocol, runtime_checkable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tool.contract import Observation, RuntimeContext, declared_inputs

_OP_POSITION = "sun_position"
_OP_SUNSET = "sunset_time"
_METHOD = "nrel_spa"
_LIBRARY = "pvlib"
_CRS_WGS84 = "wgs84"
_PREVIOUS = "$previous_tool_result"
_ACTIVE_AREA = "$active_area"
_ENGINE_KEY = "solar_ephemeris_engine"
_MAX_RANGE_DAYS = 31
_SUN_HORIZON_DEG = -0.833
_CIVIL_DEG = -6.0
_NAUTICAL_DEG = -12.0
_ASTRONOMICAL_DEG = -18.0
_TWILIGHT_FREQ = "2min"
_COORD_SPLIT_RE = re.compile(r"[,;，、\s]+")
_YEAR_RE = re.compile(r"^(\d{4})$")
_YEAR_RANGE_RE = re.compile(r"^(\d{4})\s*[-–—至到]\s*(\d{4})$")
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
_DATE_RANGE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s*(?:to|/|[-–—至到])\s*(\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)
_HAS_TIME_RE = re.compile(r"[T\s]\d{1,2}:\d{2}")
_OFFSET_RE = re.compile(
    r"^(?:utc|gmt)?\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?$",
    re.IGNORECASE,
)
_AMBIGUOUS_TZ = frozenset(
    {"cst", "cdt", "est", "edt", "mst", "mdt", "pst", "pdt", "bst", "ist"}
)
_EARLIEST_TOKENS = frozenset({"最早可用", "earliest", "earliest available", "latest"})
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
_ASSUMPTION_METHOD = "太阳位置与日出日没按 NREL SPA（pvlib nrel_numpy）本地计算，未调用在线历算 API"
_ASSUMPTION_AZIMUTH = "方位角以真北为 0°、顺时针增加（东 90°）"
_ASSUMPTION_TZ = "未把本地时当成 UTC，也未把带 Z 的时刻当成本地挂钟时间"
_ASSUMPTION_CANDIDATE = "计算结果以给定地点与时刻为条件，不表示该地点或拍摄时间已被确认"
_ASSUMPTION_ALTITUDE = "海拔按 0 m，未做地形遮挡改正"
_ASSUMPTION_CENTROID = "太阳位置按范围质心计算，不是区域内每点"
_ASSUMPTION_COORD = "无法唯一判定时，坐标串按 lon,lat 解析"
_ASSUMPTION_NO_GEOCODE = "本执行器不解析纯地名，需要真实坐标；纯地名请先 geocode"
_ASSUMPTION_HORIZON = "日出日没用地平线 -0.833°（含平均大气折射）；暮光用视高度角 -6/-12/-18°"
_ASSUMPTION_POLAR = "极昼或极夜时对应日出日没与暮光为 null，不是缺测"

_TZ_FINDER: Any = None


class EphemerisInputError(Exception):
    """area / locations / datetime / time_range / timezone 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class EngineUnavailableError(Exception):
    """pvlib 不可用或历算失败。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class GeoPoint:
    """WGS84 点；label 只回显输入标签，不表示地点已确认。"""

    lon: float
    lat: float
    label: str | None = None
    crs: str = _CRS_WGS84
    from_centroid: bool = False


@dataclass(frozen=True)
class ResolvedZone:
    """输出时区；key 为 IANA 名或 ±HH:MM 偏移。"""

    key: str
    tz: tzinfo
    source: str


@dataclass(frozen=True)
class SunPosition:
    """某一瞬间的太阳角度，单位度。"""

    elevation_deg: float
    apparent_elevation_deg: float
    azimuth_deg: float
    zenith_deg: float


@dataclass(frozen=True)
class SunTimes:
    """某一本地历日的日出日没与暮光；极昼极夜字段为 None。"""

    sunrise: datetime | None
    sunset: datetime | None
    transit: datetime | None
    civil_dawn: datetime | None
    civil_dusk: datetime | None
    nautical_dawn: datetime | None
    nautical_dusk: datetime | None
    astronomical_dawn: datetime | None
    astronomical_dusk: datetime | None
    polar: str | None


@runtime_checkable
class SolarEngine(Protocol):
    """太阳历算引擎；默认 pvlib，测试可注入。"""

    def sun_position(
        self,
        lat: float,
        lon: float,
        when: datetime,
        altitude_m: float,
    ) -> SunPosition:
        """计算 timezone-aware 瞬间的太阳高度角与方位角。"""

    def sun_times(
        self,
        lat: float,
        lon: float,
        day: date,
        tz: tzinfo,
        altitude_m: float,
    ) -> SunTimes:
        """计算指定本地日期的日出、日没、正午与暮光。"""


class PvlibEngine:
    """用 pvlib NREL SPA 做本地历算。"""

    def sun_position(
        self,
        lat: float,
        lon: float,
        when: datetime,
        altitude_m: float,
    ) -> SunPosition:
        pd, get_solarposition, _spa = _require_pvlib()
        if when.tzinfo is None:
            raise EngineUnavailableError("太阳位置计算需要带时区的时刻")
        index = pd.DatetimeIndex([pd.Timestamp(when)])
        frame = get_solarposition(
            index,
            latitude=lat,
            longitude=lon,
            altitude=altitude_m,
            method="nrel_numpy",
        )
        row = frame.iloc[0]
        return SunPosition(
            elevation_deg=_finite_angle(row["elevation"], field="elevation"),
            apparent_elevation_deg=_finite_angle(
                row["apparent_elevation"], field="apparent_elevation"
            ),
            azimuth_deg=_norm_azimuth(_finite_angle(row["azimuth"], field="azimuth")),
            zenith_deg=_finite_angle(row["zenith"], field="zenith"),
        )

    def sun_times(
        self,
        lat: float,
        lon: float,
        day: date,
        tz: tzinfo,
        altitude_m: float,
    ) -> SunTimes:
        pd, get_solarposition, sun_rise_set_transit_spa = _require_pvlib()
        noon = pd.Timestamp(
            year=day.year,
            month=day.month,
            day=day.day,
            hour=12,
            tz=tz,
        )
        spa = sun_rise_set_transit_spa(
            pd.DatetimeIndex([noon]),
            latitude=lat,
            longitude=lon,
        )
        row = spa.iloc[0]
        sunrise = _timestamp_to_datetime(row["sunrise"], tz)
        sunset = _timestamp_to_datetime(row["sunset"], tz)
        transit = _timestamp_to_datetime(row["transit"], tz)
        polar = _polar_flag(sunrise, sunset)
        twilight = _twilight_times(
            get_solarposition,
            pd,
            lat=lat,
            lon=lon,
            day=day,
            tz=tz,
            altitude_m=altitude_m,
        )
        return SunTimes(
            sunrise=sunrise,
            sunset=sunset,
            transit=transit,
            civil_dawn=twilight["civil_dawn"],
            civil_dusk=twilight["civil_dusk"],
            nautical_dawn=twilight["nautical_dawn"],
            nautical_dusk=twilight["nautical_dusk"],
            astronomical_dawn=twilight["astronomical_dawn"],
            astronomical_dusk=twilight["astronomical_dusk"],
            polar=polar,
        )


def execute_sun_position(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按地点与时刻计算太阳高度角和方位角。"""

    del purpose
    try:
        altitude_m = _parse_altitude(inputs.get("altitude_m"))
        hypothesis = _hypothesis_flag(inputs, multi=False)
        raw_area = inputs.get("area")
        declared = declared_inputs(inputs, "area", "datetime", "timezone")
        site = _parse_site(declared.get("area", raw_area), ctx, field="area")
        when_raw = declared.get("datetime")
        if when_raw in (None, ""):
            raise EphemerisInputError("缺少必填输入 datetime", "missing_input")
        zone = _resolve_timezone(declared.get("timezone"), site)
        when = _parse_datetime(when_raw, zone)
        engine = _resolve_engine(ctx)
        position = engine.sun_position(site.lat, site.lon, when, altitude_m)
        applied = _site_applied(site, raw_area)
        applied.update(
            {
                "altitude_m": altitude_m,
                "datetime_local": _iso(when.astimezone(zone.tz)),
                "datetime_utc": _iso(when.astimezone(timezone.utc)),
                "timezone": zone.key,
                "timezone_source": zone.source,
            }
        )
        payload: dict[str, Any] = {
            "operation": _OP_POSITION,
            "elevation_deg": position.elevation_deg,
            "apparent_elevation_deg": position.apparent_elevation_deg,
            "azimuth_deg": position.azimuth_deg,
            "zenith_deg": position.zenith_deg,
            "unit": "deg",
            "method": _METHOD,
            "library": _LIBRARY,
            "status": "computed_under_inputs",
            "condition": {
                "lat": site.lat,
                "lon": site.lon,
                "datetime_local": applied["datetime_local"],
                "timezone": zone.key,
            },
            "applied": applied,
            "assumptions": _assumptions(
                altitude_m=altitude_m,
                from_centroid=site.from_centroid,
                polar=False,
            ),
        }
        if hypothesis:
            payload["hypothesis"] = True
        return _ok(payload)
    except EphemerisInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)


def execute_sunset_time(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按一个或多个地点与日期计算日落和暮光时间。"""

    del purpose
    try:
        altitude_m = _parse_altitude(inputs.get("altitude_m"))
        raw_locations = inputs.get("locations")
        declared = declared_inputs(inputs, "locations", "time_range", "timezone")
        sites = _parse_locations(declared.get("locations", raw_locations), ctx)
        days = _parse_time_range(declared.get("time_range"))
        hypothesis = _hypothesis_flag(
            inputs,
            multi=len(sites) > 1 or len(days) > 1,
        )
        engine = _resolve_engine(ctx)
        results: list[dict[str, Any]] = []
        polar_any = False
        applied_sites: list[dict[str, Any]] = []
        zone_keys: list[str] = []
        for site in sites:
            zone = _resolve_timezone(declared.get("timezone"), site)
            zone_keys.append(zone.key)
            applied_sites.append(_point_payload(site))
            for day in days:
                times = engine.sun_times(site.lat, site.lon, day, zone.tz, altitude_m)
                if times.polar:
                    polar_any = True
                item: dict[str, Any] = {
                    "location": _point_payload(site),
                    "date": day.isoformat(),
                    "timezone": zone.key,
                    "sunrise": _iso(times.sunrise),
                    "sunset": _iso(times.sunset),
                    "solar_noon": _iso(times.transit),
                    "civil_dawn": _iso(times.civil_dawn),
                    "civil_dusk": _iso(times.civil_dusk),
                    "nautical_dawn": _iso(times.nautical_dawn),
                    "nautical_dusk": _iso(times.nautical_dusk),
                    "astronomical_dawn": _iso(times.astronomical_dawn),
                    "astronomical_dusk": _iso(times.astronomical_dusk),
                    "status": "computed_under_inputs",
                    "condition": {
                        "lat": site.lat,
                        "lon": site.lon,
                        "date": day.isoformat(),
                        "timezone": zone.key,
                    },
                }
                if times.polar:
                    item["polar"] = times.polar
                if hypothesis:
                    item["hypothesis"] = True
                results.append(item)
        unique_zones = list(dict.fromkeys(zone_keys))
        applied: dict[str, Any] = {
            "locations": applied_sites,
            "dates": [item.isoformat() for item in days],
            "altitude_m": altitude_m,
            "timezone": unique_zones[0] if len(unique_zones) == 1 else unique_zones,
            "timezone_source": "input" if declared.get("timezone") not in (None, "") else "inferred",
        }
        payload: dict[str, Any] = {
            "operation": _OP_SUNSET,
            "results": results,
            "method": _METHOD,
            "library": _LIBRARY,
            "status": "computed_under_inputs",
            "applied": applied,
            "assumptions": _assumptions(
                altitude_m=altitude_m,
                from_centroid=any(site.from_centroid for site in sites),
                polar=polar_any,
            ),
        }
        if hypothesis:
            payload["hypothesis"] = True
        return _ok(payload)
    except EphemerisInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)


def _resolve_engine(ctx: RuntimeContext | None) -> SolarEngine:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get(_ENGINE_KEY)
    if injected is not None:
        return injected
    return PvlibEngine()


def _require_pvlib() -> tuple[Any, Any, Any]:
    try:
        import pandas as pd
        from pvlib.solarposition import get_solarposition, sun_rise_set_transit_spa
    except ImportError as exc:
        raise EngineUnavailableError("solar_ephemeris 需要 pvlib 与 pandas") from exc
    return pd, get_solarposition, sun_rise_set_transit_spa


def _twilight_times(
    get_solarposition: Any,
    pd: Any,
    *,
    lat: float,
    lon: float,
    day: date,
    tz: tzinfo,
    altitude_m: float,
) -> dict[str, datetime | None]:
    start = pd.Timestamp(year=day.year, month=day.month, day=day.day, tz=tz)
    end = start + pd.Timedelta(days=1)
    index = pd.date_range(start, end, freq=_TWILIGHT_FREQ, inclusive="left")
    frame = get_solarposition(
        index,
        latitude=lat,
        longitude=lon,
        altitude=altitude_m,
        method="nrel_numpy",
    )
    elev = [float(value) for value in frame["apparent_elevation"].to_numpy()]
    stamps = [_timestamp_to_datetime(item, tz) for item in index]
    times = [item for item in stamps if item is not None]
    if len(times) != len(elev):
        return _empty_twilight()
    return {
        "civil_dawn": _crossing(times, elev, _CIVIL_DEG, rising=True),
        "civil_dusk": _crossing(times, elev, _CIVIL_DEG, rising=False),
        "nautical_dawn": _crossing(times, elev, _NAUTICAL_DEG, rising=True),
        "nautical_dusk": _crossing(times, elev, _NAUTICAL_DEG, rising=False),
        "astronomical_dawn": _crossing(times, elev, _ASTRONOMICAL_DEG, rising=True),
        "astronomical_dusk": _crossing(times, elev, _ASTRONOMICAL_DEG, rising=False),
    }


def _empty_twilight() -> dict[str, datetime | None]:
    return {
        "civil_dawn": None,
        "civil_dusk": None,
        "nautical_dawn": None,
        "nautical_dusk": None,
        "astronomical_dawn": None,
        "astronomical_dusk": None,
    }


def _crossing(
    times: list[datetime],
    elev: list[float],
    target: float,
    *,
    rising: bool,
) -> datetime | None:
    if len(times) < 2:
        return None
    hits: list[datetime] = []
    for index in range(len(times) - 1):
        left = elev[index]
        right = elev[index + 1]
        if not math.isfinite(left) or not math.isfinite(right):
            continue
        crossed = (left < target <= right) if rising else (left > target >= right)
        if not crossed:
            continue
        hits.append(_interpolate_time(times[index], times[index + 1], left, right, target))
    if not hits:
        return None
    return hits[0] if rising else hits[-1]


def _interpolate_time(
    start: datetime,
    end: datetime,
    left: float,
    right: float,
    target: float,
) -> datetime:
    span = (end - start).total_seconds()
    if span <= 0 or right == left:
        return start
    frac = (target - left) / (right - left)
    frac = min(1.0, max(0.0, frac))
    return start + timedelta(seconds=span * frac)


def _parse_site(raw: Any, ctx: RuntimeContext | None, *, field: str) -> GeoPoint:
    resolved = _resolve_area(raw, ctx)
    if resolved in (None, ""):
        raise EphemerisInputError(f"缺少必填输入 {field}", "missing_input")
    if isinstance(resolved, str):
        resolved = _maybe_json(resolved)
    sites = _collect_sites(resolved)
    if len(sites) == 1:
        return sites[0]
    if len(sites) > 1:
        return _centroid(sites, label=_label_of(resolved))
    if _looks_like_place_name(raw) or _looks_like_place_name(resolved):
        raise EphemerisInputError(
            f"{field} 必须是真实坐标，纯地名请先 geocode",
            "missing_input",
        )
    raise EphemerisInputError(f"{field} 无法解析为坐标", "invalid_input")


def _parse_locations(raw: Any, ctx: RuntimeContext | None) -> list[GeoPoint]:
    if raw in (None, ""):
        raise EphemerisInputError("缺少必填输入 locations", "missing_input")
    resolved = _resolve_ref(raw, ctx, field="locations")
    if isinstance(resolved, str):
        resolved = _maybe_json(resolved)
    sites = _collect_sites(resolved)
    if sites:
        return sites
    if _contains_place_name(raw) or _contains_place_name(resolved):
        raise EphemerisInputError(
            "locations 必须是真实坐标，纯地名请先 geocode",
            "missing_input",
        )
    raise EphemerisInputError("locations 无法解析为坐标", "invalid_input")


def _collect_sites(raw: Any) -> list[GeoPoint]:
    found: list[GeoPoint] = []
    _walk_sites(raw, found, depth=0)
    return found


def _walk_sites(raw: Any, found: list[GeoPoint], *, depth: int) -> None:
    if depth > 6 or raw is None or raw == "":
        return
    if isinstance(raw, GeoPoint):
        found.append(raw)
        return
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return
        parsed = _maybe_json(text)
        if parsed is not text:
            _walk_sites(parsed, found, depth=depth + 1)
            return
        point = _parse_coordinate_text(text)
        if point is not None:
            found.append(point)
        return
    if isinstance(raw, (list, tuple)):
        if _is_bbox(raw):
            bbox = _bbox_from_values(list(raw))
            if bbox is not None:
                found.append(_bbox_centroid(bbox))
            return
        if _is_coord_pair(raw):
            point = _point_from_pair(raw[0], raw[1])
            if point is not None:
                found.append(point)
            return
        if raw and all(isinstance(item, (list, tuple)) and _is_coord_pair(item) for item in raw):
            if len(raw) >= 3:
                polygon = [_point_from_pair(item[0], item[1]) for item in raw]
                points = [item for item in polygon if item is not None]
                if len(points) >= 3:
                    found.append(_centroid(points))
                    return
        for item in raw:
            _walk_sites(item, found, depth=depth + 1)
        return
    if not isinstance(raw, dict):
        return
    for key in ("results", "hits", "features", "elements", "locations"):
        nested = raw.get(key)
        if isinstance(nested, list) and nested:
            before = len(found)
            for item in nested:
                _walk_sites(item, found, depth=depth + 1)
            if len(found) > before:
                return
    geom = raw.get("geometry") if raw.get("type") == "Feature" else raw
    if isinstance(geom, dict) and geom.get("type") == "Point":
        point = _parse_point(geom.get("coordinates"))
        if point is not None:
            label = _label_of(raw)
            found.append(
                GeoPoint(lon=point.lon, lat=point.lat, label=label, crs=point.crs)
            )
            return
    if isinstance(geom, dict) and geom.get("type") in {"Polygon", "MultiPolygon"}:
        coords = geom.get("coordinates")
        ring = _first_ring(coords)
        if ring is not None:
            found.append(_centroid(ring, label=_label_of(raw)))
            return
    bbox = _bbox_from_mapping(raw)
    if bbox is not None:
        found.append(_bbox_centroid(bbox, label=_label_of(raw)))
        return
    center = _center_from_mapping(raw)
    if center is not None:
        found.append(
            GeoPoint(
                lon=center.lon,
                lat=center.lat,
                label=_label_of(raw),
                crs=center.crs,
            )
        )
        return
    for key in ("location", "center", "point", "area"):
        if key in raw:
            _walk_sites(raw.get(key), found, depth=depth + 1)
            if found:
                return


def _parse_time_range(raw: Any) -> list[date]:
    if raw in (None, ""):
        raise EphemerisInputError("缺少必填输入 time_range", "missing_input")
    if isinstance(raw, bool):
        raise EphemerisInputError("time_range 无法解析为日期", "invalid_input")
    if isinstance(raw, int):
        raise EphemerisInputError(
            "time_range 过宽，请提供具体日期或不超过 31 天的日期范围",
            "missing_input",
        )
    if isinstance(raw, datetime):
        return [raw.date()]
    if isinstance(raw, date):
        return [raw]
    if isinstance(raw, dict):
        start = _bound_to_date(raw.get("start") or raw.get("from") or raw.get("begin") or raw.get("date"))
        end_raw = raw.get("end") or raw.get("to")
        end = _bound_to_date(end_raw) if end_raw not in (None, "") else start
        if start is None:
            raise EphemerisInputError("time_range 无法解析为日期", "invalid_input")
        return _days_between(start, end or start)
    if not isinstance(raw, str):
        raise EphemerisInputError("time_range 必须是字符串或对象", "invalid_input")
    text = raw.strip()
    if not text:
        raise EphemerisInputError("缺少必填输入 time_range", "missing_input")
    if text.lower() in _EARLIEST_TOKENS or text in _EARLIEST_TOKENS:
        raise EphemerisInputError(
            "time_range 无法对应具体历日，请提供具体日期",
            "missing_input",
        )
    if _YEAR_RE.fullmatch(text) or _YEAR_RANGE_RE.fullmatch(text):
        raise EphemerisInputError(
            "time_range 过宽，请提供具体日期或不超过 31 天的日期范围",
            "missing_input",
        )
    day = _DATE_RE.fullmatch(text)
    if day:
        parsed = _parse_iso_date(day.group(1))
        if parsed is None:
            raise EphemerisInputError("time_range 无法解析为日期", "invalid_input")
        return [parsed]
    compact = re.sub(r"\s+", "", text)
    date_range = _DATE_RANGE_RE.fullmatch(compact)
    if date_range:
        start = _parse_iso_date(date_range.group(1))
        end = _parse_iso_date(date_range.group(2))
        if start is None or end is None:
            raise EphemerisInputError("time_range 无法解析为日期", "invalid_input")
        return _days_between(start, end)
    extracted = _parse_iso_date(text[:10]) if len(text) >= 10 else None
    if extracted is not None and (extracted.isoformat() == text[:10]):
        if _HAS_TIME_RE.search(text) or _DATE_RE.fullmatch(text[:10]):
            return [extracted]
    raise EphemerisInputError("time_range 无法解析为日期", "invalid_input")


def _days_between(start: date, end: date) -> list[date]:
    lo, hi = (start, end) if start <= end else (end, start)
    span = (hi - lo).days + 1
    if span > _MAX_RANGE_DAYS:
        raise EphemerisInputError(
            "time_range 过宽，请提供具体日期或不超过 31 天的日期范围",
            "missing_input",
        )
    return [lo + timedelta(days=offset) for offset in range(span)]


def _bound_to_date(raw: Any) -> date | None:
    if isinstance(raw, bool) or raw in (None, ""):
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, int):
        raise EphemerisInputError(
            "time_range 过宽，请提供具体日期或不超过 31 天的日期范围",
            "missing_input",
        )
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if _YEAR_RE.fullmatch(text):
        raise EphemerisInputError(
            "time_range 过宽，请提供具体日期或不超过 31 天的日期范围",
            "missing_input",
        )
    return _parse_iso_date(text[:10]) if len(text) >= 10 else _parse_iso_date(text)


def _parse_iso_date(text: str) -> date | None:
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_datetime(raw: Any, zone: ResolvedZone) -> datetime:
    if isinstance(raw, datetime):
        parsed = raw
        source = raw.isoformat()
    elif isinstance(raw, str) and raw.strip():
        source = raw.strip()
        parsed = _datetime_from_text(source)
    else:
        raise EphemerisInputError("datetime 必须是带时刻的字符串", "invalid_input")
    if parsed.tzinfo is None and not _HAS_TIME_RE.search(source):
        if parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0:
            raise EphemerisInputError(
                "datetime 必须包含时刻，仅日期无法计算太阳位置",
                "missing_input",
            )
    if parsed.tzinfo is None:
        try:
            return parsed.replace(tzinfo=zone.tz)
        except Exception as exc:
            raise EphemerisInputError("datetime 无法按给定时区本地化", "invalid_timezone") from exc
    return parsed


def _datetime_from_text(text: str) -> datetime:
    candidate = text.replace("Z", "+00:00").replace("z", "+00:00")
    candidate = candidate.replace(" ", "T", 1) if "T" not in candidate[:20] else candidate
    try:
        return datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise EphemerisInputError("datetime 无法解析为时刻", "invalid_input") from exc


def _resolve_timezone(raw: Any, site: GeoPoint) -> ResolvedZone:
    if raw not in (None, ""):
        if not isinstance(raw, str):
            raise EphemerisInputError("timezone 必须是 IANA 名称或 UTC 偏移", "invalid_timezone")
        return _parse_timezone(raw)
    inferred = _infer_timezone(site)
    if inferred is None:
        raise EphemerisInputError(
            "缺少 timezone，且无法从坐标推断当地时区",
            "missing_input",
        )
    return inferred


def _parse_timezone(raw: str) -> ResolvedZone:
    text = raw.strip()
    if not text:
        raise EphemerisInputError("timezone 必须是 IANA 名称或 UTC 偏移", "invalid_timezone")
    lowered = text.lower()
    if lowered in _AMBIGUOUS_TZ:
        raise EphemerisInputError(
            "timezone 缩写有歧义，请使用 IANA 名称或 UTC 偏移",
            "invalid_timezone",
        )
    if lowered in {"utc", "gmt", "z", "etc/utc", "etc/gmt"}:
        return ResolvedZone(key="UTC", tz=timezone.utc, source="input")
    offset = _OFFSET_RE.fullmatch(text.replace(" ", ""))
    if offset:
        sign = 1 if offset.group(1) == "+" else -1
        hours = int(offset.group(2))
        minutes = int(offset.group(3) or "0")
        if hours > 14 or minutes > 59:
            raise EphemerisInputError("timezone 偏移超出范围", "invalid_timezone")
        delta = timedelta(hours=sign * hours, minutes=sign * minutes)
        key = _offset_key(delta)
        return ResolvedZone(key=key, tz=timezone(delta), source="input")
    try:
        zone = ZoneInfo(text)
    except ZoneInfoNotFoundError as exc:
        raise EphemerisInputError(
            "timezone 必须是 IANA 名称或 UTC 偏移",
            "invalid_timezone",
        ) from exc
    return ResolvedZone(key=text, tz=zone, source="input")


def _infer_timezone(site: GeoPoint) -> ResolvedZone | None:
    finder = _timezone_finder()
    try:
        name = finder.timezone_at(lng=site.lon, lat=site.lat)
    except Exception:
        return None
    if not isinstance(name, str) or not name.strip():
        return None
    try:
        zone = ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return None
    return ResolvedZone(key=name, tz=zone, source="inferred")


def _timezone_finder() -> Any:
    global _TZ_FINDER
    if _TZ_FINDER is None:
        try:
            from timezonefinder import TimezoneFinder
        except ImportError as exc:
            raise EngineUnavailableError("缺少 timezone 时需要 timezonefinder 推断当地时区") from exc
        _TZ_FINDER = TimezoneFinder()
    return _TZ_FINDER


def _parse_altitude(raw: Any) -> float:
    if raw in (None, ""):
        return 0.0
    value = _as_float(raw)
    if value is None:
        raise EphemerisInputError("altitude_m 必须是数字", "invalid_input")
    return value


def _hypothesis_flag(inputs: dict[str, Any], *, multi: bool) -> bool:
    if multi:
        return True
    for key in ("candidate", "hypothesis"):
        value = inputs.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.strip().lower() in {
            "true",
            "candidate",
            "hypothesis",
            "候选",
        }:
            return True
    status = inputs.get("status")
    if isinstance(status, str) and status.strip().lower() in {"candidate", "hypothesis"}:
        return True
    return False


def _resolve_area(raw: Any, ctx: RuntimeContext | None) -> Any:
    if raw is None or raw == "" or raw == _ACTIVE_AREA:
        return ctx.active_area if ctx is not None else None
    if raw == _PREVIOUS:
        return _resolve_ref(raw, ctx, field="area")
    return raw


def _resolve_ref(raw: Any, ctx: RuntimeContext | None, *, field: str) -> Any:
    if raw != _PREVIOUS:
        return raw
    previous = ctx.previous_tool_result if ctx is not None else None
    if previous is None:
        raise EphemerisInputError(f"无法解析 {field} 的 $previous_tool_result", "missing_input")
    return _unwrap_previous(previous)


def _unwrap_previous(raw: Any) -> Any:
    if isinstance(raw, dict) and isinstance(raw.get("result"), (dict, list)) and (
        "ok" in raw or "error_code" in raw or "artifacts" in raw
    ):
        return raw["result"]
    return raw


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


def _center_from_mapping(raw: dict[str, Any]) -> GeoPoint | None:
    lat = _as_float(raw.get("lat", raw.get("latitude")))
    lon = _as_float(raw.get("lon", raw.get("lng", raw.get("longitude"))))
    nested = raw.get("center") or raw.get("location")
    if isinstance(nested, dict) and (lat is None or lon is None):
        lat = _as_float(nested.get("lat", nested.get("latitude")))
        lon = _as_float(nested.get("lon", nested.get("lng", nested.get("longitude"))))
    elif isinstance(nested, (list, tuple)) and (lat is None or lon is None):
        parsed = _parse_point(nested)
        if parsed is not None:
            return parsed
    elif isinstance(nested, str) and (lat is None or lon is None):
        parsed = _parse_coordinate_text(nested)
        if parsed is not None:
            return parsed
    if lat is None or lon is None:
        coords = raw.get("coordinates")
        if isinstance(coords, (list, tuple)):
            return _parse_point(coords)
        return None
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return GeoPoint(lon=lon, lat=lat, label=_label_of(raw))


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


def _bbox_from_mapping(raw: dict[str, Any]) -> tuple[float, float, float, float] | None:
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


def _bbox_from_values(values: list[Any]) -> tuple[float, float, float, float] | None:
    nums = [_as_float(item) for item in values]
    if any(item is None for item in nums):
        return None
    west, south, east, north = nums  # type: ignore[misc]
    if abs(west) <= 90.0 and abs(east) <= 90.0 and abs(south) > 90.0:
        south, west, north, east = west, south, east, north
    return _validated_bbox(west, south, east, north)


def _validated_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
) -> tuple[float, float, float, float] | None:
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
    return west, south, east, north


def _bbox_centroid(
    bbox: tuple[float, float, float, float],
    *,
    label: str | None = None,
) -> GeoPoint:
    west, south, east, north = bbox
    return GeoPoint(
        lon=(west + east) / 2.0,
        lat=(south + north) / 2.0,
        label=label,
        from_centroid=True,
    )


def _centroid(points: list[GeoPoint], label: str | None = None) -> GeoPoint:
    lon = sum(item.lon for item in points) / len(points)
    lat = sum(item.lat for item in points) / len(points)
    return GeoPoint(lon=lon, lat=lat, label=label, from_centroid=True)


def _first_ring(raw: Any) -> list[GeoPoint] | None:
    if not isinstance(raw, list) or not raw:
        return None
    ring = raw[0] if raw and isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], (list, tuple)) else raw
    if not isinstance(ring, list):
        return None
    points: list[GeoPoint] = []
    for item in ring:
        point = _parse_point(item)
        if point is None:
            return None
        points.append(point)
    return points if len(points) >= 3 else None


def _label_of(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    for key in ("name", "label", "text", "query", "display_name"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _site_applied(site: GeoPoint, raw_area: Any) -> dict[str, Any]:
    applied: dict[str, Any] = _point_payload(site)
    if site.from_centroid:
        applied["location_mode"] = "centroid"
    if raw_area not in (None, "", _ACTIVE_AREA, _PREVIOUS) and not isinstance(raw_area, (dict, list)):
        if isinstance(raw_area, str) and raw_area.strip() and _parse_coordinate_text(raw_area) is None:
            applied["area"] = raw_area.strip()
    return applied


def _point_payload(site: GeoPoint) -> dict[str, Any]:
    payload: dict[str, Any] = {"lon": site.lon, "lat": site.lat, "crs": site.crs}
    if site.label:
        payload["label"] = site.label
    return payload


def _assumptions(*, altitude_m: float, from_centroid: bool, polar: bool) -> list[str]:
    items = [
        _ASSUMPTION_METHOD,
        _ASSUMPTION_AZIMUTH,
        _ASSUMPTION_TZ,
        _ASSUMPTION_CANDIDATE,
        _ASSUMPTION_COORD,
        _ASSUMPTION_NO_GEOCODE,
        _ASSUMPTION_HORIZON,
    ]
    if altitude_m == 0.0:
        items.insert(4, _ASSUMPTION_ALTITUDE)
    else:
        items.insert(4, f"海拔按 {altitude_m} m，未做地形遮挡改正")
    if from_centroid:
        items.append(_ASSUMPTION_CENTROID)
    if polar:
        items.append(_ASSUMPTION_POLAR)
    return items


def _contains_place_name(raw: Any) -> bool:
    if _looks_like_place_name(raw):
        return True
    if isinstance(raw, (list, tuple)):
        return any(_looks_like_place_name(item) for item in raw)
    return False


def _looks_like_place_name(raw: Any) -> bool:
    if not isinstance(raw, str):
        return False
    text = raw.strip()
    if not text or text in {_PREVIOUS, _ACTIVE_AREA}:
        return False
    if text[0] in "{[":
        return False
    return _parse_coordinate_text(text) is None and _bbox_from_values(
        [part for part in _COORD_SPLIT_RE.split(text) if part]
    ) is None


def _is_coord_pair(raw: Any) -> bool:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return False
    return _as_float(raw[0]) is not None and _as_float(raw[1]) is not None


def _is_bbox(raw: Any) -> bool:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return False
    return all(_as_float(item) is not None for item in raw)


def _maybe_json(raw: str) -> Any:
    text = raw.strip()
    if not text or text[0] not in "{[":
        return raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw


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


def _finite_angle(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EngineUnavailableError(f"pvlib 未返回有效 {field}") from exc
    if not math.isfinite(number):
        raise EngineUnavailableError(f"pvlib 未返回有效 {field}")
    return number


def _norm_azimuth(azimuth_deg: float) -> float:
    return float(azimuth_deg) % 360.0


def _timestamp_to_datetime(value: Any, tz: tzinfo) -> datetime | None:
    if value is None:
        return None
    try:
        import pandas as pd
    except ImportError as exc:
        raise EngineUnavailableError("solar_ephemeris 需要 pandas") from exc
    try:
        if pd.isna(value):
            return None
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(timezone.utc)
    return stamp.round("s").to_pydatetime().astimezone(tz)


def _polar_flag(sunrise: datetime | None, sunset: datetime | None) -> str | None:
    if sunrise is None and sunset is None:
        return "polar"
    return None


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat(timespec="seconds")


def _offset_key(delta: timedelta) -> str:
    total = int(delta.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    return f"{sign}{hours:02d}:{minutes:02d}"


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
