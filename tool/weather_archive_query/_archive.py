"""weather_archive_query 共享执行器：Open-Meteo 时序 + NASA GIBS 图层。"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from io import BytesIO
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from tool._gate import allow_real_api as _allow_real_api, load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext, declared_inputs
from tool.runtime.image_store import put_image

_OP_WEATHER = "weather"
_OP_CLOUD = "cloud_cover"
_OP_SNOW = "snow_cover"
_OP_REFINE = "refine_range"
_CRS_WGS84 = "wgs84"
_CRS_EPSG4326 = "EPSG:4326"
_PROVIDER_OPENMETEO = "open-meteo"
_PROVIDER_GIBS = "nasa_gibs"
_DATA_ERA5 = "ERA5"
_DATA_TYPE_REANALYSIS = "reanalysis_grid"
_DATA_TYPE_LAYER = "remote_sensing_layer"
_EVIDENCE_SERIES = "time_series"
_EVIDENCE_IMAGERY = "imagery"
_SESSIONS_KEY = "weather_archive_sessions"
_SESSION_COUNTER_KEY = "weather_archive_session_counter"
_TIMESERIES_EXTRAS_KEY = "weather_archive_timeseries_provider"
_LAYER_EXTRAS_KEY = "weather_archive_layer_provider"
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_UA = "geoagent-dataset/1.0 (weather_archive_query; local)"
_DEFAULT_OPENMETEO = "https://archive-api.open-meteo.com/v1/archive"
_DEFAULT_GIBS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
_DEFAULT_WMS_VERSION = "1.3.0"
_DEFAULT_WIDTH = 512
_DEFAULT_HEIGHT = 512
_DEFAULT_SEARCH_RADIUS_M = 2000
_METERS_PER_DEG_LAT = 111_320.0
_YEAR_MIN = 1940
_YEAR_MAX = 2100
_HOURLY_SPAN_DAYS = 14
_MAX_SERIES_POINTS = 366
_GIBS_CLOUD_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"
_GIBS_SNOW_LAYER = "MODIS_Terra_NDSI_Snow_Cover"
_GIBS_CLOUD_FRACTION_LAYER = "MODIS_Terra_Cloud_Fraction_Day"
_OPENMETEO_ALIASES = frozenset(
    {
        "open-meteo",
        "openmeteo",
        "open_meteo",
        "era5",
        "era5-land",
        "era5_land",
    }
)
_GIBS_ALIASES = frozenset(
    {
        "gibs",
        "nasa",
        "nasa_gibs",
        "nasa-gibs",
        "modis",
        "earthdata",
    }
)
_YEAR_RE = re.compile(r"^(\d{4})$")
_YEAR_RANGE_RE = re.compile(r"^(\d{4})\s*[-–—至到]\s*(\d{4})$")
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
_DATE_RANGE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s*(?:to|/|[-–—至到])\s*(\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)
_COORD_SPLIT_RE = re.compile(r"[,;，、\s]+")
_CONDITION_RE = re.compile(
    r"^\s*([A-Za-z0-9_\u4e00-\u9fff]+)\s*(<=|>=|==|=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$"
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
    "Open-Meteo Archive 为再分析格点，不是气象站实况",
    "格点结果不能直接等价于照片拍摄点当时的精确天气",
    "GIBS 图层是区域图像证据，未从像素读出云量百分比或积雪比例",
)
_QUERY_INPUT_FIELDS = ("area", "time_range", "variables", "provider")
_REFINE_INPUT_FIELDS = ("source_result", "time_range", "area", "condition")

@dataclass(frozen=True)
class VariableSpec:
    """一个气象变量的时序/图层映射。"""

    name: str
    unit: str
    kind: str
    om_daily: str | None = None
    om_hourly: str | None = None
    aggregation: str = "mean"
    layer: str | None = None

_VARIABLES: dict[str, VariableSpec] = {
    "temperature_2m_mean": VariableSpec(
        "temperature_2m_mean", "°C", "temperature", om_daily="temperature_2m_mean",
        om_hourly="temperature_2m", aggregation="mean",
    ),
    "precipitation_sum": VariableSpec(
        "precipitation_sum", "mm", "precipitation", om_daily="precipitation_sum",
        om_hourly="precipitation", aggregation="sum",
    ),
    "weather_code": VariableSpec(
        "weather_code", "wmo", "weather_code", om_daily="weather_code",
        om_hourly="weather_code", aggregation="mode",
    ),
    "cloud_cover": VariableSpec(
        "cloud_cover", "%", "cloud_cover", om_hourly="cloud_cover", aggregation="mean",
        layer=_GIBS_CLOUD_LAYER,
    ),
    "visibility": VariableSpec(
        "visibility", "m", "visibility", om_hourly="visibility", aggregation="mean",
    ),
    "wind_speed_10m": VariableSpec(
        "wind_speed_10m", "km/h", "wind", om_daily="wind_speed_10m_max",
        om_hourly="wind_speed_10m", aggregation="mean",
    ),
    "snowfall_sum": VariableSpec(
        "snowfall_sum", "cm", "snowfall", om_daily="snowfall_sum",
        om_hourly="snowfall", aggregation="sum",
    ),
    "snow_depth": VariableSpec(
        "snow_depth", "m", "snow_depth", om_hourly="snow_depth", aggregation="mean",
    ),
    "snow_cover_fraction": VariableSpec(
        "snow_cover_fraction", "%", "snow_cover_fraction", layer=_GIBS_SNOW_LAYER,
    ),
}
_VARIABLE_ALIASES: dict[str, str] = {
    "temperature": "temperature_2m_mean",
    "temperature_2m": "temperature_2m_mean",
    "temp": "temperature_2m_mean",
    "温度": "temperature_2m_mean",
    "precipitation": "precipitation_sum",
    "rain": "precipitation_sum",
    "rainfall": "precipitation_sum",
    "降水": "precipitation_sum",
    "降雨": "precipitation_sum",
    "weather": "weather_code",
    "weathercode": "weather_code",
    "天气": "weather_code",
    "天气现象": "weather_code",
    "cloud": "cloud_cover",
    "clouds": "cloud_cover",
    "云量": "cloud_cover",
    "云图": "cloud_cover",
    "visibility": "visibility",
    "能见度": "visibility",
    "wind": "wind_speed_10m",
    "wind_speed": "wind_speed_10m",
    "风": "wind_speed_10m",
    "风速": "wind_speed_10m",
    "snowfall": "snowfall_sum",
    "snow": "snowfall_sum",
    "降雪": "snowfall_sum",
    "snow_depth": "snow_depth",
    "积雪深度": "snow_depth",
    "snow_cover": "snow_cover_fraction",
    "ndsi": "snow_cover_fraction",
    "积雪": "snow_cover_fraction",
    "积雪覆盖": "snow_cover_fraction",
    "积雪范围": "snow_cover_fraction",
    "cloud_fraction": "cloud_cover",
    "cloud_cover_fraction": "cloud_cover",
}
_DEFAULT_VARS: dict[str, tuple[str, ...]] = {
    _OP_WEATHER: ("temperature_2m_mean", "precipitation_sum", "weather_code"),
    _OP_CLOUD: ("cloud_cover",),
    _OP_SNOW: ("snow_depth", "snowfall_sum", "snow_cover_fraction"),
}
_SEMANTIC_CONDITIONS: dict[str, tuple[str, str, float]] = {
    "clear": ("cloud_cover", "<", 20.0),
    "晴": ("cloud_cover", "<", 20.0),
    "少云": ("cloud_cover", "<", 20.0),
    "snow": ("snowfall_sum", ">", 0.0),
    "雪": ("snowfall_sum", ">", 0.0),
    "下雪": ("snowfall_sum", ">", 0.0),
    "有雪": ("snowfall_sum", ">", 0.0),
}

class WeatherInputError(Exception):
    """area / time_range / variables / condition 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """真实气象服务未配置、被闸门拒绝或调用失败。"""

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
    polygon: tuple[tuple[float, float], ...] | None = None
    unsupported: Any = None
    applied: Any = None

@dataclass(frozen=True)
class TimeWindow:
    """闭区间日期窗。"""

    start: date
    end: date

@dataclass(frozen=True)
class SeriesRequest:
    """Open-Meteo Archive 时序请求。"""

    latitude: float
    longitude: float
    start: date
    end: date
    daily: tuple[str, ...] = ()
    hourly: tuple[str, ...] = ()
    timezone: str = "GMT"

@dataclass(frozen=True)
class LayerFetchRequest:
    """GIBS WMS GetMap 请求。"""

    bbox: BBox
    layer: str
    time: date
    width: int = _DEFAULT_WIDTH
    height: int = _DEFAULT_HEIGHT
    crs: str = _CRS_EPSG4326

@dataclass(frozen=True)
class ParsedCondition:
    """本地细化使用的数值条件。"""

    variable: str
    op: str
    value: float
    applied: dict[str, Any]

@dataclass
class WeatherSession:
    """一次气象查询会话，供 refine_range 复用时序。"""

    session_id: str
    operation: str
    location: dict[str, Any]
    time_range: dict[str, str]
    series: dict[str, Any] | None = None
    layers: list[dict[str, Any]] = field(default_factory=list)
    applied: dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class WeatherTimeseriesProvider(Protocol):
    """可注入的历史时序后端；测试用 extras 替换。"""

    def fetch_series(self, request: SeriesRequest) -> dict[str, Any]:
        """按点位与日期窗返回 Open-Meteo 风格 JSON。"""

@runtime_checkable
class WeatherLayerProvider(Protocol):
    """可注入的气象图层后端；测试用 extras 替换。"""

    def fetch_map(self, request: LayerFetchRequest) -> bytes:
        """按 bbox 与日期返回 PNG 字节。"""

class OpenMeteoArchiveProvider:
    """Open-Meteo Historical Weather Archive 适配器。"""

    name = _PROVIDER_OPENMETEO

    def __init__(
        self,
        *,
        endpoint: str,
        timeout_sec: float,
        user_agent: str,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent

    def fetch_series(self, request: SeriesRequest) -> dict[str, Any]:
        params: dict[str, str] = {
            "latitude": f"{request.latitude:.6f}",
            "longitude": f"{request.longitude:.6f}",
            "start_date": request.start.isoformat(),
            "end_date": request.end.isoformat(),
            "timezone": request.timezone,
        }
        if request.daily:
            params["daily"] = ",".join(request.daily)
        if request.hourly:
            params["hourly"] = ",".join(request.hourly)
        payload = _http_json(
            _append_query(self._endpoint, urllib.parse.urlencode(params)),
            headers={"Accept": "application/json", "User-Agent": self._user_agent},
            timeout_sec=self._timeout_sec,
            error_prefix="Open-Meteo Archive",
        )
        if not isinstance(payload, dict):
            raise EngineUnavailableError("Open-Meteo Archive 回执不是 JSON 对象")
        return payload

class NasaGibsWmsProvider:
    """NASA GIBS WMS GetMap 适配器。"""

    name = _PROVIDER_GIBS

    def __init__(
        self,
        *,
        endpoint: str,
        timeout_sec: float,
        user_agent: str,
        width: int,
        height: int,
        wms_version: str = _DEFAULT_WMS_VERSION,
    ) -> None:
        self._endpoint = endpoint
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent
        self._width = width
        self._height = height
        self._wms_version = wms_version

    def fetch_map(self, request: LayerFetchRequest) -> bytes:
        width = request.width or self._width
        height = request.height or self._height
        params: dict[str, str] = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": self._wms_version,
            "LAYERS": request.layer,
            "STYLES": "",
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
            "WIDTH": str(width),
            "HEIGHT": str(height),
            "BBOX": _wms_bbox(request.bbox, request.crs, self._wms_version),
            "TIME": request.time.isoformat(),
        }
        if self._wms_version.startswith("1.3"):
            params["CRS"] = request.crs
        else:
            params["SRS"] = request.crs
        raw = _http_bytes(
            _append_query(self._endpoint, urllib.parse.urlencode(params)),
            headers={"Accept": "image/png,application/xml,*/*", "User-Agent": self._user_agent},
            timeout_sec=self._timeout_sec,
            error_prefix="NASA GIBS GetMap",
        )
        if _looks_like_xml(raw):
            raise EngineUnavailableError(_xml_exception_message(raw, "NASA GIBS GetMap"))
        return raw

def execute_weather(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """查询历史温度、降水、风或能见度的再分析时序。"""

    return _execute_query(_OP_WEATHER, purpose=purpose, inputs=inputs, ctx=ctx)

def execute_cloud_cover(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """查询历史云量时序，并可选返回代表日云图。"""

    return _execute_query(_OP_CLOUD, purpose=purpose, inputs=inputs, ctx=ctx)

def execute_snow_cover(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """查询积雪深度/降雪时序，并可选返回积雪覆盖图层。"""

    return _execute_query(_OP_SNOW, purpose=purpose, inputs=inputs, ctx=ctx)

def execute_refine_range(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """在已有气象时序上按时间窗和数值条件本地筛选日期。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, *_REFINE_INPUT_FIELDS)
        session = _resolve_source_session(inputs.get("source_result"), ctx)
        window = _require_time_window(inputs.get("time_range"))
        condition = _parse_condition(inputs.get("condition"), session.series)
        area = None
        if inputs.get("area") not in (None, ""):
            area = _parse_area(_resolve_area(inputs.get("area"), ctx))
        series, matched = _filter_series(session.series, window, condition)
        applied: dict[str, Any] = {
            "source_result": session.session_id,
            "time_range": _window_applied(window),
        }
        if condition is not None:
            applied["condition"] = condition.applied
        if area is not None and area.applied is not None:
            applied["area"] = area.applied
        result_id = _new_session_id(ctx)
        refined = WeatherSession(
            session_id=result_id,
            operation=_OP_REFINE,
            location=dict(session.location),
            time_range=_window_applied(window),
            series=series,
            layers=[],
            applied=applied,
        )
        _store_session(ctx, refined)
        result: dict[str, Any] = {
            "operation": _OP_REFINE,
            "result_id": result_id,
            "source_result": session.session_id,
            "location": dict(session.location),
            "time_range": _window_applied(window),
            "matched_dates": matched,
            "series": series,
            "applied": applied,
            "assumptions": [
                "本步只按数值条件筛选已有时序，不下『当天一定下雪』等结论",
                *_BASE_ASSUMPTIONS,
            ],
        }
        if ctx is not None:
            ctx.previous_tool_result = result
            ctx.active_session = result_id
        return Observation(
            ok=True,
            result=_strip_forbidden(result),
            session=result_id,
        )
    except WeatherInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def _execute_query(
    operation: str,
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    del purpose
    try:
        inputs = declared_inputs(inputs, *_QUERY_INPUT_FIELDS)
        location = _resolve_location(inputs, ctx)
        window = _require_time_window(inputs.get("time_range"))
        specs = _parse_variables(inputs.get("variables"), operation)
        want_series, want_layers = _backends_for(operation, inputs.get("provider"))
        layer_names = _layer_names(operation, specs) if want_layers else ()
        series_block: dict[str, Any] | None = None
        layers: list[dict[str, Any]] = []
        artifacts: dict[str, Any] = {}
        if want_series:
            om_specs = [spec for spec in specs if spec.om_daily or spec.om_hourly]
            if om_specs:
                series_provider = _resolve_timeseries_provider(ctx)
                series_request = _series_request(location, window, om_specs)
                payload = series_provider.fetch_series(series_request)
                series_block = _normalize_series(payload, om_specs, window)
        if want_layers and layer_names:
            layer_provider = _resolve_layer_provider(ctx)
            captured_on = window.start
            image_ids: list[str] = []
            image_paths: dict[str, str] = {}
            for layer_name in layer_names:
                png = layer_provider.fetch_map(
                    LayerFetchRequest(
                        bbox=location.bbox,
                        layer=layer_name,
                        time=captured_on,
                        width=_preview_width(),
                        height=_preview_height(),
                    )
                )
                image_id, image_path = _store_png(png, source_id=layer_name, ctx=ctx)
                row = {
                    "provider": _PROVIDER_GIBS,
                    "data_source": layer_name,
                    "data_type": _DATA_TYPE_LAYER,
                    "evidence_kind": _EVIDENCE_IMAGERY,
                    "layer": layer_name,
                    "captured_on": captured_on.isoformat(),
                    "image_id": image_id,
                    "variable_kind": _layer_kind(layer_name),
                }
                layers.append(row)
                image_ids.append(image_id)
                image_paths[image_id] = image_path
            if image_ids:
                artifacts["image_ids"] = image_ids
                artifacts["images"] = image_paths
        applied = _applied(location, window, inputs.get("provider"), specs, want_series, want_layers)
        result_id = _new_session_id(ctx)
        session = WeatherSession(
            session_id=result_id,
            operation=operation,
            location=_location_payload(location),
            time_range=_window_applied(window),
            series=series_block,
            layers=list(layers),
            applied=applied,
        )
        _store_session(ctx, session)
        result: dict[str, Any] = {
            "operation": operation,
            "result_id": result_id,
            "location": _location_payload(location),
            "time_range": _window_applied(window),
            "applied": applied,
            "assumptions": _assumptions(operation, want_series, want_layers),
        }
        if series_block is not None:
            result["series"] = series_block
        if layers:
            result["layers"] = layers
        if ctx is not None:
            ctx.previous_tool_result = result
            ctx.active_session = result_id
            ctx.active_area = _bbox_applied(location.bbox)
        return Observation(
            ok=True,
            result=_strip_forbidden(result),
            artifacts=artifacts,
            session=result_id,
        )
    except WeatherInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

@dataclass(frozen=True)
class _ResolvedLocation:
    center: GeoPoint
    bbox: BBox
    area: ParsedArea
    radius_m: int

def _resolve_location(inputs: dict[str, Any], ctx: RuntimeContext | None) -> _ResolvedLocation:
    raw = _resolve_area(inputs.get("area"), ctx)
    if raw in (None, ""):
        raise WeatherInputError("缺少必填输入 area", "missing_input")
    area = _parse_area(raw)
    center = area.center
    bbox = area.bbox
    if bbox is None and area.polygon is not None:
        bbox = _bbox_from_polygon(area.polygon)
        if center is None:
            center = GeoPoint(
                lat=(bbox.south + bbox.north) / 2.0,
                lon=(bbox.west + bbox.east) / 2.0,
            )
    radius = area.radius_m if area.radius_m is not None else _DEFAULT_SEARCH_RADIUS_M
    if center is not None and bbox is None:
        bbox = _bbox_from_center(center, radius)
    if bbox is not None and center is None:
        center = GeoPoint(
            lat=(bbox.south + bbox.north) / 2.0,
            lon=(bbox.west + bbox.east) / 2.0,
        )
    if bbox is None or center is None:
        if area.text:
            raise WeatherInputError(
                "需要 bbox 或中心点加半径；纯地名请先 geocode",
                "missing_input",
            )
        raise WeatherInputError("area 无法解析为范围", "invalid_area")
    return _ResolvedLocation(center=center, bbox=bbox, area=area, radius_m=radius)

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
        polygon = _polygon_from_points(raw)
        if polygon is not None:
            return ParsedArea(polygon=polygon, applied=_polygon_applied(polygon))
        if len(raw) == 4:
            bbox = _bbox_from_values(list(raw))
            if bbox is not None:
                return ParsedArea(bbox=bbox, applied=_bbox_applied(bbox))
        if len(raw) == 2:
            point = _parse_lonlat_pair(raw[0], raw[1])
            if point is not None:
                return ParsedArea(
                    center=point,
                    applied={"lat": point.lat, "lon": point.lon},
                )
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
            if radius_raw not in (None, ""):
                parsed = _as_float(radius_raw)
                if parsed is None or parsed < 0:
                    raise WeatherInputError("radius_m 必须是非负数", "invalid_area")
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

def _bbox_from_polygon(points: tuple[tuple[float, float], ...]) -> BBox:
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    return BBox(west=min(lons), south=min(lats), east=max(lons), north=max(lats))

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

def _bbox_applied(bbox: BBox) -> dict[str, float | str]:
    return {
        "west": bbox.west,
        "south": bbox.south,
        "east": bbox.east,
        "north": bbox.north,
        "crs": _CRS_WGS84,
    }

def _polygon_applied(points: tuple[tuple[float, float], ...]) -> list[list[float]]:
    return [[lon, lat] for lon, lat in points]

def _location_payload(location: _ResolvedLocation) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "lat": location.center.lat,
        "lon": location.center.lon,
        "crs": _CRS_WGS84,
        "bbox": _bbox_applied(location.bbox),
    }
    if location.area.applied is not None:
        payload["area"] = location.area.applied
    return payload

def _require_time_window(raw: Any) -> TimeWindow:
    if raw in (None, ""):
        raise WeatherInputError("缺少必填输入 time_range", "missing_input")
    window, unsupported = _parse_time_window(raw)
    if window is None:
        raise WeatherInputError(
            "time_range 无法解析为年份、日期或起止范围",
            "invalid_time_range",
        )
    del unsupported
    return window

def _parse_time_window(raw: Any) -> tuple[TimeWindow | None, Any | None]:
    if raw is None or raw == "":
        return None, None
    if isinstance(raw, bool):
        return None, raw
    if isinstance(raw, int) and _YEAR_MIN <= raw <= _YEAR_MAX:
        return TimeWindow(date(raw, 1, 1), date(raw, 12, 31)), None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None, None
        year = _YEAR_RE.fullmatch(text)
        if year:
            value = int(year.group(1))
            if _YEAR_MIN <= value <= _YEAR_MAX:
                return TimeWindow(date(value, 1, 1), date(value, 12, 31)), None
        year_range = _YEAR_RANGE_RE.fullmatch(text)
        if year_range:
            start, end = int(year_range.group(1)), int(year_range.group(2))
            if _YEAR_MIN <= start <= _YEAR_MAX and _YEAR_MIN <= end <= _YEAR_MAX:
                lo, hi = (start, end) if start <= end else (end, start)
                return TimeWindow(date(lo, 1, 1), date(hi, 12, 31)), None
        day = _DATE_RE.fullmatch(text)
        if day:
            parsed = _parse_iso_date(day.group(1))
            if parsed is not None:
                return TimeWindow(parsed, parsed), None
        compact = re.sub(r"\s+", "", text)
        date_range = _DATE_RANGE_RE.fullmatch(compact)
        if date_range:
            start_d = _parse_iso_date(date_range.group(1))
            end_d = _parse_iso_date(date_range.group(2))
            if start_d is not None and end_d is not None:
                lo, hi = (start_d, end_d) if start_d <= end_d else (end_d, start_d)
                return TimeWindow(lo, hi), None
        return None, raw
    if isinstance(raw, dict):
        start = _bound_to_date(raw.get("start") or raw.get("from") or raw.get("begin"), start=True)
        end = _bound_to_date(raw.get("end") or raw.get("to"), start=False)
        if start is not None and end is not None:
            lo, hi = (start, end) if start <= end else (end, start)
            return TimeWindow(lo, hi), None
        return None, raw
    return None, raw

def _bound_to_date(raw: Any, *, start: bool) -> date | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int) and _YEAR_MIN <= raw <= _YEAR_MAX:
        return date(raw, 1, 1) if start else date(raw, 12, 31)
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if _YEAR_RE.fullmatch(text):
        value = int(text)
        if _YEAR_MIN <= value <= _YEAR_MAX:
            return date(value, 1, 1) if start else date(value, 12, 31)
    return _parse_iso_date(text)

def _parse_iso_date(text: str) -> date | None:
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None

def _window_applied(window: TimeWindow) -> dict[str, str]:
    return {"start": window.start.isoformat(), "end": window.end.isoformat()}

def _parse_variables(raw: Any, operation: str) -> list[VariableSpec]:
    names = _variable_names(raw)
    if not names:
        names = list(_DEFAULT_VARS[operation])
    specs: list[VariableSpec] = []
    seen: set[str] = set()
    for name in names:
        canonical = _canonical_variable(name)
        if canonical is None:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        spec = _VARIABLES[canonical]
        if canonical == "cloud_cover" and _is_cloud_fraction_token(name):
            spec = VariableSpec(
                spec.name,
                spec.unit,
                spec.kind,
                om_daily=spec.om_daily,
                om_hourly=spec.om_hourly,
                aggregation=spec.aggregation,
                layer=_GIBS_CLOUD_FRACTION_LAYER,
            )
        specs.append(spec)
    if not specs:
        raise WeatherInputError("variables 无法映射到已知气象变量", "invalid_input")
    return specs

def _variable_names(raw: Any) -> list[str]:
    if raw in (None, ""):
        return []
    if isinstance(raw, str):
        parts = [part.strip() for part in re.split(r"[,，、;/|]+", raw) if part.strip()]
        return parts or [raw.strip()]
    if isinstance(raw, (list, tuple)):
        names: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise WeatherInputError("variables 必须是字符串或字符串列表", "invalid_input")
            names.append(item.strip())
        return names
    raise WeatherInputError("variables 必须是字符串或字符串列表", "invalid_input")

def _canonical_variable(raw: str) -> str | None:
    token = raw.strip()
    if not token:
        return None
    lowered = token.lower().replace(" ", "_")
    if lowered in _VARIABLES:
        return lowered
    if lowered in _VARIABLE_ALIASES:
        return _VARIABLE_ALIASES[lowered]
    if token in _VARIABLE_ALIASES:
        return _VARIABLE_ALIASES[token]
    return None

def _is_cloud_fraction_token(raw: str) -> bool:
    lowered = raw.strip().lower().replace(" ", "_")
    return lowered in {"cloud_fraction", "cloud_cover_fraction", "云量比例"}

def _backends_for(operation: str, raw_provider: Any) -> tuple[bool, bool]:
    if raw_provider in (None, ""):
        if operation == _OP_WEATHER:
            return True, False
        return True, True
    if not isinstance(raw_provider, str):
        raise WeatherInputError("provider 必须是字符串", "unsupported_provider")
    name = raw_provider.strip().lower().replace(" ", "_")
    if name in _OPENMETEO_ALIASES:
        return True, False
    if name in _GIBS_ALIASES:
        return False, True
    raise EngineUnavailableError(f"未知 weather provider: {raw_provider}", "unsupported_provider")

def _layer_names(operation: str, specs: list[VariableSpec]) -> tuple[str, ...]:
    names: list[str] = []
    for spec in specs:
        if spec.layer and spec.layer not in names:
            names.append(spec.layer)
    if names:
        return tuple(names)
    if operation == _OP_SNOW:
        return (_GIBS_SNOW_LAYER,)
    return (_GIBS_CLOUD_LAYER,)

def _layer_kind(layer_name: str) -> str:
    if layer_name == _GIBS_SNOW_LAYER:
        return "snow_cover_fraction"
    if layer_name == _GIBS_CLOUD_FRACTION_LAYER:
        return "cloud_cover"
    return "cloud_cover"

def _series_request(
    location: _ResolvedLocation,
    window: TimeWindow,
    specs: list[VariableSpec],
) -> SeriesRequest:
    span = (window.end - window.start).days + 1
    use_hourly = span <= _HOURLY_SPAN_DAYS and any(spec.om_hourly and not spec.om_daily for spec in specs)
    if not use_hourly and any(spec.om_hourly and not spec.om_daily for spec in specs):
        use_hourly = True
    daily: list[str] = []
    hourly: list[str] = []
    for spec in specs:
        if use_hourly and spec.om_hourly:
            if spec.om_hourly not in hourly:
                hourly.append(spec.om_hourly)
            continue
        if spec.om_daily and spec.om_daily not in daily:
            daily.append(spec.om_daily)
        elif spec.om_hourly and spec.om_hourly not in hourly:
            hourly.append(spec.om_hourly)
    if not daily and not hourly:
        raise WeatherInputError("当前变量无法映射到 Open-Meteo 时序", "invalid_input")
    return SeriesRequest(
        latitude=location.center.lat,
        longitude=location.center.lon,
        start=window.start,
        end=window.end,
        daily=tuple(daily),
        hourly=tuple(hourly),
    )

def _normalize_series(
    payload: dict[str, Any],
    specs: list[VariableSpec],
    window: TimeWindow,
) -> dict[str, Any]:
    daily = payload.get("daily") if isinstance(payload.get("daily"), dict) else {}
    hourly = payload.get("hourly") if isinstance(payload.get("hourly"), dict) else {}
    daily_units = payload.get("daily_units") if isinstance(payload.get("daily_units"), dict) else {}
    hourly_units = payload.get("hourly_units") if isinstance(payload.get("hourly_units"), dict) else {}
    times: list[str] = []
    values: dict[str, list[Any]] = {}
    units: dict[str, str] = {}
    variables: list[dict[str, str]] = []
    span = (window.end - window.start).days + 1
    hourly_times = _as_str_list(hourly.get("time"))
    need_daily_from_hourly = bool(hourly_times) and span > _HOURLY_SPAN_DAYS
    for spec in specs:
        if spec.om_daily and spec.om_daily in daily:
            series_times = _as_str_list(daily.get("time"))
            series_vals = list(daily.get(spec.om_daily) or [])
            if not times:
                times = series_times
            values[spec.name] = _align_values(times, series_times, series_vals)
            unit = _optional_str(daily_units.get(spec.om_daily)) or spec.unit
            units[spec.name] = unit
            variables.append({"name": spec.name, "unit": unit, "kind": spec.kind})
            continue
        if spec.om_hourly and spec.om_hourly in hourly:
            series_vals = list(hourly.get(spec.om_hourly) or [])
            unit = _optional_str(hourly_units.get(spec.om_hourly)) or spec.unit
            if need_daily_from_hourly:
                agg_times, agg_vals = _aggregate_hourly(hourly_times, series_vals, spec.aggregation)
                if not times:
                    times = agg_times
                values[spec.name] = _align_values(times, agg_times, agg_vals)
            else:
                if not times:
                    times = hourly_times
                values[spec.name] = _align_values(times, hourly_times, series_vals)
            units[spec.name] = unit
            variables.append({"name": spec.name, "unit": unit, "kind": spec.kind})
    truncated = False
    if len(times) > _MAX_SERIES_POINTS:
        times = times[:_MAX_SERIES_POINTS]
        values = {name: vals[:_MAX_SERIES_POINTS] for name, vals in values.items()}
        truncated = True
    block: dict[str, Any] = {
        "provider": _PROVIDER_OPENMETEO,
        "data_source": _DATA_ERA5,
        "data_type": _DATA_TYPE_REANALYSIS,
        "evidence_kind": _EVIDENCE_SERIES,
        "variables": variables,
        "times": times,
        "values": values,
        "units": units,
    }
    if truncated:
        block["truncated"] = True
    return block

def _as_str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]

def _align_values(target_times: list[str], source_times: list[str], source_vals: list[Any]) -> list[Any]:
    if target_times == source_times:
        padded = list(source_vals)
        if len(padded) < len(target_times):
            padded.extend([None] * (len(target_times) - len(padded)))
        return padded[: len(target_times)]
    date_only = not (
        target_times
        and source_times
        and len(target_times[0]) > 10
        and len(source_times[0]) > 10
    )
    lookup: dict[str, Any] = {}
    for index, stamp in enumerate(source_times):
        key = _date_key(stamp) if date_only else stamp
        lookup[key] = source_vals[index] if index < len(source_vals) else None
    return [
        lookup.get(_date_key(stamp) if date_only else stamp) for stamp in target_times
    ]

def _date_key(raw: str) -> str:
    return raw[:10] if len(raw) >= 10 else raw

def _aggregate_hourly(
    times: list[str],
    values: list[Any],
    aggregation: str,
) -> tuple[list[str], list[Any]]:
    buckets: dict[str, list[float]] = {}
    order: list[str] = []
    for index, stamp in enumerate(times):
        key = _date_key(stamp)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        if index >= len(values):
            continue
        number = _as_float(values[index])
        if number is not None:
            buckets[key].append(number)
    aggregated: list[Any] = []
    for key in order:
        numbers = buckets[key]
        if not numbers:
            aggregated.append(None)
            continue
        if aggregation == "sum":
            aggregated.append(sum(numbers))
        elif aggregation == "mode":
            aggregated.append(statistics.multimode(numbers)[0])
        else:
            aggregated.append(sum(numbers) / len(numbers))
    return order, aggregated

def _store_png(raw: bytes, *, source_id: str, ctx: RuntimeContext | None) -> tuple[str, str]:
    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except OSError as exc:
        raise EngineUnavailableError(f"GIBS 回执不是可解码图片: {exc}") from exc
    image_id, path = put_image(image, source_id=source_id, suffix="png", ctx=ctx)
    return image_id, str(path)

def _applied(
    location: _ResolvedLocation,
    window: TimeWindow,
    provider: Any,
    specs: list[VariableSpec],
    want_series: bool,
    want_layers: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": provider if isinstance(provider, str) and provider.strip() else None,
        "crs": _CRS_WGS84,
        "time_range": _window_applied(window),
        "variables": [spec.name for spec in specs],
        "backends": {
            "timeseries": want_series,
            "layers": want_layers,
        },
        "search_radius_m": location.radius_m,
        "bbox": _bbox_applied(location.bbox),
    }
    if location.area.applied is not None:
        payload["area"] = location.area.applied
    unsupported: dict[str, Any] = {}
    if location.area.unsupported is not None:
        unsupported["area"] = location.area.unsupported
    if unsupported:
        payload["unsupported"] = unsupported
    return payload

def _assumptions(operation: str, want_series: bool, want_layers: bool) -> list[str]:
    items = list(_BASE_ASSUMPTIONS)
    if operation == _OP_SNOW:
        items.append("snow_depth 是积雪深度，不是地表积雪覆盖比例")
        items.append("GIBS NDSI 图层表示积雪覆盖，不能与积雪深度混用")
    if want_layers:
        items.append("本步只返回代表日图层，不是逐日动画")
    if want_series:
        items.append("Open-Meteo 使用查询范围质心一点，不是区域平均")
    return items

def _resolve_timeseries_provider(ctx: RuntimeContext | None) -> WeatherTimeseriesProvider:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get(_TIMESERIES_EXTRAS_KEY)
    if injected is not None:
        if not isinstance(injected, WeatherTimeseriesProvider):
            raise EngineUnavailableError(
                "weather_archive_timeseries_provider 必须提供 fetch_series(request)",
            )
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_API=false，禁止调用真实气象档案 API")
    return OpenMeteoArchiveProvider(
        endpoint=_env_value("WEATHER_ARCHIVE_OPENMETEO_ENDPOINT", _DEFAULT_OPENMETEO),
        timeout_sec=_env_timeout("WEATHER_ARCHIVE_TIMEOUT_SEC"),
        user_agent=_env_value("WEATHER_ARCHIVE_USER_AGENT", _DEFAULT_UA),
    )

def _resolve_layer_provider(ctx: RuntimeContext | None) -> WeatherLayerProvider:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get(_LAYER_EXTRAS_KEY)
    if injected is not None:
        if not isinstance(injected, WeatherLayerProvider):
            raise EngineUnavailableError(
                "weather_archive_layer_provider 必须提供 fetch_map(request)",
            )
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_API=false，禁止调用真实气象图层 API")
    return NasaGibsWmsProvider(
        endpoint=_env_value("WEATHER_ARCHIVE_GIBS_ENDPOINT", _DEFAULT_GIBS),
        timeout_sec=_env_timeout("WEATHER_ARCHIVE_TIMEOUT_SEC"),
        user_agent=_env_value("WEATHER_ARCHIVE_USER_AGENT", _DEFAULT_UA),
        width=_preview_width(),
        height=_preview_height(),
    )

def _new_session_id(ctx: RuntimeContext | None) -> str:
    extras = _extras(ctx)
    counter = int(extras.get(_SESSION_COUNTER_KEY) or 0) + 1
    extras[_SESSION_COUNTER_KEY] = counter
    return f"weather_{counter:04d}"

def _store_session(ctx: RuntimeContext | None, session: WeatherSession) -> None:
    extras = _extras(ctx)
    sessions = extras.setdefault(_SESSIONS_KEY, {})
    if not isinstance(sessions, dict):
        sessions = {}
        extras[_SESSIONS_KEY] = sessions
    sessions[session.session_id] = session

def _resolve_source_session(raw: Any, ctx: RuntimeContext | None) -> WeatherSession:
    if raw in (None, ""):
        raise WeatherInputError("缺少必填输入 source_result", "missing_input")
    payload: Any = raw
    if payload == "$previous_tool_result":
        payload = ctx.previous_tool_result if ctx is not None else None
        if payload in (None, ""):
            raise WeatherInputError("无法解析 $previous_tool_result", "missing_input")
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            raise WeatherInputError("缺少必填输入 source_result", "missing_input")
        session = _session_by_id(text, ctx)
        if session is not None:
            return session
        if _looks_like_natural_language(text):
            raise WeatherInputError(
                "source_result 必须是真实结果对象、结果 ID 或 $previous_tool_result，不能用自然语言描述替代",
                "invalid_input",
            )
        raise WeatherInputError(f"source_result 引用的结果不存在: {text}", "missing_input")
    if isinstance(payload, dict):
        for key in ("result_id", "session", "session_id", "source_result"):
            token = payload.get(key)
            if isinstance(token, str) and token.strip():
                found = _session_by_id(token.strip(), ctx)
                if found is not None:
                    return found
        nested = payload.get("result")
        if isinstance(nested, dict):
            return _resolve_source_session(nested, ctx)
        series = payload.get("series")
        if isinstance(series, dict) and isinstance(series.get("times"), list):
            result_id = _optional_str(payload.get("result_id")) or "inline"
            location = payload.get("location") if isinstance(payload.get("location"), dict) else {}
            time_range = payload.get("time_range") if isinstance(payload.get("time_range"), dict) else {}
            return WeatherSession(
                session_id=result_id,
                operation=str(payload.get("operation") or _OP_WEATHER),
                location=location,
                time_range={
                    "start": str(time_range.get("start") or ""),
                    "end": str(time_range.get("end") or ""),
                },
                series=series,
            )
        raise WeatherInputError("source_result 无法解析为气象结果", "invalid_input")
    raise WeatherInputError("source_result 必须是结果对象或引用", "invalid_input")

def _session_by_id(result_id: str, ctx: RuntimeContext | None) -> WeatherSession | None:
    extras = ctx.extras if ctx is not None else {}
    sessions = extras.get(_SESSIONS_KEY)
    if not isinstance(sessions, dict):
        return None
    found = sessions.get(result_id)
    return found if isinstance(found, WeatherSession) else None

def _looks_like_natural_language(text: str) -> bool:
    if " " in text or "。" in text or "，" in text:
        return True
    return any("\u4e00" <= char <= "\u9fff" for char in text)

def _parse_condition(raw: Any, series: dict[str, Any] | None) -> ParsedCondition | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, dict):
        variable = _optional_str(raw.get("variable") or raw.get("field") or raw.get("name"))
        op = _optional_str(raw.get("op") or raw.get("operator"))
        value = _as_float(raw.get("value") or raw.get("threshold"))
        if variable is None or op is None or value is None:
            raise WeatherInputError("condition 对象需要 variable、op 与 value", "invalid_input")
        resolved = _resolve_condition_variable(variable, series)
        return ParsedCondition(
            variable=resolved,
            op=_normalize_op(op),
            value=value,
            applied={"variable": resolved, "op": _normalize_op(op), "value": value},
        )
    if isinstance(raw, str):
        text = raw.strip()
        semantic = _SEMANTIC_CONDITIONS.get(text) or _SEMANTIC_CONDITIONS.get(text.lower())
        if semantic is not None:
            variable, op, value = semantic
            resolved = _resolve_condition_variable(variable, series)
            return ParsedCondition(
                variable=resolved,
                op=op,
                value=value,
                applied={"variable": resolved, "op": op, "value": value, "from": text},
            )
        matched = _CONDITION_RE.fullmatch(text)
        if matched is None:
            raise WeatherInputError("condition 无法解析为变量比较式", "invalid_input")
        resolved = _resolve_condition_variable(matched.group(1), series)
        op = _normalize_op(matched.group(2))
        value = float(matched.group(3))
        return ParsedCondition(
            variable=resolved,
            op=op,
            value=value,
            applied={"variable": resolved, "op": op, "value": value},
        )
    raise WeatherInputError("condition 必须是字符串或对象", "invalid_input")

def _normalize_op(raw: str) -> str:
    op = raw.strip()
    if op == "=":
        return "=="
    if op not in {"<", ">", "<=", ">=", "=="}:
        raise WeatherInputError("condition 比较符必须是 <、>、<=、>= 或 ==", "invalid_input")
    return op

def _resolve_condition_variable(raw: str, series: dict[str, Any] | None) -> str:
    available = _series_value_keys(series)
    canonical = _canonical_variable(raw) or raw.strip()
    if canonical in available:
        return canonical
    if raw in available:
        return raw
    if canonical == "snowfall_sum":
        for candidate in ("snowfall_sum", "snowfall", "snow_depth"):
            if candidate in available:
                return candidate
    if canonical == "precipitation_sum":
        for candidate in ("precipitation_sum", "precipitation", "rain"):
            if candidate in available:
                return candidate
    if available:
        raise WeatherInputError(f"条件变量不在已有结果中: {raw}", "invalid_input")
    return canonical

def _series_value_keys(series: dict[str, Any] | None) -> set[str]:
    if not isinstance(series, dict):
        return set()
    values = series.get("values")
    if isinstance(values, dict):
        return {str(key) for key in values}
    return set()

def _filter_series(
    series: dict[str, Any] | None,
    window: TimeWindow,
    condition: ParsedCondition | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(series, dict):
        raise WeatherInputError("source_result 缺少可筛选的时序", "invalid_input")
    times = _as_str_list(series.get("times"))
    raw_values = series.get("values")
    if not isinstance(raw_values, dict):
        raise WeatherInputError("source_result 缺少可筛选的时序", "invalid_input")
    kept_indices: list[int] = []
    matched_dates: list[str] = []
    for index, stamp in enumerate(times):
        day = _parse_iso_date(_date_key(stamp))
        if day is None or day < window.start or day > window.end:
            continue
        if condition is not None:
            column = raw_values.get(condition.variable)
            sample = column[index] if isinstance(column, list) and index < len(column) else None
            number = _as_float(sample)
            if number is None or not _compare(number, condition.op, condition.value):
                continue
        kept_indices.append(index)
        key = _date_key(stamp)
        if key not in matched_dates:
            matched_dates.append(key)
    filtered_values: dict[str, list[Any]] = {}
    for name, column in raw_values.items():
        if not isinstance(column, list):
            continue
        filtered_values[str(name)] = [
            column[index] if index < len(column) else None for index in kept_indices
        ]
    filtered = dict(series)
    filtered["times"] = [times[index] for index in kept_indices]
    filtered["values"] = filtered_values
    return filtered, matched_dates

def _compare(sample: float, op: str, threshold: float) -> bool:
    if op == "<":
        return sample < threshold
    if op == ">":
        return sample > threshold
    if op == "<=":
        return sample <= threshold
    if op == ">=":
        return sample >= threshold
    return sample == threshold

def _extras(ctx: RuntimeContext | None) -> dict[str, Any]:
    if ctx is None:
        return {}
    return ctx.extras

def _preview_width() -> int:
    parsed = _as_int(os.environ.get("WEATHER_ARCHIVE_PREVIEW_WIDTH", "").strip())
    if parsed is None or parsed <= 0:
        return _DEFAULT_WIDTH
    return parsed

def _preview_height() -> int:
    parsed = _as_int(os.environ.get("WEATHER_ARCHIVE_PREVIEW_HEIGHT", "").strip())
    if parsed is None or parsed <= 0:
        return _DEFAULT_HEIGHT
    return parsed

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
    raw = _http_bytes(url, headers=headers, timeout_sec=timeout_sec, error_prefix=error_prefix)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EngineUnavailableError(f"{error_prefix} 回执不是 JSON: {exc}") from exc
    if isinstance(payload, dict) and payload.get("error"):
        reason = payload.get("reason") or payload.get("error")
        raise EngineUnavailableError(f"{error_prefix} 失败: {reason}")
    return payload

def _http_bytes(
    url: str,
    *,
    headers: dict[str, str],
    timeout_sec: float,
    error_prefix: str,
) -> bytes:
    request = urllib.request.Request(url, method="GET")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200] if exc.fp else b""
        text = detail.decode("utf-8", errors="replace")
        raise EngineUnavailableError(f"{error_prefix} HTTP {exc.code}: {text}") from exc
    except urllib.error.URLError as exc:
        raise EngineUnavailableError(f"{error_prefix} 网络失败: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise EngineUnavailableError(f"{error_prefix} 调用失败: {exc}") from exc

def _append_query(endpoint: str, query: str) -> str:
    separator = "&" if urllib.parse.urlparse(endpoint).query else "?"
    return f"{endpoint}{separator}{query}"

def _wms_bbox(bbox: BBox, crs: str, version: str) -> str:
    if version.startswith("1.3") and crs.upper() == "EPSG:4326":
        return f"{bbox.south},{bbox.west},{bbox.north},{bbox.east}"
    return f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}"

def _looks_like_xml(raw: bytes) -> bool:
    stripped = raw.lstrip()
    return stripped.startswith(b"<?xml") or stripped.startswith(b"<")

def _xml_exception_message(raw: bytes, prefix: str) -> str:
    text = raw.decode("utf-8", errors="replace")[:200]
    return f"{prefix} 返回服务异常: {text}"

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
