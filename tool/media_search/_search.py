"""media_search 共享执行器：Wikimedia Commons，以及显式指定时的 YouTube 视频检索。"""

from __future__ import annotations

import html
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol, runtime_checkable

from tool._gate import allow_real_api as _allow_real_api
from tool._gate import load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext

_DEFAULT_TOP_K = 10
_TOP_K_MIN = 1
_TOP_K_MAX = 100
_COMMONS_MAX_TOP_K = 50
_PROVIDER_COMMONS = "wikimedia_commons"
_PROVIDER_YOUTUBE = "youtube"
_KIND_PHOTO = "photo"
_KIND_VIDEO = "video"
_PHOTO_FILETYPE = "bitmap|drawing"
_VIDEO_FILETYPE = "video"
_FILE_NAMESPACE = "6"
_DEFAULT_COMMONS_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
_DEFAULT_YOUTUBE_ENDPOINT = "https://www.googleapis.com/youtube/v3/search"
_YOUTUBE_MAX_TOP_K = 50
_DEFAULT_TIMEOUT_SEC = 30.0
_HTTP_ATTEMPTS = 3
_RETRY_HTTP_STATUS = frozenset({429, 503})
_RETRY_DELAYS_SEC = (1.0, 2.0)
_RETRY_DELAY_CAP_SEC = 5.0
_DEFAULT_USER_AGENT = "geoagent-dataset/1.0 (media_search; local)"
_GEO_RADIUS_MIN = 10
_GEO_RADIUS_MAX = 10000
_DEFAULT_GEO_RADIUS_M = 5000
_YEAR_MIN = 1000
_YEAR_MAX = 2100
_YEAR_RE = re.compile(r"^(\d{4})$")
_YEAR_RANGE_RE = re.compile(r"^(\d{4})\s*[-–—至到]\s*(\d{4})$")
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
_DATE_RANGE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s*to\s*(\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_ISO_DATETIME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)(?:Z|[+-]\d{2}:\d{2})?$",
)
_YEAR_IN_TEXT_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_COMMONS_ALIASES = {
    "wikimedia_commons": _PROVIDER_COMMONS,
    "commons": _PROVIDER_COMMONS,
    "wikimedia": _PROVIDER_COMMONS,
    "mediawiki": _PROVIDER_COMMONS,
}
_YOUTUBE_ALIASES = frozenset({"youtube", "you_tube", "yt"})
_SECRET_QUERY_RE = re.compile(r"(?i)(key=)[^&\s\"']+")
_EXT_FILTER = (
    "DateTimeOriginal|LicenseShortName|LicenseUrl|Artist|"
    "ImageDescription|Credit|UsageTerms"
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
_ASSUMPTIONS = [
    "uploaded_at 是上传或发布时间，不能当作画面发生或拍摄时间",
    "captured_at 仅在来源元数据明确给出时填写，否则为 null",
    "YouTube 的 time_range 按视频发布时间过滤，不代表拍摄时间",
    "检索命中不授予下载或训练使用权",
    "本步只返回元数据与链接，未下载媒体",
    "未指定 platforms 时只检索 Wikimedia Commons；YouTube 需在 platforms 中写明",
]

class SearchInputError(Exception):
    """query / area / top_k 等输入无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """真实媒体检索未配置、被闸门拒绝或调用失败。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code

@dataclass(frozen=True)
class GeoArea:
    """已有坐标的检索范围；半径单位为米。"""

    lat: float
    lon: float
    radius_m: int

@dataclass(frozen=True)
class TimeWindow:
    """闭区间日期窗，只用于过滤 captured_at。"""

    start: date
    end: date

@dataclass(frozen=True)
class ParsedArea:
    """解析后的 area：地名文本、坐标，或无法映射的原值。"""

    text: str | None = None
    geo: GeoArea | None = None
    unsupported: Any = None
    applied: Any = None

@dataclass(frozen=True)
class MediaSearchRequest:
    """组装后的媒体检索请求。"""

    query: str
    media_kind: str
    top_k: int
    geo: GeoArea | None = None
    published_window: TimeWindow | None = None


@dataclass(frozen=True)
class SourceSelection:
    """本次要调用的来源；未显式指定时只走 Commons。"""

    commons: bool
    youtube: bool
    unsupported: tuple[Any, ...]
    explicit: bool

@runtime_checkable
class MediaSearchProvider(Protocol):
    """可注入的媒体检索后端；测试用 extras['media_search_provider'] 替换。"""

    def search(self, request: MediaSearchRequest) -> dict[str, Any]:
        """提交检索，返回 Commons `{query: {pages}}` 或 `{results: [...]}`。"""

class CommonsMediaSearchProvider:
    """Wikimedia Commons MediaWiki API 适配器；免 API Key，必须带 User-Agent。"""

    name = _PROVIDER_COMMONS
    max_top_k = _COMMONS_MAX_TOP_K

    def __init__(
        self,
        *,
        endpoint: str,
        timeout_sec: float,
        user_agent: str,
    ) -> None:
        self._endpoint = endpoint
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent

    def search(self, request: MediaSearchRequest) -> dict[str, Any]:
        params: dict[str, str] = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "imageinfo",
            "iiprop": "url|timestamp|user|mime|size|extmetadata",
            "iiextmetadatafilter": _EXT_FILTER,
        }
        if request.geo is not None:
            params.update(
                {
                    "generator": "geosearch",
                    "ggscoord": f"{request.geo.lat}|{request.geo.lon}",
                    "ggsradius": str(request.geo.radius_m),
                    "ggsnamespace": _FILE_NAMESPACE,
                    "ggslimit": str(request.top_k),
                }
            )
        else:
            filetype = (
                _VIDEO_FILETYPE if request.media_kind == _KIND_VIDEO else _PHOTO_FILETYPE
            )
            params.update(
                {
                    "generator": "search",
                    "gsrsearch": f"filetype:{filetype} {request.query}".strip(),
                    "gsrnamespace": _FILE_NAMESPACE,
                    "gsrlimit": str(request.top_k),
                }
            )
        url = _append_query(self._endpoint, urllib.parse.urlencode(params, safe=":|"))
        raw = _http_json(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self._user_agent,
            },
            timeout_sec=self._timeout_sec,
            error_prefix="Wikimedia Commons",
        )
        if not isinstance(raw, dict):
            raise EngineUnavailableError("Wikimedia Commons 回执不是 JSON 对象")
        return raw


class YouTubeMediaSearchProvider:
    """YouTube Data API v3 search.list；只查公开视频，不下载。"""

    name = _PROVIDER_YOUTUBE
    max_top_k = _YOUTUBE_MAX_TOP_K

    def __init__(self, *, endpoint: str, timeout_sec: float, api_key: str) -> None:
        self._endpoint = endpoint
        self._timeout_sec = timeout_sec
        self._api_key = api_key

    def search(self, request: MediaSearchRequest) -> dict[str, Any]:
        if request.media_kind != _KIND_VIDEO:
            raise EngineUnavailableError("YouTube 只支持 video_search")
        params: dict[str, str] = {
            "part": "snippet",
            "type": "video",
            "maxResults": str(request.top_k),
            "q": request.query,
            "key": self._api_key,
        }
        if request.geo is not None:
            params["location"] = f"{request.geo.lat},{request.geo.lon}"
            params["locationRadius"] = f"{request.geo.radius_m}m"
        if request.published_window is not None:
            window = request.published_window
            params["publishedAfter"] = f"{window.start.isoformat()}T00:00:00Z"
            before = window.end + timedelta(days=1)
            params["publishedBefore"] = f"{before.isoformat()}T00:00:00Z"
        url = _append_query(self._endpoint, urllib.parse.urlencode(params))
        raw = _http_json(
            url,
            headers={"Accept": "application/json"},
            timeout_sec=self._timeout_sec,
            error_prefix="YouTube",
        )
        if not isinstance(raw, dict):
            raise EngineUnavailableError("YouTube 回执不是 JSON 对象")
        return raw

def execute_video_search(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按关键词检索公开视频；默认 Wikimedia Commons，platforms 含 youtube 时走 YouTube。"""

    del purpose
    return _run_search(_KIND_VIDEO, inputs, ctx)

def execute_photo_search(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """检索历史照片或公共图库；第一期仅 Wikimedia Commons。"""

    del purpose
    return _run_search(_KIND_PHOTO, inputs, ctx)

def _run_search(
    media_kind: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    operation = "video_search" if media_kind == _KIND_VIDEO else "photo_search"
    try:
        query = _parse_query(
            inputs.get("query"),
            required=media_kind == _KIND_VIDEO,
        )
        area = _parse_area(_resolve_area(inputs.get("area"), ctx))
        if media_kind == _KIND_PHOTO and not query and area.text is None and area.geo is None:
            if area.unsupported is not None:
                raise SearchInputError("area 无法解析且缺少 query", "invalid_area")
            raise SearchInputError("缺少必填输入 query 或 area", "missing_input")
        time_window, unsupported_time = _parse_time_range(inputs.get("time_range"))
        requested_top_k = _parse_top_k(inputs.get("top_k"))
        selection = _parse_requested_providers(operation, inputs)
        providers, provider_errors = _select_providers(selection, ctx)
        assembled = " ".join(part for part in (query, area.text) if part)
        buckets: list[list[dict[str, Any]]] = []
        used_names: list[str] = []
        for provider in providers:
            provider_name = str(getattr(provider, "name", "injected"))
            server_side = provider_name == _PROVIDER_YOUTUBE
            max_top_k = _provider_max_top_k(provider)
            provider_top_k = min(requested_top_k, max_top_k)
            need_extra = (not server_side) and (
                time_window is not None or (area.geo is not None and bool(assembled))
            )
            fetch_k = max_top_k if need_extra else provider_top_k
            request = MediaSearchRequest(
                query=assembled,
                media_kind=media_kind,
                top_k=fetch_k,
                geo=area.geo,
                published_window=time_window if server_side else None,
            )
            try:
                payload = provider.search(request)
            except EngineUnavailableError as exc:
                provider_errors.append(_redact_secrets(str(exc)))
                continue
            provider_hits = _normalize_results(
                payload,
                media_kind=media_kind,
                provider_name=provider_name,
            )
            if (not server_side) and area.geo is not None and assembled:
                provider_hits = [
                    item for item in provider_hits if _matches_query(item, assembled)
                ]
            if time_window is not None and not server_side:
                provider_hits = [
                    item for item in provider_hits if _captured_in_window(item, time_window)
                ]
            buckets.append(provider_hits[:provider_top_k])
            used_names.append(provider_name)
        if not buckets:
            message = provider_errors[-1] if provider_errors else "没有可用的媒体检索来源"
            raise EngineUnavailableError(message)
        top_k = min(
            requested_top_k,
            max(_provider_max_top_k(provider) for provider in providers),
        )
        hits = _interleave(buckets, top_k)
    except SearchInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(_redact_secrets(str(exc)), exc.error_code)

    results = []
    for item in hits:
        row = dict(item)
        row["result_id"] = f"ms_{len(results) + 1}"
        results.append(row)

    applied: dict[str, Any] = {
        "provider": used_names[0] if len(used_names) == 1 else "+".join(used_names),
        "query": assembled,
        "top_k": top_k,
    }
    if area.applied is not None:
        applied["area"] = area.applied
    if time_window is not None:
        applied["time_range"] = (
            f"{time_window.start.isoformat()}to{time_window.end.isoformat()}"
        )
    unsupported: dict[str, Any] = {}
    source_key = "platforms" if operation == "video_search" else "sources"
    if selection.unsupported:
        unsupported[source_key] = list(selection.unsupported)
    if provider_errors:
        unsupported["provider_errors"] = list(provider_errors)
    if area.unsupported is not None:
        unsupported["area"] = area.unsupported
    if unsupported_time is not None:
        unsupported["time_range"] = unsupported_time
    if unsupported:
        applied["unsupported"] = unsupported

    result: dict[str, Any] = {
        "operation": operation,
        "results": results,
        "applied": applied,
        "assumptions": list(_ASSUMPTIONS),
    }
    return Observation(ok=True, result=_strip_forbidden(result))

def _parse_query(raw: Any, *, required: bool) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if required:
            raise SearchInputError("缺少必填输入 query", "missing_input")
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise SearchInputError("query 必须是字符串或字符串列表", "invalid_query")
            stripped = item.strip()
            if stripped:
                parts.append(stripped)
        if not parts:
            if required:
                raise SearchInputError("缺少必填输入 query", "missing_input")
            return ""
        return " ".join(parts)
    raise SearchInputError("query 必须是字符串或字符串列表", "invalid_query")

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
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        geo = _geo_from_bbox_values(list(raw))
        if geo is None:
            return ParsedArea(unsupported=raw)
        return ParsedArea(geo=geo, applied=_geo_applied(geo))
    if isinstance(raw, dict):
        geo = _geo_from_mapping(raw)
        if geo is not None:
            return ParsedArea(geo=geo, applied=_geo_applied(geo))
        return ParsedArea(unsupported=raw)
    return ParsedArea(unsupported=raw)

def _geo_from_mapping(raw: dict[str, Any]) -> GeoArea | None:
    lat = _as_float(raw.get("lat", raw.get("latitude")))
    lon = _as_float(raw.get("lon", raw.get("lng", raw.get("longitude"))))
    if lat is not None and lon is not None:
        radius = _as_float(raw.get("radius_m", raw.get("radius", _DEFAULT_GEO_RADIUS_M)))
        if radius is None:
            radius = float(_DEFAULT_GEO_RADIUS_M)
        return _validated_geo(lat, lon, int(round(radius)))
    bbox = raw.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return _geo_from_bbox_values(list(bbox))
    west = _as_float(raw.get("west", raw.get("min_lon")))
    south = _as_float(raw.get("south", raw.get("min_lat")))
    east = _as_float(raw.get("east", raw.get("max_lon")))
    north = _as_float(raw.get("north", raw.get("max_lat")))
    if None not in (west, south, east, north):
        return _geo_from_corners(south, west, north, east)
    return None

def _geo_from_bbox_values(values: list[Any]) -> GeoArea | None:
    nums = [_as_float(item) for item in values]
    if any(item is None for item in nums):
        return None
    first, second, third, fourth = nums[0], nums[1], nums[2], nums[3]
    # GeoJSON [min_lon, min_lat, max_lon, max_lat]：经度绝对值常大于纬度。
    if abs(first) <= 90 and abs(third) <= 90 and (abs(second) > 90 or abs(fourth) > 90):
        return _geo_from_corners(first, second, third, fourth)
    return _geo_from_corners(second, first, fourth, third)

def _geo_from_corners(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> GeoArea | None:
    lat = (min_lat + max_lat) / 2.0
    lon = (min_lon + max_lon) / 2.0
    center_lat_rad = math.radians(lat)
    lat_m = abs(max_lat - min_lat) * 111320.0 / 2.0
    lon_m = abs(max_lon - min_lon) * 111320.0 * math.cos(center_lat_rad) / 2.0
    radius = int(round(math.hypot(lat_m, lon_m)))
    return _validated_geo(lat, lon, radius)

def _validated_geo(lat: float, lon: float, radius_m: int) -> GeoArea | None:
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    radius = max(_GEO_RADIUS_MIN, min(_GEO_RADIUS_MAX, radius_m))
    return GeoArea(lat=lat, lon=lon, radius_m=radius)

def _geo_applied(geo: GeoArea) -> dict[str, float | int]:
    return {"lat": geo.lat, "lon": geo.lon, "radius_m": geo.radius_m}

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

def _parse_requested_providers(
    operation: str,
    inputs: dict[str, Any],
) -> SourceSelection:
    key = "platforms" if operation == "video_search" else "sources"
    raw = inputs.get(key)
    if raw is None or raw == "" or raw == []:
        return SourceSelection(
            commons=True,
            youtube=False,
            unsupported=(),
            explicit=False,
        )
    if isinstance(raw, str):
        items: list[Any] = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        raise SearchInputError(f"{key} 必须是字符串列表", "invalid_query")
    want_commons = False
    want_youtube = False
    unsupported: list[Any] = []
    for item in items:
        if not isinstance(item, str):
            raise SearchInputError(f"{key} 必须是字符串列表", "invalid_query")
        lowered = item.strip().lower()
        if _COMMONS_ALIASES.get(lowered) == _PROVIDER_COMMONS:
            want_commons = True
        elif operation == "video_search" and lowered in _YOUTUBE_ALIASES:
            want_youtube = True
        else:
            unsupported.append(item)
    if not want_commons and not want_youtube:
        raise EngineUnavailableError(f"未实现来源: {unsupported}")
    return SourceSelection(
        commons=want_commons,
        youtube=want_youtube,
        unsupported=tuple(unsupported),
        explicit=True,
    )

def _parse_time_range(raw: Any) -> tuple[TimeWindow | None, Any | None]:
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
        date_range = _DATE_RANGE_RE.fullmatch(re.sub(r"\s+", "", text))
        if date_range:
            start_d = _parse_iso_date(date_range.group(1))
            end_d = _parse_iso_date(date_range.group(2))
            if start_d is not None and end_d is not None:
                lo, hi = (start_d, end_d) if start_d <= end_d else (end_d, start_d)
                return TimeWindow(lo, hi), None
        return None, raw
    if isinstance(raw, dict):
        start = _bound_to_date(raw.get("start"), start=True)
        end = _bound_to_date(raw.get("end"), start=False)
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
        return date.fromisoformat(text)
    except ValueError:
        return None

def _parse_top_k(raw: Any) -> int:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DEFAULT_TOP_K
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SearchInputError("top_k 必须是整数", "invalid_top_k") from exc
    return max(_TOP_K_MIN, min(_TOP_K_MAX, value))

def _provider_max_top_k(provider: MediaSearchProvider) -> int:
    raw = getattr(provider, "max_top_k", _COMMONS_MAX_TOP_K)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _COMMONS_MAX_TOP_K
    if value < _TOP_K_MIN:
        return _COMMONS_MAX_TOP_K
    return min(value, _TOP_K_MAX)

def _select_providers(
    selection: SourceSelection,
    ctx: RuntimeContext | None,
) -> tuple[list[MediaSearchProvider], list[str]]:
    providers: list[MediaSearchProvider] = []
    errors: list[str] = []
    if selection.commons:
        providers.append(_resolve_provider(ctx))
    if selection.youtube:
        try:
            providers.append(_resolve_youtube(ctx))
        except EngineUnavailableError as exc:
            if not providers:
                raise
            errors.append(_redact_secrets(str(exc)))
    if not providers:
        raise EngineUnavailableError("没有可用的媒体检索来源")
    return providers, errors

def _resolve_youtube(ctx: RuntimeContext | None) -> MediaSearchProvider:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("youtube_search_provider")
    if injected is not None:
        if not isinstance(injected, MediaSearchProvider):
            raise EngineUnavailableError("youtube_search_provider 必须提供 search(request)")
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_TOOL_API=false，禁止调用 YouTube Data API")
    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise EngineUnavailableError("未配置 YOUTUBE_API_KEY，无法检索 YouTube")
    return YouTubeMediaSearchProvider(
        endpoint=_env_value("YOUTUBE_DATA_API_ENDPOINT", _DEFAULT_YOUTUBE_ENDPOINT),
        timeout_sec=_env_timeout(),
        api_key=api_key,
    )

def _resolve_provider(ctx: RuntimeContext | None) -> MediaSearchProvider:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("media_search_provider")
    if injected is not None:
        if not isinstance(injected, MediaSearchProvider):
            raise EngineUnavailableError(
                "media_search_provider 必须提供 search(request)",
            )
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError(
            "ALLOW_REAL_TOOL_API=false，禁止调用真实媒体检索 API",
        )
    return CommonsMediaSearchProvider(
        endpoint=_env_value("MEDIA_SEARCH_COMMONS_ENDPOINT", _DEFAULT_COMMONS_ENDPOINT),
        timeout_sec=_env_timeout(),
        user_agent=_env_value("MEDIA_SEARCH_USER_AGENT", _DEFAULT_USER_AGENT),
    )

def _env_value(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default

def _env_timeout() -> float:
    raw = os.environ.get("MEDIA_SEARCH_TIMEOUT_SEC", "").strip()
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
    """请求 JSON。429/503 和连接中断再试两次；其他 HTTP 状态立即失败。"""

    last_transport: Exception | None = None
    for attempt in range(_HTTP_ATTEMPTS):
        request = urllib.request.Request(url, method="GET")
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRY_HTTP_STATUS or attempt == _HTTP_ATTEMPTS - 1:
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
                raise EngineUnavailableError(
                    _redact_secrets(f"{error_prefix} HTTP {exc.code}: {detail[:200]}"),
                ) from exc
            time.sleep(_retry_delay(attempt, exc))
            continue
        except urllib.error.URLError as exc:
            last_transport = exc
            if attempt == _HTTP_ATTEMPTS - 1:
                raise EngineUnavailableError(f"{error_prefix} 网络失败: {exc.reason}") from exc
            time.sleep(_retry_delay(attempt))
            continue
        except json.JSONDecodeError as exc:
            raise EngineUnavailableError(f"{error_prefix} 调用失败: {exc}") from exc
        except (TimeoutError, OSError) as exc:
            last_transport = exc
            if attempt == _HTTP_ATTEMPTS - 1:
                raise EngineUnavailableError(f"{error_prefix} 调用失败: {exc}") from exc
            time.sleep(_retry_delay(attempt))
            continue
        if isinstance(raw, dict) and raw.get("error"):
            raise EngineUnavailableError(
                _redact_secrets(f"{error_prefix} 失败: {raw['error']}")[:400],
            )
        return raw
    raise EngineUnavailableError(f"{error_prefix} 网络失败: {last_transport}")


def _retry_delay(attempt: int, exc: urllib.error.HTTPError | None = None) -> float:
    """限流时优先采用 Retry-After，并限制单次等待。"""

    if exc is not None and exc.headers is not None:
        raw = exc.headers.get("Retry-After")
        if raw:
            try:
                parsed = float(str(raw).strip())
            except ValueError:
                parsed = -1.0
            if parsed >= 0:
                return min(parsed, _RETRY_DELAY_CAP_SEC)
    index = min(attempt, len(_RETRY_DELAYS_SEC) - 1)
    return _RETRY_DELAYS_SEC[index]

def _append_query(endpoint: str, query: str) -> str:
    separator = "&" if urllib.parse.urlparse(endpoint).query else "?"
    return f"{endpoint}{separator}{query}"

def _interleave(buckets: list[list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    pools = [list(bucket) for bucket in buckets]
    merged: list[dict[str, Any]] = []
    while len(merged) < limit and any(pools):
        for pool in pools:
            if pool and len(merged) < limit:
                merged.append(pool.pop(0))
    return merged

def _redact_secrets(text: str) -> str:
    return _SECRET_QUERY_RE.sub(r"\1REDACTED", text)

def _normalize_results(
    payload: dict[str, Any],
    *,
    media_kind: str,
    provider_name: str,
) -> list[dict[str, Any]]:
    if (
        provider_name == _PROVIDER_YOUTUBE
        and isinstance(payload, dict)
        and isinstance(payload.get("items"), list)
    ):
        return _normalize_youtube_items(payload)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        hits: list[dict[str, Any]] = []
        for item in payload["results"]:
            if not isinstance(item, dict):
                continue
            hit = _normalize_injected_hit(item, media_kind, provider_name)
            if hit is not None:
                hits.append(hit)
        return hits
    hits = []
    for page in _pages_from_payload(payload):
        hit = _normalize_commons_page(page, media_kind, provider_name)
        if hit is not None:
            hits.append(hit)
    return hits

def _normalize_youtube_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    items = payload.get("items")
    if not isinstance(items, list):
        return hits
    for item in items:
        if not isinstance(item, dict):
            continue
        id_block = item.get("id") if isinstance(item.get("id"), dict) else {}
        video_id = str(id_block.get("videoId") or "").strip()
        if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{6,}", video_id):
            continue
        snippet = item.get("snippet") if isinstance(item.get("snippet"), dict) else {}
        page_url = f"https://www.youtube.com/watch?v={video_id}"
        hits.append(
            {
                "media_id": video_id,
                "title": html.unescape(str(snippet.get("title") or "")).strip(),
                "url": page_url,
                "source_url": page_url,
                "media_kind": _KIND_VIDEO,
                "provider": _PROVIDER_YOUTUBE,
                "author": html.unescape(str(snippet.get("channelTitle") or "")).strip(),
                "license": "",
                "license_url": "",
                "uploaded_at": str(snippet.get("publishedAt") or "").strip(),
                "captured_at": None,
                "snippet": html.unescape(str(snippet.get("description") or "")).strip(),
            }
        )
    return hits

def _pages_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    query = payload.get("query") if isinstance(payload, dict) else None
    if not isinstance(query, dict):
        return []
    pages = query.get("pages")
    if isinstance(pages, list):
        return [item for item in pages if isinstance(item, dict)]
    if isinstance(pages, dict):
        return [item for item in pages.values() if isinstance(item, dict)]
    return []

def _normalize_commons_page(
    page: dict[str, Any],
    media_kind: str,
    provider_name: str,
) -> dict[str, Any] | None:
    infos = page.get("imageinfo")
    info = infos[0] if isinstance(infos, list) and infos and isinstance(infos[0], dict) else {}
    mime = str(info.get("mime") or "")
    if not _mime_matches(mime, media_kind):
        return None
    title = str(page.get("title") or "").strip()
    if not title:
        return None
    page_url = _clean_url(info.get("descriptionurl")) or _commons_page_url(title)
    source_url = _clean_url(info.get("url"))
    if not page_url and not source_url:
        return None
    ext = info.get("extmetadata") if isinstance(info.get("extmetadata"), dict) else {}
    display_title = title[5:] if title.lower().startswith("file:") else title
    captured = _strip_html(_ext_value(ext, "DateTimeOriginal")) or None
    return {
        "media_id": title,
        "title": display_title,
        "url": page_url or source_url,
        "source_url": source_url or page_url,
        "media_kind": media_kind,
        "provider": provider_name,
        "author": _strip_html(_ext_value(ext, "Artist")),
        "license": _strip_html(_ext_value(ext, "LicenseShortName")),
        "license_url": _clean_url(_ext_value(ext, "LicenseUrl")),
        "uploaded_at": str(info.get("timestamp") or "").strip(),
        "captured_at": captured,
        "snippet": _strip_html(_ext_value(ext, "ImageDescription")),
    }

def _normalize_injected_hit(
    item: dict[str, Any],
    media_kind: str,
    provider_name: str,
) -> dict[str, Any] | None:
    kind = str(item.get("media_kind") or media_kind).strip() or media_kind
    if kind not in {_KIND_PHOTO, _KIND_VIDEO}:
        kind = media_kind
    if kind != media_kind:
        return None
    url = _clean_url(item.get("url"))
    source_url = _clean_url(item.get("source_url"))
    media_id = str(item.get("media_id") or "").strip()
    if not url and not source_url and not media_id:
        return None
    captured_raw = item.get("captured_at")
    captured = None
    if captured_raw is not None and not isinstance(captured_raw, bool):
        text = _strip_html(str(captured_raw)).strip()
        captured = text or None
    return {
        "media_id": media_id,
        "title": str(item.get("title") or "").strip(),
        "url": url or source_url,
        "source_url": source_url or url,
        "media_kind": kind,
        "provider": str(item.get("provider") or provider_name),
        "author": _strip_html(str(item.get("author") or "")),
        "license": _strip_html(str(item.get("license") or "")),
        "license_url": _clean_url(item.get("license_url")),
        "uploaded_at": str(item.get("uploaded_at") or "").strip(),
        "captured_at": captured,
        "snippet": _strip_html(str(item.get("snippet") or "")),
    }

def _mime_matches(mime: str, media_kind: str) -> bool:
    lowered = mime.strip().lower()
    if not lowered:
        return True
    if media_kind == _KIND_VIDEO:
        return lowered.startswith("video/")
    if lowered.startswith("video/") or lowered.startswith("audio/"):
        return False
    return lowered.startswith("image/") or lowered.startswith("application/")

def _ext_value(ext: dict[str, Any], key: str) -> str:
    block = ext.get(key)
    if isinstance(block, dict):
        value = block.get("value")
    else:
        value = block
    if value is None or isinstance(value, bool):
        return ""
    return str(value)

def _commons_page_url(title: str) -> str:
    quoted = urllib.parse.quote(title.replace(" ", "_"), safe=":/()")
    return f"https://commons.wikimedia.org/wiki/{quoted}"

def _matches_query(hit: dict[str, Any], query: str) -> bool:
    tokens = [token.casefold() for token in query.split() if token.strip()]
    if not tokens:
        return True
    haystack = " ".join(
        str(hit.get(key) or "")
        for key in ("title", "snippet", "media_id", "url")
    ).casefold()
    return all(token in haystack for token in tokens)

def _captured_in_window(hit: dict[str, Any], window: TimeWindow) -> bool:
    bounds = _captured_bounds(hit.get("captured_at"))
    if bounds is None:
        return False
    start, end = bounds
    return start <= window.end and end >= window.start

def _captured_bounds(raw: Any) -> tuple[date, date] | None:
    if raw is None or isinstance(raw, bool):
        return None
    text = _strip_html(str(raw)).strip()
    if not text:
        return None
    iso_dt = _ISO_DATETIME_RE.fullmatch(text)
    if iso_dt:
        parsed = _parse_iso_date(iso_dt.group(1))
        if parsed is not None:
            return parsed, parsed
    iso_day = _ISO_DATE_RE.search(text)
    if iso_day:
        parsed = _parse_iso_date(iso_day.group(1))
        if parsed is not None:
            return parsed, parsed
    if _YEAR_RE.fullmatch(text):
        year = int(text)
        if _YEAR_MIN <= year <= _YEAR_MAX:
            return date(year, 1, 1), date(year, 12, 31)
    years = [
        int(match.group(1))
        for match in _YEAR_IN_TEXT_RE.finditer(text)
        if _YEAR_MIN <= int(match.group(1)) <= _YEAR_MAX
    ]
    if len(years) == 1:
        year = years[0]
        return date(year, 1, 1), date(year, 12, 31)
    if len(years) >= 2:
        lo, hi = min(years), max(years)
        return date(lo, 1, 1), date(hi, 12, 31)
    return None

def _strip_html(raw: str) -> str:
    text = html.unescape(raw)
    text = _HTML_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def _clean_url(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    if not text:
        return ""
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return text

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
