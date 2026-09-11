"""administrative_registry 共享执行器：GeoNames 通用地名库 + 可选官方档案。"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tool._gate import allow_real_api as _allow_real_api, load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext, declared_inputs

_OP_ADMIN = "administrative"
_OP_DIRECTORY = "directory"
_PROVIDER_GEONAMES = "geonames"
_PROVIDER_OFFICIAL = "official"
_CRS_WGS84 = "wgs84"
_ADMIN_TOP_K = 5
_DIRECTORY_TOP_K = 10
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_GEONAMES_ENDPOINT = "https://secure.geonames.org"
_DEFAULT_USER_AGENT = "geoagent-dataset/1.0 (administrative_registry; local)"
_METERS_PER_DEG_LAT = 111_320.0
_YEAR_MIN = 1000
_YEAR_MAX = 2100
_GEONAMES_PAGE = "https://www.geonames.org"
_INPUT_FIELDS = ("query", "area", "time_range", "registry", "fields")
_ALWAYS_RESULT_KEYS = ("result_id", "validity", "evidence")
_YEAR_RE = re.compile(r"^(\d{4})$")
_YEAR_RANGE_RE = re.compile(r"^(\d{4})\s*[-–—至到]\s*(\d{4})$")
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
_DATE_RANGE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s*(?:to|/|[-–—至到])\s*(\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)
_GEONAMES_ALIASES = frozenset({"geonames", "geoname", "geo_names"})
_OFFICIAL_ALIASES = frozenset(
    {
        "official",
        "mca",
        "china",
        "gazetteer",
        "china_place_names",
        "国家地名信息库",
        "民政",
        "地名信息库",
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
_PHOTO_POINT_ASSUMPTION = "候选记录是地名或档案匹配，不是照片拍摄点"
_CRS_ASSUMPTION = "当前后端坐标为 WGS84，未做坐标系转换"
_NO_SESSION_ASSUMPTION = "本步只返回档案条目，未打开街景或地图会话"
_GEONAMES_LICENSE_ASSUMPTION = "GeoNames 结果需按来源条款署名，不授予训练使用权"
_HISTORICAL_ASSUMPTION = (
    "现在属于某区，不代表拍摄当年也属于该区；本回执只表示当前资料适用范围，不能自行向过去延伸"
)
_NO_OFFICIAL_ASSUMPTION = "未配置地区官方来源，结果仅为通用地名库当前资料"

class RegistryInputError(Exception):
    """query / area / time_range / registry / fields 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """真实地名档案未配置、被闸门拒绝或调用失败。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code

@dataclass(frozen=True)
class GeoPoint:
    """WGS84 点。"""

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
class TimeWindow:
    """闭区间日期窗；不作为 GeoNames 历史快照参数。"""

    start: date
    end: date

@dataclass(frozen=True)
class ParsedArea:
    """解析后的可选范围。"""

    text: str | None = None
    center: GeoPoint | None = None
    radius_m: int | None = None
    bbox: BBox | None = None
    applied: Any = None
    unsupported: Any = None

@dataclass(frozen=True)
class RegistryRequest:
    """组装后的地名档案检索。"""

    operation: str
    query: str
    top_k: int
    area_text: str | None = None
    bbox: BBox | None = None
    time_window: TimeWindow | None = None

@runtime_checkable
class GazetteerProvider(Protocol):
    """可注入的地名档案后端；测试用 extras 替换。"""

    def search(self, request: RegistryRequest) -> list[dict[str, Any]]:
        """按查询返回可归一化的原始记录列表。"""

    def hierarchy(self, provider_id: str) -> list[dict[str, Any]]:
        """返回从根到该对象的上下级链；没有则返回空列表。"""

class GeoNamesProvider:
    """GeoNames Web 服务适配器；坐标为 WGS84。"""

    name = _PROVIDER_GEONAMES
    crs = _CRS_WGS84

    def __init__(
        self,
        *,
        username: str,
        endpoint: str,
        timeout_sec: float,
        user_agent: str,
    ) -> None:
        self._username = username
        self._endpoint = endpoint.rstrip("/")
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent

    def search(self, request: RegistryRequest) -> list[dict[str, Any]]:
        query = _search_query(request)
        params: dict[str, str] = {
            "q": query,
            "maxRows": str(request.top_k),
            "username": self._username,
            "style": "FULL",
        }
        if _has_cjk(query):
            params["lang"] = "zh"
        if request.bbox is not None:
            params["north"] = _fmt_coord(request.bbox.north)
            params["south"] = _fmt_coord(request.bbox.south)
            params["east"] = _fmt_coord(request.bbox.east)
            params["west"] = _fmt_coord(request.bbox.west)
        payload = self._get("searchJSON", params)
        hits = payload.get("geonames")
        if not isinstance(hits, list):
            return []
        return [item for item in hits if isinstance(item, dict)]

    def hierarchy(self, provider_id: str) -> list[dict[str, Any]]:
        if not provider_id:
            return []
        payload = self._get(
            "hierarchyJSON",
            {"geonameId": provider_id, "username": self._username},
        )
        hits = payload.get("geonames")
        if not isinstance(hits, list):
            return []
        return [item for item in hits if isinstance(item, dict)]

    def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        url = _append_query(f"{self._endpoint}/{path}", urllib.parse.urlencode(params))
        raw = _http_json(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self._user_agent,
            },
            timeout_sec=self._timeout_sec,
            error_prefix="GeoNames",
        )
        if not isinstance(raw, dict):
            raise EngineUnavailableError("GeoNames 回执不是 JSON 对象")
        status = raw.get("status")
        if isinstance(status, dict):
            message = _text(status.get("message")) or "unknown"
            raise EngineUnavailableError(f"GeoNames 失败: {message}")
        return raw

class OfficialDumpProvider:
    """本地官方地名 dump；无统一远程 API。"""

    name = _PROVIDER_OFFICIAL
    crs = _CRS_WGS84

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def search(self, request: RegistryRequest) -> list[dict[str, Any]]:
        tokens = [part.casefold() for part in request.query.split() if part]
        area_l = (request.area_text or "").strip().casefold()
        hits: list[dict[str, Any]] = []
        for record in self._records:
            blob = _record_blob(record)
            if tokens and not all(token in blob for token in tokens):
                continue
            if area_l and area_l not in blob:
                continue
            if request.time_window is not None and _has_stated_interval(record):
                if not _overlaps_interval(request.time_window, record):
                    continue
            hits.append(record)
            if len(hits) >= request.top_k:
                break
        return hits

    def hierarchy(self, provider_id: str) -> list[dict[str, Any]]:
        if not provider_id:
            return []
        for record in self._records:
            if _record_provider_id(record) == provider_id:
                return _parents_to_hierarchy(record)
        return []

def execute_administrative(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """查询地点的行政归属、标准地名或水体名称。"""

    return _execute(_OP_ADMIN, purpose=purpose, inputs=inputs, ctx=ctx)

def execute_directory(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """查询特定类别设施、机构或对象名录。"""

    return _execute(_OP_DIRECTORY, purpose=purpose, inputs=inputs, ctx=ctx)

def _execute(
    operation: str,
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    del purpose
    try:
        inputs = declared_inputs(inputs, *_INPUT_FIELDS)
        query = _parse_query(inputs.get("query"))
        if not query:
            raise RegistryInputError("缺少必填输入 query", "missing_input")
        area = _parse_area(_resolve_area(inputs.get("area"), ctx))
        time_window, unsupported_time = _parse_time_range(inputs.get("time_range"))
        registry = _parse_registry(inputs.get("registry"))
        fields = _parse_fields(inputs.get("fields"))
        top_k = _ADMIN_TOP_K if operation == _OP_ADMIN else _DIRECTORY_TOP_K
        providers, used_official = _resolve_providers(ctx, registry)
        request_bbox, unused_area = _route_area(area)
        request = RegistryRequest(
            operation=operation,
            query=query,
            top_k=top_k,
            area_text=area.text,
            bbox=request_bbox,
            time_window=time_window,
        )
        hits = _collect_hits(
            providers,
            request,
            operation=operation,
            time_window=time_window,
        )
        return _ok(
            hits[:top_k],
            operation=operation,
            providers=providers,
            query=query,
            top_k=top_k,
            area=area,
            registry=registry,
            fields=fields,
            time_applied=_window_applied(time_window),
            unused_area=unused_area,
            unused_time=unsupported_time,
            used_official=used_official,
        )
    except RegistryInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def _collect_hits(
    providers: list[GazetteerProvider],
    request: RegistryRequest,
    *,
    operation: str,
    time_window: TimeWindow | None,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for provider in providers:
        provider_name = _provider_name(provider)
        for raw in provider.search(request):
            if not isinstance(raw, dict):
                continue
            hit = _normalize_record(
                raw,
                provider=provider,
                provider_name=provider_name,
                operation=operation,
                time_window=time_window,
            )
            if hit is not None:
                hits.append(hit)
    return _dedupe(hits)

def _ok(
    results: list[dict[str, Any]],
    *,
    operation: str,
    providers: list[GazetteerProvider],
    query: str,
    top_k: int,
    area: ParsedArea,
    registry: str | None,
    fields: list[str] | None,
    time_applied: dict[str, str] | None,
    unused_area: dict[str, Any],
    unused_time: Any,
    used_official: bool,
) -> Observation:
    prefix = "admin" if operation == _OP_ADMIN else "directory"
    numbered: list[dict[str, Any]] = []
    for item in results:
        row = dict(item)
        row["result_id"] = f"{prefix}_{len(numbered) + 1}"
        numbered.append(_project_fields(row, fields))
    names = [_provider_name(provider) for provider in providers]
    applied: dict[str, Any] = {
        "provider": "+".join(names) if names else _PROVIDER_GEONAMES,
        "providers": names,
        "query": query,
        "top_k": top_k,
        "crs": _CRS_WGS84,
    }
    if registry:
        applied["registry"] = registry
    if area.applied is not None:
        applied["area"] = area.applied
    if time_applied is not None:
        applied["time_range"] = time_applied
    if fields:
        applied["fields"] = fields
    unused: dict[str, Any] = dict(unused_area)
    if unused_time is not None:
        unused["time_range"] = unused_time
    if unused:
        applied["unsupported"] = unused
    result: dict[str, Any] = {
        "operation": operation,
        "results": numbered,
        "applied": applied,
        "assumptions": _assumptions(
            names,
            time_requested=time_applied is not None,
            used_official=used_official,
            results=numbered,
        ),
    }
    return Observation(ok=True, result=_strip_forbidden(result))

def _assumptions(
    provider_names: list[str],
    *,
    time_requested: bool,
    used_official: bool,
    results: list[dict[str, Any]],
) -> list[str]:
    items = [_PHOTO_POINT_ASSUMPTION, _CRS_ASSUMPTION, _NO_SESSION_ASSUMPTION]
    if _PROVIDER_GEONAMES in provider_names:
        items.append(_GEONAMES_LICENSE_ASSUMPTION)
    if not used_official and _PROVIDER_GEONAMES in provider_names:
        items.append(_NO_OFFICIAL_ASSUMPTION)
    stated = any(
        isinstance(item.get("validity"), dict)
        and item["validity"].get("historical_affiliation_stated") is True
        for item in results
    )
    if time_requested and not stated:
        items.append(_HISTORICAL_ASSUMPTION)
    return items

def _project_fields(hit: dict[str, Any], fields: list[str] | None) -> dict[str, Any]:
    if not fields:
        return hit
    keep = set(fields) | set(_ALWAYS_RESULT_KEYS)
    return {key: value for key, value in hit.items() if key in keep}

def _normalize_record(
    raw: dict[str, Any],
    *,
    provider: GazetteerProvider,
    provider_name: str,
    operation: str,
    time_window: TimeWindow | None,
) -> dict[str, Any] | None:
    standard_name = _standard_name(raw)
    if not standard_name:
        return None
    hit: dict[str, Any] = {"standard_name": standard_name}
    aliases = _aliases(raw, standard_name)
    if aliases:
        hit["aliases"] = aliases
    admin_level = _text(raw.get("admin_level") or raw.get("fcode") or raw.get("feature_code"))
    if admin_level:
        hit["admin_level"] = admin_level
        hit["feature_code"] = admin_level
    fclass = _text(raw.get("fcl") or raw.get("feature_class"))
    if fclass:
        hit["feature_class"] = fclass
    location = _location(raw)
    if location is not None:
        hit["location"] = location
    provider_id = _record_provider_id(raw)
    evidence: dict[str, Any] = {"source": _text(raw.get("source")) or provider_name}
    if provider_id:
        evidence["provider_id"] = provider_id
    url = _text(raw.get("url")) or _evidence_url(provider_name, provider_id)
    if url:
        evidence["url"] = url
    hit["evidence"] = evidence
    hit["validity"] = _validity(raw, time_window=time_window)
    hierarchy = _resolve_hierarchy(
        raw,
        provider=provider,
        provider_id=provider_id,
        operation=operation,
    )
    if operation == _OP_ADMIN or hierarchy:
        hit["hierarchy"] = hierarchy
    return hit

def _resolve_hierarchy(
    raw: dict[str, Any],
    *,
    provider: GazetteerProvider,
    provider_id: str,
    operation: str,
) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    if operation == _OP_ADMIN and provider_id:
        try:
            chain = _normalize_hierarchy(provider.hierarchy(provider_id) or [])
        except EngineUnavailableError:
            chain = []
    if not chain:
        chain = _normalize_hierarchy(raw.get("hierarchy") or [])
    if not chain:
        chain = _hierarchy_from_admin_names(raw)
    if not chain:
        chain = _parents_to_hierarchy(raw)
    return chain

def _validity(raw: dict[str, Any], *, time_window: TimeWindow | None) -> dict[str, Any]:
    start, end = _record_interval(raw)
    stated = start is not None or end is not None
    historical = stated and (time_window is None or _overlaps_interval(time_window, raw))
    payload: dict[str, Any] = {
        "scope": "stated_interval" if stated else "current_source",
        "historical_affiliation_stated": historical,
    }
    if start is not None:
        payload["valid_from"] = start.isoformat()
    if end is not None:
        payload["valid_to"] = end.isoformat()
    if time_window is not None:
        payload["time_range_requested"] = _window_applied(time_window)
    return payload

def _standard_name(raw: dict[str, Any]) -> str:
    return _text(
        raw.get("standard_name")
        or raw.get("toponymName")
        or raw.get("name")
        or raw.get("asciiName")
    )

def _aliases(raw: dict[str, Any], standard_name: str) -> list[str]:
    aliases: list[str] = []
    seen = {standard_name}
    extra = raw.get("aliases")
    if isinstance(extra, list):
        for item in extra:
            text = _text(item) if not isinstance(item, dict) else _text(item.get("name"))
            if text and text not in seen:
                seen.add(text)
                aliases.append(text)
    alternates = raw.get("alternateNames")
    if isinstance(alternates, list):
        for item in alternates:
            text = _text(item.get("name")) if isinstance(item, dict) else _text(item)
            if text and text not in seen:
                seen.add(text)
                aliases.append(text)
    for candidate in (_text(raw.get("toponymName")), _text(raw.get("name"))):
        if candidate and candidate not in seen:
            seen.add(candidate)
            aliases.append(candidate)
    return aliases

def _location(raw: dict[str, Any]) -> dict[str, float | str] | None:
    nested = raw.get("location")
    if isinstance(nested, dict):
        lat = _as_float(nested.get("lat", nested.get("latitude")))
        lon = _as_float(nested.get("lon", nested.get("lng", nested.get("longitude"))))
        if lat is not None and lon is not None:
            return {"lon": lon, "lat": lat, "crs": _text(nested.get("crs")) or _CRS_WGS84}
    lat = _as_float(raw.get("lat", raw.get("latitude")))
    lon = _as_float(raw.get("lon", raw.get("lng", raw.get("longitude"))))
    if lat is None or lon is None:
        return None
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return {"lon": lon, "lat": lat, "crs": _CRS_WGS84}

def _evidence_url(provider_name: str, provider_id: str) -> str:
    if provider_name == _PROVIDER_GEONAMES and provider_id:
        return f"{_GEONAMES_PAGE}/{provider_id}"
    return ""

def _normalize_hierarchy(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    chain: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            name = item.strip()
            if name:
                chain.append({"name": name})
            continue
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name") or item.get("toponymName") or item.get("standard_name"))
        if not name:
            continue
        node: dict[str, Any] = {"name": name}
        level = _text(item.get("admin_level") or item.get("fcode") or item.get("feature_code"))
        if level:
            node["admin_level"] = level
        provider_id = _record_provider_id(item)
        if provider_id:
            node["provider_id"] = provider_id
        chain.append(node)
    return chain

def _hierarchy_from_admin_names(raw: dict[str, Any]) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    country = _text(raw.get("countryName") or raw.get("country"))
    if country:
        chain.append({"name": country, "admin_level": "PCL"})
    for index, key in enumerate(
        ("adminName1", "adminName2", "adminName3", "adminName4", "adminName5"),
        start=1,
    ):
        name = _text(raw.get(key))
        if name:
            chain.append({"name": name, "admin_level": f"ADM{index}"})
    return chain

def _parents_to_hierarchy(raw: dict[str, Any]) -> list[dict[str, Any]]:
    parents = raw.get("parents") or raw.get("hierarchy")
    if isinstance(parents, str):
        parts = [part.strip() for part in re.split(r"[|,，、]", parents) if part.strip()]
        return [{"name": part} for part in parts]
    return _normalize_hierarchy(parents)

def _dedupe(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for hit in hits:
        evidence = hit.get("evidence") if isinstance(hit.get("evidence"), dict) else {}
        key = (evidence.get("source"), evidence.get("provider_id"), hit.get("standard_name"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique

def _parse_query(raw: Any) -> str:
    if raw is None or raw == "":
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise RegistryInputError("query 必须是字符串或字符串列表", "invalid_query")
            stripped = item.strip()
            if stripped:
                parts.append(stripped)
        return " ".join(parts)
    raise RegistryInputError("query 必须是字符串或字符串列表", "invalid_query")

def _parse_registry(raw: Any) -> str | None:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise RegistryInputError("registry 必须是字符串", "invalid_registry")
    text = raw.strip()
    return text or None

def _parse_fields(raw: Any) -> list[str] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        parts = [part.strip() for part in re.split(r"[|,，、\s]+", raw) if part.strip()]
        return parts or None
    if isinstance(raw, list):
        fields: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise RegistryInputError("fields 必须是字符串或字符串列表", "invalid_fields")
            fields.append(item.strip())
        return fields or None
    raise RegistryInputError("fields 必须是字符串或字符串列表", "invalid_fields")

def _parse_time_range(raw: Any) -> tuple[TimeWindow | None, Any | None]:
    if raw is None or raw == "":
        return None, None
    if isinstance(raw, bool):
        raise RegistryInputError("time_range 必须是字符串或对象", "invalid_time_range")
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
                lo_d, hi_d = (start_d, end_d) if start_d <= end_d else (end_d, start_d)
                return TimeWindow(lo_d, hi_d), None
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

def _window_applied(window: TimeWindow | None) -> dict[str, str] | None:
    if window is None:
        return None
    return {"start": window.start.isoformat(), "end": window.end.isoformat()}

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
        if len(raw) == 4:
            bbox = _bbox_from_values(list(raw))
            if bbox is not None:
                return ParsedArea(bbox=bbox, applied=_bbox_applied(bbox))
        return ParsedArea(unsupported=raw)
    if isinstance(raw, dict):
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
                    raise RegistryInputError("radius_m 必须是非负数", "invalid_area")
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

def _route_area(area: ParsedArea) -> tuple[BBox | None, dict[str, Any]]:
    unused: dict[str, Any] = {}
    if area.unsupported is not None:
        unused["area"] = area.unsupported
    bbox = area.bbox
    if bbox is None and area.center is not None and area.radius_m is not None:
        bbox = _bbox_from_center(area.center, area.radius_m)
    elif area.center is not None and area.radius_m is None and area.bbox is None:
        unused["center"] = {"lat": area.center.lat, "lon": area.center.lon}
    return bbox, unused

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

def _bbox_from_center(center: GeoPoint, radius_m: int) -> BBox:
    dlat = radius_m / _METERS_PER_DEG_LAT
    meters_per_deg_lon = _METERS_PER_DEG_LAT * max(math.cos(math.radians(center.lat)), 1e-6)
    dlon = radius_m / meters_per_deg_lon
    return BBox(
        west=max(-180.0, center.lon - dlon),
        south=max(-90.0, center.lat - dlat),
        east=min(180.0, center.lon + dlon),
        north=min(90.0, center.lat + dlat),
    )

def _bbox_applied(bbox: BBox) -> dict[str, float]:
    return {"west": bbox.west, "south": bbox.south, "east": bbox.east, "north": bbox.north}

def _registry_kind(registry: str | None) -> str:
    if not registry:
        return "auto"
    folded = registry.strip().casefold()
    if folded in _GEONAMES_ALIASES:
        return "geonames"
    if folded in _OFFICIAL_ALIASES or registry.strip() in _OFFICIAL_ALIASES:
        return "official"
    return "unknown"

def _resolve_providers(
    ctx: RuntimeContext | None,
    registry: str | None,
) -> tuple[list[GazetteerProvider], bool]:
    extras = ctx.extras if ctx is not None else {}
    injected_primary = extras.get("administrative_registry_provider")
    injected_geo = extras.get("geonames_provider")
    injected_official = extras.get("official_gazetteer_provider")
    if injected_primary is not None:
        _ensure_provider(injected_primary, "administrative_registry_provider")
    if injected_geo is not None:
        _ensure_provider(injected_geo, "geonames_provider")
    if injected_official is not None:
        _ensure_provider(injected_official, "official_gazetteer_provider")

    kind = _registry_kind(registry)
    if kind == "unknown":
        if injected_primary is not None and _provider_name(injected_primary) == (registry or ""):
            return [injected_primary], _provider_name(injected_primary) == _PROVIDER_OFFICIAL
        raise EngineUnavailableError(f"未知或不支持的 registry: {registry}")

    want_geo = kind in {"auto", "geonames"}
    want_official = kind in {"auto", "official"}
    providers: list[GazetteerProvider] = []
    if want_geo:
        geo = injected_primary or injected_geo
        if geo is None:
            geo = _build_geonames()
        providers.append(geo)
    used_official = False
    if want_official:
        official = injected_official
        if official is None and kind == "official" and injected_primary is not None:
            official = injected_primary
        if official is None:
            official = _build_official(required=kind == "official")
        if official is not None:
            providers.append(official)
            used_official = True
    if not providers:
        raise EngineUnavailableError("未配置可用的地名档案后端")
    return providers, used_official

def _ensure_provider(raw: Any, label: str) -> None:
    if not isinstance(raw, GazetteerProvider):
        raise EngineUnavailableError(f"{label} 必须提供 search(request) 与 hierarchy(provider_id)")

def _build_geonames() -> GeoNamesProvider:
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_API=false，禁止调用真实地名档案 API")
    username = os.environ.get("GEONAMES_USERNAME", "").strip()
    if not username:
        raise EngineUnavailableError("未配置 GEONAMES_USERNAME")
    return GeoNamesProvider(
        username=username,
        endpoint=_env_value("GEONAMES_ENDPOINT", _DEFAULT_GEONAMES_ENDPOINT),
        timeout_sec=_env_timeout("GEONAMES_TIMEOUT_SEC"),
        user_agent=_env_value("GEONAMES_USER_AGENT", _DEFAULT_USER_AGENT),
    )

def _build_official(*, required: bool) -> OfficialDumpProvider | None:
    _load_dotenv()
    path = os.environ.get("ADMINISTRATIVE_REGISTRY_OFFICIAL_PATH", "").strip()
    if not path:
        if required:
            raise EngineUnavailableError(
                "未配置官方地名档案；请注入 official_gazetteer_provider "
                "或设置 ADMINISTRATIVE_REGISTRY_OFFICIAL_PATH"
            )
        return None
    return _load_official_dump(path)

def _load_official_dump(path: str) -> OfficialDumpProvider:
    file_path = Path(path)
    if not file_path.is_file():
        raise EngineUnavailableError(f"官方地名档案不存在: {path}")
    if file_path.suffix.lower() == ".csv":
        return OfficialDumpProvider(_load_official_csv(file_path))
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineUnavailableError(f"官方地名档案无法读取: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("records", raw.get("items", []))
    if not isinstance(raw, list):
        raise EngineUnavailableError("官方地名档案必须是 JSON 数组")
    return OfficialDumpProvider([item for item in raw if isinstance(item, dict)])

def _load_official_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]

def _provider_name(provider: GazetteerProvider) -> str:
    return str(getattr(provider, "name", "injected"))

def _search_query(request: RegistryRequest) -> str:
    if request.area_text:
        return f"{request.query} {request.area_text}".strip()
    return request.query

def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)

def _record_blob(record: dict[str, Any]) -> str:
    parts = [_text(record.get("standard_name")), _text(record.get("name")), _text(record.get("adcode"))]
    aliases = record.get("aliases")
    if isinstance(aliases, list):
        parts.extend(_text(item) for item in aliases)
    elif isinstance(aliases, str):
        parts.append(aliases)
    parents = record.get("parents")
    if isinstance(parents, list):
        parts.extend(
            _text(item) if not isinstance(item, dict) else _text(item.get("name")) for item in parents
        )
    elif isinstance(parents, str):
        parts.append(parents)
    return " ".join(part for part in parts if part).casefold()

def _record_provider_id(raw: dict[str, Any]) -> str:
    evidence = raw.get("evidence")
    if isinstance(evidence, dict):
        nested = _text(evidence.get("provider_id"))
        if nested:
            return nested
    return _text(
        raw.get("geonameId")
        or raw.get("geoname_id")
        or raw.get("adcode")
        or raw.get("provider_id")
        or raw.get("id")
    )

def _has_stated_interval(raw: dict[str, Any]) -> bool:
    start, end = _record_interval(raw)
    return start is not None or end is not None

def _record_interval(raw: dict[str, Any]) -> tuple[date | None, date | None]:
    start = _bound_to_date(raw.get("valid_from") or raw.get("from"), start=True)
    end = _bound_to_date(raw.get("valid_to") or raw.get("to"), start=False)
    return start, end

def _overlaps_interval(window: TimeWindow, raw: dict[str, Any]) -> bool:
    start, end = _record_interval(raw)
    lo = start or date.min
    hi = end or date.max
    return not (hi < window.start or lo > window.end)

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
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise EngineUnavailableError(f"{error_prefix} HTTP {exc.code}: {detail[:200]}") from exc
    except urllib.error.URLError as exc:
        raise EngineUnavailableError(f"{error_prefix} 网络失败: {exc.reason}") from exc
    except (json.JSONDecodeError, TimeoutError, OSError) as exc:
        raise EngineUnavailableError(f"{error_prefix} 调用失败: {exc}") from exc

def _append_query(endpoint: str, query: str) -> str:
    separator = "&" if urllib.parse.urlparse(endpoint).query else "?"
    return f"{endpoint}{separator}{query}"

def _text(raw: Any) -> str:
    if raw is None or isinstance(raw, bool):
        return ""
    if isinstance(raw, list):
        return ""
    if isinstance(raw, str):
        return raw.strip()
    return str(raw).strip()

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

def _fmt_coord(value: float) -> str:
    return f"{value:.6f}"

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
