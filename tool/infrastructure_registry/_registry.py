"""infrastructure_registry 共享执行器：地区建设/许可档案连接器。

无全球统一接口。英格兰可用 Planning Data；国内走授权 dump 或已知门户入口，
不爬取住建验证码站。
"""

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

_OP_CONSTRUCTION = "construction"
_OP_PERMIT = "permit"
_PROVIDER_PLANNING = "planning_data"
_PROVIDER_DUMP = "dump"
_PROVIDER_PORTAL = "jzsc"
_CRS_WGS84 = "wgs84"
_TOP_K = 10
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_PLANNING_ENDPOINT = "https://www.planning.data.gov.uk/entity.json"
_DEFAULT_USER_AGENT = "geoagent-dataset/1.0 (infrastructure_registry; local)"
_PLANNING_ENTITY_PAGE = "https://www.planning.data.gov.uk/entity"
_JZSC_PORTAL = "https://jzsc.mohurd.gov.cn/"
_METERS_PER_DEG_LAT = 111_320.0
_YEAR_MIN = 1000
_YEAR_MAX = 2100
_INPUT_FIELDS = ("query", "area", "time_range", "registry", "fields")
_ALWAYS_RESULT_KEYS = ("result_id", "validity", "evidence")
_YEAR_RE = re.compile(r"^(\d{4})$")
_YEAR_RANGE_RE = re.compile(r"^(\d{4})\s*[-–—至到]\s*(\d{4})$")
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
_DATE_RANGE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s*(?:to|/|[-–—至到])\s*(\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)
_UK_POSTCODE_RE = re.compile(
    r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$",
    re.IGNORECASE,
)
_UPRN_RE = re.compile(r"^\d{5,12}$")
_POINT_WKT_RE = re.compile(
    r"POINT\s*\(\s*([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s*\)",
    re.IGNORECASE,
)
_PLANNING_ALIASES = frozenset(
    {
        "planning_data",
        "planning-data",
        "planningdata",
        "england",
        "uk",
        "planning",
    }
)
_DUMP_ALIASES = frozenset({"dump", "official", "authorized", "file", "授权"})
_PORTAL_ALIASES = frozenset(
    {
        "jzsc",
        "mohurd",
        "建筑市场",
        "住建",
        "全国建筑市场监管公共服务平台",
        "住房和城乡建设",
    }
)
_CONSTRUCTION_DATASETS = (
    "listed-building",
    "listed-building-outline",
    "conservation-area",
    "brownfield-land",
    "heritage-at-risk",
)
_PERMIT_DATASETS = ("planning-application",)
_CONSTRUCTION_TYPES = frozenset(
    {
        "construction",
        "listed_building",
        "listed-building",
        "listed-building-outline",
        "conservation-area",
        "conservation_area",
        "brownfield-land",
        "brownfield",
        "heritage-at-risk",
        "heritage",
        "bridge",
        "facility",
        "infrastructure",
        "building",
        "建设",
        "竣工",
        "开工",
    }
)
_PERMIT_TYPES = frozenset(
    {
        "permit",
        "planning_application",
        "planning-application",
        "license",
        "licence",
        "registration",
        "许可",
        "施工许可",
        "规划许可",
        "登记",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "body",
        "confirmed_location",
        "confirmed_place",
        "construction_year",
        "content",
        "full_text",
        "location_confirmed",
        "markdown",
        "raw_content",
        "taken_at",
    }
)
_DATE_KIND_ASSUMPTION = "许可颁发、开工、竣工和正式开放是不同时间，不能混成修建年份"
_ABSENCE_ASSUMPTION = "查不到记录不代表建筑从未存在"
_COVERAGE_ASSUMPTION = "小型设施、历史建筑和非完整公开档案覆盖不足"
_NO_SESSION_ASSUMPTION = "本步只返回档案条目，未打开街景或地图会话"
_PHOTO_POINT_ASSUMPTION = "候选记录是建设或许可档案匹配，不是照片拍摄点"
_CRS_ASSUMPTION = "当前后端坐标为 WGS84，未做坐标系转换"
_PLANNING_SCOPE_ASSUMPTION = (
    "Planning Data 仅覆盖英格兰；规划申请数据本身不完整，且 q 仅支持邮编或 UPRN"
)
_PORTAL_ASSUMPTION = "国内无统一公开建设/许可 API，仅提供官方入口；正文请用 web_page_read，未绕过登录或验证码"
_NO_BACKEND_ERROR = (
    "未配置建设/许可档案后端；请设置 INFRASTRUCTURE_REGISTRY_DUMP_PATH、"
    "注入 provider，或指定 registry=planning_data。"
    "ALLOW_REAL_API=false 时不会默认调用境外 Planning Data API"
)

class RegistryInputError(Exception):
    """query / area / time_range / registry / fields 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """建设/许可档案未配置、被闸门拒绝或调用失败。"""

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
    """闭区间日期窗。"""

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
    """组装后的建设/许可档案检索。"""

    operation: str
    query: str
    top_k: int
    area_text: str | None = None
    bbox: BBox | None = None
    center: GeoPoint | None = None
    time_window: TimeWindow | None = None

@dataclass(frozen=True)
class PortalInfo:
    """已知官方门户的覆盖说明；不发起抓取。"""

    official_portal: str | None = None
    portal_only: bool = False

@runtime_checkable
class RegistryProvider(Protocol):
    """可注入的建设/许可档案后端；测试用 extras 替换。"""

    def search(self, request: RegistryRequest) -> list[dict[str, Any]]:
        """按查询返回可归一化的原始记录列表。"""

class PlanningDataProvider:
    """英格兰 Planning Data entity.json 适配器；坐标为 WGS84。"""

    name = _PROVIDER_PLANNING
    crs = _CRS_WGS84

    def __init__(self, *, endpoint: str, timeout_sec: float, user_agent: str) -> None:
        self._endpoint = endpoint
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent

    def search(self, request: RegistryRequest) -> list[dict[str, Any]]:
        datasets = (
            _PERMIT_DATASETS if request.operation == _OP_PERMIT else _CONSTRUCTION_DATASETS
        )
        postcode_or_uprn = _is_postcode_or_uprn(request.query)
        if (
            not postcode_or_uprn
            and request.bbox is None
            and request.center is None
        ):
            return []
        params: list[tuple[str, str]] = [("limit", str(request.top_k))]
        for dataset in datasets:
            params.append(("dataset", dataset))
        if postcode_or_uprn:
            params.append(("q", request.query.strip()))
        if request.bbox is not None:
            params.append(("geometry", _bbox_wkt(request.bbox)))
            params.append(("geometry_relation", "intersects"))
        elif request.center is not None:
            params.append(("latitude", _fmt_coord(request.center.lat)))
            params.append(("longitude", _fmt_coord(request.center.lon)))
        if request.time_window is not None:
            params.append(("start_date_year", str(request.time_window.start.year)))
            params.append(("start_date_month", f"{request.time_window.start.month:02d}"))
            params.append(("start_date_day", f"{request.time_window.start.day:02d}"))
            params.append(("start_date_match", "since"))
        payload = self._get(params)
        hits = payload.get("entities", payload.get("results", []))
        if not isinstance(hits, list):
            return []
        rows = [item for item in hits if isinstance(item, dict)]
        if not postcode_or_uprn:
            rows = _filter_query_tokens(rows, request.query)
        return rows[: request.top_k]

    def _get(self, params: list[tuple[str, str]]) -> dict[str, Any]:
        query = urllib.parse.urlencode(params, doseq=True)
        url = _append_query(self._endpoint, query)
        raw = _http_json(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self._user_agent,
            },
            timeout_sec=self._timeout_sec,
            error_prefix="Planning Data",
        )
        if not isinstance(raw, dict):
            raise EngineUnavailableError("Planning Data 回执不是 JSON 对象")
        return raw

class AuthorizedDumpProvider:
    """本地授权建设/许可 dump；无统一远程 API。"""

    name = _PROVIDER_DUMP
    crs = _CRS_WGS84

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def search(self, request: RegistryRequest) -> list[dict[str, Any]]:
        tokens = [part.casefold() for part in request.query.split() if part]
        area_l = (request.area_text or "").strip().casefold()
        hits: list[dict[str, Any]] = []
        for record in self._records:
            if not _record_matches_operation(record, request.operation):
                continue
            blob = _record_blob(record)
            if tokens and not all(token in blob for token in tokens):
                continue
            if area_l and area_l not in blob:
                continue
            if request.bbox is not None and not _record_in_bbox(record, request.bbox):
                continue
            hits.append(record)
            if len(hits) >= request.top_k:
                break
        return hits

class OfficialPortalProvider:
    """已知国内门户：只声明入口，不抓取页面。"""

    name = _PROVIDER_PORTAL
    crs = _CRS_WGS84
    official_portal = _JZSC_PORTAL

    def search(self, request: RegistryRequest) -> list[dict[str, Any]]:
        del request
        return []

def execute_construction(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """查询建筑、桥梁或设施的建设和历史记录。"""

    return _execute(_OP_CONSTRUCTION, purpose=purpose, inputs=inputs, ctx=ctx)

def execute_permit(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """查询许可、登记、编号或行业记录。"""

    return _execute(_OP_PERMIT, purpose=purpose, inputs=inputs, ctx=ctx)

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
        providers, portal = _resolve_providers(ctx, registry)
        request_bbox, unused_area, request_center = _route_area(area)
        request = RegistryRequest(
            operation=operation,
            query=query,
            top_k=_TOP_K,
            area_text=area.text,
            bbox=request_bbox,
            center=request_center,
            time_window=time_window,
        )
        hits = _collect_hits(providers, request, time_window=time_window)
        unused: dict[str, Any] = dict(unused_area)
        if unsupported_time is not None:
            unused["time_range"] = unsupported_time
        if (
            _PROVIDER_PLANNING in [_provider_name(item) for item in providers]
            and not _is_postcode_or_uprn(query)
            and request_bbox is None
            and request_center is None
        ):
            unused["query"] = "Planning Data 的 q 仅支持邮编或 UPRN，项目名检索需要 area"
        return _ok(
            hits[:_TOP_K],
            operation=operation,
            providers=providers,
            query=query,
            area=area,
            registry=registry,
            fields=fields,
            time_applied=_window_applied(time_window),
            unused=unused,
            portal=portal,
        )
    except RegistryInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def _collect_hits(
    providers: list[RegistryProvider],
    request: RegistryRequest,
    *,
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
                provider_name=provider_name,
                operation=request.operation,
                time_window=time_window,
            )
            if hit is None:
                continue
            if time_window is not None and not _hit_overlaps_window(
                hit,
                time_window,
                operation=request.operation,
            ):
                continue
            hits.append(hit)
    return _dedupe(hits)

def _ok(
    results: list[dict[str, Any]],
    *,
    operation: str,
    providers: list[RegistryProvider],
    query: str,
    area: ParsedArea,
    registry: str | None,
    fields: list[str] | None,
    time_applied: dict[str, str] | None,
    unused: dict[str, Any],
    portal: PortalInfo,
) -> Observation:
    numbered: list[dict[str, Any]] = []
    for item in results:
        row = dict(item)
        row["result_id"] = f"{operation}_{len(numbered) + 1}"
        numbered.append(_project_fields(row, fields))
    names = [_provider_name(provider) for provider in providers]
    applied: dict[str, Any] = {
        "provider": "+".join(names) if names else _PROVIDER_DUMP,
        "providers": names,
        "query": query,
        "top_k": _TOP_K,
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
    if portal.official_portal:
        applied["official_portal"] = portal.official_portal
    if unused:
        applied["unsupported"] = unused
    coverage = {
        "insufficient": _coverage_insufficient(numbered, names, portal),
        "reason": _coverage_reason(numbered, names, portal),
    }
    if portal.official_portal:
        coverage["official_portal"] = portal.official_portal
    result: dict[str, Any] = {
        "operation": operation,
        "results": numbered,
        "coverage": coverage,
        "applied": applied,
        "assumptions": _assumptions(names, portal=portal),
    }
    return Observation(ok=True, result=_strip_forbidden(result))

def _coverage_insufficient(
    results: list[dict[str, Any]],
    provider_names: list[str],
    portal: PortalInfo,
) -> bool:
    if not results:
        return True
    if portal.portal_only:
        return True
    return _PROVIDER_PLANNING in provider_names or _PROVIDER_PORTAL in provider_names

def _coverage_reason(
    results: list[dict[str, Any]],
    provider_names: list[str],
    portal: PortalInfo,
) -> str:
    if portal.portal_only or (not results and _PROVIDER_PORTAL in provider_names):
        return "国内门户无统一公开 API，当前未导入授权档案"
    if not results:
        return "当前来源未命中记录；不代表建筑或许可从未存在"
    if _PROVIDER_PLANNING in provider_names:
        return "Planning Data 覆盖随地区变化，规划申请本身不完整"
    return "公开建设/许可档案覆盖不全"

def _assumptions(provider_names: list[str], *, portal: PortalInfo) -> list[str]:
    items = [
        _DATE_KIND_ASSUMPTION,
        _ABSENCE_ASSUMPTION,
        _COVERAGE_ASSUMPTION,
        _PHOTO_POINT_ASSUMPTION,
        _CRS_ASSUMPTION,
        _NO_SESSION_ASSUMPTION,
    ]
    if _PROVIDER_PLANNING in provider_names:
        items.append(_PLANNING_SCOPE_ASSUMPTION)
    if portal.official_portal or _PROVIDER_PORTAL in provider_names:
        items.append(_PORTAL_ASSUMPTION)
    return items

def _project_fields(hit: dict[str, Any], fields: list[str] | None) -> dict[str, Any]:
    if not fields:
        return hit
    keep = set(fields) | set(_ALWAYS_RESULT_KEYS)
    return {key: value for key, value in hit.items() if key in keep}

def _normalize_record(
    raw: dict[str, Any],
    *,
    provider_name: str,
    operation: str,
    time_window: TimeWindow | None,
) -> dict[str, Any] | None:
    project_name = _project_name(raw)
    if not project_name:
        return None
    dates = _dates_from_raw(raw, operation=operation)
    hit: dict[str, Any] = {
        "project_name": project_name,
        "record_type": _record_type(raw, operation=operation),
        "dates": dates,
    }
    reference = _reference(raw)
    if reference:
        hit["reference"] = reference
    location = _location(raw)
    if location is not None:
        hit["location"] = location
    evidence: dict[str, Any] = {"source": _text(raw.get("source")) or provider_name}
    provider_id = _record_provider_id(raw)
    if provider_id:
        evidence["provider_id"] = provider_id
    url = _text(raw.get("source_url") or raw.get("url")) or _evidence_url(
        provider_name,
        provider_id,
    )
    if url:
        evidence["url"] = url
    file_name = _text(raw.get("file") or raw.get("source_file"))
    if file_name:
        evidence["file"] = file_name
    hit["evidence"] = evidence
    hit["validity"] = _validity(dates, time_window=time_window)
    return hit

def _project_name(raw: dict[str, Any]) -> str:
    return _text(
        raw.get("project_name")
        or raw.get("name")
        or raw.get("title")
        or raw.get("address")
    )

def _record_type(raw: dict[str, Any], *, operation: str) -> str:
    stated = _text(
        raw.get("record_type") or raw.get("dataset") or raw.get("type") or raw.get("kind")
    )
    if stated:
        return stated.replace("_", "-") if stated in {"planning_application"} else stated
    return "planning-application" if operation == _OP_PERMIT else "construction"

def _reference(raw: dict[str, Any]) -> str:
    return _text(
        raw.get("reference")
        or raw.get("number")
        or raw.get("permit_no")
        or raw.get("ref")
    )

def _dates_from_raw(raw: dict[str, Any], *, operation: str) -> dict[str, str]:
    nested = raw.get("dates") if isinstance(raw.get("dates"), dict) else {}
    permit = _bound_to_date(
        raw.get("permit_date") or raw.get("issued_at") or nested.get("permit"),
        start=True,
    )
    started = _bound_to_date(
        raw.get("started_at") or raw.get("construction_start") or nested.get("started"),
        start=True,
    )
    completed = _bound_to_date(
        raw.get("completed_at")
        or raw.get("completion_date")
        or nested.get("completed"),
        start=True,
    )
    opened = _bound_to_date(
        raw.get("opened_at") or raw.get("opened_date") or nested.get("opened"),
        start=True,
    )
    recorded = _bound_to_date(
        raw.get("recorded")
        or raw.get("start-date")
        or raw.get("start_date")
        or raw.get("entry-date")
        or nested.get("recorded"),
        start=True,
    )
    kind = _text(raw.get("recorded_kind") or nested.get("recorded_kind"))
    if not kind:
        dataset = _text(raw.get("dataset") or raw.get("record_type"))
        if permit is not None:
            kind = "permit"
        elif started is not None:
            kind = "start"
        elif completed is not None:
            kind = "completion"
        elif opened is not None:
            kind = "opening"
        elif dataset == "planning-application" or operation == _OP_PERMIT:
            kind = "permit" if recorded is not None else ""
        elif recorded is not None:
            kind = "entry"
    if recorded is None and permit is not None:
        recorded = permit
        kind = kind or "permit"
    if recorded is None and started is not None:
        recorded = started
        kind = kind or "start"
    payload: dict[str, str] = {}
    if permit is not None:
        payload["permit"] = permit.isoformat()
    if started is not None:
        payload["started"] = started.isoformat()
    if completed is not None:
        payload["completed"] = completed.isoformat()
    if opened is not None:
        payload["opened"] = opened.isoformat()
    if recorded is not None:
        payload["recorded"] = recorded.isoformat()
        payload["recorded_kind"] = kind or "unknown"
    return payload

def _validity(
    dates: dict[str, str],
    *,
    time_window: TimeWindow | None,
) -> dict[str, Any]:
    stated = bool(
        dates.get("permit")
        or dates.get("started")
        or dates.get("completed")
        or dates.get("opened")
        or dates.get("recorded")
    )
    payload: dict[str, Any] = {
        "scope": "stated_interval" if stated else "unstated",
    }
    if time_window is not None:
        payload["time_range_requested"] = _window_applied(time_window)
    return payload

def _hit_overlaps_window(
    hit: dict[str, Any],
    window: TimeWindow,
    *,
    operation: str,
) -> bool:
    dates = hit.get("dates") if isinstance(hit.get("dates"), dict) else {}
    primary = _primary_dates(dates, operation=operation)
    if not primary:
        return True
    return any(_date_in_window(value, window) for value in primary)

def _primary_dates(dates: dict[str, Any], *, operation: str) -> list[date]:
    keys = (
        ("permit", "recorded")
        if operation == _OP_PERMIT
        else ("started", "completed", "opened", "recorded")
    )
    if operation == _OP_PERMIT and dates.get("permit"):
        keys = ("permit",)
    elif operation == _OP_CONSTRUCTION and any(
        dates.get(key) for key in ("started", "completed", "opened")
    ):
        keys = ("started", "completed", "opened")
    values: list[date] = []
    for key in keys:
        parsed = _parse_iso_date(str(dates.get(key) or ""))
        if parsed is not None:
            values.append(parsed)
    return values

def _date_in_window(value: date, window: TimeWindow) -> bool:
    return window.start <= value <= window.end

def _location(raw: dict[str, Any]) -> dict[str, float | str] | None:
    nested = raw.get("location")
    if isinstance(nested, dict):
        lat = _as_float(nested.get("lat", nested.get("latitude")))
        lon = _as_float(nested.get("lon", nested.get("lng", nested.get("longitude"))))
        if lat is not None and lon is not None:
            return {"lon": lon, "lat": lat, "crs": _text(nested.get("crs")) or _CRS_WGS84}
    lat = _as_float(raw.get("lat", raw.get("latitude")))
    lon = _as_float(raw.get("lon", raw.get("lng", raw.get("longitude"))))
    if lat is not None and lon is not None and _valid_lonlat(lat, lon):
        return {"lon": lon, "lat": lat, "crs": _CRS_WGS84}
    point = _text(raw.get("point") or raw.get("geometry"))
    parsed = _point_from_wkt(point)
    if parsed is not None:
        return {"lon": parsed.lon, "lat": parsed.lat, "crs": _CRS_WGS84}
    return None

def _point_from_wkt(text: str) -> GeoPoint | None:
    match = _POINT_WKT_RE.search(text)
    if not match:
        return None
    lon = float(match.group(1))
    lat = float(match.group(2))
    if not _valid_lonlat(lat, lon):
        return None
    return GeoPoint(lat=lat, lon=lon)

def _evidence_url(provider_name: str, provider_id: str) -> str:
    if provider_name == _PROVIDER_PLANNING and provider_id:
        return f"{_PLANNING_ENTITY_PAGE}/{provider_id}"
    return ""

def _dedupe(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for hit in hits:
        evidence = hit.get("evidence") if isinstance(hit.get("evidence"), dict) else {}
        key = (evidence.get("source"), evidence.get("provider_id"), hit.get("project_name"), hit.get("reference"))
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
    if isinstance(raw, bool) or raw is None or raw == "":
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
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
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

def _route_area(area: ParsedArea) -> tuple[BBox | None, dict[str, Any], GeoPoint | None]:
    unused: dict[str, Any] = {}
    if area.unsupported is not None:
        unused["area"] = area.unsupported
    bbox = area.bbox
    center = area.center
    if bbox is None and center is not None and area.radius_m is not None:
        bbox = _bbox_from_center(center, area.radius_m)
        center = None
    return bbox, unused, center if bbox is None else None

def _center_from_mapping(raw: dict[str, Any]) -> GeoPoint | None:
    lat = _as_float(raw.get("lat", raw.get("latitude")))
    lon = _as_float(raw.get("lon", raw.get("lng", raw.get("longitude"))))
    nested = raw.get("center")
    if isinstance(nested, dict) and (lat is None or lon is None):
        lat = _as_float(nested.get("lat", nested.get("latitude")))
        lon = _as_float(nested.get("lon", nested.get("lng", nested.get("longitude"))))
    if lat is None or lon is None:
        return None
    if not _valid_lonlat(lat, lon):
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

def _bbox_wkt(bbox: BBox) -> str:
    return (
        f"POLYGON(({bbox.west} {bbox.south},{bbox.east} {bbox.south},"
        f"{bbox.east} {bbox.north},{bbox.west} {bbox.north},{bbox.west} {bbox.south}))"
    )

def _registry_kind(registry: str | None) -> str:
    if not registry:
        return "auto"
    folded = registry.strip().casefold()
    if folded in _PLANNING_ALIASES:
        return "planning_data"
    if folded in _DUMP_ALIASES:
        return "dump"
    if folded in _PORTAL_ALIASES or registry.strip() in _PORTAL_ALIASES:
        return "portal"
    return "unknown"

def _resolve_providers(
    ctx: RuntimeContext | None,
    registry: str | None,
) -> tuple[list[RegistryProvider], PortalInfo]:
    extras = ctx.extras if ctx is not None else {}
    injected_primary = extras.get("infrastructure_registry_provider")
    injected_planning = extras.get("planning_data_provider")
    injected_dump = extras.get("infrastructure_dump_provider")
    if injected_primary is not None:
        _ensure_provider(injected_primary, "infrastructure_registry_provider")
    if injected_planning is not None:
        _ensure_provider(injected_planning, "planning_data_provider")
    if injected_dump is not None:
        _ensure_provider(injected_dump, "infrastructure_dump_provider")

    kind = _registry_kind(registry)
    if kind == "unknown":
        if injected_primary is not None and _provider_name(injected_primary) == (registry or ""):
            return [injected_primary], PortalInfo()
        raise EngineUnavailableError(f"未知或不支持的 registry: {registry}")

    if kind == "planning_data":
        provider = injected_primary or injected_planning
        if provider is None:
            provider = _build_planning_data()
        return [provider], PortalInfo()

    if kind == "dump":
        dump = injected_dump or injected_primary or _build_dump(required=True)
        return [dump], PortalInfo()

    if kind == "portal":
        dump = injected_dump or _build_dump(required=False)
        info = PortalInfo(official_portal=_JZSC_PORTAL, portal_only=dump is None)
        if dump is not None:
            return [dump], info
        return [OfficialPortalProvider()], info

    if injected_primary is not None:
        return [injected_primary], PortalInfo()
    dump = injected_dump or _build_dump(required=False)
    if dump is not None:
        return [dump], PortalInfo()
    raise EngineUnavailableError(_NO_BACKEND_ERROR)

def _ensure_provider(raw: Any, label: str) -> None:
    if not isinstance(raw, RegistryProvider):
        raise EngineUnavailableError(f"{label} 必须提供 search(request)")

def _build_planning_data() -> PlanningDataProvider:
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_API=false，禁止调用真实建设/许可档案 API")
    return PlanningDataProvider(
        endpoint=_env_value("PLANNING_DATA_ENDPOINT", _DEFAULT_PLANNING_ENDPOINT),
        timeout_sec=_env_timeout("PLANNING_DATA_TIMEOUT_SEC"),
        user_agent=_env_value("PLANNING_DATA_USER_AGENT", _DEFAULT_USER_AGENT),
    )

def _build_dump(*, required: bool) -> AuthorizedDumpProvider | None:
    _load_dotenv()
    path = os.environ.get("INFRASTRUCTURE_REGISTRY_DUMP_PATH", "").strip()
    if not path:
        if required:
            raise EngineUnavailableError(
                "未配置授权建设/许可档案；请注入 infrastructure_dump_provider "
                "或设置 INFRASTRUCTURE_REGISTRY_DUMP_PATH"
            )
        return None
    return _load_dump(path)

def _load_dump(path: str) -> AuthorizedDumpProvider:
    file_path = Path(path)
    if not file_path.is_file():
        raise EngineUnavailableError(f"授权建设/许可档案不存在: {path}")
    if file_path.suffix.lower() == ".csv":
        return AuthorizedDumpProvider(_load_csv(file_path))
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineUnavailableError(f"授权建设/许可档案无法读取: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("records", raw.get("items", []))
    if not isinstance(raw, list):
        raise EngineUnavailableError("授权建设/许可档案必须是 JSON 数组")
    return AuthorizedDumpProvider([item for item in raw if isinstance(item, dict)])

def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]

def _provider_name(provider: RegistryProvider) -> str:
    return str(getattr(provider, "name", "injected"))

def _record_matches_operation(record: dict[str, Any], operation: str) -> bool:
    stated = _text(record.get("record_type") or record.get("dataset") or record.get("type"))
    if not stated:
        return True
    folded = stated.casefold()
    if operation == _OP_PERMIT:
        if folded in {item.casefold() for item in _PERMIT_TYPES}:
            return True
        if folded in {item.casefold() for item in _CONSTRUCTION_TYPES}:
            return False
        return True
    if folded in {item.casefold() for item in _CONSTRUCTION_TYPES}:
        return True
    if folded in {item.casefold() for item in _PERMIT_TYPES}:
        return False
    return True

def _record_blob(record: dict[str, Any]) -> str:
    parts = [
        _text(record.get("project_name")),
        _text(record.get("name")),
        _text(record.get("reference")),
        _text(record.get("number")),
        _text(record.get("area")),
        _text(record.get("city")),
        _text(record.get("address")),
    ]
    aliases = record.get("aliases")
    if isinstance(aliases, list):
        parts.extend(_text(item) for item in aliases)
    elif isinstance(aliases, str):
        parts.append(aliases)
    return " ".join(part for part in parts if part).casefold()

def _record_in_bbox(record: dict[str, Any], bbox: BBox) -> bool:
    location = _location(record)
    if location is None:
        return True
    lat = float(location["lat"])
    lon = float(location["lon"])
    return bbox.south <= lat <= bbox.north and bbox.west <= lon <= bbox.east

def _record_provider_id(raw: dict[str, Any]) -> str:
    evidence = raw.get("evidence")
    if isinstance(evidence, dict):
        nested = _text(evidence.get("provider_id"))
        if nested:
            return nested
    entity = raw.get("entity")
    if isinstance(entity, int):
        return str(entity)
    return _text(
        raw.get("entity")
        or raw.get("provider_id")
        or raw.get("reference")
        or raw.get("id")
    )

def _filter_query_tokens(rows: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    tokens = [part.casefold() for part in query.split() if part]
    if not tokens:
        return rows
    matched: list[dict[str, Any]] = []
    for row in rows:
        blob = _record_blob(row)
        if all(token in blob for token in tokens):
            matched.append(row)
    return matched

def _is_postcode_or_uprn(query: str) -> bool:
    text = query.strip()
    return bool(_UK_POSTCODE_RE.fullmatch(text) or _UPRN_RE.fullmatch(text))

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

def _valid_lonlat(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0

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
