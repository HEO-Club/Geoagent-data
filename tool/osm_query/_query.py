"""osm_query 共享执行器：Overpass 查询与统计 OSM 要素。"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tool._gate import allow_real_api as _allow_real_api, load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext, declared_inputs

_PROVIDER_OVERPASS = "overpass"
_CRS_WGS84 = "wgs84"
_OP_QUERY = "query"
_OP_COUNT = "count"
_QUERY_INPUT_FIELDS = (
    "area",
    "bbox",
    "center",
    "radius_m",
    "tags",
    "feature_types",
    "element_types",
    "spatial_relation",
    "return_geometry",
    "limit",
    "overpass_ql",
)
_COUNT_INPUT_FIELDS = ("source_result", "area", "bbox", "tags", "group_by")
_KIND_BBOX = "bbox"
_KIND_AROUND_POINT = "around_point"
_KIND_AROUND_NAME = "around_name"
_KIND_AREA_NAME = "area_name"
_KIND_POLY = "poly"
_REL_WITHIN = "within"
_REL_NEAR = "near"
_REL_INTERSECTS = "intersects"
_ELEMENT_NODE = "node"
_ELEMENT_WAY = "way"
_ELEMENT_RELATION = "relation"
_ELEMENT_TYPES = (_ELEMENT_NODE, _ELEMENT_WAY, _ELEMENT_RELATION)
_DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"
_DEFAULT_UA = "geoagent-dataset/1.0 (osm_query; local)"
_DEFAULT_TIMEOUT_SEC = 25.0
_DEFAULT_LIMIT = 200
_LIMIT_MIN = 1
_LIMIT_MAX = 10000
_DEFAULT_MAXSIZE = 33_554_432
_HTTP_TIMEOUT_PAD_SEC = 5.0
_COORD_SPLIT_RE = re.compile(r"[,;，、\s]+")
_TIMEOUT_RE = re.compile(r"\[timeout\s*:\s*(\d+)\s*\]", re.IGNORECASE)
_MAXSIZE_RE = re.compile(r"\[maxsize\s*:\s*(\d+)\s*\]", re.IGNORECASE)
_OUT_RE = re.compile(r"\[out\s*:\s*([^\]]+)\]", re.IGNORECASE)
_BBOX_FILTER_RE = re.compile(
    r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)"
)
_AROUND_RE = re.compile(r"around\s*:", re.IGNORECASE)
_AREA_FILTER_RE = re.compile(r"\(\s*area\b", re.IGNORECASE)
_POLY_RE = re.compile(r"poly\s*:", re.IGNORECASE)
_AREA_STMT_RE = re.compile(r"\barea\s*\[", re.IGNORECASE)
_SPATIAL_KEYS = frozenset({"area", "bbox", "center", "overpass_ql", "source_result"})
_IMAGE_KEYS = frozenset(
    {"image", "current_image", "image_id", "photo", "picture", "frame"}
)
_SPATIAL_RELATIONS = {
    "within": _REL_WITHIN,
    "inside": _REL_WITHIN,
    "in": _REL_WITHIN,
    "intersects": _REL_INTERSECTS,
    "intersect": _REL_INTERSECTS,
    "overlap": _REL_INTERSECTS,
    "near": _REL_NEAR,
    "around": _REL_NEAR,
    "nearby": _REL_NEAR,
    "附近": _REL_NEAR,
    "周边": _REL_NEAR,
    "范围内": _REL_WITHIN,
    "相交": _REL_INTERSECTS,
}
_FEATURE_TYPE_TAGS: dict[str, dict[str, str]] = {
    "桥梁": {"bridge": "yes"},
    "桥": {"bridge": "yes"},
    "bridge": {"bridge": "yes"},
    "bridges": {"bridge": "yes"},
    "铁路": {"railway": "*"},
    "铁轨": {"railway": "*"},
    "railway": {"railway": "*"},
    "rail": {"railway": "*"},
    "railroad": {"railway": "*"},
    "道路": {"highway": "*"},
    "公路": {"highway": "*"},
    "road": {"highway": "*"},
    "roads": {"highway": "*"},
    "highway": {"highway": "*"},
    "电塔": {"power": "tower"},
    "电力塔": {"power": "tower"},
    "输电塔": {"power": "tower"},
    "power tower": {"power": "tower"},
    "power_tower": {"power": "tower"},
    "电力线": {"power": "line"},
    "输电线": {"power": "line"},
    "power line": {"power": "line"},
    "power_line": {"power": "line"},
    "powerline": {"power": "line"},
    "建筑": {"building": "*"},
    "建筑物": {"building": "*"},
    "building": {"building": "*"},
    "buildings": {"building": "*"},
    "河流": {"waterway": "river"},
    "河": {"waterway": "river"},
    "river": {"waterway": "river"},
    "rivers": {"waterway": "river"},
    "湖泊": {"natural": "water"},
    "湖": {"natural": "water"},
    "lake": {"natural": "water"},
    "lakes": {"natural": "water"},
    "水域": {"natural": "water"},
    "water": {"natural": "water"},
    "塔": {"man_made": "tower"},
    "tower": {"man_made": "tower"},
    "towers": {"man_made": "tower"},
}
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
_COUNT_ASSUMPTION = "count 只表示当前 OSM 库中满足条件的对象数，不代表现实世界完整数量"
_TIMESTAMP_ASSUMPTION = "data_timestamp 是 OSM 库快照时间，不是照片拍摄时间"
_NO_SESSION_ASSUMPTION = "本步只返回 OSM 查询结果，未打开街景或地图会话"
_CRS_ASSUMPTION = "当前后端为 Overpass，坐标为 WGS84，未做坐标系转换"
_NO_IMAGE_ASSUMPTION = "Overpass 不会看图，本步未从图片提取地物"
_STRUCTURED_UNUSED = (
    "area",
    "bbox",
    "center",
    "radius_m",
    "tags",
    "feature_types",
    "element_types",
    "spatial_relation",
    "return_geometry",
    "limit",
)

class OsmQueryInputError(Exception):
    """area / bbox / tags / overpass_ql 等无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """真实 Overpass 服务未配置、被闸门拒绝或调用失败。"""

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
class QueryScope:
    """结构化查询的空间范围。"""

    kind: str
    bbox: BBox | None = None
    center: GeoPoint | None = None
    radius_m: int | None = None
    name: str | None = None
    polygon: tuple[tuple[float, float], ...] | None = None
    applied: Any = None

@dataclass(frozen=True)
class OsmQueryRequest:
    """组装后的 Overpass 请求。"""

    ql: str
    timeout_sec: float
    operation: str

@runtime_checkable
class OsmQueryProvider(Protocol):
    """可注入的 OSM 查询后端；测试用 extras['osm_query_provider'] 替换。"""

    def query(self, request: OsmQueryRequest) -> dict[str, Any]:
        """提交 Overpass QL，返回可归一化的 JSON 对象。"""

class OverpassProvider:
    """Overpass interpreter 适配器；端点与 UA 只读环境变量。"""

    name = _PROVIDER_OVERPASS
    crs = _CRS_WGS84

    def __init__(
        self,
        *,
        endpoint: str,
        user_agent: str,
        timeout_sec: float,
    ) -> None:
        self._endpoint = endpoint
        self._user_agent = user_agent
        self._timeout_sec = timeout_sec

    def query(self, request: OsmQueryRequest) -> dict[str, Any]:
        body = urllib.parse.urlencode({"data": request.ql}).encode("utf-8")
        http_timeout = max(self._timeout_sec, request.timeout_sec) + _HTTP_TIMEOUT_PAD_SEC
        raw = _http_post_json(
            self._endpoint,
            body=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
                "User-Agent": self._user_agent,
            },
            timeout_sec=http_timeout,
            error_prefix="Overpass",
        )
        if not isinstance(raw, dict):
            raise EngineUnavailableError("Overpass 回执不是 JSON 对象")
        remark = raw.get("remark")
        if isinstance(remark, str) and "error" in remark.lower():
            raise EngineUnavailableError(f"Overpass 失败: {remark[:200]}")
        return raw

def execute_query(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按区域、标签和空间关系查询 OSM 要素。"""

    del purpose
    try:
        original = dict(inputs)
        inputs = declared_inputs(inputs, *_QUERY_INPUT_FIELDS)
        parsed = _parse_query_inputs(inputs, ctx, original=original)
        provider = _resolve_provider(ctx)
        payload = provider.query(
            OsmQueryRequest(
                ql=parsed["ql"],
                timeout_sec=float(parsed["timeout_sec"]),
                operation=_OP_QUERY,
            )
        )
        return _ok_query(
            payload,
            provider_name=_provider_name(provider),
            parsed=parsed,
        )
    except OsmQueryInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def execute_count(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """统计符合条件的 OSM 要素数量或分布。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, *_COUNT_INPUT_FIELDS)
        source = _resolve_source_result(inputs.get("source_result"), ctx)
        group_by = _parse_group_by(inputs.get("group_by"))
        if source is not None:
            return _ok_local_count(source, group_by=group_by)
        parsed = _parse_count_inputs(inputs, ctx)
        provider = _resolve_provider(ctx)
        payload = provider.query(
            OsmQueryRequest(
                ql=parsed["ql"],
                timeout_sec=float(parsed["timeout_sec"]),
                operation=_OP_COUNT,
            )
        )
        return _ok_remote_count(
            payload,
            provider_name=_provider_name(provider),
            parsed=parsed,
            group_by=group_by,
        )
    except OsmQueryInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def _parse_query_inputs(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    original: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timeout_sec = _env_timeout_int()
    overpass_ql = _optional_str(inputs.get("overpass_ql"))
    if overpass_ql:
        ql = _sanitize_overpass_ql(
            overpass_ql,
            timeout_sec=timeout_sec,
            maxsize=_DEFAULT_MAXSIZE,
        )
        unused = _structured_unused(inputs)
        return {
            "ql": ql,
            "timeout_sec": timeout_sec,
            "limit": None,
            "return_geometry": None,
            "scope": None,
            "tags": None,
            "feature_types": None,
            "element_types": None,
            "spatial_relation": None,
            "unused": unused,
        }
    _ensure_spatial_or_raise(inputs, ctx, for_count=False, original=original)
    return _compile_structured_ql(inputs, ctx, as_count=False, timeout_sec=timeout_sec)

def _parse_count_inputs(inputs: dict[str, Any], ctx: RuntimeContext | None) -> dict[str, Any]:
    timeout_sec = _env_timeout_int()
    _ensure_spatial_or_raise(inputs, ctx, for_count=True)
    return _compile_structured_ql(inputs, ctx, as_count=True, timeout_sec=timeout_sec)

def _compile_structured_ql(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    as_count: bool,
    timeout_sec: int,
) -> dict[str, Any]:
    spatial_relation = _parse_spatial_relation(inputs.get("spatial_relation"))
    radius_m = _parse_radius_m(inputs.get("radius_m"))
    scope = _resolve_scope(
        inputs,
        ctx,
        spatial_relation=spatial_relation,
        radius_m=radius_m,
    )
    feature_types = _parse_string_list(inputs.get("feature_types"), field="feature_types")
    tags = _parse_tags(inputs.get("tags"))
    tag_sets = _tag_sets(tags, feature_types)
    element_types = _parse_element_types(inputs.get("element_types"))
    limit = None if as_count else _parse_limit(inputs.get("limit"))
    return_geometry = False if as_count else _parse_return_geometry(inputs.get("return_geometry"))
    ql = _build_structured_ql(
        scope=scope,
        tag_sets=tag_sets,
        element_types=element_types,
        timeout_sec=timeout_sec,
        limit=limit or _DEFAULT_LIMIT,
        return_geometry=return_geometry,
        as_count=as_count,
    )
    return {
        "ql": ql,
        "timeout_sec": timeout_sec,
        "limit": limit,
        "return_geometry": None if as_count else return_geometry,
        "scope": scope,
        "tags": tags,
        "feature_types": feature_types,
        "element_types": element_types,
        "spatial_relation": spatial_relation,
        "unused": {},
    }

def _build_structured_ql(
    *,
    scope: QueryScope,
    tag_sets: list[dict[str, str]],
    element_types: tuple[str, ...],
    timeout_sec: int,
    limit: int,
    return_geometry: bool,
    as_count: bool,
) -> str:
    prefix, spatial = _scope_prefix_and_filter(scope)
    selector = _element_selector(element_types)
    statements = [_selector_statement(selector, tag_set, spatial) for tag_set in tag_sets]
    if len(statements) == 1:
        body = f"{statements[0]};"
    else:
        inner = "\n  ".join(f"{item};" for item in statements)
        body = f"(\n  {inner}\n);"
    if as_count:
        out_line = "out count;"
    elif return_geometry:
        out_line = f"out geom {limit};"
    else:
        out_line = f"out tags {limit};"
    parts = [f"[out:json][timeout:{timeout_sec}];"]
    if prefix:
        parts.append(prefix.rstrip())
    parts.append(body)
    parts.append(out_line)
    return "\n".join(parts)

def _scope_prefix_and_filter(scope: QueryScope) -> tuple[str, str]:
    if scope.kind == _KIND_BBOX and scope.bbox is not None:
        bbox = scope.bbox
        spatial = (
            f"({_fmt_num(bbox.south)},{_fmt_num(bbox.west)},"
            f"{_fmt_num(bbox.north)},{_fmt_num(bbox.east)})"
        )
        return "", spatial
    if scope.kind == _KIND_AROUND_POINT and scope.center is not None and scope.radius_m is not None:
        center = scope.center
        spatial = f"(around:{scope.radius_m},{_fmt_num(center.lat)},{_fmt_num(center.lon)})"
        return "", spatial
    if scope.kind == _KIND_AROUND_NAME and scope.name and scope.radius_m is not None:
        escaped = _ql_escape(scope.name)
        prefix = f'nwr["name"="{escaped}"]->.center;\n'
        return prefix, f"(around.center:{scope.radius_m})"
    if scope.kind == _KIND_AREA_NAME and scope.name:
        escaped = _ql_escape(scope.name)
        prefix = f'area["name"="{escaped}"]->.searchArea;\n'
        return prefix, "(area.searchArea)"
    if scope.kind == _KIND_POLY and scope.polygon:
        coords = " ".join(
            f"{_fmt_num(lat)} {_fmt_num(lon)}" for lon, lat in scope.polygon
        )
        return "", f'(poly:"{coords}")'
    raise OsmQueryInputError("无法构造空间过滤条件", "invalid_input")

def _selector_statement(selector: str, tags: dict[str, str], spatial: str) -> str:
    return f"{selector}{_tag_filters(tags)}{spatial}"

def _element_selector(element_types: tuple[str, ...]) -> str:
    chosen = set(element_types)
    if not chosen or chosen >= set(_ELEMENT_TYPES):
        return "nwr"
    letters = {"node": "n", "way": "w", "relation": "r"}
    ordered = [name for name in _ELEMENT_TYPES if name in chosen]
    if len(ordered) == 1:
        return ordered[0]
    return "".join(letters[name] for name in ordered)

def _tag_filters(tags: dict[str, str]) -> str:
    parts: list[str] = []
    for key, value in tags.items():
        escaped_key = _ql_escape(key)
        if value in {"", "*"}:
            parts.append(f'["{escaped_key}"]')
        else:
            parts.append(f'["{escaped_key}"="{_ql_escape(value)}"]')
    return "".join(parts)

def _tag_sets(tags: dict[str, str], feature_types: tuple[str, ...]) -> list[dict[str, str]]:
    mapped = [_map_feature_type(item) for item in feature_types]
    if not mapped:
        return [dict(tags)]
    merged: list[dict[str, str]] = []
    for item in mapped:
        row = dict(item)
        row.update(tags)
        merged.append(row)
    return merged

def _map_feature_type(raw: str) -> dict[str, str]:
    key = raw.strip()
    mapped = _FEATURE_TYPE_TAGS.get(key) or _FEATURE_TYPE_TAGS.get(key.lower())
    if mapped is None:
        raise OsmQueryInputError(
            f"无法映射 feature_types={raw!r}，请改传 tags",
            "missing_input",
        )
    return dict(mapped)

def _sanitize_overpass_ql(ql: str, *, timeout_sec: int, maxsize: int) -> str:
    text = ql.strip()
    if not text:
        raise OsmQueryInputError("缺少 overpass_ql", "missing_input")
    if "{{" in text:
        raise OsmQueryInputError("不支持 Overpass Turbo 宏，请改用结构化参数", "unsupported_query")
    out_match = _OUT_RE.search(text)
    if out_match:
        kind = out_match.group(1).strip().lower()
        if kind != "json":
            raise OsmQueryInputError("overpass_ql 仅支持 [out:json]", "unsupported_query")
    else:
        text = f"[out:json];{text}"

    def _cap_timeout(match: re.Match[str]) -> str:
        value = min(int(match.group(1)), timeout_sec)
        return f"[timeout:{value}]"

    if _TIMEOUT_RE.search(text):
        text = _TIMEOUT_RE.sub(_cap_timeout, text, count=1)
    else:
        text = _OUT_RE.sub(lambda match: f"{match.group(0)}[timeout:{timeout_sec}]", text, count=1)

    def _cap_maxsize(match: re.Match[str]) -> str:
        value = min(int(match.group(1)), maxsize)
        return f"[maxsize:{value}]"

    if _MAXSIZE_RE.search(text):
        text = _MAXSIZE_RE.sub(_cap_maxsize, text, count=1)
    if not _has_spatial_constraint(text):
        raise OsmQueryInputError(
            "overpass_ql 必须包含 bbox、around、area 或 poly 空间约束",
            "unsupported_query",
        )
    return text

def _has_spatial_constraint(ql: str) -> bool:
    return bool(
        _BBOX_FILTER_RE.search(ql)
        or _AROUND_RE.search(ql)
        or _AREA_FILTER_RE.search(ql)
        or _POLY_RE.search(ql)
        or _AREA_STMT_RE.search(ql)
    )

def _resolve_scope(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    spatial_relation: str,
    radius_m: int | None,
) -> QueryScope:
    bbox = _parse_bbox(inputs.get("bbox"))
    center, center_name = _parse_center(inputs.get("center"))
    area = _parse_area(_resolve_area(inputs.get("area"), ctx))
    effective_radius = radius_m if radius_m is not None else area.radius_m
    if spatial_relation == _REL_NEAR:
        return _near_scope(
            bbox=bbox,
            center=center,
            center_name=center_name,
            area=area,
            radius_m=effective_radius,
        )
    if bbox is not None:
        return QueryScope(kind=_KIND_BBOX, bbox=bbox, applied=_bbox_applied(bbox))
    if center is not None:
        if effective_radius is None:
            raise OsmQueryInputError("center 需要配合 radius_m", "missing_input")
        return QueryScope(
            kind=_KIND_AROUND_POINT,
            center=center,
            radius_m=effective_radius,
            applied=_center_applied(center, effective_radius),
        )
    if center_name:
        if effective_radius is None:
            raise OsmQueryInputError("center 需要配合 radius_m", "missing_input")
        return QueryScope(
            kind=_KIND_AROUND_NAME,
            name=center_name,
            radius_m=effective_radius,
            applied={"name": center_name, "radius_m": effective_radius},
        )
    if area.bbox is not None:
        return QueryScope(kind=_KIND_BBOX, bbox=area.bbox, applied=area.applied)
    if area.polygon is not None:
        return QueryScope(kind=_KIND_POLY, polygon=area.polygon, applied=area.applied)
    if area.center is not None:
        if effective_radius is None:
            raise OsmQueryInputError("center 需要配合 radius_m", "missing_input")
        return QueryScope(
            kind=_KIND_AROUND_POINT,
            center=area.center,
            radius_m=effective_radius,
            applied=_center_applied(area.center, effective_radius),
        )
    if area.text:
        return QueryScope(kind=_KIND_AREA_NAME, name=area.text, applied=area.text)
    if area.unsupported is not None:
        raise OsmQueryInputError("area 无法解析为地名、bbox 或中心点", "invalid_input")
    raise OsmQueryInputError(_missing_spatial_message(inputs, for_count=False), "missing_input")

def _near_scope(
    *,
    bbox: BBox | None,
    center: GeoPoint | None,
    center_name: str | None,
    area: ParsedArea,
    radius_m: int | None,
) -> QueryScope:
    if radius_m is None:
        raise OsmQueryInputError("spatial_relation=near 需要 radius_m", "missing_input")
    if center is not None:
        return QueryScope(
            kind=_KIND_AROUND_POINT,
            center=center,
            radius_m=radius_m,
            applied=_center_applied(center, radius_m),
        )
    if center_name:
        return QueryScope(
            kind=_KIND_AROUND_NAME,
            name=center_name,
            radius_m=radius_m,
            applied={"name": center_name, "radius_m": radius_m},
        )
    if bbox is not None:
        mid = _bbox_center(bbox)
        return QueryScope(
            kind=_KIND_AROUND_POINT,
            center=mid,
            radius_m=radius_m,
            applied=_center_applied(mid, radius_m),
        )
    if area.center is not None:
        return QueryScope(
            kind=_KIND_AROUND_POINT,
            center=area.center,
            radius_m=radius_m,
            applied=_center_applied(area.center, radius_m),
        )
    if area.bbox is not None:
        mid = _bbox_center(area.bbox)
        return QueryScope(
            kind=_KIND_AROUND_POINT,
            center=mid,
            radius_m=radius_m,
            applied=_center_applied(mid, radius_m),
        )
    name = area.text
    if name:
        return QueryScope(
            kind=_KIND_AROUND_NAME,
            name=name,
            radius_m=radius_m,
            applied={"name": name, "radius_m": radius_m},
        )
    raise OsmQueryInputError("spatial_relation=near 需要 area、bbox 或 center", "missing_input")

def _ensure_spatial_or_raise(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    for_count: bool,
    original: dict[str, Any] | None = None,
) -> None:
    if _has_spatial_input(inputs, ctx, for_count=for_count):
        return
    raise OsmQueryInputError(
        _missing_spatial_message(original or inputs, for_count=for_count),
        "missing_input",
    )

def _has_spatial_input(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    for_count: bool,
) -> bool:
    if _filled(inputs.get("bbox")):
        return True
    if not for_count and (
        _filled(inputs.get("overpass_ql")) or _filled(inputs.get("center"))
    ):
        return True
    area = _resolve_area(inputs.get("area"), ctx)
    if _filled(area):
        return True
    if for_count and _filled(inputs.get("source_result")):
        return True
    return False

def _missing_spatial_message(inputs: dict[str, Any], *, for_count: bool) -> str:
    if _looks_like_image_only(inputs):
        if for_count:
            return "Overpass 不会看图；请提供 source_result、area 或 bbox"
        return "Overpass 不会看图；请提供 area、bbox、center 或 overpass_ql"
    if for_count:
        return "缺少 source_result、area 或 bbox"
    return "缺少 area、bbox、center 或 overpass_ql"

def _looks_like_image_only(inputs: dict[str, Any]) -> bool:
    keys = {str(key).lower() for key in inputs}
    has_image = bool(keys & _IMAGE_KEYS) or any(
        isinstance(value, str) and value.strip() == "$current_image" for value in inputs.values()
    )
    has_spatial = any(_filled(inputs.get(key)) for key in _SPATIAL_KEYS)
    return has_image and not has_spatial

def _ok_query(
    payload: dict[str, Any],
    *,
    provider_name: str,
    parsed: dict[str, Any],
) -> Observation:
    elements = _normalize_elements(
        payload,
        return_geometry=parsed.get("return_geometry") is not False,
        limit=parsed.get("limit"),
    )
    result = _result_envelope(
        operation=_OP_QUERY,
        provider_name=provider_name,
        parsed=parsed,
        payload=payload,
        extra={"elements": elements, "count": len(elements)},
    )
    return Observation(ok=True, result=_strip_forbidden(result))

def _ok_remote_count(
    payload: dict[str, Any],
    *,
    provider_name: str,
    parsed: dict[str, Any],
    group_by: tuple[str, ...],
) -> Observation:
    extra: dict[str, Any] = {"count": _count_from_payload(payload)}
    if group_by:
        extra["groups"] = []
    result = _result_envelope(
        operation=_OP_COUNT,
        provider_name=provider_name,
        parsed=parsed,
        payload=payload,
        extra=extra,
    )
    return Observation(ok=True, result=_strip_forbidden(result))

def _ok_local_count(
    source: dict[str, Any],
    *,
    group_by: tuple[str, ...],
) -> Observation:
    elements = _elements_from_source(source)
    extra: dict[str, Any] = {"count": len(elements), "applied": {"provider": "local"}}
    if group_by:
        extra["groups"] = _group_elements(elements, group_by)
    timestamp = source.get("data_timestamp") if isinstance(source, dict) else None
    result: dict[str, Any] = {
        "operation": _OP_COUNT,
        "count": extra["count"],
        "applied": extra["applied"],
        "assumptions": _assumptions(provider_name="local"),
    }
    if "groups" in extra:
        result["groups"] = extra["groups"]
    if isinstance(timestamp, str) and timestamp.strip():
        result["data_timestamp"] = timestamp.strip()
    return Observation(ok=True, result=_strip_forbidden(result))

def _result_envelope(
    *,
    operation: str,
    provider_name: str,
    parsed: dict[str, Any],
    payload: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    applied: dict[str, Any] = {
        "provider": provider_name,
        "crs": _CRS_WGS84,
        "overpass_ql": parsed["ql"],
        "timeout_sec": parsed["timeout_sec"],
    }
    if parsed.get("limit") is not None:
        applied["limit"] = parsed["limit"]
    if parsed.get("return_geometry") is not None:
        applied["return_geometry"] = parsed["return_geometry"]
    scope = parsed.get("scope")
    if isinstance(scope, QueryScope) and scope.applied is not None:
        if scope.kind == _KIND_BBOX:
            applied["bbox"] = scope.applied
        elif scope.kind in {_KIND_AROUND_POINT, _KIND_AROUND_NAME}:
            applied["center"] = scope.applied
        else:
            applied["area"] = scope.applied
    if parsed.get("tags"):
        applied["tags"] = parsed["tags"]
    if parsed.get("feature_types"):
        applied["feature_types"] = list(parsed["feature_types"])
    if parsed.get("element_types"):
        applied["element_types"] = list(parsed["element_types"])
    if parsed.get("spatial_relation"):
        applied["spatial_relation"] = parsed["spatial_relation"]
    unused = parsed.get("unused") or {}
    if unused:
        applied["unused"] = unused
    result: dict[str, Any] = {
        "operation": operation,
        "applied": applied,
        "assumptions": _assumptions(provider_name=provider_name),
    }
    result.update(extra)
    timestamp = _data_timestamp(payload)
    if timestamp:
        result["data_timestamp"] = timestamp
    return result

def _assumptions(*, provider_name: str) -> list[str]:
    items = [
        _COUNT_ASSUMPTION,
        _TIMESTAMP_ASSUMPTION,
        _NO_IMAGE_ASSUMPTION,
        _NO_SESSION_ASSUMPTION,
    ]
    if provider_name == _PROVIDER_OVERPASS:
        items.insert(2, _CRS_ASSUMPTION)
    elif provider_name not in {"local"}:
        items.insert(2, f"当前后端为 {provider_name}，坐标为 WGS84，未做坐标系转换")
    return items

def _normalize_elements(
    payload: dict[str, Any],
    *,
    return_geometry: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    raw_elements = payload.get("elements")
    if not isinstance(raw_elements, list):
        return []
    elements: list[dict[str, Any]] = []
    for item in raw_elements:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").lower() == "count":
            continue
        row = _normalize_element(item, return_geometry=return_geometry)
        if row is None:
            continue
        row["result_id"] = f"osm_{len(elements) + 1}"
        elements.append(row)
        if limit is not None and len(elements) >= limit:
            break
    return elements

def _normalize_element(item: dict[str, Any], *, return_geometry: bool) -> dict[str, Any] | None:
    osm_type = str(item.get("type") or item.get("osm_type") or "").strip().lower()
    osm_id = _as_int(item.get("id", item.get("osm_id")))
    if osm_type not in _ELEMENT_TYPES or osm_id is None:
        return None
    row: dict[str, Any] = {"osm_type": osm_type, "osm_id": osm_id}
    tags = item.get("tags")
    if isinstance(tags, dict):
        cleaned = {
            str(key): _stringify_tag(value)
            for key, value in tags.items()
            if key not in _FORBIDDEN_KEYS and _stringify_tag(value) is not None
        }
        if cleaned:
            row["tags"] = cleaned
    if return_geometry:
        geometry = _element_geometry(item, osm_type)
        if geometry is not None:
            row["geometry"] = geometry
    return row

def _element_geometry(item: dict[str, Any], osm_type: str) -> dict[str, Any] | None:
    if osm_type == _ELEMENT_NODE:
        lat = _as_float(item.get("lat"))
        lon = _as_float(item.get("lon", item.get("lng")))
        if lat is None or lon is None:
            return None
        return {"type": "Point", "coordinates": [lon, lat]}
    if osm_type == _ELEMENT_WAY:
        points = _geometry_points(item.get("geometry"))
        if points is None:
            return None
        return {"type": "LineString", "coordinates": points}
    return None

def _geometry_points(raw: Any) -> list[list[float]] | None:
    if not isinstance(raw, list) or not raw:
        return None
    points: list[list[float]] = []
    for item in raw:
        if isinstance(item, dict):
            lat = _as_float(item.get("lat"))
            lon = _as_float(item.get("lon", item.get("lng")))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lon = _as_float(item[0])
            lat = _as_float(item[1])
        else:
            return None
        if lat is None or lon is None:
            return None
        points.append([lon, lat])
    if len(points) < 2:
        return None
    return points

def _count_from_payload(payload: dict[str, Any]) -> int:
    raw_elements = payload.get("elements")
    if not isinstance(raw_elements, list):
        return 0
    for item in raw_elements:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").lower() != "count":
            continue
        tags = item.get("tags")
        if isinstance(tags, dict):
            total = _as_int(tags.get("total"))
            if total is not None:
                return total
    return len(
        [
            item
            for item in raw_elements
            if isinstance(item, dict) and str(item.get("type") or "").lower() != "count"
        ]
    )

def _data_timestamp(payload: dict[str, Any]) -> str | None:
    osm3s = payload.get("osm3s")
    if isinstance(osm3s, dict):
        stamp = osm3s.get("timestamp_osm_base")
        if isinstance(stamp, str) and stamp.strip():
            return stamp.strip()
    remark = payload.get("remark")
    if isinstance(remark, dict):
        stamp = remark.get("timestamp_osm_base")
        if isinstance(stamp, str) and stamp.strip():
            return stamp.strip()
    return None

def _resolve_source_result(raw: Any, ctx: RuntimeContext | None) -> dict[str, Any] | None:
    if raw is None or raw == "":
        return None
    if raw == "$previous_tool_result":
        previous = ctx.previous_tool_result if ctx is not None else None
        if not isinstance(previous, dict):
            raise OsmQueryInputError("无法解析 $previous_tool_result", "missing_input")
        return _unwrap_source(previous)
    if isinstance(raw, (dict, list)):
        return _unwrap_source(raw)
    if isinstance(raw, str):
        previous = ctx.previous_tool_result if ctx is not None else None
        if isinstance(previous, dict):
            return _unwrap_source(previous)
        raise OsmQueryInputError("无法解析 source_result，请传入上一轮 osm_query 结果", "missing_input")
    raise OsmQueryInputError("source_result 必须是结果对象或引用", "invalid_input")

def _unwrap_source(raw: Any) -> dict[str, Any]:
    if isinstance(raw, list):
        return {"elements": raw}
    if not isinstance(raw, dict):
        raise OsmQueryInputError("source_result 无法解析为 OSM 结果", "invalid_input")
    if isinstance(raw.get("elements"), list):
        return raw
    nested = raw.get("result")
    if isinstance(nested, dict) and isinstance(nested.get("elements"), list):
        return nested
    raise OsmQueryInputError("source_result 缺少 elements", "invalid_input")

def _elements_from_source(source: dict[str, Any]) -> list[dict[str, Any]]:
    raw = source.get("elements")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]

def _group_elements(elements: list[dict[str, Any]], group_by: tuple[str, ...]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, ...], int] = {}
    for item in elements:
        tags = item.get("tags") if isinstance(item.get("tags"), dict) else {}
        key = tuple(_stringify_tag(tags.get(field)) or "" for field in group_by)
        counts[key] = counts.get(key, 0) + 1
    groups: list[dict[str, Any]] = []
    for key, total in counts.items():
        row: dict[str, Any] = {"count": total}
        for field, value in zip(group_by, key, strict=True):
            row[field] = value
        groups.append(row)
    return groups

def _parse_group_by(raw: Any) -> tuple[str, ...]:
    return _parse_string_list(raw, field="group_by")

def _parse_tags(raw: Any) -> dict[str, str]:
    if raw is None or raw == "":
        return {}
    if not isinstance(raw, dict):
        raise OsmQueryInputError("tags 必须是对象", "invalid_input")
    tags: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            continue
        if value is None or value == "" or value == "*":
            tags[name] = "*"
            continue
        if isinstance(value, bool):
            tags[name] = "yes" if value else "no"
            continue
        text = _stringify_tag(value)
        if text is None:
            raise OsmQueryInputError("tags 取值必须是字符串或标量", "invalid_input")
        tags[name] = text
    return tags

def _parse_element_types(raw: Any) -> tuple[str, ...]:
    values = _parse_string_list(raw, field="element_types")
    if not values:
        return _ELEMENT_TYPES
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        name = item.strip().lower()
        aliases = {"nodes": _ELEMENT_NODE, "ways": _ELEMENT_WAY, "relations": _ELEMENT_RELATION, "rel": _ELEMENT_RELATION}
        name = aliases.get(name, name)
        if name not in _ELEMENT_TYPES:
            raise OsmQueryInputError("element_types 仅允许 node、way、relation", "invalid_input")
        if name not in seen:
            seen.add(name)
            normalized.append(name)
    return tuple(normalized)

def _parse_spatial_relation(raw: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _REL_WITHIN
    if not isinstance(raw, str):
        raise OsmQueryInputError("spatial_relation 必须是字符串", "invalid_input")
    mapped = _SPATIAL_RELATIONS.get(raw.strip()) or _SPATIAL_RELATIONS.get(raw.strip().lower())
    if mapped is None:
        raise OsmQueryInputError("spatial_relation 仅允许 within、near、intersects", "invalid_input")
    return mapped

def _parse_return_geometry(raw: Any) -> bool:
    if raw is None or raw == "":
        return True
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    raise OsmQueryInputError("return_geometry 必须是布尔值", "invalid_input")

def _parse_limit(raw: Any) -> int:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _env_limit_default()
    value = _as_int(raw)
    if value is None:
        raise OsmQueryInputError("limit 必须是整数", "invalid_input")
    if value < _LIMIT_MIN or value > _LIMIT_MAX:
        raise OsmQueryInputError(f"limit 必须在 {_LIMIT_MIN} 到 {_LIMIT_MAX} 之间", "invalid_input")
    return value

def _parse_radius_m(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    value = _as_float(raw)
    if value is None or value < 0:
        raise OsmQueryInputError("radius_m 必须是非负数", "invalid_input")
    return int(round(value))

def _parse_bbox(raw: Any) -> BBox | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, dict):
        bbox = _bbox_from_mapping(raw)
        if bbox is None:
            raise OsmQueryInputError("bbox 必须是 [west,south,east,north]", "invalid_input")
        return bbox
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        bbox = _bbox_from_values(list(raw))
        if bbox is None:
            raise OsmQueryInputError("bbox 必须是 [west,south,east,north]", "invalid_input")
        return bbox
    raise OsmQueryInputError("bbox 必须是 [west,south,east,north]", "invalid_input")

def _parse_center(raw: Any) -> tuple[GeoPoint | None, str | None]:
    if raw is None or raw == "":
        return None, None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None, None
        point = _parse_coordinate_text(text)
        if point is not None:
            return point, None
        return None, text
    if isinstance(raw, dict):
        point = _center_from_mapping(raw)
        if point is not None:
            return point, None
        name = _optional_str(raw.get("name", raw.get("text", raw.get("place"))))
        if name:
            return None, name
        raise OsmQueryInputError("center 必须是坐标或地名", "invalid_input")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        point = _parse_coordinate_pair(raw[0], raw[1])
        if point is None:
            raise OsmQueryInputError("center 坐标无效", "invalid_input")
        return point, None
    raise OsmQueryInputError("center 必须是坐标或地名", "invalid_input")

def _parse_string_list(raw: Any, *, field: str) -> tuple[str, ...]:
    if raw is None or raw == "":
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    if isinstance(raw, (list, tuple)):
        values: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise OsmQueryInputError(f"{field} 必须是字符串或字符串列表", "invalid_input")
            values.append(item.strip())
        return tuple(values)
    raise OsmQueryInputError(f"{field} 必须是字符串或字符串列表", "invalid_input")

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
            return ParsedArea(center=point, applied=_center_applied(point, None))
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
            radius = _parse_radius_m(raw.get("radius_m", raw.get("radius")))
            return ParsedArea(
                center=center,
                radius_m=radius,
                applied=_center_applied(center, radius),
            )
        text = _optional_str(raw.get("city", raw.get("text", raw.get("name", raw.get("area")))))
        if text:
            return ParsedArea(text=text, applied=text)
        return ParsedArea(unsupported=raw)
    return ParsedArea(unsupported=raw)

def _bbox_from_mapping(raw: dict[str, Any]) -> BBox | None:
    nested = raw.get("bbox")
    if isinstance(nested, (list, tuple)) and len(nested) == 4:
        return _bbox_from_values(list(nested))
    west = _as_float(raw.get("west", raw.get("min_lon", raw.get("minx"))))
    south = _as_float(raw.get("south", raw.get("min_lat", raw.get("miny"))))
    east = _as_float(raw.get("east", raw.get("max_lon", raw.get("maxx"))))
    north = _as_float(raw.get("north", raw.get("max_lat", raw.get("maxy"))))
    if west is None or south is None or east is None or north is None:
        return None
    return _bbox_from_values([west, south, east, north])

def _bbox_from_values(raw: list[Any]) -> BBox | None:
    west = _as_float(raw[0])
    south = _as_float(raw[1])
    east = _as_float(raw[2])
    north = _as_float(raw[3])
    if west is None or south is None or east is None or north is None:
        return None
    if not -180.0 <= west <= 180.0 or not -180.0 <= east <= 180.0:
        return None
    if not -90.0 <= south <= 90.0 or not -90.0 <= north <= 90.0:
        return None
    if west >= east or south >= north:
        return None
    return BBox(west=west, south=south, east=east, north=north)

def _center_from_mapping(raw: dict[str, Any]) -> GeoPoint | None:
    lat = _as_float(raw.get("lat", raw.get("latitude")))
    lon = _as_float(raw.get("lon", raw.get("lng", raw.get("longitude"))))
    nested = raw.get("center")
    if isinstance(nested, dict) and (lat is None or lon is None):
        lat = _as_float(nested.get("lat", nested.get("latitude")))
        lon = _as_float(nested.get("lon", nested.get("lng", nested.get("longitude"))))
    if lat is None or lon is None:
        return None
    return _validated_point(lat, lon)

def _parse_coordinate_text(raw: str) -> GeoPoint | None:
    parts = [part for part in _COORD_SPLIT_RE.split(raw.strip()) if part]
    if len(parts) != 2:
        return None
    return _parse_coordinate_pair(parts[0], parts[1])

def _parse_coordinate_pair(first_raw: Any, second_raw: Any) -> GeoPoint | None:
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
    return _validated_point(lat, lon)

def _validated_point(lat: float, lon: float) -> GeoPoint | None:
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return GeoPoint(lat=lat, lon=lon)

def _polygon_from_points(raw: Any) -> tuple[tuple[float, float], ...] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
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
    if len(points) < 3:
        return None
    return tuple(points)

def _bbox_center(bbox: BBox) -> GeoPoint:
    return GeoPoint(lat=(bbox.south + bbox.north) / 2.0, lon=(bbox.west + bbox.east) / 2.0)

def _bbox_applied(bbox: BBox) -> dict[str, float | str]:
    return {
        "west": bbox.west,
        "south": bbox.south,
        "east": bbox.east,
        "north": bbox.north,
        "crs": _CRS_WGS84,
    }

def _center_applied(center: GeoPoint, radius_m: int | None) -> dict[str, Any]:
    applied: dict[str, Any] = {"lat": center.lat, "lon": center.lon, "crs": _CRS_WGS84}
    if radius_m is not None:
        applied["radius_m"] = radius_m
    return applied

def _polygon_applied(points: tuple[tuple[float, float], ...]) -> list[list[float]]:
    return [[lon, lat] for lon, lat in points]

def _structured_unused(inputs: dict[str, Any]) -> dict[str, Any]:
    unused: dict[str, Any] = {}
    for key in _STRUCTURED_UNUSED:
        if _filled(inputs.get(key)):
            unused[key] = inputs[key]
    return unused

def _resolve_provider(ctx: RuntimeContext | None) -> OsmQueryProvider:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("osm_query_provider")
    if injected is not None:
        if not isinstance(injected, OsmQueryProvider):
            raise EngineUnavailableError("osm_query_provider 必须提供 query(request)")
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_API=false，禁止调用真实 OSM 查询 API")
    return OverpassProvider(
        endpoint=_env_value("OSM_QUERY_OVERPASS_ENDPOINT", _DEFAULT_ENDPOINT),
        user_agent=_env_value("OSM_QUERY_OVERPASS_USER_AGENT", _DEFAULT_UA),
        timeout_sec=_env_timeout("OSM_QUERY_TIMEOUT_SEC"),
    )

def _provider_name(provider: OsmQueryProvider) -> str:
    return str(getattr(provider, "name", "injected"))

def _env_value(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default

def _env_timeout(name: str) -> float:
    return _parse_timeout(os.environ.get(name, "").strip(), _DEFAULT_TIMEOUT_SEC)

def _env_timeout_int() -> int:
    return int(round(_env_timeout("OSM_QUERY_TIMEOUT_SEC")))

def _env_limit_default() -> int:
    parsed = _as_int(os.environ.get("OSM_QUERY_MAX_LIMIT", "").strip())
    if parsed is None:
        return _DEFAULT_LIMIT
    return max(_LIMIT_MIN, min(_LIMIT_MAX, parsed))

def _parse_timeout(raw: str, default: float) -> float:
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    return default

def _http_post_json(
    url: str,
    *,
    body: bytes,
    headers: dict[str, str],
    timeout_sec: float,
    error_prefix: str,
) -> Any:
    request = urllib.request.Request(url, data=body, method="POST")
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
        raise EngineUnavailableError(f"{error_prefix} 网络失败: {exc.reason}") from exc
    except (json.JSONDecodeError, TimeoutError, OSError) as exc:
        raise EngineUnavailableError(f"{error_prefix} 调用失败: {exc}") from exc
    return raw

def _ql_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')

def _fmt_num(value: float) -> str:
    if value == int(value) and abs(value) < 1_000_000_000:
        return str(int(value))
    return f"{value:.7f}".rstrip("0").rstrip(".")

def _stringify_tag(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return "yes" if raw else "no"
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if isinstance(raw, float) and raw.is_integer():
            return str(int(raw))
        return str(raw)
    if isinstance(raw, str):
        return raw.strip()
    return None

def _optional_str(raw: Any) -> str | None:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None

def _filled(raw: Any) -> bool:
    if raw is None or raw == "":
        return False
    if isinstance(raw, (list, tuple, dict)) and not raw:
        return False
    return True

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
