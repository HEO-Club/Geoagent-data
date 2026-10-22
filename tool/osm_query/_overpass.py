"""OSM/Overpass 查询构造、网络适配和严格结果归一。"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol, runtime_checkable

from tool.contract import Observation, RuntimeContext
from tool.runtime.result_store import resolve_result, store_result

_DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"
_DEFAULT_LIMIT = 100
_MAX_LIMIT = 1000
_MAX_RESPONSE_BYTES = 20 * 1024 * 1024
_MAX_RAW_QUERY_CHARS = 20_000
_TAG_KEY_RE = re.compile(r"^[A-Za-z0-9_:.-]+$")
_ALLOWED_ELEMENT_TYPES = {"node", "way", "relation"}

_FEATURE_TAGS: dict[str, list[dict[str, str]]] = {
    "bridge": [{"bridge": "*"}],
    "桥": [{"bridge": "*"}],
    "桥梁": [{"bridge": "*"}],
    "river": [{"waterway": "river"}],
    "河流": [{"waterway": "river"}],
    "water": [{"natural": "water"}, {"waterway": "*"}],
    "水体": [{"natural": "water"}, {"waterway": "*"}],
    "railway": [{"railway": "*"}],
    "铁路": [{"railway": "*"}],
    "road": [{"highway": "*"}],
    "道路": [{"highway": "*"}],
    "school": [{"amenity": "school"}],
    "学校": [{"amenity": "school"}],
    "park": [{"leisure": "park"}],
    "公园": [{"leisure": "park"}],
    "power_tower": [{"power": "tower"}],
    "电塔": [{"power": "tower"}],
    "电线杆": [{"power": "pole"}],
    "building": [{"building": "*"}],
    "建筑": [{"building": "*"}],
    "tower": [{"man_made": "tower"}],
    "塔": [{"man_made": "tower"}],
    "dam": [{"waterway": "dam"}],
    "水坝": [{"waterway": "dam"}],
    "station": [{"public_transport": "station"}, {"railway": "station"}],
    "车站": [{"public_transport": "station"}, {"railway": "station"}],
    "airport": [{"aeroway": "aerodrome"}],
    "机场": [{"aeroway": "aerodrome"}],
    "tunnel": [{"tunnel": "*"}],
    "隧道": [{"tunnel": "*"}],
}


class OverpassInputError(Exception):
    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class OverpassProviderError(Exception):
    def __init__(self, message: str, error_code: str = "provider_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code


@runtime_checkable
class OverpassClient(Protocol):
    name: str

    def query(self, overpass_ql: str) -> dict[str, Any]:
        """执行 Overpass QL 并返回 JSON 对象。"""


class HttpOverpassClient:
    name = "overpass"

    def __init__(self, *, endpoint: str, user_agent: str, timeout_sec: float) -> None:
        self._endpoint = endpoint
        self._user_agent = user_agent
        self._timeout_sec = timeout_sec

    def query(self, overpass_ql: str) -> dict[str, Any]:
        body = urllib.parse.urlencode({"data": overpass_ql}).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={
                "User-Agent": self._user_agent,
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_sec) as response:
                payload = response.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise OverpassProviderError(
                f"Overpass HTTP {exc.code}: {detail[:200]}",
                "provider_error",
            ) from exc
        except urllib.error.URLError as exc:
            raise OverpassProviderError(f"Overpass 网络失败: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise OverpassProviderError(f"Overpass 调用失败: {exc}") from exc
        if len(payload) > _MAX_RESPONSE_BYTES:
            raise OverpassProviderError("Overpass 回执超过 20 MiB 上限", "response_too_large")
        try:
            raw = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OverpassProviderError("Overpass 回执不是有效 JSON", "provider_error") from exc
        if not isinstance(raw, dict):
            raise OverpassProviderError("Overpass 回执不是 JSON 对象", "provider_error")
        return raw


def execute_query(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    del purpose
    try:
        provider, ql, raw = run_overpass(inputs, ctx)
        elements = normalize_elements(raw, include_geometry=bool(inputs.get("return_geometry")))
    except OverpassInputError as exc:
        return _fail(str(exc), exc.error_code)
    except OverpassProviderError as exc:
        return _fail(str(exc), exc.error_code)
    result: dict[str, Any] = {
        "provider": provider,
        "query_language": "Overpass QL",
        "overpass_ql": ql,
        "element_count": len(elements),
        "elements": elements,
        "crs": "EPSG:4326",
        "attribution": "© OpenStreetMap contributors",
        "assumptions": [
            "OSM 要素来自社区维护数据，缺失要素不等于现实中不存在",
            "area 名称按 OSM 行政边界精确匹配；模糊地理描述应先调用 geocode 获取 bbox",
        ],
    }
    result_id = store_result(result, namespace="osm", ctx=ctx)
    if result_id:
        result["result_id"] = result_id
    return Observation(ok=True, result=result)


def execute_count(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    del purpose
    source_ref = inputs.get("source_result")
    if inputs.get("group_by") is not None and source_ref is None:
        return _fail(
            "group_by 需要真实 source_result；请先执行 query，再对完整结果进行本地分组",
            "needs_acquisition",
        )
    if source_ref is not None:
        source = resolve_result(source_ref, ctx)
        if source is None:
            return _fail(
                "source_result 尚未解析；请先执行 osm_query.query 并传入真实 result_id",
                "needs_acquisition",
            )
        elements = source.get("elements")
        if not isinstance(elements, list):
            return _fail("source_result 不包含 OSM elements", "invalid_source_result")
        result = _count_local(elements, inputs.get("group_by"))
        result["source_result"] = source_ref
        result["provider"] = "local_result_store"
        return Observation(ok=True, result=result)
    try:
        provider, ql, raw = run_overpass(inputs, ctx, count=True)
        counts = _provider_counts(raw)
    except OverpassInputError as exc:
        return _fail(str(exc), exc.error_code)
    except OverpassProviderError as exc:
        return _fail(str(exc), exc.error_code)
    result = {
        "provider": provider,
        "query_language": "Overpass QL",
        "overpass_ql": ql,
        "count": counts["total"],
        "counts": counts,
        "attribution": "© OpenStreetMap contributors",
    }
    return Observation(ok=True, result=result)


def run_overpass(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    count: bool = False,
    name_pattern: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    raw_query = inputs.get("overpass_ql")
    if raw_query is not None:
        ql = _raw_query(raw_query, ctx)
    else:
        ql = build_query(inputs, count=count, name_pattern=name_pattern)
    client = _client(ctx)
    raw = client.query(ql)
    if raw.get("remark") and not isinstance(raw.get("elements"), list):
        raise OverpassProviderError(f"Overpass 返回错误: {raw['remark']}", "provider_error")
    return client.name, ql, raw


def build_query(
    inputs: dict[str, Any],
    *,
    count: bool = False,
    name_pattern: str | None = None,
) -> str:
    prefix, scope, scope_kind = _scope(inputs)
    relation = inputs.get("spatial_relation")
    if relation is not None:
        if not isinstance(relation, str) or relation.strip().lower() not in {"within", "near"}:
            raise OverpassInputError(
                "当前 OSM 查询只支持 spatial_relation=within 或 near",
                "unsupported_spatial_relation",
            )
        expected = "near" if scope_kind == "center" else "within"
        if relation.strip().lower() != expected:
            raise OverpassInputError(
                f"当前查询范围对应 spatial_relation={expected}",
                "invalid_spatial_relation",
            )
    selectors = _selectors(inputs.get("tags"), inputs.get("feature_types"))
    if name_pattern:
        name_selector = {"name": f"~{name_pattern}"}
        selectors = [{**selector, **name_selector} for selector in selectors] if selectors else [name_selector]
    if not selectors:
        raise OverpassInputError(
            "查询必须提供 tags、可识别的 feature_types 或 POI 名称，禁止无过滤遍历整个区域",
            "missing_filter",
        )
    element_types = _element_types(inputs.get("element_types"))
    lines: list[str] = []
    for selector in selectors:
        tags = _render_tags(selector)
        for element_type in element_types:
            lines.append(f"  {element_type}{tags}{scope};")
    body = "\n".join(lines)
    if count:
        output = "out count;"
    else:
        geometry = "geom" if bool(inputs.get("return_geometry")) else "center"
        output = f"out tags {geometry} qt {_limit(inputs.get('limit'))};"
    return f"[out:json][timeout:25];\n{prefix}(\n{body}\n);\n{output}"


def normalize_elements(raw: dict[str, Any], *, include_geometry: bool) -> list[dict[str, Any]]:
    elements = raw.get("elements")
    if not isinstance(elements, list):
        raise OverpassProviderError("Overpass 回执缺少 elements", "provider_error")
    rows: list[dict[str, Any]] = []
    for item in elements:
        if not isinstance(item, dict) or item.get("type") == "count":
            continue
        osm_type = str(item.get("type") or "")
        osm_id = item.get("id")
        raw_tags = item.get("tags")
        tags: dict[str, Any] = raw_tags if isinstance(raw_tags, dict) else {}
        lat, lon = _element_coordinates(item)
        row: dict[str, Any] = {
            "osm_type": osm_type,
            "osm_id": osm_id,
            "name": str(tags.get("name") or tags.get("name:zh") or ""),
            "latitude": lat,
            "longitude": lon,
            "tags": {str(key): _safe_scalar(value) for key, value in tags.items()},
        }
        raw_geometry = item.get("geometry")
        if include_geometry and isinstance(raw_geometry, list):
            geometry = []
            for point in raw_geometry:
                if isinstance(point, dict) and "lat" in point and "lon" in point:
                    try:
                        geometry.append([float(point["lon"]), float(point["lat"])])
                    except (TypeError, ValueError):
                        continue
            row["geometry"] = geometry
        bounds = item.get("bounds")
        if isinstance(bounds, dict):
            try:
                row["bbox"] = [
                    float(bounds["minlon"]),
                    float(bounds["minlat"]),
                    float(bounds["maxlon"]),
                    float(bounds["maxlat"]),
                ]
            except (KeyError, TypeError, ValueError):
                pass
        rows.append(row)
    return rows


def _scope(inputs: dict[str, Any]) -> tuple[str, str, str]:
    bbox = inputs.get("bbox")
    area = inputs.get("area")
    center = inputs.get("center")
    if bbox is None and isinstance(area, dict):
        bbox = area.get("bbox")
        center = center or area.get("center")
        if center is None and {"lat", "lon"} <= set(area):
            center = area
        area = area.get("name")
    if bbox is not None:
        west, south, east, north = _bbox(bbox)
        return "", f"({south:g},{west:g},{north:g},{east:g})", "bbox"
    if center is not None:
        lat, lon = _coordinates(center)
        radius = _positive_number(inputs.get("radius_m"), "radius_m")
        return "", f"(around:{radius:g},{lat:g},{lon:g})", "center"
    if isinstance(area, str) and area.strip():
        name = _escape(area.strip())
        prefix = f'area["name"="{name}"]["boundary"="administrative"]->.searchArea;\n'
        return prefix, "(area.searchArea)", "area"
    has_filter = any(
        inputs.get(key) is not None for key in ("tags", "feature_types")
    )
    raise OverpassInputError(
        "查询需要标准行政区 area、bbox，或 center+radius_m；模糊地点请先调用 geocode",
        "needs_acquisition" if has_filter else "missing_input",
    )


def _selectors(tags_raw: Any, features_raw: Any) -> list[dict[str, str]]:
    base = _tags(tags_raw)
    features = _string_list(features_raw, "feature_types")
    mapped: list[dict[str, str]] = []
    unknown: list[str] = []
    for feature in features:
        options = _FEATURE_TAGS.get(feature.strip().lower())
        if options is None:
            unknown.append(feature)
        else:
            mapped.extend(options)
    if unknown:
        raise OverpassInputError(
            f"无法安全映射为 OSM 标签的 feature_types: {unknown}；请显式提供 tags",
            "unsupported_feature_type",
        )
    if mapped:
        merged = [_merge_selector(base, option) for option in mapped]
        unique: list[dict[str, str]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for selector in merged:
            key = tuple(sorted(selector.items()))
            if key not in seen:
                seen.add(key)
                unique.append(selector)
        return unique
    return [base] if base else []


def _merge_selector(base: dict[str, str], mapped: dict[str, str]) -> dict[str, str]:
    result = dict(mapped)
    for key, value in base.items():
        if key not in result:
            result[key] = value
            continue
        expected = result[key]
        if expected == value:
            continue
        if expected == "*":
            if value.lower() in {"no", "false"}:
                raise OverpassInputError(
                    f"tags.{key}={value!r} 与 feature_types 映射冲突",
                    "conflicting_filters",
                )
            result[key] = value
            continue
        if value == "*":
            continue
        raise OverpassInputError(
            f"tags.{key}={value!r} 与 feature_types 要求 {expected!r} 冲突",
            "conflicting_filters",
        )
    return result


def _tags(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise OverpassInputError("tags 必须是对象", "invalid_tags")
    result: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not _TAG_KEY_RE.fullmatch(name):
            raise OverpassInputError(f"非法 OSM tag key: {name!r}", "invalid_tags")
        if isinstance(value, bool):
            result[name] = "yes" if value else "no"
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            result[name] = str(value).strip() or "*"
        else:
            raise OverpassInputError(f"tags.{name} 必须是标量", "invalid_tags")
    return result


def _render_tags(tags: dict[str, str]) -> str:
    chunks: list[str] = []
    for key, value in tags.items():
        escaped_key = _escape(key)
        if value == "*":
            chunks.append(f'["{escaped_key}"]')
        elif value.startswith("~"):
            pattern = _escape_regex(value[1:])
            chunks.append(f'["{escaped_key}"~"{pattern}",i]')
        else:
            chunks.append(f'["{escaped_key}"="{_escape(value)}"]')
    return "".join(chunks)


def _element_types(raw: Any) -> list[str]:
    values = _string_list(raw, "element_types") or ["node", "way", "relation"]
    normalized = [value.strip().lower() for value in values]
    if any(value not in _ALLOWED_ELEMENT_TYPES for value in normalized):
        raise OverpassInputError("element_types 只能是 node、way、relation", "invalid_element_types")
    return list(dict.fromkeys(normalized))


def _raw_query(raw: Any, ctx: RuntimeContext | None) -> str:
    allow = bool(ctx and ctx.extras.get("allow_raw_overpass_ql") is True)
    if not allow:
        raise OverpassInputError(
            "原始 overpass_ql 默认禁用；应优先传 area/tags 等结构化条件",
            "raw_query_disabled",
        )
    if not isinstance(raw, str) or not raw.strip():
        raise OverpassInputError("overpass_ql 必须是非空字符串", "invalid_query")
    ql = raw.strip()
    lowered = ql.lower()
    if len(ql) > _MAX_RAW_QUERY_CHARS or "{{" in ql or "[maxsize:" in lowered:
        raise OverpassInputError("overpass_ql 包含宏、maxsize 或长度超限", "unsafe_query")
    if "out" not in lowered:
        raise OverpassInputError("overpass_ql 必须显式包含 out", "invalid_query")
    if "[out:" in lowered and "[out:json]" not in lowered:
        raise OverpassInputError("overpass_ql 只允许 JSON 输出", "unsafe_query")
    prefix = "" if "[out:json]" in lowered else "[out:json];\n"
    timeout = "" if "[timeout:" in lowered else "[timeout:25];\n"
    return prefix + timeout + ql


def _client(ctx: RuntimeContext | None) -> OverpassClient:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("overpass_client")
    if injected is not None:
        if not isinstance(injected, OverpassClient):
            raise OverpassProviderError("overpass_client 必须实现 query(overpass_ql)", "invalid_provider")
        return injected
    _load_dotenv()
    if not _enabled():
        raise OverpassProviderError("ALLOW_REAL_TOOL_API=false，禁止调用真实 Overpass 服务")
    user_agent = os.environ.get("GEOAGENT_USER_AGENT", "").strip()
    if not user_agent:
        raise OverpassProviderError("请设置可识别的 GEOAGENT_USER_AGENT 后再调用公共 Overpass")
    return HttpOverpassClient(endpoint=_endpoint(), user_agent=user_agent, timeout_sec=_timeout())


def _endpoint() -> str:
    endpoint = os.environ.get("OVERPASS_ENDPOINT", _DEFAULT_ENDPOINT).strip()
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise OverpassProviderError("OVERPASS_ENDPOINT 必须是有效 HTTPS 地址")
    allowed = {"overpass-api.de", "overpass.kumi.systems"}
    custom = os.environ.get("ALLOW_CUSTOM_TOOL_ENDPOINTS", "false").strip().lower()
    if parsed.hostname.lower() not in allowed and custom not in {"1", "true", "yes", "on"}:
        raise OverpassProviderError("自定义 Overpass 主机需显式开启 ALLOW_CUSTOM_TOOL_ENDPOINTS")
    return endpoint


def _provider_counts(raw: dict[str, Any]) -> dict[str, int]:
    elements = raw.get("elements")
    if not isinstance(elements, list):
        raise OverpassProviderError("Overpass count 回执缺少 elements", "provider_error")
    for item in elements:
        if isinstance(item, dict) and item.get("type") == "count" and isinstance(item.get("tags"), dict):
            tags = item["tags"]
            return {
                "nodes": _int(tags.get("nodes")),
                "ways": _int(tags.get("ways")),
                "relations": _int(tags.get("relations")),
                "total": _int(tags.get("total")),
            }
    normalized = normalize_elements(raw, include_geometry=False)
    return {
        "nodes": sum(item["osm_type"] == "node" for item in normalized),
        "ways": sum(item["osm_type"] == "way" for item in normalized),
        "relations": sum(item["osm_type"] == "relation" for item in normalized),
        "total": len(normalized),
    }


def _count_local(elements: list[Any], group_by_raw: Any) -> dict[str, Any]:
    rows = [item for item in elements if isinstance(item, dict)]
    result: dict[str, Any] = {
        "count": len(rows),
        "counts": {
            "nodes": sum(item.get("osm_type") == "node" for item in rows),
            "ways": sum(item.get("osm_type") == "way" for item in rows),
            "relations": sum(item.get("osm_type") == "relation" for item in rows),
            "total": len(rows),
        },
    }
    group_fields = _string_list(group_by_raw, "group_by")
    if group_fields:
        groups: dict[str, int] = {}
        for item in rows:
            raw_tags = item.get("tags")
            tags: dict[str, Any] = raw_tags if isinstance(raw_tags, dict) else {}
            key = " | ".join(f"{field}={tags.get(field, '<missing>')}" for field in group_fields)
            groups[key] = groups.get(key, 0) + 1
        result["groups"] = groups
    return result


def _element_coordinates(item: dict[str, Any]) -> tuple[float | None, float | None]:
    raw = item if "lat" in item and "lon" in item else item.get("center")
    if isinstance(raw, dict):
        try:
            return float(raw["lat"]), float(raw["lon"])
        except (KeyError, TypeError, ValueError):
            pass
    bounds = item.get("bounds")
    if isinstance(bounds, dict):
        try:
            return (
                (float(bounds["minlat"]) + float(bounds["maxlat"])) / 2,
                (float(bounds["minlon"]) + float(bounds["maxlon"])) / 2,
            )
        except (KeyError, TypeError, ValueError):
            pass
    return None, None


def _bbox(raw: Any) -> tuple[float, float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise OverpassInputError("bbox 必须是 [west,south,east,north]", "invalid_bbox")
    try:
        west, south, east, north = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise OverpassInputError("bbox 必须由四个数字组成", "invalid_bbox") from exc
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise OverpassInputError("bbox 超出 WGS84 范围或边界顺序错误", "invalid_bbox")
    return west, south, east, north


def _coordinates(raw: Any) -> tuple[float, float]:
    lat: Any = None
    lon: Any = None
    try:
        if isinstance(raw, dict):
            lat = raw.get("lat", raw.get("latitude"))
            lon = raw.get("lon", raw.get("longitude"))
        elif isinstance(raw, (list, tuple)) and len(raw) == 2:
            lat, lon = raw
        else:
            raise ValueError
        latitude, longitude = float(lat), float(lon)
    except (TypeError, ValueError) as exc:
        raise OverpassInputError("center 必须是 [lat,lon] 或坐标对象", "needs_acquisition") from exc
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise OverpassInputError("center 超出 WGS84 有效范围", "invalid_coordinates")
    return latitude, longitude


def _positive_number(raw: Any, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise OverpassInputError(f"{name} 必须是正数", "missing_input") from exc
    if value <= 0:
        raise OverpassInputError(f"{name} 必须是正数", "invalid_input")
    return value


def _limit(raw: Any) -> int:
    if raw is None:
        return _DEFAULT_LIMIT
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise OverpassInputError("limit 必须是整数", "invalid_limit") from exc
    if value < 1:
        raise OverpassInputError("limit 必须 >= 1", "invalid_limit")
    return min(value, _MAX_LIMIT)


def _string_list(raw: Any, name: str) -> list[str]:
    if raw is None:
        return []
    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, list):
        raise OverpassInputError(f"{name} 必须是字符串或字符串数组", f"invalid_{name}")
    result = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    if len(result) != len(values):
        raise OverpassInputError(f"{name} 只能包含非空字符串", f"invalid_{name}")
    return result


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _escape_regex(value: str) -> str:
    return re.escape(value)[:200].replace('"', '\\"')


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _int(raw: Any) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _enabled() -> bool:
    return os.environ.get("ALLOW_REAL_TOOL_API", "false").strip().lower() in {"1", "true", "yes", "on"}


def _timeout() -> float:
    try:
        value = float(os.environ.get("OVERPASS_TIMEOUT_SEC", "45"))
    except ValueError:
        value = 45.0
    return min(120.0, value) if value > 0 else 45.0


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _fail(message: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=message, error_code=error_code)


__all__ = [
    "OverpassClient",
    "build_query",
    "execute_count",
    "execute_query",
    "normalize_elements",
    "run_overpass",
]
