"""osm_query 的邻近、沿线与相交查询。near 在块内逐点筛选，不做全省组合。"""

from __future__ import annotations

import math
import re
from typing import Any

from tool.contract import Observation, RuntimeContext, declared_inputs
from tool.osm_query._place import (
    _LANDUSE_NOTE,
    _PARTIAL_NOTE,
    parse_layout,
    search_places,
)
from tool.osm_query._query import (
    EngineUnavailableError,
    OsmQueryInputError,
    OsmQueryRequest,
    QueryScope,
    _KIND_AREA_NAME,
    _env_timeout_int,
    _fail,
    _map_feature_type,
    _normalize_elements,
    _parse_limit,
    _parse_radius_m,
    _parse_tags,
    _provider_name,
    _resolve_admin_scope,
    _resolve_provider,
    _resolve_scope,
    _result_envelope,
    _scope_prefix_and_filter,
    _strip_forbidden,
    _tag_filters,
)

_OP_NEAR = "near"
_OP_CROSSINGS = "crossings"
_OP_INTERSECT = "intersect"
_DEFAULT_WITHIN_M = 200
_DEFAULT_BUFFER_M = 40
_INTERSECT_AROUND_M = 30
_DEFAULT_CLUSTER_M = 100
_SCOPE_FIELDS = ("area", "bbox", "center", "radius_m", "limit")
_NEAR_INPUT_FIELDS = _SCOPE_FIELDS + (
    "subject",
    "reference",
    "within_m",
    "also",
    "within_also_m",
    "subject_tags",
    "reference_tags",
    "also_tags",
    "layout",
)
_CROSSINGS_INPUT_FIELDS = _SCOPE_FIELDS + ("line", "line_tags", "kinds", "buffer_m")
_INTERSECT_INPUT_FIELDS = _SCOPE_FIELDS + (
    "subject",
    "reference",
    "subject_tags",
    "reference_tags",
    "cluster_m",
)
_CROSSING_KINDS = ("bridge", "dam", "ferry", "level_crossing")
_EMPTY_NOTE = "查不到只表示 OSM 没有返回，不能当成该地没有"


def execute_near(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """在范围内逐点筛选与参照地物相邻的地点。范围过大时内部分块。"""

    del purpose
    try:
        parsed_inputs = declared_inputs(inputs, *_NEAR_INPUT_FIELDS)
        subject_name = _text(parsed_inputs.get("subject"))
        reference_name = _text(parsed_inputs.get("reference"))
        also_name = _text(parsed_inputs.get("also"))
        subject = _filters(
            parsed_inputs.get("subject"),
            _parse_tags(parsed_inputs.get("subject_tags")),
            field="subject",
        )
        reference = _filters(
            parsed_inputs.get("reference"),
            _parse_tags(parsed_inputs.get("reference_tags")),
            field="reference",
        )
        also_tags = _parse_tags(parsed_inputs.get("also_tags"))
        also = _optional_filters(parsed_inputs.get("also"), also_tags, field="also")
        within_m = _positive_meters(parsed_inputs.get("within_m"), default=_DEFAULT_WITHIN_M, field="within_m")
        within_also_m = _positive_meters(
            parsed_inputs.get("within_also_m"),
            default=_DEFAULT_WITHIN_M,
            field="within_also_m",
        )
        layout = parse_layout(
            parsed_inputs.get("layout"),
            subject=subject_name,
            reference=reference_name,
            also=also_name,
        )
        provider, scope, timeout_sec, limit = _bound_scope(parsed_inputs, ctx)
        found = search_places(
            provider=provider,
            scope=scope,
            timeout_sec=timeout_sec,
            limit=limit,
            subject_name=subject_name,
            reference_name=reference_name,
            also_name=also_name,
            subject_filter=subject,
            reference_filter=reference,
            also_filter=also,
            within_m=within_m,
            within_also_m=within_also_m,
            layout=layout,
            ctx=ctx,
        )
        applied = {
            "subject": subject_name,
            "reference": reference_name,
            "within_m": within_m,
            "tiles": found.tile_queries,
        }
        if also:
            applied["also"] = also_name
            applied["within_also_m"] = within_also_m
        if layout.bearings or layout.directions:
            applied["layout"] = True
        if found.landuse_missing:
            applied["landuse_missing"] = found.landuse_missing
        parsed = {
            "ql": found.last_ql,
            "timeout_sec": timeout_sec,
            "limit": limit,
            "return_geometry": True,
            "scope": scope,
            "tags": None,
            "feature_types": (),
            "element_types": (),
            "spatial_relation": None,
            "unused": {},
        }
        result = _result_envelope(
            operation=_OP_NEAR,
            provider_name=_provider_name(provider),
            parsed=parsed,
            payload={},
            extra={
                "elements": found.places,
                "places": found.places,
                "count": len(found.places),
                "partial": found.partial,
            },
        )
        if found.unscanned_bbox is not None:
            result["unscanned_bbox"] = found.unscanned_bbox
        result["applied"].update(applied)
        notes = [_EMPTY_NOTE]
        if found.landuse_missing:
            notes.append(_LANDUSE_NOTE)
        if found.partial:
            notes.append(_PARTIAL_NOTE)
        result["assumptions"] = list(result.get("assumptions") or []) + notes
        return Observation(ok=True, result=_strip_forbidden(result))
    except OsmQueryInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)


def execute_crossings(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """返回落在一条线附近的桥、坝、渡口或道口。"""

    del purpose
    try:
        parsed_inputs = declared_inputs(inputs, *_CROSSINGS_INPUT_FIELDS)
        line = _filters(
            parsed_inputs.get("line"),
            _parse_tags(parsed_inputs.get("line_tags")),
            field="line",
        )
        kinds = _parse_kinds(parsed_inputs.get("kinds"))
        buffer_m = _positive_meters(
            parsed_inputs.get("buffer_m"),
            default=_DEFAULT_BUFFER_M,
            field="buffer_m",
        )
        provider, scope, timeout_sec, limit = _bound_scope(parsed_inputs, ctx)
        prefix, spatial = _scope_prefix_and_filter(scope)
        parts = [_crossing_selector(kind, spatial, buffer_m) for kind in kinds]
        inner = "\n  ".join(f"{item};" for item in parts)
        body = f"way{line}{spatial}->.line;\n(\n  {inner}\n);\n"
        ql = _ql(prefix, body, timeout_sec=timeout_sec, limit=limit)
        return _run(
            operation=_OP_CROSSINGS,
            ql=ql,
            provider=provider,
            scope=scope,
            timeout_sec=timeout_sec,
            limit=limit,
            applied_extra={"line": _text(parsed_inputs.get("line")), "kinds": list(kinds), "buffer_m": buffer_m},
        )
    except OsmQueryInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)


def execute_intersect(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """返回两类线的交叉点。"""

    del purpose
    try:
        parsed_inputs = declared_inputs(inputs, *_INTERSECT_INPUT_FIELDS)
        subject = _filters(
            parsed_inputs.get("subject"),
            _parse_tags(parsed_inputs.get("subject_tags")),
            field="subject",
        )
        reference = _filters(
            parsed_inputs.get("reference"),
            _parse_tags(parsed_inputs.get("reference_tags")),
            field="reference",
        )
        cluster_m = _positive_meters(
            parsed_inputs.get("cluster_m"),
            default=_DEFAULT_CLUSTER_M,
            field="cluster_m",
        )
        provider, scope, timeout_sec, limit = _bound_scope(parsed_inputs, ctx)
        prefix, spatial = _scope_prefix_and_filter(scope)
        body = (
            f"way{subject}{spatial}->.a;\n"
            f"way{reference}(around.a:{_INTERSECT_AROUND_M}){spatial}->.b;\n"
            ".a out geom;\n.b out geom;"
        )
        ql = _statement(prefix, body, timeout_sec=timeout_sec)
        payload = provider.query(
            OsmQueryRequest(ql=ql, timeout_sec=float(timeout_sec), operation=_OP_INTERSECT)
        )
        candidates = _intersections(payload, subject_filter=subject, reference_filter=reference, cluster_m=cluster_m)
        if limit is not None:
            candidates = candidates[:limit]
        parsed = {
            "ql": ql,
            "timeout_sec": timeout_sec,
            "limit": limit,
            "return_geometry": True,
            "scope": scope,
            "tags": None,
            "feature_types": (),
            "element_types": (),
            "spatial_relation": None,
            "unused": {},
        }
        result = _result_envelope(
            operation=_OP_INTERSECT,
            provider_name=_provider_name(provider),
            parsed=parsed,
            payload=payload if isinstance(payload, dict) else {},
            extra={"elements": candidates, "candidates": candidates, "count": len(candidates)},
        )
        result["applied"]["subject"] = _text(parsed_inputs.get("subject"))
        result["applied"]["reference"] = _text(parsed_inputs.get("reference"))
        result["applied"]["cluster_m"] = cluster_m
        result["assumptions"] = list(result.get("assumptions") or []) + [_EMPTY_NOTE]
        return Observation(ok=True, result=_strip_forbidden(result))
    except OsmQueryInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)


def _bound_scope(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> tuple[Any, QueryScope, int, int]:
    timeout_sec = _env_timeout_int()
    scope = _resolve_scope(
        inputs,
        ctx,
        spatial_relation="within",
        radius_m=_parse_radius_m(inputs.get("radius_m")),
    )
    provider = _resolve_provider(ctx)
    if scope.kind == _KIND_AREA_NAME and scope.name:
        scope = _resolve_admin_scope(scope.name, provider, ctx, timeout_sec=timeout_sec)
    return provider, scope, timeout_sec, _parse_limit(inputs.get("limit"))


def _run(
    *,
    operation: str,
    ql: str,
    provider: Any,
    scope: QueryScope,
    timeout_sec: int,
    limit: int,
    applied_extra: dict[str, Any],
) -> Observation:
    payload = provider.query(OsmQueryRequest(ql=ql, timeout_sec=float(timeout_sec), operation=operation))
    if not isinstance(payload, dict):
        raise EngineUnavailableError("Overpass 回执不是 JSON 对象")
    elements = _normalize_elements(payload, return_geometry=True, limit=limit)
    parsed = {
        "ql": ql,
        "timeout_sec": timeout_sec,
        "limit": limit,
        "return_geometry": True,
        "scope": scope,
        "tags": None,
        "feature_types": (),
        "element_types": (),
        "spatial_relation": None,
        "unused": {},
    }
    result = _result_envelope(
        operation=operation,
        provider_name=_provider_name(provider),
        parsed=parsed,
        payload=payload,
        extra={"elements": elements, "count": len(elements)},
    )
    result["applied"].update(applied_extra)
    result["assumptions"] = list(result.get("assumptions") or []) + [_EMPTY_NOTE]
    return Observation(ok=True, result=_strip_forbidden(result))


def _filters(feature: Any, tags: dict[str, str], *, field: str) -> str:
    text = _text(feature)
    parts: list[str] = []
    if text:
        parts.append(_tag_filters(_map_feature_type(text)))
    if tags:
        parts.append(_tag_filters(tags))
    if not parts:
        raise OsmQueryInputError(f"缺少 {field}", "missing_input")
    return "".join(parts)


def _optional_filters(feature: Any, tags: dict[str, str], *, field: str) -> str:
    if _text(feature) or tags:
        return _filters(feature, tags, field=field)
    return ""


def _positive_meters(raw: Any, *, default: int, field: str) -> int:
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise OsmQueryInputError(f"{field} 必须是正数", "invalid_input") from None
    if value <= 0:
        raise OsmQueryInputError(f"{field} 必须是正数", "invalid_input")
    return int(round(value))


def _parse_kinds(raw: Any) -> tuple[str, ...]:
    if raw is None or raw == "" or raw == []:
        return _CROSSING_KINDS
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item).strip()]
    else:
        raise OsmQueryInputError("kinds 必须是字符串或数组", "invalid_input")
    unknown = [item for item in items if item not in _CROSSING_KINDS]
    if unknown:
        raise OsmQueryInputError(f"kinds 不支持 {unknown[0]!r}", "invalid_input")
    return tuple(dict.fromkeys(items))


def _crossing_selector(kind: str, spatial: str, buffer_m: int) -> str:
    around = f"(around.line:{buffer_m}){spatial}"
    if kind == "bridge":
        return f'way["bridge"]["bridge"!="no"]{around}'
    if kind == "dam":
        return f'nwr["waterway"~"^(dam|weir)$"]{around}'
    if kind == "ferry":
        return f'way["route"="ferry"]{around}'
    return f'node["railway"~"^(level_crossing|crossing)$"]{around}'


def _ql(prefix: str, body: str, *, timeout_sec: int, limit: int) -> str:
    return _statement(prefix, f"{body}.hit out geom {limit};" if body.rstrip().endswith("->.hit;") or ".hit;" in body else f"{body}out geom {limit};", timeout_sec=timeout_sec)


def _statement(prefix: str, body: str, *, timeout_sec: int) -> str:
    parts = [f"[out:json][timeout:{timeout_sec}];"]
    if prefix:
        parts.append(prefix.rstrip())
    parts.append(body.rstrip())
    return "\n".join(parts)


def _text(raw: Any) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def _intersections(
    payload: dict[str, Any],
    *,
    subject_filter: str,
    reference_filter: str,
    cluster_m: int,
) -> list[dict[str, Any]]:
    elements = payload.get("elements") if isinstance(payload, dict) else None
    if not isinstance(elements, list):
        return []
    ways = [item for item in elements if isinstance(item, dict) and item.get("geometry")]
    primary = [item for item in ways if _matches(item, subject_filter)]
    secondary = [item for item in ways if item not in primary and _matches(item, reference_filter)]
    if not secondary:
        secondary = [item for item in ways if item not in primary]
    hits: list[tuple[float, float]] = []
    for left in primary:
        for right in secondary:
            hits.extend(_line_crossings(left.get("geometry"), right.get("geometry")))
    clustered = _cluster(hits, cluster_m)
    rows: list[dict[str, Any]] = []
    for index, (lat, lon) in enumerate(clustered, start=1):
        rows.append(
            {
                "result_id": f"intersect_{index}",
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
            }
        )
    return rows


def _matches(element: dict[str, Any], filters: str) -> bool:
    tags = element.get("tags") if isinstance(element.get("tags"), dict) else {}
    for key, value in re.findall(r'\["([^"]+)"="([^"]+)"\]', filters):
        if str(tags.get(key)) != value:
            return False
    for key in re.findall(r'\["([^"=\]]+)"\]', filters):
        if key not in tags:
            return False
    return True


def _line_crossings(left: Any, right: Any) -> list[tuple[float, float]]:
    left_points = _latlon(left)
    right_points = _latlon(right)
    if len(left_points) < 2 or len(right_points) < 2:
        return []
    lat0 = left_points[0][0]
    kx = 111320 * math.cos(math.radians(lat0))
    ky = 110540

    def project(point: tuple[float, float]) -> tuple[float, float]:
        return point[1] * kx, point[0] * ky

    found: list[tuple[float, float]] = []
    for start, end in zip(left_points, left_points[1:]):
        p1, p2 = project(start), project(end)
        for other_start, other_end in zip(right_points, right_points[1:]):
            hit = _segment_cross(p1, p2, project(other_start), project(other_end))
            if hit is not None:
                found.append((hit[1] / ky, hit[0] / kx))
    return found


def _segment_cross(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float] | None:
    determinant = (p2[0] - p1[0]) * (p4[1] - p3[1]) - (p2[1] - p1[1]) * (p4[0] - p3[0])
    if abs(determinant) < 1e-9:
        return None
    along = ((p3[0] - p1[0]) * (p4[1] - p3[1]) - (p3[1] - p1[1]) * (p4[0] - p3[0])) / determinant
    across = ((p3[0] - p1[0]) * (p2[1] - p1[1]) - (p3[1] - p1[1]) * (p2[0] - p1[0])) / determinant
    if 0 <= along <= 1 and 0 <= across <= 1:
        return (p1[0] + along * (p2[0] - p1[0]), p1[1] + along * (p2[1] - p1[1]))
    return None


def _latlon(raw: Any) -> list[tuple[float, float]]:
    if not isinstance(raw, list):
        return []
    points: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        lat = item.get("lat")
        lon = item.get("lon", item.get("lng"))
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            points.append((float(lat), float(lon)))
    return points


def _cluster(points: list[tuple[float, float]], radius_m: int) -> list[tuple[float, float]]:
    groups: list[dict[str, Any]] = []
    for lat, lon in points:
        for group in groups:
            center = group["center"]
            if _distance_m(center, (lat, lon)) <= radius_m:
                count = group["count"] + 1
                group["center"] = (
                    (center[0] * (count - 1) + lat) / count,
                    (center[1] * (count - 1) + lon) / count,
                )
                group["count"] = count
                break
        else:
            groups.append({"center": (lat, lon), "count": 1})
    return [group["center"] for group in groups]


def _distance_m(origin: tuple[float, float], target: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*origin, *target))
    arc = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * math.asin(math.sqrt(arc))
