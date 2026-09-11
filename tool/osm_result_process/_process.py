"""osm_result_process 共享执行器：本地筛选与导出已有 OSM 结果，不回源 Overpass。"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from io import StringIO
from typing import Any
from xml.etree import ElementTree as ET

from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box as shapely_box
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from tool.contract import Observation, RuntimeContext, declared_inputs

_OP_FILTER = "filter"
_OP_EXPORT = "export"
_PROVIDER_LOCAL = "local"
_CRS_WGS84 = "wgs84"
_REGISTRY_KEY = "osm_results"
_SEQ_KEY = "_osm_result_seq"
_FORMATS = frozenset({"geojson", "json", "csv", "kml", "osm_xml"})
_ELEMENT_TYPES = frozenset({"node", "way", "relation"})
_RESERVED_FILTER_KEYS = frozenset({"tags", "osm_type", "element_types", "dedupe"})
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_METERS_PER_DEG_LAT = 111_320.0
_SPATIAL_RELATIONS: dict[str, str] = {
    "within": "within",
    "inside": "within",
    "in": "within",
    "intersects": "intersects",
    "intersect": "intersects",
    "overlap": "intersects",
    "contains": "contains",
    "near": "near",
    "around": "near",
    "nearby": "near",
    "crosses": "crosses",
    "cross": "crosses",
    "附近": "near",
    "周边": "near",
    "相交": "intersects",
    "范围内": "within",
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
_LOCAL_ASSUMPTION = "本步只做本地矢量处理，未回源 OSM / Overpass"
_NO_INVENT_ASSUMPTION = "未根据模型印象补齐缺失属性或几何；缺字段则无法执行该条件"
_NO_SESSION_ASSUMPTION = "本步只返回处理后的 OSM 结果，未打开街景或地图会话"
_CRS_ASSUMPTION = "坐标按 WGS84 解释，未做坐标系转换"


class ProcessInputError(Exception):
    """source_result / filters / spatial_filter / format 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class ParsedSource:
    """已解析的真实 OSM 要素集合。"""

    elements: list[dict[str, Any]]
    data_timestamp: str | None


@dataclass(frozen=True)
class ParsedFilters:
    """已解析的属性筛选条件。"""

    tags: dict[str, str]
    osm_types: tuple[str, ...]
    dedupe: bool
    applied: dict[str, Any]
    has_condition: bool


@dataclass(frozen=True)
class ParsedSpatial:
    """已解析的空间筛选条件。"""

    relation: str
    geometry: BaseGeometry
    distance_m: float | None
    applied: dict[str, Any]


def execute_filter(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按属性或几何条件筛选已有 OSM 结果，不访问外部 API。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "source_result", "filters", "spatial_filter")
        source = _resolve_source_result(inputs.get("source_result"), ctx)
        if inputs.get("filters") in (None, ""):
            raise ProcessInputError("缺少必填输入 filters", "missing_input")
        spatial = _parse_spatial_filter(inputs.get("spatial_filter"))
        parsed_filters = _parse_filters(inputs.get("filters"), spatial_present=spatial is not None)
        if not parsed_filters.has_condition and spatial is None:
            raise ProcessInputError("缺少必填输入 filters", "missing_input")
        kept = _apply_filters(source.elements, parsed_filters, spatial)
        return _ok_process(
            operation=_OP_FILTER,
            source=source,
            elements=kept,
            original_count=len(source.elements),
            applied=_applied_envelope(
                filters=parsed_filters.applied,
                spatial=None if spatial is None else spatial.applied,
            ),
            ctx=ctx,
        )
    except ProcessInputError as exc:
        return _fail(str(exc), exc.error_code)


def execute_export(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """把已有 OSM 结果导出为目录允许的格式，不访问外部 API。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "source_result", "format", "include_geometry")
        source = _resolve_source_result(inputs.get("source_result"), ctx)
        fmt = _parse_format(inputs.get("format"))
        include_geometry = _parse_optional_bool(inputs.get("include_geometry"), default=True)
        payload, geojson = _export_payload(source.elements, fmt, include_geometry=include_geometry)
        extra: dict[str, Any] = {"payload": payload, "format": fmt, "include_geometry": include_geometry}
        if geojson is not None:
            extra["geojson"] = geojson
        return _ok_process(
            operation=_OP_EXPORT,
            source=source,
            elements=source.elements,
            original_count=len(source.elements),
            applied=_applied_envelope(fmt=fmt, include_geometry=include_geometry),
            ctx=ctx,
            extra=extra,
        )
    except ProcessInputError as exc:
        return _fail(str(exc), exc.error_code)


def _resolve_source_result(raw: Any, ctx: RuntimeContext | None) -> ParsedSource:
    """只接受真实结果对象、上下文引用或已登记 ID，拒绝自然语言描述。"""

    if raw is None or raw == "":
        raise ProcessInputError("缺少必填输入 source_result", "missing_input")
    if raw == "$previous_tool_result":
        previous = ctx.previous_tool_result if ctx is not None else None
        if not isinstance(previous, dict) and not isinstance(previous, list):
            raise ProcessInputError("无法解析 $previous_tool_result", "missing_input")
        return _source_from_unwrapped(_unwrap_source(previous))
    if isinstance(raw, (dict, list)):
        return _source_from_unwrapped(_unwrap_source(raw))
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ProcessInputError("缺少必填输入 source_result", "missing_input")
        if not _ID_RE.fullmatch(text):
            raise ProcessInputError(
                "source_result 必须是真实结果对象、结果 ID 或 $previous_tool_result，不能用自然语言描述替代",
                "invalid_input",
            )
        return _lookup_registry(text, ctx)
    raise ProcessInputError("source_result 必须是结果对象或引用", "invalid_input")


def _lookup_registry(result_id: str, ctx: RuntimeContext | None) -> ParsedSource:
    if ctx is None:
        raise ProcessInputError(f"source_result 引用的结果不存在: {result_id}", "missing_input")
    registry = ctx.extras.get(_REGISTRY_KEY)
    if not isinstance(registry, dict) or result_id not in registry:
        raise ProcessInputError(f"source_result 引用的结果不存在: {result_id}", "missing_input")
    stored = registry[result_id]
    if not isinstance(stored, (dict, list)):
        raise ProcessInputError("登记的 OSM 结果无法解析", "invalid_input")
    return _source_from_unwrapped(_unwrap_source(stored))


def _unwrap_source(raw: Any) -> dict[str, Any]:
    if isinstance(raw, list):
        return {"elements": [_coerce_element(item) for item in raw if isinstance(item, dict)]}
    if not isinstance(raw, dict):
        raise ProcessInputError("source_result 无法解析为 OSM 结果", "invalid_input")
    if str(raw.get("type") or "").lower() == "featurecollection":
        return {"elements": _elements_from_features(raw.get("features")), "data_timestamp": _timestamp_of(raw)}
    if str(raw.get("type") or "").lower() == "feature":
        element = _element_from_feature(raw)
        return {"elements": [] if element is None else [element], "data_timestamp": _timestamp_of(raw)}
    if isinstance(raw.get("elements"), list):
        return {
            "elements": [_coerce_element(item) for item in raw["elements"] if isinstance(item, dict)],
            "data_timestamp": _timestamp_of(raw),
        }
    nested = raw.get("result")
    if isinstance(nested, dict):
        return _unwrap_source(nested)
    raise ProcessInputError("source_result 缺少 elements", "invalid_input")


def _source_from_unwrapped(payload: dict[str, Any]) -> ParsedSource:
    raw_elements = payload.get("elements")
    if not isinstance(raw_elements, list):
        raise ProcessInputError("source_result 缺少 elements", "invalid_input")
    elements = [item for item in raw_elements if isinstance(item, dict)]
    stamp = payload.get("data_timestamp")
    timestamp = stamp.strip() if isinstance(stamp, str) and stamp.strip() else None
    return ParsedSource(elements=elements, data_timestamp=timestamp)


def _elements_from_features(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    elements: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        element = _element_from_feature(item)
        if element is not None:
            elements.append(element)
    return elements


def _element_from_feature(feature: dict[str, Any]) -> dict[str, Any] | None:
    props = feature.get("properties")
    properties = props if isinstance(props, dict) else {}
    geometry = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else None
    osm_type = _optional_str(properties.get("osm_type") or properties.get("type"))
    osm_id = _as_int(properties.get("osm_id", feature.get("id")))
    tags_raw = properties.get("tags")
    if isinstance(tags_raw, dict):
        tags = _clean_tags(tags_raw)
    else:
        tags = _clean_tags(
            {
                key: value
                for key, value in properties.items()
                if key not in {"osm_type", "osm_id", "result_id", "type"}
            }
        )
    row: dict[str, Any] = {}
    result_id = _optional_str(properties.get("result_id") or feature.get("id"))
    if result_id is not None:
        row["result_id"] = result_id
    if osm_type in _ELEMENT_TYPES:
        row["osm_type"] = osm_type
    if osm_id is not None:
        row["osm_id"] = osm_id
    if tags:
        row["tags"] = tags
    if geometry is not None:
        row["geometry"] = geometry
    return row or None


def _coerce_element(item: dict[str, Any]) -> dict[str, Any]:
    if str(item.get("type") or "").lower() == "feature":
        converted = _element_from_feature(item)
        return converted if converted is not None else {}
    row: dict[str, Any] = {}
    result_id = _optional_str(item.get("result_id"))
    if result_id is not None:
        row["result_id"] = result_id
    osm_type = _optional_str(item.get("osm_type") or item.get("type"))
    if osm_type in _ELEMENT_TYPES:
        row["osm_type"] = osm_type
    osm_id = _as_int(item.get("osm_id", item.get("id")))
    if osm_id is not None:
        row["osm_id"] = osm_id
    tags = item.get("tags")
    if isinstance(tags, dict):
        cleaned = _clean_tags(tags)
        if cleaned:
            row["tags"] = cleaned
    geometry = item.get("geometry")
    if isinstance(geometry, dict):
        row["geometry"] = geometry
    return row


def _parse_filters(raw: Any, *, spatial_present: bool = False) -> ParsedFilters:
    if raw is None or raw == "":
        if spatial_present:
            return ParsedFilters(
                tags={},
                osm_types=(),
                dedupe=True,
                applied={"dedupe": True},
                has_condition=False,
            )
        raise ProcessInputError("缺少必填输入 filters", "missing_input")
    if not isinstance(raw, dict):
        raise ProcessInputError("filters 必须是对象", "invalid_input")
    tags: dict[str, str] = {}
    nested = raw.get("tags")
    if nested is not None:
        if not isinstance(nested, dict):
            raise ProcessInputError("filters.tags 必须是对象", "invalid_input")
        tags.update(_parse_tag_map(nested))
    for key, value in raw.items():
        if key in _RESERVED_FILTER_KEYS:
            continue
        parsed = _stringify_tag(value)
        if parsed is None:
            raise ProcessInputError(f"无法解析筛选条件 {key}", "invalid_input")
        tags[str(key)] = parsed
    osm_types = _parse_osm_types(raw.get("osm_type"), raw.get("element_types"))
    dedupe = _parse_optional_bool(raw.get("dedupe"), default=True)
    applied: dict[str, Any] = {"dedupe": dedupe}
    if tags:
        applied["tags"] = dict(tags)
    if osm_types:
        applied["osm_type"] = list(osm_types)
    has_condition = bool(tags) or bool(osm_types)
    return ParsedFilters(
        tags=tags,
        osm_types=osm_types,
        dedupe=dedupe,
        applied=applied,
        has_condition=has_condition,
    )


def _parse_tag_map(raw: dict[str, Any]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for key, value in raw.items():
        parsed = _stringify_tag(value)
        if parsed is None:
            raise ProcessInputError(f"无法解析 tag 条件 {key}", "invalid_input")
        tags[str(key)] = parsed
    return tags


def _parse_osm_types(osm_type: Any, element_types: Any) -> tuple[str, ...]:
    values: list[str] = []
    if osm_type is not None and osm_type != "":
        values.extend(_as_string_list(osm_type, field="osm_type"))
    if element_types is not None and element_types != "":
        values.extend(_as_string_list(element_types, field="element_types"))
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        kind = item.strip().lower()
        if kind not in _ELEMENT_TYPES:
            raise ProcessInputError(f"不支持的 osm_type: {item}", "invalid_input")
        if kind not in seen:
            seen.add(kind)
            normalized.append(kind)
    return tuple(normalized)


def _parse_spatial_filter(raw: Any) -> ParsedSpatial | None:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, dict):
        raise ProcessInputError("spatial_filter 必须是对象", "invalid_input")
    relation_raw = raw.get("relation")
    if relation_raw is None or relation_raw == "":
        raise ProcessInputError("spatial_filter 缺少 relation", "missing_input")
    relation_key = str(relation_raw).strip().lower() if not isinstance(relation_raw, str) else relation_raw.strip()
    relation = _SPATIAL_RELATIONS.get(relation_key) or _SPATIAL_RELATIONS.get(relation_key.lower())
    if relation is None:
        raise ProcessInputError(f"不支持的空间关系: {relation_raw}", "invalid_input")
    if raw.get("geometry") is None or raw.get("geometry") == "":
        raise ProcessInputError("spatial_filter 缺少 geometry", "missing_input")
    geometry = _parse_reference_geometry(raw.get("geometry"))
    distance_m: float | None = None
    if relation == "near":
        distance_m = _parse_distance_m(raw.get("distance_m"))
        if distance_m is None:
            raise ProcessInputError("near 需要 distance_m", "missing_input")
    applied: dict[str, Any] = {
        "relation": relation,
        "geometry": mapping(geometry),
    }
    if distance_m is not None:
        applied["distance_m"] = distance_m
    return ParsedSpatial(relation=relation, geometry=geometry, distance_m=distance_m, applied=applied)


def _parse_reference_geometry(raw: Any) -> BaseGeometry:
    """参照几何必须是坐标/GeoJSON/bbox/要素几何，不能用地名。"""

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ProcessInputError("spatial_filter.geometry 不能为空", "missing_input")
        if text[0] not in "[{":
            raise ProcessInputError("参照几何必须是坐标，不能用地名", "invalid_input")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProcessInputError("参照几何必须是坐标，不能用地名", "invalid_input") from exc
        return _parse_reference_geometry(parsed)
    if isinstance(raw, (list, tuple)):
        return _geometry_from_sequence(raw)
    if isinstance(raw, dict):
        return _geometry_from_mapping(raw)
    raise ProcessInputError("参照几何必须是坐标，不能用地名", "invalid_input")


def _geometry_from_sequence(raw: list[Any] | tuple[Any, ...]) -> BaseGeometry:
    if len(raw) == 4 and all(_as_float(item) is not None for item in raw):
        west, south, east, north = (_as_float(item) for item in raw)
        assert west is not None and south is not None and east is not None and north is not None
        return shapely_box(west, south, east, north)
    if len(raw) == 2 and all(_as_float(item) is not None for item in raw):
        lon, lat = _as_float(raw[0]), _as_float(raw[1])
        assert lon is not None and lat is not None
        return shape({"type": "Point", "coordinates": [lon, lat]})
    geoms: list[BaseGeometry] = []
    for item in raw:
        if isinstance(item, dict):
            geoms.append(_geometry_from_mapping(item))
        elif isinstance(item, (list, tuple)):
            geoms.append(_geometry_from_sequence(item))
        else:
            raise ProcessInputError("参照几何必须是坐标，不能用地名", "invalid_input")
    if not geoms:
        raise ProcessInputError("参照几何缺少可解析坐标", "invalid_input")
    return unary_union(geoms)


def _geometry_from_mapping(raw: dict[str, Any]) -> BaseGeometry:
    kind = str(raw.get("type") or "").lower()
    if kind == "featurecollection":
        geoms = [_geometry_from_mapping(item) for item in raw.get("features") or [] if isinstance(item, dict)]
        if not geoms:
            raise ProcessInputError("参照几何缺少可解析坐标", "invalid_input")
        return unary_union(geoms)
    if kind == "feature":
        geom = raw.get("geometry")
        if not isinstance(geom, dict):
            raise ProcessInputError("参照几何缺少可解析坐标", "invalid_input")
        return _shape_geojson(geom)
    if "coordinates" in raw and kind:
        return _shape_geojson(raw)
    if all(key in raw for key in ("west", "south", "east", "north")):
        west = _as_float(raw["west"])
        south = _as_float(raw["south"])
        east = _as_float(raw["east"])
        north = _as_float(raw["north"])
        if None in (west, south, east, north):
            raise ProcessInputError("bbox 坐标无效", "invalid_input")
        return shapely_box(west, south, east, north)
    lat = _as_float(raw.get("lat"))
    lon = _as_float(raw.get("lon", raw.get("lng")))
    if lat is not None and lon is not None:
        return shape({"type": "Point", "coordinates": [lon, lat]})
    if isinstance(raw.get("geometry"), dict):
        return _geometry_from_mapping(raw["geometry"])
    if isinstance(raw.get("elements"), list):
        geoms: list[BaseGeometry] = []
        for item in raw["elements"]:
            if not isinstance(item, dict):
                continue
            geom = item.get("geometry")
            if isinstance(geom, dict):
                geoms.append(_shape_geojson(geom))
        if not geoms:
            raise ProcessInputError("参照几何缺少可解析坐标", "invalid_input")
        return unary_union(geoms)
    raise ProcessInputError("参照几何必须是坐标，不能用地名", "invalid_input")


def _shape_geojson(raw: dict[str, Any]) -> BaseGeometry:
    try:
        geom = shape(raw)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ProcessInputError("无法解析参照几何", "invalid_input") from exc
    if geom.is_empty:
        raise ProcessInputError("参照几何为空", "invalid_input")
    return geom


def _parse_distance_m(raw: Any) -> float | None:
    value = _as_float(raw)
    if value is None:
        return None
    if value <= 0:
        raise ProcessInputError("distance_m 必须为正数", "invalid_input")
    return value


def _parse_format(raw: Any) -> str:
    if raw is None or raw == "":
        raise ProcessInputError("缺少必填输入 format", "missing_input")
    text = str(raw).strip().lower()
    if text not in _FORMATS:
        raise ProcessInputError(
            f"不支持的导出格式: {raw}；允许 geojson/json/csv/kml/osm_xml",
            "invalid_input",
        )
    return text


def _apply_filters(
    elements: list[dict[str, Any]],
    parsed: ParsedFilters,
    spatial: ParsedSpatial | None,
) -> list[dict[str, Any]]:
    if parsed.tags:
        for key in parsed.tags:
            if not _any_has_tag(elements, key):
                raise ProcessInputError(
                    f"源结果缺少属性 {key}，无法执行该筛选条件，不能补齐缺失标签",
                    "unsupported_filter",
                )
    if parsed.osm_types and not any("osm_type" in item for item in elements):
        raise ProcessInputError("源结果缺少 osm_type，无法执行该筛选条件", "unsupported_filter")
    if spatial is not None and any(not isinstance(item.get("geometry"), dict) for item in elements):
        raise ProcessInputError("源结果缺少几何，无法执行空间筛选条件", "unsupported_filter")

    kept: list[dict[str, Any]] = []
    for item in elements:
        if parsed.osm_types:
            osm_type = str(item.get("osm_type") or "")
            if osm_type not in parsed.osm_types:
                continue
        if parsed.tags and not _match_tags(item, parsed.tags):
            continue
        if spatial is not None and not _match_spatial(item, spatial):
            continue
        kept.append(_clean_element(item))
    if parsed.dedupe:
        kept = _dedupe_elements(kept)
    return kept


def _any_has_tag(elements: list[dict[str, Any]], key: str) -> bool:
    for item in elements:
        tags = item.get("tags")
        if isinstance(tags, dict) and key in tags:
            return True
    return False


def _match_tags(element: dict[str, Any], tags: dict[str, str]) -> bool:
    el_tags = element.get("tags") if isinstance(element.get("tags"), dict) else {}
    for key, expected in tags.items():
        actual = el_tags.get(key)
        if actual is None:
            return False
        actual_text = _stringify_tag(actual)
        if actual_text is None:
            return False
        if expected != "*" and actual_text != expected:
            return False
    return True


def _match_spatial(element: dict[str, Any], spatial: ParsedSpatial) -> bool:
    raw_geom = element.get("geometry")
    if not isinstance(raw_geom, dict):
        raise ProcessInputError("源结果缺少几何，无法执行空间筛选条件", "unsupported_filter")
    try:
        geom = shape(raw_geom)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ProcessInputError("要素几何无法解析，无法执行空间筛选条件", "unsupported_filter") from exc
    if geom.is_empty:
        raise ProcessInputError("要素几何为空，无法执行空间筛选条件", "unsupported_filter")
    relation = spatial.relation
    if relation == "near":
        assert spatial.distance_m is not None
        origin = spatial.geometry.centroid
        left = _to_local_meters(geom, origin.x, origin.y)
        right = _to_local_meters(spatial.geometry, origin.x, origin.y)
        return float(left.distance(right)) <= spatial.distance_m
    if relation == "within":
        return bool(geom.within(spatial.geometry))
    if relation == "intersects":
        return bool(geom.intersects(spatial.geometry))
    if relation == "contains":
        return bool(geom.contains(spatial.geometry))
    if relation == "crosses":
        return bool(geom.crosses(spatial.geometry))
    raise ProcessInputError(f"不支持的空间关系: {relation}", "invalid_input")


def _to_local_meters(geom: BaseGeometry, origin_lon: float, origin_lat: float) -> BaseGeometry:
    meters_per_deg_lon = _METERS_PER_DEG_LAT * max(abs(math.cos(math.radians(origin_lat))), 1e-6)

    def _project(x: Any, y: Any, z: Any = None) -> tuple[Any, ...]:
        mx = (x - origin_lon) * meters_per_deg_lon
        my = (y - origin_lat) * _METERS_PER_DEG_LAT
        if z is None:
            return (mx, my)
        return (mx, my, z)

    return shapely_transform(_project, geom)


def _dedupe_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    unique: list[dict[str, Any]] = []
    for item in elements:
        osm_type = _optional_str(item.get("osm_type"))
        osm_id = _as_int(item.get("osm_id"))
        if osm_type is None or osm_id is None:
            unique.append(item)
            continue
        key = (osm_type, osm_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _export_payload(
    elements: list[dict[str, Any]],
    fmt: str,
    *,
    include_geometry: bool,
) -> tuple[str, dict[str, Any] | None]:
    cleaned = [_clean_element(item, include_geometry=include_geometry) for item in elements]
    geojson = _to_geojson(cleaned, include_geometry=include_geometry)
    if fmt == "geojson":
        return json.dumps(geojson, ensure_ascii=False), geojson
    if fmt == "json":
        return json.dumps(cleaned, ensure_ascii=False), None
    if fmt == "csv":
        return _to_csv(cleaned, include_geometry=include_geometry), None
    if fmt == "kml":
        return _to_kml(cleaned, include_geometry=include_geometry), None
    if fmt == "osm_xml":
        return _to_osm_xml(cleaned, include_geometry=include_geometry), None
    raise ProcessInputError(f"不支持的导出格式: {fmt}", "invalid_input")


def _to_geojson(elements: list[dict[str, Any]], *, include_geometry: bool) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for item in elements:
        properties: dict[str, Any] = {}
        if "result_id" in item:
            properties["result_id"] = item["result_id"]
        if "osm_type" in item:
            properties["osm_type"] = item["osm_type"]
        if "osm_id" in item:
            properties["osm_id"] = item["osm_id"]
        tags = item.get("tags")
        if isinstance(tags, dict):
            properties["tags"] = tags
        geometry = item.get("geometry") if include_geometry else None
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": geometry if isinstance(geometry, dict) else None,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _to_csv(elements: list[dict[str, Any]], *, include_geometry: bool) -> str:
    tag_keys: list[str] = []
    seen: set[str] = set()
    for item in elements:
        tags = item.get("tags") if isinstance(item.get("tags"), dict) else {}
        for key in tags:
            if key not in seen:
                seen.add(key)
                tag_keys.append(key)
    fieldnames = ["result_id", "osm_type", "osm_id", *tag_keys]
    if include_geometry:
        fieldnames.append("geometry")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for item in elements:
        row: dict[str, Any] = {
            "result_id": item.get("result_id", ""),
            "osm_type": item.get("osm_type", ""),
            "osm_id": item.get("osm_id", ""),
        }
        tags = item.get("tags") if isinstance(item.get("tags"), dict) else {}
        for key in tag_keys:
            row[key] = tags.get(key, "")
        if include_geometry:
            geom = item.get("geometry")
            row["geometry"] = json.dumps(geom, ensure_ascii=False) if isinstance(geom, dict) else ""
        writer.writerow(row)
    return buffer.getvalue()


def _to_kml(elements: list[dict[str, Any]], *, include_geometry: bool) -> str:
    root = ET.Element("kml", {"xmlns": "http://www.opengis.net/kml/2.2"})
    document = ET.SubElement(root, "Document")
    for item in elements:
        placemark = ET.SubElement(document, "Placemark")
        name = _element_name(item)
        if name:
            ET.SubElement(placemark, "name").text = name
        tags = item.get("tags") if isinstance(item.get("tags"), dict) else {}
        if tags or "osm_id" in item or "osm_type" in item:
            ext = ET.SubElement(placemark, "ExtendedData")
            if "osm_type" in item:
                _kml_data(ext, "osm_type", str(item["osm_type"]))
            if "osm_id" in item:
                _kml_data(ext, "osm_id", str(item["osm_id"]))
            for key, value in tags.items():
                _kml_data(ext, str(key), str(value))
        if include_geometry:
            _append_kml_geometry(placemark, item.get("geometry"))
    return _xml_bytes(root)


def _kml_data(parent: ET.Element, name: str, value: str) -> None:
    node = ET.SubElement(parent, "Data", {"name": name})
    ET.SubElement(node, "value").text = value


def _append_kml_geometry(parent: ET.Element, geometry: Any) -> None:
    if not isinstance(geometry, dict):
        return
    kind = str(geometry.get("type") or "")
    coords = geometry.get("coordinates")
    if kind == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        point = ET.SubElement(parent, "Point")
        ET.SubElement(point, "coordinates").text = _kml_coord_pair(coords)
        return
    if kind == "LineString" and isinstance(coords, list):
        line = ET.SubElement(parent, "LineString")
        ET.SubElement(line, "coordinates").text = " ".join(
            _kml_coord_pair(item) for item in coords if isinstance(item, (list, tuple)) and len(item) >= 2
        )
        return
    if kind == "Polygon" and isinstance(coords, list) and coords:
        polygon = ET.SubElement(parent, "Polygon")
        outer = ET.SubElement(polygon, "outerBoundaryIs")
        ring = ET.SubElement(outer, "LinearRing")
        ring_coords = coords[0] if isinstance(coords[0], list) else coords
        ET.SubElement(ring, "coordinates").text = " ".join(
            _kml_coord_pair(item)
            for item in ring_coords
            if isinstance(item, (list, tuple)) and len(item) >= 2
        )


def _kml_coord_pair(raw: Any) -> str:
    lon = _as_float(raw[0]) if isinstance(raw, (list, tuple)) and raw else None
    lat = _as_float(raw[1]) if isinstance(raw, (list, tuple)) and len(raw) > 1 else None
    if lon is None or lat is None:
        return ""
    return f"{lon},{lat}"


def _to_osm_xml(elements: list[dict[str, Any]], *, include_geometry: bool) -> str:
    root = ET.Element("osm", {"version": "0.6", "generator": "geoagent-osm_result_process"})
    next_node_id = -1
    way_entries: list[tuple[Any, dict[str, Any], list[list[float]]]] = []
    relation_entries: list[tuple[Any, dict[str, Any]]] = []
    for item in elements:
        osm_type = str(item.get("osm_type") or "node")
        osm_id = item.get("osm_id", 0)
        tags = item.get("tags") if isinstance(item.get("tags"), dict) else {}
        geom = item.get("geometry") if include_geometry else None
        if osm_type == "way":
            way_entries.append((osm_id, tags, _linestring_coords(geom)))
            continue
        if osm_type == "relation":
            relation_entries.append((osm_id, tags))
            continue
        attrib = {"id": str(osm_id)}
        point = _point_coords(geom)
        if point is not None:
            attrib["lon"] = str(point[0])
            attrib["lat"] = str(point[1])
        node = ET.SubElement(root, "node", attrib)
        _append_osm_tags(node, tags)
    for osm_id, tags, coords in way_entries:
        nd_refs: list[str] = []
        for lon, lat in coords:
            node_id = str(next_node_id)
            next_node_id -= 1
            ET.SubElement(root, "node", {"id": node_id, "lon": str(lon), "lat": str(lat)})
            nd_refs.append(node_id)
        way = ET.SubElement(root, "way", {"id": str(osm_id)})
        for node_id in nd_refs:
            ET.SubElement(way, "nd", {"ref": node_id})
        _append_osm_tags(way, tags)
    for osm_id, tags in relation_entries:
        rel = ET.SubElement(root, "relation", {"id": str(osm_id)})
        _append_osm_tags(rel, tags)
    return _xml_bytes(root)


def _append_osm_tags(parent: ET.Element, tags: dict[str, Any]) -> None:
    for key, value in tags.items():
        text = _stringify_tag(value)
        if text is None:
            continue
        ET.SubElement(parent, "tag", {"k": str(key), "v": text})


def _linestring_coords(geometry: Any) -> list[list[float]]:
    if not isinstance(geometry, dict) or str(geometry.get("type") or "") != "LineString":
        return []
    raw = geometry.get("coordinates")
    if not isinstance(raw, list):
        return []
    points: list[list[float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        lon = _as_float(item[0])
        lat = _as_float(item[1])
        if lon is None or lat is None:
            continue
        points.append([lon, lat])
    return points


def _point_coords(geometry: Any) -> tuple[float, float] | None:
    if not isinstance(geometry, dict) or str(geometry.get("type") or "") != "Point":
        return None
    raw = geometry.get("coordinates")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    lon = _as_float(raw[0])
    lat = _as_float(raw[1])
    if lon is None or lat is None:
        return None
    return (lon, lat)


def _element_name(item: dict[str, Any]) -> str | None:
    tags = item.get("tags") if isinstance(item.get("tags"), dict) else {}
    name = tags.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    result_id = item.get("result_id")
    if isinstance(result_id, str) and result_id.strip():
        return result_id.strip()
    return None


def _xml_bytes(root: ET.Element) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def _ok_process(
    *,
    operation: str,
    source: ParsedSource,
    elements: list[dict[str, Any]],
    original_count: int,
    applied: dict[str, Any],
    ctx: RuntimeContext | None,
    extra: dict[str, Any] | None = None,
) -> Observation:
    kept = len(elements)
    result_id = _next_result_id(ctx)
    result: dict[str, Any] = {
        "operation": operation,
        "result_id": result_id,
        "elements": elements,
        "count": kept,
        "kept": kept,
        "removed": max(original_count - kept, 0),
        "applied": applied,
        "assumptions": [
            _LOCAL_ASSUMPTION,
            _NO_INVENT_ASSUMPTION,
            _CRS_ASSUMPTION,
            _NO_SESSION_ASSUMPTION,
        ],
    }
    if source.data_timestamp:
        result["data_timestamp"] = source.data_timestamp
    if extra:
        result.update(extra)
    cleaned = _strip_forbidden(result)
    _register_result(ctx, result_id, cleaned if isinstance(cleaned, dict) else result)
    return Observation(ok=True, result=cleaned if isinstance(cleaned, dict) else result)


def _applied_envelope(
    *,
    filters: dict[str, Any] | None = None,
    spatial: dict[str, Any] | None = None,
    fmt: str | None = None,
    include_geometry: bool | None = None,
) -> dict[str, Any]:
    applied: dict[str, Any] = {"provider": _PROVIDER_LOCAL, "crs": _CRS_WGS84}
    if filters is not None:
        applied["filters"] = filters
    if spatial is not None:
        applied["spatial_filter"] = spatial
    if fmt is not None:
        applied["format"] = fmt
    if include_geometry is not None:
        applied["include_geometry"] = include_geometry
    return applied


def _next_result_id(ctx: RuntimeContext | None) -> str:
    if ctx is None:
        return "osm_processed_1"
    current = ctx.extras.get(_SEQ_KEY, 0)
    seq = current + 1 if isinstance(current, int) else 1
    ctx.extras[_SEQ_KEY] = seq
    return f"osm_processed_{seq}"


def _register_result(ctx: RuntimeContext | None, result_id: str, result: dict[str, Any]) -> None:
    if ctx is None:
        return
    registry = ctx.extras.get(_REGISTRY_KEY)
    if not isinstance(registry, dict):
        registry = {}
        ctx.extras[_REGISTRY_KEY] = registry
    registry[result_id] = {
        "result_id": result_id,
        "elements": result.get("elements") if isinstance(result.get("elements"), list) else [],
        "count": result.get("count"),
        "data_timestamp": result.get("data_timestamp"),
    }


def _clean_element(item: dict[str, Any], *, include_geometry: bool = True) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if isinstance(item.get("result_id"), str) and item["result_id"].strip():
        row["result_id"] = item["result_id"].strip()
    if item.get("osm_type") in _ELEMENT_TYPES:
        row["osm_type"] = item["osm_type"]
    osm_id = _as_int(item.get("osm_id"))
    if osm_id is not None:
        row["osm_id"] = osm_id
    tags = item.get("tags")
    if isinstance(tags, dict):
        cleaned = _clean_tags(tags)
        if cleaned:
            row["tags"] = cleaned
    if include_geometry and isinstance(item.get("geometry"), dict):
        row["geometry"] = item["geometry"]
    return row


def _clean_tags(tags: dict[str, Any]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in tags.items():
        if key in _FORBIDDEN_KEYS:
            continue
        text = _stringify_tag(value)
        if text is None:
            continue
        cleaned[str(key)] = text
    return cleaned


def _parse_optional_bool(raw: Any, *, default: bool) -> bool:
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if raw in (0, 1):
            return bool(raw)
        raise ProcessInputError("布尔值无效", "invalid_input")
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    raise ProcessInputError("布尔值无效", "invalid_input")


def _as_string_list(raw: Any, *, field: str) -> list[str]:
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        values: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise ProcessInputError(f"{field} 必须是字符串或字符串列表", "invalid_input")
            values.append(item.strip())
        return values
    raise ProcessInputError(f"{field} 必须是字符串或字符串列表", "invalid_input")


def _stringify_tag(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip()
    return text if text else None


def _optional_str(raw: Any) -> str | None:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return str(raw)
    return None


def _timestamp_of(raw: dict[str, Any]) -> str | None:
    stamp = raw.get("data_timestamp")
    if isinstance(stamp, str) and stamp.strip():
        return stamp.strip()
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
