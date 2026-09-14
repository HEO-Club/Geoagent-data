"""OSM 已有结果的本地属性/空间筛选和安全导出。"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from tool.contract import Observation, RuntimeContext
from tool.runtime.geo import (
    GeoRuntimeError,
    filter_items_spatially,
    geometry_from_item,
    geometry_from_reference,
    require_wgs84,
    resolve_structured_source,
    source_items,
    source_label,
)
from tool.runtime.result_store import store_result

_SUPPORTED_FILTERS = {
    "osm_type",
    "osm_id",
    "name",
    "name_contains",
    "name_regex",
    "tags",
    "has_geometry",
    "bbox",
    "limit",
}
_SUPPORTED_EXPORT_FORMATS = {"geojson", "json", "csv", "kml"}
_MAX_REGEX_LENGTH = 200
_MAX_EXPORT_ITEMS = 100_000


def execute_filter(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    del purpose
    try:
        source_raw = inputs.get("source_result")
        source = resolve_structured_source(source_raw, ctx)
        require_wgs84(source)
        elements, source_key = source_items(source)
        filters = _filters(inputs.get("filters"))
        bbox_geometry = (
            geometry_from_reference(filters["bbox"], None, field="filters.bbox")
            if "bbox" in filters
            else None
        )
        name_pattern = (
            _safe_name_pattern(filters["name_regex"])
            if "name_regex" in filters
            else None
        )
        filtered = [
            item
            for item in elements
            if _matches(
                item,
                filters,
                bbox_geometry=bbox_geometry,
                name_pattern=name_pattern,
            )
        ]
        spatial = inputs.get("spatial_filter")
        spatial_stats: dict[str, Any] | None = None
        if spatial is not None:
            if not isinstance(spatial, dict):
                raise GeoRuntimeError("spatial_filter 必须是对象", "invalid_spatial_filter")
            relation = spatial.get("relation")
            if not isinstance(relation, str) or not relation.strip():
                raise GeoRuntimeError(
                    "spatial_filter.relation 为必填字符串",
                    "missing_input",
                )
            if "geometry" not in spatial:
                raise GeoRuntimeError("spatial_filter.geometry 为必填", "missing_input")
            reference = geometry_from_reference(spatial["geometry"], ctx)
            filtered, spatial_stats = filter_items_spatially(
                filtered,
                relation=relation,
                reference=reference,
                distance_m=spatial.get("distance_m"),
            )
        limit = filters.get("limit")
        if isinstance(limit, int):
            filtered = filtered[:limit]
    except GeoRuntimeError as exc:
        return _fail(str(exc), exc.error_code)
    except re.error as exc:
        return _fail(f"name_regex 无效: {exc}", "invalid_filter")

    result: dict[str, Any] = {
        "operation": "filter",
        "source_result": source_label(source_raw),
        "source_collection": source_key,
        "input_count": len(elements),
        "output_count": len(filtered),
        "dropped_count": len(elements) - len(filtered),
        "elements": filtered,
        "filters": filters,
        "spatial_filter": spatial_stats,
        "crs": str(source.get("crs") or "EPSG:4326"),
        "attribution": source.get("attribution") or "© OpenStreetMap contributors",
        "assumptions": [
            "筛选只作用于 source_result 中实际存在的字段和几何",
            "缺失标签或几何不会被推断为现实中不存在该要素",
        ],
    }
    result_id = store_result(result, namespace="osm_filtered", ctx=ctx)
    if result_id:
        result["result_id"] = result_id
    return Observation(ok=True, result=result)


def execute_export(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    del purpose
    try:
        source_raw = inputs.get("source_result")
        source = resolve_structured_source(source_raw, ctx)
        require_wgs84(source)
        elements, source_key = source_items(source)
        if len(elements) > _MAX_EXPORT_ITEMS:
            raise GeoRuntimeError(
                f"导出要素超过 {_MAX_EXPORT_ITEMS} 条上限",
                "source_too_large",
            )
        export_format = _export_format(inputs.get("format"))
        include_geometry = _boolean(inputs.get("include_geometry"), default=True)
        content, media_type = _render_export(
            elements,
            export_format=export_format,
            include_geometry=include_geometry,
            source=source,
        )
        path = _write_artifact(content, export_format, ctx)
    except GeoRuntimeError as exc:
        return _fail(str(exc), exc.error_code)
    except OSError as exc:
        return _fail(f"导出文件写入失败: {exc}", "artifact_write_failed")

    digest = hashlib.sha256(content).hexdigest()
    result = {
        "operation": "export",
        "source_result": source_label(source_raw),
        "source_collection": source_key,
        "format": export_format,
        "include_geometry": include_geometry,
        "exported_count": len(elements),
        "crs": str(source.get("crs") or "EPSG:4326"),
        "media_type": media_type,
        "size_bytes": len(content),
        "sha256": digest,
        "attribution": source.get("attribution") or "© OpenStreetMap contributors",
        "assumptions": [
            "导出文件来自已有 source_result，不执行新的 OSM 网络查询",
            "GeoJSON/KML 几何由返回坐标、完整 geometry 或 bbox 按此顺序构造",
        ],
    }
    return Observation(
        ok=True,
        result=result,
        artifacts={"export_path": str(path), "media_type": media_type},
    )


def _filters(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise GeoRuntimeError("filters 必须是对象", "invalid_filter")
    unknown = sorted(set(raw) - _SUPPORTED_FILTERS)
    if unknown:
        raise GeoRuntimeError(f"不支持的 filters 字段: {unknown}", "unsupported_filter")
    filters = copy.deepcopy(raw)
    if "has_geometry" in filters and not isinstance(filters["has_geometry"], bool):
        raise GeoRuntimeError("filters.has_geometry 必须是布尔值", "invalid_filter")
    if "tags" in filters and not isinstance(filters["tags"], dict):
        raise GeoRuntimeError("filters.tags 必须是对象", "invalid_filter")
    if "name_contains" in filters and not isinstance(filters["name_contains"], str):
        raise GeoRuntimeError("filters.name_contains 必须是字符串", "invalid_filter")
    if "name_regex" in filters:
        pattern = filters["name_regex"]
        if not isinstance(pattern, str) or not pattern or len(pattern) > _MAX_REGEX_LENGTH:
            raise GeoRuntimeError(
                f"filters.name_regex 必须是 1-{_MAX_REGEX_LENGTH} 字符字符串",
                "invalid_filter",
            )
        re.compile(pattern, re.IGNORECASE)
    if "bbox" in filters:
        geometry_from_reference(filters["bbox"], None, field="filters.bbox")
    if "limit" in filters:
        value = filters["limit"]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_EXPORT_ITEMS:
            raise GeoRuntimeError(
                f"filters.limit 必须是 1-{_MAX_EXPORT_ITEMS} 的整数",
                "invalid_filter",
            )
    return filters


def _matches(
    item: dict[str, Any],
    filters: dict[str, Any],
    *,
    bbox_geometry: BaseGeometry | None,
    name_pattern: re.Pattern[str] | None,
) -> bool:
    if "osm_type" in filters and not _matches_scalar(item.get("osm_type"), filters["osm_type"]):
        return False
    if "osm_id" in filters and not _matches_scalar(item.get("osm_id"), filters["osm_id"]):
        return False
    name = str(item.get("name") or "")
    if "name" in filters and not _matches_scalar(name, filters["name"]):
        return False
    if "name_contains" in filters:
        needle = filters["name_contains"]
        if not isinstance(needle, str) or needle.casefold() not in name.casefold():
            return False
    if name_pattern is not None and name_pattern.search(name) is None:
        return False
    raw_tags = item.get("tags")
    tags: dict[str, Any] = raw_tags if isinstance(raw_tags, dict) else {}
    for key, expected in (filters.get("tags") or {}).items():
        actual = tags.get(str(key))
        if expected == "*":
            if str(key) not in tags:
                return False
        elif isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    if "has_geometry" in filters or "bbox" in filters:
        geometry, _ = geometry_from_item(item)
        if "has_geometry" in filters and (geometry is not None) != filters["has_geometry"]:
            return False
        if "bbox" in filters:
            assert bbox_geometry is not None
            if geometry is None or not geometry.intersects(bbox_geometry):
                return False
    return True


def _matches_scalar(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return actual in expected or str(actual) in {str(value) for value in expected}
    return actual == expected or str(actual) == str(expected)


def _export_format(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise GeoRuntimeError("format 为必填字符串", "missing_input")
    value = raw.strip().lower()
    if value == "osm_xml":
        raise GeoRuntimeError(
            "规范化 OSM payload 已丢失 relation members/node refs，拒绝伪造可回灌的 osm_xml",
            "lossy_format_unsupported",
        )
    if value not in _SUPPORTED_EXPORT_FORMATS:
        raise GeoRuntimeError(f"不支持导出格式: {value}", "unsupported_format")
    return value


def _boolean(raw: Any, *, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    raise GeoRuntimeError("include_geometry 必须是布尔值", "invalid_input")


def _render_export(
    elements: list[dict[str, Any]],
    *,
    export_format: str,
    include_geometry: bool,
    source: dict[str, Any],
) -> tuple[bytes, str]:
    if export_format == "json":
        payload = {
            "crs": source.get("crs") or "EPSG:4326",
            "attribution": source.get("attribution") or "© OpenStreetMap contributors",
            "elements": [_without_geometry(item) for item in elements] if not include_geometry else elements,
        }
        return _json_bytes(payload), "application/json"
    features = [_feature(item, include_geometry=include_geometry) for item in elements]
    if export_format == "geojson":
        payload = {
            "type": "FeatureCollection",
            "features": features,
            "attribution": source.get("attribution") or "© OpenStreetMap contributors",
        }
        return _json_bytes(payload), "application/geo+json"
    if export_format == "csv":
        return _csv_bytes(elements, include_geometry=include_geometry), "text/csv"
    return _kml_bytes(features), "application/vnd.google-earth.kml+xml"


def _feature(item: dict[str, Any], *, include_geometry: bool) -> dict[str, Any]:
    geometry, geometry_source = geometry_from_item(item)
    properties = copy.deepcopy(item.get("properties")) if item.get("type") == "Feature" else copy.deepcopy(item)
    if not isinstance(properties, dict):
        properties = {}
    for key in ("geometry", "geojson", "bbox", "latitude", "longitude", "lat", "lon"):
        properties.pop(key, None)
    properties["geometry_source"] = geometry_source
    return {
        "type": "Feature",
        "geometry": mapping(geometry) if include_geometry and geometry is not None else None,
        "properties": properties,
    }


def _without_geometry(item: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(item)
    for key in ("geometry", "geojson", "bbox", "latitude", "longitude", "lat", "lon", "center"):
        copied.pop(key, None)
    return copied


def _csv_bytes(elements: list[dict[str, Any]], *, include_geometry: bool) -> bytes:
    buffer = io.StringIO(newline="")
    fields = ["osm_type", "osm_id", "name", "latitude", "longitude", "geometry", "tags"]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for item in elements:
        geometry, _ = geometry_from_item(item)
        writer.writerow(
            {
                "osm_type": item.get("osm_type") or item.get("type"),
                "osm_id": item.get("osm_id") or item.get("id"),
                "name": _csv_safe(_feature_name(item)),
                "latitude": item.get("latitude") or item.get("lat"),
                "longitude": item.get("longitude") or item.get("lon"),
                "geometry": (
                    json.dumps(mapping(geometry), ensure_ascii=False, separators=(",", ":"))
                    if include_geometry and geometry is not None
                    else ""
                ),
                "tags": json.dumps(item.get("tags") or {}, ensure_ascii=False, separators=(",", ":")),
            }
        )
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def _kml_bytes(features: list[dict[str, Any]]) -> bytes:
    root = Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = SubElement(root, "Document")
    for index, feature in enumerate(features):
        placemark = SubElement(document, "Placemark")
        properties = feature.get("properties") or {}
        SubElement(placemark, "name").text = str(properties.get("name") or f"feature-{index + 1}")
        geometry = feature.get("geometry")
        if isinstance(geometry, dict):
            _append_kml_geometry(placemark, geometry)
    return b'<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(root, encoding="utf-8")


def _append_kml_geometry(parent: Element, geometry: dict[str, Any]) -> None:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Point" and isinstance(coordinates, (list, tuple)):
        node = SubElement(parent, "Point")
        SubElement(node, "coordinates").text = _kml_coordinate(coordinates)
    elif geometry_type == "LineString" and isinstance(coordinates, list):
        node = SubElement(parent, "LineString")
        SubElement(node, "coordinates").text = " ".join(_kml_coordinate(point) for point in coordinates)
    elif geometry_type == "Polygon" and isinstance(coordinates, list) and coordinates:
        polygon = SubElement(parent, "Polygon")
        boundary = SubElement(polygon, "outerBoundaryIs")
        ring = SubElement(boundary, "LinearRing")
        SubElement(ring, "coordinates").text = " ".join(
            _kml_coordinate(point) for point in coordinates[0]
        )
    elif geometry_type and isinstance(geometry.get("geometries"), list):
        multi = SubElement(parent, "MultiGeometry")
        for child in geometry["geometries"]:
            if isinstance(child, dict):
                _append_kml_geometry(multi, child)
    elif isinstance(coordinates, list):
        multi = SubElement(parent, "MultiGeometry")
        child_type = geometry_type.removeprefix("Multi") if isinstance(geometry_type, str) else ""
        for child_coordinates in coordinates:
            _append_kml_geometry(
                multi,
                {"type": child_type, "coordinates": child_coordinates},
            )


def _kml_coordinate(raw: Any) -> str:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return ""
    altitude = raw[2] if len(raw) > 2 else 0
    return f"{raw[0]},{raw[1]},{altitude}"


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _safe_name_pattern(raw: Any) -> re.Pattern[str]:
    assert isinstance(raw, str)
    if "(?" in raw or re.search(r"\\[1-9]", raw):
        raise GeoRuntimeError(
            "name_regex 不允许 lookaround、命名组或反向引用",
            "unsafe_regex",
        )
    if re.search(r"(?:\([^)]*[+*][^)]*\)|\[[^]]*\][+*])[+*{]", raw):
        raise GeoRuntimeError("name_regex 包含高风险嵌套量词", "unsafe_regex")
    try:
        return re.compile(raw, re.IGNORECASE)
    except re.error as exc:
        raise GeoRuntimeError(f"name_regex 无效: {exc}", "invalid_filter") from exc


def _feature_name(item: dict[str, Any]) -> Any:
    properties = item.get("properties")
    property_name = properties.get("name") if isinstance(properties, dict) else None
    return item.get("name") or property_name or ""


def _csv_safe(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _write_artifact(content: bytes, export_format: str, ctx: RuntimeContext | None) -> Path:
    raw_dir = ctx.extras.get("artifact_dir") if ctx is not None else None
    root = Path(raw_dir) if isinstance(raw_dir, str) and raw_dir.strip() else Path.cwd() / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    call_id = ctx.extras.get("harness_call_id") if ctx is not None else None
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(call_id or uuid.uuid4().hex))[:100]
    path = root / f"osm_export_{stem}.{export_format}"
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return path.resolve()


def _fail(error: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=error, error_code=error_code)


__all__ = ["execute_export", "execute_filter"]
