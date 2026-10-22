"""POI 名称检索与区域浏览：名称走地理编码，类别/属性走 Overpass。"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.geocode._nominatim import execute_geocode
from tool.osm_query._overpass import (
    OverpassInputError,
    OverpassProviderError,
    normalize_elements,
    run_overpass,
)
from tool.runtime.result_store import store_result

_DEFAULT_TOP_K = 20
_MAX_TOP_K = 200


class PoiInputError(Exception):
    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def execute_poi_search(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    """按名称或类别返回候选 POI，不根据匹配结果直接确定最终地点。"""

    del purpose
    prepared = _with_context_scope(inputs, ctx)
    try:
        query = _query_text(prepared.get("query"))
        categories = _string_list(prepared.get("categories"), "categories")
    except PoiInputError as exc:
        return _fail(str(exc), exc.error_code)
    filters = prepared.get("filters")
    has_scope = any(prepared.get(key) is not None for key in ("area", "bbox", "center"))
    if not query and not categories:
        return _fail("poi_search 至少需要 query 或 categories", "missing_input")

    if query and not categories and filters is None and not has_scope:
        return _name_search(query, prepared, ctx)
    if not has_scope:
        return _fail(
            "类别或属性 POI 查询需要 area、bbox 或 center+radius_m；请先调用 geocode 获取范围",
            "needs_acquisition",
        )
    return _overpass_search(
        "poi_search",
        prepared,
        ctx,
        query=query,
        categories=categories,
    )


def execute_browse(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    """在明确区域内按名称、类别或 OSM 属性浏览候选。"""

    del purpose
    prepared = _with_context_scope(inputs, ctx)
    try:
        query = _query_text(prepared.get("query"))
        categories = _string_list(prepared.get("categories"), "categories")
    except PoiInputError as exc:
        return _fail(str(exc), exc.error_code)
    if not any(prepared.get(key) is not None for key in ("area", "bbox", "center")):
        code = "needs_acquisition" if query or categories or prepared.get("filters") else "missing_input"
        return _fail("browse 需要 area、bbox 或 center+radius_m", code)
    if not query and not categories and not isinstance(prepared.get("filters"), dict):
        return _fail("browse 必须提供 query、categories 或 filters，禁止无过滤遍历", "missing_filter")
    return _overpass_search("browse", prepared, ctx, query=query, categories=categories)


def _name_search(
    query: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    try:
        top_k = _top_k(inputs.get("top_k"))
    except PoiInputError as exc:
        return _fail(str(exc), exc.error_code)
    observation = execute_geocode(
        purpose="POI 名称检索",
        inputs={
            "query": query,
            "direction": "forward",
            "top_k": min(top_k, 10),
            **({"language": inputs["language"]} if inputs.get("language") else {}),
        },
        ctx=ctx,
    )
    if not observation.ok or observation.result is None:
        return observation
    candidates = [_candidate_from_geocode(item) for item in observation.result["candidates"]]
    return _finish(
        operation="poi_search",
        provider=str(observation.result.get("provider") or "geocode"),
        candidates=candidates,
        applied={"query": query, "top_k": top_k, "strategy": "name_geocode"},
        ctx=ctx,
    )


def _overpass_search(
    operation: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    query: str,
    categories: list[str],
) -> Observation:
    try:
        top_k = _top_k(inputs.get("top_k"))
    except PoiInputError as exc:
        return _fail(str(exc), exc.error_code)
    osm_inputs: dict[str, Any] = {
        key: inputs[key]
        for key in ("area", "bbox", "center", "radius_m")
        if inputs.get(key) is not None
    }
    if categories:
        osm_inputs["feature_types"] = categories
    filters = inputs.get("filters")
    if filters is not None:
        if not isinstance(filters, dict):
            return _fail("filters 必须是 OSM tag 对象", "invalid_filters")
        osm_inputs["tags"] = filters
    osm_inputs["limit"] = top_k
    try:
        provider, ql, raw = run_overpass(
            osm_inputs,
            ctx,
            name_pattern=query or None,
        )
        elements = normalize_elements(raw, include_geometry=False)
    except OverpassInputError as exc:
        return _fail(str(exc), exc.error_code)
    except OverpassProviderError as exc:
        return _fail(str(exc), exc.error_code)
    candidates = [_candidate_from_osm(item) for item in elements[:top_k]]
    return _finish(
        operation=operation,
        provider=provider,
        candidates=candidates,
        applied={
            "query": query or None,
            "categories": categories,
            "filters": filters,
            "top_k": top_k,
            "strategy": "overpass",
            "overpass_ql": ql,
        },
        ctx=ctx,
    )


def _finish(
    *,
    operation: str,
    provider: str,
    candidates: list[dict[str, Any]],
    applied: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    result: dict[str, Any] = {
        "provider": provider,
        "operation": operation,
        "candidate_count": len(candidates),
        "candidates": _dedupe(candidates),
        "applied": applied,
        "crs": "EPSG:4326",
        "attribution": "© OpenStreetMap contributors",
        "assumptions": [
            "POI 返回值是候选清单，不代表已确认拍摄地点",
            "零结果只说明当前数据源和查询条件未命中，不代表现实中不存在该对象",
        ],
    }
    result["candidate_count"] = len(result["candidates"])
    result_id = store_result(result, namespace="poi", ctx=ctx)
    if result_id:
        result["result_id"] = result_id
    return Observation(ok=True, result=result)


def _candidate_from_geocode(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "poi_id": f"{item.get('osm_type', '')}/{item.get('osm_id', '')}",
        "name": item.get("display_name") or "",
        "display_name": item.get("display_name") or "",
        "latitude": item.get("latitude"),
        "longitude": item.get("longitude"),
        "category": item.get("category") or item.get("type") or "",
        "osm_type": item.get("osm_type"),
        "osm_id": item.get("osm_id"),
        "tags": {},
    }


def _candidate_from_osm(item: dict[str, Any]) -> dict[str, Any]:
    raw_tags = item.get("tags")
    tags: dict[str, Any] = raw_tags if isinstance(raw_tags, dict) else {}
    category = next(
        (
            f"{key}={tags[key]}"
            for key in ("amenity", "leisure", "tourism", "shop", "railway", "highway", "natural", "waterway", "power", "man_made")
            if key in tags
        ),
        "",
    )
    return {
        "poi_id": f"{item.get('osm_type', '')}/{item.get('osm_id', '')}",
        "name": item.get("name") or "",
        "latitude": item.get("latitude"),
        "longitude": item.get("longitude"),
        "category": category,
        "osm_type": item.get("osm_type"),
        "osm_id": item.get("osm_id"),
        "tags": tags,
    }


def _with_context_scope(inputs: dict[str, Any], ctx: RuntimeContext | None) -> dict[str, Any]:
    prepared = dict(inputs)
    if (
        not any(prepared.get(key) is not None for key in ("area", "bbox", "center"))
        and ctx is not None
        and ctx.active_area
    ):
        prepared["area"] = ctx.active_area
    return prepared


def _query_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list) and raw and all(isinstance(item, str) and item.strip() for item in raw):
        return " ".join(item.strip() for item in raw)
    raise PoiInputError("query 必须是字符串或字符串数组", "invalid_query")


def _string_list(raw: Any, name: str) -> list[str]:
    if raw is None:
        return []
    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, list) or not values:
        raise PoiInputError(
            f"{name} 必须是字符串或非空字符串数组",
            f"invalid_{name}",
        )
    result = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    if len(result) != len(values):
        raise PoiInputError(f"{name} 只能包含非空字符串", f"invalid_{name}")
    return result


def _top_k(raw: Any) -> int:
    if raw is None:
        return _DEFAULT_TOP_K
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise PoiInputError("top_k 必须是整数", "invalid_top_k") from exc
    if value < 1:
        raise PoiInputError("top_k 必须 >= 1", "invalid_top_k")
    return min(value, _MAX_TOP_K)


def _dedupe(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (
            candidate.get("osm_type"),
            candidate.get("osm_id"),
            candidate.get("name"),
            candidate.get("latitude"),
            candidate.get("longitude"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _fail(message: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=message, error_code=error_code)


__all__ = ["execute_browse", "execute_poi_search"]
