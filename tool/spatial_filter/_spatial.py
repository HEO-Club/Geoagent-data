"""基于 Shapely 与局部 AEQD 投影的本地空间关系筛选。"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext
from tool.runtime.geo import (
    GeoRuntimeError,
    filter_items_spatially,
    geometry_from_reference,
    geometry_mapping,
    require_wgs84,
    resolve_structured_source,
    source_items,
    source_label,
)
from tool.runtime.result_store import store_result


def execute_geometry_filter(
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
        items, source_key = source_items(source)
        relation = inputs.get("relation")
        if not isinstance(relation, str) or not relation.strip():
            raise GeoRuntimeError("relation 为必填字符串", "missing_input")
        if "geometry" not in inputs:
            raise GeoRuntimeError("geometry 为必填", "missing_input")
        reference = geometry_from_reference(inputs["geometry"], ctx)
        matched, stats = filter_items_spatially(
            items,
            relation=relation,
            reference=reference,
            distance_m=inputs.get("distance_m"),
        )
    except GeoRuntimeError as exc:
        return Observation(ok=False, error=str(exc), error_code=exc.error_code)

    result: dict[str, Any] = {
        "operation": "geometry_filter",
        "source_result": source_label(source_raw),
        "source_collection": source_key,
        "relation": stats["relation"],
        "distance_m": stats["distance_m"],
        "input_count": stats["input_count"],
        "matched_count": stats["matched_count"],
        "rejected_count": stats["rejected_count"],
        "missing_geometry_count": stats["missing_geometry_count"],
        "elements": matched,
        "reference_geometry": geometry_mapping(reference),
        "source_crs": str(source.get("crs") or "EPSG:4326"),
        "metric_projection": stats["metric_projection"],
        "attribution": source.get("attribution") or "© OpenStreetMap contributors",
        "assumptions": [
            "输入和输出几何按 WGS84 经度/纬度解释",
            "near 使用以参照几何中心为原点的局部 AEQD 米制投影",
            "缺少 geometry 的要素不会被猜测，而是计入 missing_geometry_count",
        ],
    }
    result_id = store_result(result, namespace="spatial", ctx=ctx)
    if result_id:
        result["result_id"] = result_id
    return Observation(ok=True, result=result)


__all__ = ["execute_geometry_filter"]
