"""spatial_filter 共享执行器：按几何与空间关系本地筛选已有矢量要素。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError, ProjError
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box as shapely_box
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from tool._crs import gcj02_to_wgs84, is_gcj02
from tool.contract import Observation, RuntimeContext, declared_inputs

_OP_FILTER = "geometry_filter"
_PROVIDER_LOCAL = "local"
_CRS_WGS84 = "wgs84"
_PREVIOUS = "$previous_tool_result"
_REGISTRY_KEY = "spatial_results"
_OSM_REGISTRY_KEY = "osm_results"
_SEQ_KEY = "_spatial_result_seq"
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_GEOJSON_GEOM_TYPES = frozenset(
    {
        "point",
        "multipoint",
        "linestring",
        "multilinestring",
        "polygon",
        "multipolygon",
        "geometrycollection",
    }
)
_RELATIONS: dict[str, str] = {
    "within": "within",
    "inside": "within",
    "in": "within",
    "范围内": "within",
    "intersects": "intersects",
    "intersect": "intersects",
    "overlap": "intersects",
    "overlapping": "intersects",
    "相交": "intersects",
    "contains": "contains",
    "contain": "contains",
    "near": "near",
    "around": "near",
    "nearby": "near",
    "附近": "near",
    "周边": "near",
    "crosses": "crosses",
    "cross": "crosses",
}
_WGS84_ALIASES = frozenset(
    {
        "wgs84",
        "wgs 84",
        "wgs1984",
        "epsg:4326",
        "epsg4326",
        "4326",
        "crs84",
        "ogc:crs84",
        "urn:ogc:def:crs:ogc:1.3:crs84",
        "urn:ogc:def:crs:epsg::4326",
        "urn:ogc:def:crs:epsg:4326",
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
_LOCAL_ASSUMPTION = "本步只做本地矢量处理，未访问空间数据库或外部 API"
_NO_INVENT_ASSUMPTION = "未根据模型印象补齐缺失几何；缺几何则无法执行该条件"
_AEQD_ASSUMPTION = "near 距离在局部 AEQD 投影面以米计算，未把经纬度当平面坐标相减"
_GEOM_NOT_LABEL_ASSUMPTION = "邻近距离算到参照几何本身，不是到地名标注点"
_NO_SESSION_ASSUMPTION = "本步只返回筛选后的要素，未打开街景或地图会话"
_CRS_ASSUMPTION = "无 crs 标签时按 WGS84 解释"
_SKIP_PROP_KEYS = frozenset({"geometry", "crs", "datum", "coordinates", "features", "elements"})


class SpatialFilterError(Exception):
    """source_result / relation / geometry / distance_m 无法按合同解析或执行。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class ParsedFeature:
    """已解析的候选要素；geometry 为 GeoJSON 坐标顺序 lon,lat。"""

    geometry: BaseGeometry
    properties: dict[str, Any]
    crs: str


@dataclass(frozen=True)
class ParsedRef:
    """参照几何。"""

    geometry: BaseGeometry
    crs: str
    applied: dict[str, Any]


def execute_geometry_filter(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按空间关系筛选已有矢量要素，不访问外部服务。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "source_result", "relation", "geometry", "distance_m")
        source = _resolve_source_result(inputs.get("source_result"), ctx)
        relation = _parse_relation(inputs.get("relation"))
        distance_m = _parse_distance_m(inputs.get("distance_m"), required=relation == "near")
        reference = _parse_reference(inputs.get("geometry"), ctx)
        features, reference, result_crs = _harmonize_crs(source, reference)
        kept = _apply_filter(features, reference, relation=relation, distance_m=distance_m)
        return _ok_filter(
            features=kept,
            original_count=len(features),
            relation=relation,
            reference=reference,
            distance_m=distance_m,
            crs=result_crs,
            ctx=ctx,
        )
    except SpatialFilterError as exc:
        return _fail(str(exc), exc.error_code)


def _resolve_source_result(raw: Any, ctx: RuntimeContext | None) -> list[ParsedFeature]:
    """只接受真实结果对象、上下文引用或已登记 ID，拒绝自然语言描述。"""

    payload = _resolve_payload(raw, ctx, field="source_result")
    features = _features_from_payload(payload)
    if not features and _payload_claims_features(payload):
        raise SpatialFilterError("源结果缺少几何，无法执行空间筛选条件", "unsupported_filter")
    return features


def _parse_reference(raw: Any, ctx: RuntimeContext | None) -> ParsedRef:
    """参照几何必须是坐标 / GeoJSON / bbox / 前置结果几何，不能用地名。"""

    if raw is None or raw == "":
        raise SpatialFilterError("缺少必填输入 geometry", "missing_input")
    geom, crs = _geometry_and_crs(raw, ctx)
    if geom.is_empty:
        raise SpatialFilterError("参照几何为空", "invalid_input")
    return ParsedRef(geometry=geom, crs=crs, applied=dict(mapping(geom)))


def _parse_relation(raw: Any) -> str:
    if raw is None or raw == "":
        raise SpatialFilterError("缺少必填输入 relation", "missing_input")
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise SpatialFilterError(f"不支持的空间关系: {raw}", "invalid_input")
    key = str(raw).strip()
    relation = _RELATIONS.get(key) or _RELATIONS.get(key.lower())
    if relation is None:
        raise SpatialFilterError(f"不支持的空间关系: {raw}", "invalid_input")
    return relation


def _parse_distance_m(raw: Any, *, required: bool) -> float | None:
    if raw is None or raw == "":
        if required:
            raise SpatialFilterError("near 需要 distance_m", "missing_input")
        return None
    value = _as_float(raw)
    if value is None:
        raise SpatialFilterError("distance_m 必须是数字", "invalid_input")
    if value <= 0:
        raise SpatialFilterError("distance_m 必须为正数", "invalid_input")
    return value


def _resolve_payload(raw: Any, ctx: RuntimeContext | None, *, field: str) -> Any:
    if raw is None or raw == "":
        raise SpatialFilterError(f"缺少必填输入 {field}", "missing_input")
    if raw == _PREVIOUS:
        previous = ctx.previous_tool_result if ctx is not None else None
        if not isinstance(previous, (dict, list)):
            raise SpatialFilterError(f"无法解析 {field} 的 $previous_tool_result", "missing_input")
        return previous
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise SpatialFilterError(f"缺少必填输入 {field}", "missing_input")
        if text[0] in "{[":
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                if field == "geometry":
                    raise SpatialFilterError("参照几何必须是坐标，不能用地名", "invalid_input") from exc
                raise SpatialFilterError(
                    f"{field} 必须是真实结果对象、结果 ID 或 $previous_tool_result，不能用自然语言描述替代",
                    "invalid_input",
                ) from exc
        if _ID_RE.fullmatch(text):
            return _lookup_registry(text, ctx, field=field)
        if field == "geometry":
            raise SpatialFilterError("参照几何必须是坐标，不能用地名", "invalid_input")
        raise SpatialFilterError(
            f"{field} 必须是真实结果对象、结果 ID 或 $previous_tool_result，不能用自然语言描述替代",
            "invalid_input",
        )
    raise SpatialFilterError(f"{field} 必须是结果对象或引用", "invalid_input")


def _lookup_registry(result_id: str, ctx: RuntimeContext | None, *, field: str) -> Any:
    if ctx is None:
        raise SpatialFilterError(f"{field} 引用的结果不存在: {result_id}", "missing_input")
    for key in (_REGISTRY_KEY, _OSM_REGISTRY_KEY):
        registry = ctx.extras.get(key)
        if isinstance(registry, dict) and result_id in registry:
            stored = registry[result_id]
            if isinstance(stored, (dict, list)):
                return stored
            raise SpatialFilterError(f"{field} 引用的结果无法解析", "invalid_input")
    direct = ctx.extras.get(result_id)
    if isinstance(direct, (dict, list)):
        return direct
    raise SpatialFilterError(f"{field} 引用的结果不存在: {result_id}", "missing_input")


def _payload_claims_features(raw: Any) -> bool:
    if isinstance(raw, list):
        return any(isinstance(item, dict) for item in raw)
    if not isinstance(raw, dict):
        return False
    if isinstance(raw.get("features"), list) and raw["features"]:
        return True
    if isinstance(raw.get("elements"), list) and raw["elements"]:
        return True
    if isinstance(raw.get("geometry"), dict) or str(raw.get("type") or "").lower() in _GEOJSON_GEOM_TYPES:
        return True
    nested = raw.get("result")
    if isinstance(nested, (dict, list)):
        return _payload_claims_features(nested)
    return False


def _features_from_payload(raw: Any, *, default_crs: str | None = None) -> list[ParsedFeature]:
    collection_crs = default_crs
    payload = raw
    if isinstance(payload, dict):
        collection_crs = _crs_of(payload) or collection_crs
        kind = str(payload.get("type") or "").lower()
        if kind == "featurecollection":
            return _features_from_sequence(payload.get("features"), default_crs=collection_crs or _CRS_WGS84)
        if kind == "feature" or (isinstance(payload.get("geometry"), dict) and "features" not in payload and "elements" not in payload):
            feature = _feature_from_mapping(payload, default_crs=collection_crs or _CRS_WGS84)
            return [feature]
        if isinstance(payload.get("features"), list):
            return _features_from_sequence(payload.get("features"), default_crs=collection_crs or _CRS_WGS84)
        if isinstance(payload.get("elements"), list):
            return _features_from_sequence(payload.get("elements"), default_crs=collection_crs or _CRS_WGS84)
        nested = payload.get("result")
        if isinstance(nested, (dict, list)):
            return _features_from_payload(nested, default_crs=collection_crs)
        if kind in _GEOJSON_GEOM_TYPES and "coordinates" in payload:
            return [_feature_from_mapping(payload, default_crs=collection_crs or _CRS_WGS84)]
        raise SpatialFilterError("source_result 缺少可解析要素", "invalid_input")
    if isinstance(payload, list):
        return _features_from_sequence(payload, default_crs=collection_crs or _CRS_WGS84)
    raise SpatialFilterError("source_result 必须是结果对象或引用", "invalid_input")


def _features_from_sequence(raw: Any, *, default_crs: str) -> list[ParsedFeature]:
    if not isinstance(raw, list):
        return []
    features: list[ParsedFeature] = []
    saw_object = False
    for item in raw:
        if not isinstance(item, dict):
            continue
        saw_object = True
        features.append(_feature_from_mapping(item, default_crs=default_crs))
    if saw_object and not features:
        raise SpatialFilterError("源结果缺少几何，无法执行空间筛选条件", "unsupported_filter")
    return features


def _feature_from_mapping(raw: dict[str, Any], *, default_crs: str) -> ParsedFeature:
    crs = _normalize_crs_tag(_crs_of(raw) or default_crs)
    geom: BaseGeometry | None = None
    properties: dict[str, Any] = {}
    kind = str(raw.get("type") or "").lower()
    if kind == "feature":
        geom = _shape_required(raw.get("geometry"), field="源结果")
        props = raw.get("properties")
        properties = dict(props) if isinstance(props, dict) else {}
        ident = raw.get("id")
        if ident is not None and "id" not in properties:
            properties["id"] = ident
    elif isinstance(raw.get("geometry"), dict):
        geom = _shape_required(raw.get("geometry"), field="源结果")
        properties = _properties_from_generic(raw)
    elif kind in _GEOJSON_GEOM_TYPES and "coordinates" in raw:
        geom = _shape_required(raw, field="源结果")
        properties = _properties_from_generic(raw)
    else:
        raise SpatialFilterError("源结果缺少几何，无法执行空间筛选条件", "unsupported_filter")
    if geom.is_empty:
        raise SpatialFilterError("源结果缺少几何，无法执行空间筛选条件", "unsupported_filter")
    return ParsedFeature(geometry=geom, properties=_strip_forbidden(properties), crs=crs)


def _properties_from_generic(raw: dict[str, Any]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    nested = raw.get("properties")
    if isinstance(nested, dict):
        props.update(nested)
    for key, value in raw.items():
        if key in _SKIP_PROP_KEYS or key == "properties":
            continue
        if key == "type" and str(value).lower() in {"feature", *_GEOJSON_GEOM_TYPES}:
            continue
        if key not in props:
            props[key] = value
    return props


def _geometry_and_crs(raw: Any, ctx: RuntimeContext | None) -> tuple[BaseGeometry, str]:
    if isinstance(raw, str):
        payload = _resolve_payload(raw, ctx, field="geometry")
        return _geometry_and_crs(payload, ctx)
    if raw == _PREVIOUS:
        payload = _resolve_payload(raw, ctx, field="geometry")
        return _geometry_and_crs(payload, ctx)
    if isinstance(raw, (list, tuple)):
        return _geometry_from_sequence(raw), _CRS_WGS84
    if isinstance(raw, dict):
        crs = _normalize_crs_tag(_crs_of(raw) or _CRS_WGS84)
        return _geometry_from_mapping(raw, ctx), crs
    raise SpatialFilterError("参照几何必须是坐标，不能用地名", "invalid_input")


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
            geoms.append(_geometry_from_mapping(item, None))
        elif isinstance(item, (list, tuple)):
            geoms.append(_geometry_from_sequence(item))
        else:
            raise SpatialFilterError("参照几何必须是坐标，不能用地名", "invalid_input")
    if not geoms:
        raise SpatialFilterError("参照几何缺少可解析坐标", "invalid_input")
    return unary_union(geoms)


def _geometry_from_mapping(raw: dict[str, Any], ctx: RuntimeContext | None) -> BaseGeometry:
    kind = str(raw.get("type") or "").lower()
    if kind == "featurecollection":
        geoms = [
            _geometry_from_mapping(item, ctx)
            for item in raw.get("features") or []
            if isinstance(item, dict)
        ]
        if not geoms:
            raise SpatialFilterError("参照几何缺少可解析坐标", "invalid_input")
        return unary_union(geoms)
    if kind == "feature":
        geom = raw.get("geometry")
        if not isinstance(geom, dict):
            raise SpatialFilterError("参照几何缺少可解析坐标", "invalid_input")
        return _shape_required(geom, field="参照几何")
    if "coordinates" in raw and kind:
        return _shape_required(raw, field="参照几何")
    if all(key in raw for key in ("west", "south", "east", "north")):
        west = _as_float(raw["west"])
        south = _as_float(raw["south"])
        east = _as_float(raw["east"])
        north = _as_float(raw["north"])
        if None in (west, south, east, north):
            raise SpatialFilterError("bbox 坐标无效", "invalid_input")
        return shapely_box(west, south, east, north)
    lat = _as_float(raw.get("lat"))
    lon = _as_float(raw.get("lon", raw.get("lng")))
    if lat is not None and lon is not None:
        return shape({"type": "Point", "coordinates": [lon, lat]})
    if isinstance(raw.get("geometry"), dict):
        return _geometry_from_mapping(raw["geometry"], ctx)
    if isinstance(raw.get("features"), list):
        geoms = [
            _geometry_from_mapping(item, ctx)
            for item in raw["features"]
            if isinstance(item, dict)
        ]
        if not geoms:
            raise SpatialFilterError("参照几何缺少可解析坐标", "invalid_input")
        return unary_union(geoms)
    if isinstance(raw.get("elements"), list):
        geoms = []
        for item in raw["elements"]:
            if not isinstance(item, dict):
                continue
            geom = item.get("geometry")
            if isinstance(geom, dict):
                geoms.append(_shape_required(geom, field="参照几何"))
        if not geoms:
            raise SpatialFilterError("参照几何缺少可解析坐标", "invalid_input")
        return unary_union(geoms)
    nested = raw.get("result")
    if isinstance(nested, dict):
        return _geometry_from_mapping(nested, ctx)
    if isinstance(nested, list):
        return _geometry_from_sequence(nested)
    raise SpatialFilterError("参照几何必须是坐标，不能用地名", "invalid_input")


def _shape_required(raw: Any, *, field: str) -> BaseGeometry:
    if not isinstance(raw, dict):
        code = "unsupported_filter" if field == "源结果" else "invalid_input"
        if field == "源结果":
            raise SpatialFilterError("源结果缺少几何，无法执行空间筛选条件", code)
        raise SpatialFilterError("参照几何缺少可解析坐标", code)
    try:
        geom = shape(raw)
    except (ValueError, TypeError, AttributeError) as exc:
        if field == "源结果":
            raise SpatialFilterError("要素几何无法解析，无法执行空间筛选条件", "unsupported_filter") from exc
        raise SpatialFilterError("无法解析参照几何", "invalid_input") from exc
    if geom.is_empty:
        if field == "源结果":
            raise SpatialFilterError("源结果缺少几何，无法执行空间筛选条件", "unsupported_filter")
        raise SpatialFilterError("参照几何为空", "invalid_input")
    return geom


def _harmonize_crs(
    features: list[ParsedFeature],
    reference: ParsedRef,
) -> tuple[list[ParsedFeature], ParsedRef, str]:
    """把候选与参照转到可比较的 WGS84；无法转换则失败。"""

    converted: list[ParsedFeature] = []
    for item in features:
        converted.append(
            ParsedFeature(
                geometry=_to_wgs84(item.geometry, item.crs),
                properties=item.properties,
                crs=_CRS_WGS84,
            )
        )
    ref_geom = _to_wgs84(reference.geometry, reference.crs)
    return (
        converted,
        ParsedRef(geometry=ref_geom, crs=_CRS_WGS84, applied=dict(mapping(ref_geom))),
        _CRS_WGS84,
    )


def _to_wgs84(geom: BaseGeometry, crs_tag: str) -> BaseGeometry:
    normalized = _normalize_crs_tag(crs_tag)
    if normalized == _CRS_WGS84:
        return geom
    if is_gcj02(crs_tag):
        def _gcj_project(x: float, y: float, z: float | None = None) -> tuple[float, ...]:
            lon, lat = gcj02_to_wgs84(x, y)
            if z is None:
                return lon, lat
            return lon, lat, z

        try:
            projected = shapely_transform(_gcj_project, geom)
        except (ValueError, TypeError) as exc:
            raise SpatialFilterError(f"无法将 crs={crs_tag} 转到可比较坐标系", "invalid_input") from exc
        if projected.is_empty:
            raise SpatialFilterError(f"无法将 crs={crs_tag} 转到可比较坐标系", "invalid_input")
        return projected
    try:
        source = _crs_from_tag(crs_tag)
        transformer = Transformer.from_crs(source, CRS.from_epsg(4326), always_xy=True)
    except (ValueError, CRSError, ProjError) as exc:
        raise SpatialFilterError(f"无法将 crs={crs_tag} 转到可比较坐标系", "invalid_input") from exc

    def _project(x: float, y: float, z: float | None = None) -> tuple[float, ...]:
        lon, lat = transformer.transform(x, y)
        if z is None:
            return lon, lat
        return lon, lat, z

    try:
        projected = shapely_transform(_project, geom)
    except (ValueError, TypeError, CRSError, ProjError) as exc:
        raise SpatialFilterError(f"无法将 crs={crs_tag} 转到可比较坐标系", "invalid_input") from exc
    if projected.is_empty:
        raise SpatialFilterError(f"无法将 crs={crs_tag} 转到可比较坐标系", "invalid_input")
    return projected


def _crs_from_tag(tag: str) -> CRS:
    normalized = _normalize_crs_tag(tag)
    if normalized == _CRS_WGS84:
        return CRS.from_epsg(4326)
    text = tag.strip()
    lowered = text.lower().replace(" ", "")
    if lowered.startswith("epsg:"):
        code = lowered.split(":", 1)[1]
        if code.isdigit():
            return CRS.from_epsg(int(code))
    if lowered.isdigit():
        return CRS.from_epsg(int(lowered))
    return CRS.from_user_input(text)


def _apply_filter(
    features: list[ParsedFeature],
    reference: ParsedRef,
    *,
    relation: str,
    distance_m: float | None,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in features:
        matched, dist = _match_spatial(item.geometry, reference.geometry, relation, distance_m)
        if not matched:
            continue
        row: dict[str, Any] = {
            "geometry": mapping(item.geometry),
            "properties": item.properties,
            "relation": relation,
        }
        if dist is not None:
            row["distance_m"] = dist
        kept.append(row)
    return kept


def _match_spatial(
    geom: BaseGeometry,
    reference: BaseGeometry,
    relation: str,
    distance_m: float | None,
) -> tuple[bool, float | None]:
    if relation == "near":
        assert distance_m is not None
        dist = _aeqd_distance_m(geom, reference)
        return dist <= distance_m, dist
    if relation == "within":
        return bool(geom.within(reference)), None
    if relation == "intersects":
        return bool(geom.intersects(reference)), None
    if relation == "contains":
        return bool(geom.contains(reference)), None
    if relation == "crosses":
        return bool(geom.crosses(reference)), None
    raise SpatialFilterError(f"不支持的空间关系: {relation}", "invalid_input")


def _aeqd_distance_m(left: BaseGeometry, right: BaseGeometry) -> float:
    lon_0, lat_0 = _mid_centroid(left, right)
    projected_left = _project_aeqd(left, lon_0=lon_0, lat_0=lat_0)
    projected_right = _project_aeqd(right, lon_0=lon_0, lat_0=lat_0)
    return float(projected_left.distance(projected_right))


def _project_aeqd(geom: BaseGeometry, *, lon_0: float, lat_0: float) -> BaseGeometry:
    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)

    def _project(x: float, y: float, z: float | None = None) -> tuple[float, ...]:
        east, north = transformer.transform(x, y)
        if z is None:
            return east, north
        return east, north, z

    return shapely_transform(_project, geom)


def _mid_centroid(left: BaseGeometry, right: BaseGeometry) -> tuple[float, float]:
    lon = (float(left.centroid.x) + float(right.centroid.x)) / 2.0
    lat = (float(left.centroid.y) + float(right.centroid.y)) / 2.0
    return lon, lat


def _ok_filter(
    *,
    features: list[dict[str, Any]],
    original_count: int,
    relation: str,
    reference: ParsedRef,
    distance_m: float | None,
    crs: str,
    ctx: RuntimeContext | None,
) -> Observation:
    kept = len(features)
    result_id = _next_result_id(ctx)
    applied: dict[str, Any] = {
        "provider": _PROVIDER_LOCAL,
        "crs": crs,
        "relation": relation,
        "geometry": reference.applied,
    }
    if distance_m is not None:
        applied["distance_m"] = distance_m
    result: dict[str, Any] = {
        "operation": _OP_FILTER,
        "result_id": result_id,
        "features": features,
        "count": kept,
        "kept": kept,
        "removed": max(original_count - kept, 0),
        "applied": applied,
        "assumptions": [
            _LOCAL_ASSUMPTION,
            _NO_INVENT_ASSUMPTION,
            _AEQD_ASSUMPTION,
            _GEOM_NOT_LABEL_ASSUMPTION,
            _CRS_ASSUMPTION,
            _NO_SESSION_ASSUMPTION,
        ],
    }
    cleaned = _strip_forbidden(result)
    payload = cleaned if isinstance(cleaned, dict) else result
    _register_result(ctx, result_id, payload)
    return Observation(ok=True, result=payload)


def _next_result_id(ctx: RuntimeContext | None) -> str:
    if ctx is None:
        return "spatial_1"
    current = ctx.extras.get(_SEQ_KEY, 0)
    seq = current + 1 if isinstance(current, int) else 1
    ctx.extras[_SEQ_KEY] = seq
    return f"spatial_{seq}"


def _register_result(ctx: RuntimeContext | None, result_id: str, result: dict[str, Any]) -> None:
    if ctx is None:
        return
    registry = ctx.extras.get(_REGISTRY_KEY)
    if not isinstance(registry, dict):
        registry = {}
        ctx.extras[_REGISTRY_KEY] = registry
    registry[result_id] = result


def _normalize_crs_tag(tag: str | None) -> str:
    if tag is None or not str(tag).strip():
        return _CRS_WGS84
    text = str(tag).strip().lower()
    compact = text.replace(" ", "")
    if text in _WGS84_ALIASES or compact in _WGS84_ALIASES:
        return _CRS_WGS84
    if compact.startswith("urn:ogc:def:crs:epsg::") and compact.endswith("4326"):
        return _CRS_WGS84
    return text


def _crs_of(raw: dict[str, Any]) -> str | None:
    value = raw.get("crs", raw.get("datum"))
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        props = value.get("properties", value)
        if isinstance(props, dict):
            text = props.get("name", props.get("code"))
            if isinstance(text, str) and text.strip():
                return text.strip()
        name = value.get("name", value.get("code"))
        if isinstance(name, str) and name.strip():
            return name.strip()
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
