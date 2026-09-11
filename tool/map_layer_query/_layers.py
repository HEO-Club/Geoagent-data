"""map_layer_query 共享执行器：OGC WMS GetMap，可选 GeoServer WFS。"""

from __future__ import annotations

import base64
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from tool._gate import allow_real_api as _allow_real_api, load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext, declared_inputs
from tool.runtime.image_store import put_image

_PROVIDER_WMS = "wms"
_PROVIDER_ALIASES = frozenset({"wms", "geoserver", "ogc"})
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_WMS_VERSION = "1.3.0"
_DEFAULT_WFS_VERSION = "2.0.0"
_DEFAULT_CRS = "EPSG:4326"
_DEFAULT_WIDTH = 800
_DEFAULT_HEIGHT = 600
_DEFAULT_MAX_FEATURES = 200
_DEFAULT_UA = "geoagent-dataset/1.0 (map_layer_query; local)"
_METERS_PER_DEG_LAT = 111_320.0
_VECTOR_LAYERS = frozenset({"hydrology", "administrative", "roads"})
_HISTORICAL_LAYER = "historical_map"
_LAYER_ALIASES: dict[str, str] = {
    "hydrology": "hydrology",
    "water": "hydrology",
    "waters": "hydrology",
    "hydro": "hydrology",
    "river": "hydrology",
    "rivers": "hydrology",
    "水系": "hydrology",
    "河流": "hydrology",
    "terrain": "terrain",
    "dem": "terrain",
    "hillshade": "terrain",
    "elevation": "terrain",
    "地形": "terrain",
    "administrative": "administrative",
    "admin": "administrative",
    "boundary": "administrative",
    "boundaries": "administrative",
    "行政区": "administrative",
    "行政边界": "administrative",
    "roads": "roads",
    "road": "roads",
    "道路": "roads",
    "historical_map": "historical_map",
    "historical": "historical_map",
    "history": "historical_map",
    "历史地图": "historical_map",
    "历史底图": "historical_map",
}
_SEMANTIC_HINTS: dict[str, tuple[str, ...]] = {
    "hydrology": ("hydrology", "hydro", "water", "river", "水系", "河流"),
    "terrain": ("terrain", "dem", "hillshade", "elevation", "地形"),
    "administrative": ("administrative", "admin", "boundary", "行政区", "行政"),
    "roads": ("road", "transport", "道路"),
    "historical_map": ("historical", "history", "archive", "历史"),
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
_IMAGE_ASSUMPTION = (
    "WMS 图片不能用于量距、相交或空间筛选，也不得从截图读出精确坐标"
)
_VECTOR_ASSUMPTION = "需要可计算几何时使用本步 WFS 矢量，或改用 osm_query"
_CRS_ASSUMPTION = "当前后端未做坐标系转换"
_NO_SESSION_ASSUMPTION = "本步只返回图层，未打开街景或地图会话"

class LayerInputError(Exception):
    """area / layers / time_range 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """真实图层服务未配置、被闸门拒绝或调用失败。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code

@dataclass(frozen=True)
class GeoPoint:
    """WGS84 点；本执行器不转换坐标系。"""

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
class LayerRequest:
    """组装后的图层加载请求。"""

    layers: tuple[str, ...]
    bbox: BBox | None
    area_text: str | None = None
    time_range: str | None = None
    crs: str = _DEFAULT_CRS
    width: int = _DEFAULT_WIDTH
    height: int = _DEFAULT_HEIGHT
    max_features: int = _DEFAULT_MAX_FEATURES

@dataclass
class Capabilities:
    """WMS GetCapabilities 解析结果。"""

    layers: dict[str, str] = field(default_factory=dict)
    timed: set[str] = field(default_factory=set)

@runtime_checkable
class MapLayerProvider(Protocol):
    """可注入的图层后端；测试用 extras['map_layer_query_provider'] 替换。"""

    def load(self, request: LayerRequest) -> dict[str, Any]:
        """加载图层，返回可归一化的 JSON 对象。"""

class OgcWmsProvider:
    """OGC WMS GetMap 适配器，可选同一 GeoServer 的 WFS GetFeature。"""

    name = _PROVIDER_WMS
    crs = _DEFAULT_CRS

    def __init__(
        self,
        *,
        wms_endpoint: str,
        wfs_endpoint: str | None = None,
        wms_version: str = _DEFAULT_WMS_VERSION,
        wfs_version: str = _DEFAULT_WFS_VERSION,
        crs: str = _DEFAULT_CRS,
        layer_mapping: dict[str, str] | None = None,
        wfs_mapping: dict[str, str] | None = None,
        user: str | None = None,
        password: str | None = None,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
        user_agent: str = _DEFAULT_UA,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        max_features: int = _DEFAULT_MAX_FEATURES,
    ) -> None:
        self._wms_endpoint = wms_endpoint.rstrip("/")
        self._wfs_endpoint = wfs_endpoint.rstrip("/") if wfs_endpoint else None
        self._wms_version = wms_version
        self._wfs_version = wfs_version
        self.crs = crs
        self._layer_mapping = dict(layer_mapping or {})
        self._wfs_mapping = dict(wfs_mapping or {})
        self._user = user
        self._password = password
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent
        self._width = width
        self._height = height
        self._max_features = max_features
        self._capabilities: Capabilities | None = None

    def load(self, request: LayerRequest) -> dict[str, Any]:
        if request.bbox is None:
            raise LayerInputError(
                "WMS 需要 bbox 或中心点加半径；纯地名请先 geocode",
                "missing_input",
            )
        mapping = self._resolve_mapping(request.layers)
        unused_time = bool(
            request.time_range
            and any(not self._layer_has_time(wms_name) for wms_name in mapping.values())
        )
        layers_out: list[dict[str, Any]] = []
        for semantic, wms_name in mapping.items():
            png = self._get_map(wms_name, request, apply_time=not unused_time)
            layers_out.append(
                {
                    "name": semantic,
                    "provider_layer": wms_name,
                    "representation": "image",
                    "image_png": png,
                    "data_date": request.time_range if request.time_range and not unused_time else None,
                    "crs": request.crs,
                }
            )
            if self._wfs_endpoint and semantic in _VECTOR_LAYERS:
                features = self._get_feature(mapping[semantic], request)
                layers_out.append(
                    {
                        "name": semantic,
                        "provider_layer": self._wfs_layer_name(semantic, wms_name),
                        "representation": "vector",
                        "features": features,
                        "crs": request.crs,
                        "data_date": request.time_range if request.time_range and not unused_time else None,
                    }
                )
        payload: dict[str, Any] = {
            "layers": layers_out,
            "bbox": _bbox_applied(request.bbox, request.crs),
            "crs": request.crs,
            "wms_version": self._wms_version,
            "layer_mapping": mapping,
            "width": request.width,
            "height": request.height,
            "time": None if unused_time else request.time_range,
        }
        if unused_time and request.time_range:
            payload["unsupported_time"] = request.time_range
        return payload

    def _resolve_mapping(self, layers: tuple[str, ...]) -> dict[str, str]:
        catalog = dict(self._layer_mapping)
        if not catalog:
            catalog.update(self._capabilities_catalog().layers)
        resolved: dict[str, str] = {}
        missing: list[str] = []
        for semantic in layers:
            mapped = catalog.get(semantic)
            if not mapped:
                mapped = self._match_hint(semantic, catalog)
            if not mapped:
                missing.append(semantic)
                continue
            resolved[semantic] = mapped
        if missing:
            raise LayerInputError(
                "图层无法映射: " + ", ".join(missing),
                "unsupported_layer",
            )
        return resolved

    def _match_hint(self, semantic: str, catalog: dict[str, str]) -> str | None:
        hints = _SEMANTIC_HINTS.get(semantic, (semantic,))
        for _key, wms_name in catalog.items():
            haystack = f"{_key} {wms_name}".lower()
            if any(hint.lower() in haystack for hint in hints):
                return wms_name
        for wms_name in catalog.values():
            haystack = wms_name.lower()
            if any(hint.lower() in haystack for hint in hints):
                return wms_name
        return None

    def _capabilities_catalog(self) -> Capabilities:
        if self._capabilities is not None:
            return self._capabilities
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetCapabilities",
            "VERSION": self._wms_version,
        }
        raw = self._http(self._wms_endpoint, params, error_prefix="WMS GetCapabilities")
        caps = _parse_capabilities(raw)
        self._capabilities = caps
        return caps

    def _layer_has_time(self, wms_name: str) -> bool:
        if self._layer_mapping and self._capabilities is None:
            return True
        return wms_name in self._capabilities_catalog().timed

    def _get_map(self, wms_name: str, request: LayerRequest, *, apply_time: bool) -> bytes:
        assert request.bbox is not None
        params: dict[str, str] = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": self._wms_version,
            "LAYERS": wms_name,
            "STYLES": "",
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
            "WIDTH": str(request.width),
            "HEIGHT": str(request.height),
            "BBOX": _wms_bbox(request.bbox, request.crs, self._wms_version),
        }
        if self._wms_version.startswith("1.3"):
            params["CRS"] = request.crs
        else:
            params["SRS"] = request.crs
        if apply_time and request.time_range:
            params["TIME"] = request.time_range
        raw = self._http(self._wms_endpoint, params, error_prefix="WMS GetMap")
        if _looks_like_xml(raw):
            raise EngineUnavailableError(_xml_exception_message(raw, "WMS GetMap"))
        return raw

    def _get_feature(self, wms_name: str, request: LayerRequest) -> dict[str, Any]:
        assert self._wfs_endpoint is not None
        assert request.bbox is not None
        type_name = wms_name
        bbox = request.bbox
        params: dict[str, str] = {
            "SERVICE": "WFS",
            "REQUEST": "GetFeature",
            "VERSION": self._wfs_version,
            "OUTPUTFORMAT": "application/json",
            "BBOX": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north},{request.crs}",
        }
        if self._wfs_version.startswith("2"):
            params["TYPENAMES"] = type_name
            params["COUNT"] = str(request.max_features)
        else:
            params["TYPENAME"] = type_name
            params["MAXFEATURES"] = str(request.max_features)
        raw = self._http(self._wfs_endpoint, params, error_prefix="WFS GetFeature")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if _looks_like_xml(raw):
                raise EngineUnavailableError(_xml_exception_message(raw, "WFS GetFeature")) from exc
            raise EngineUnavailableError(f"WFS GetFeature 回执不是 JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise EngineUnavailableError("WFS GetFeature 回执不是 JSON 对象")
        return payload

    def _wfs_layer_name(self, semantic: str, wms_name: str) -> str:
        return self._wfs_mapping.get(semantic) or wms_name

    def _http(self, endpoint: str, params: dict[str, str], *, error_prefix: str) -> bytes:
        url = _append_query(endpoint, urllib.parse.urlencode(params))
        request = urllib.request.Request(url, method="GET")
        request.add_header("Accept", "*/*")
        request.add_header("User-Agent", self._user_agent)
        if self._user and self._password:
            token = base64.b64encode(f"{self._user}:{self._password}".encode("utf-8")).decode("ascii")
            request.add_header("Authorization", f"Basic {token}")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_sec) as response:
                return bytes(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:200] if exc.fp else b""
            text = detail.decode("utf-8", errors="replace")
            raise EngineUnavailableError(f"{error_prefix} HTTP {exc.code}: {text}") from exc
        except urllib.error.URLError as exc:
            raise EngineUnavailableError(f"{error_prefix} 网络失败: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise EngineUnavailableError(f"{error_prefix} 调用失败: {exc}") from exc

def execute_load_layer(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """加载指定区域的水系、地形、道路、行政区或历史地图图层。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "area", "layers", "time_range", "provider")
        _reject_unsupported_provider(inputs.get("provider"))
        layers = _parse_layers(inputs.get("layers"))
        if not layers:
            raise LayerInputError("缺少必填输入 layers", "missing_input")
        area_raw = _resolve_area(inputs.get("area"), ctx)
        if area_raw is None or area_raw == "":
            raise LayerInputError("缺少必填输入 area", "missing_input")
        area = _parse_area(area_raw)
        if (
            area.bbox is None
            and area.center is None
            and area.polygon is None
            and not area.text
        ):
            raise LayerInputError("area 无法解析为范围", "invalid_area")
        time_range = _parse_time_range(inputs.get("time_range"))
        bbox = _bbox_from_area(area)
        provider = _resolve_provider(ctx)
        if bbox is None and _is_ogc_provider(provider):
            raise LayerInputError(
                "WMS 需要 bbox 或中心点加半径；纯地名请先 geocode",
                "missing_input",
            )
        crs = _provider_crs(provider)
        width, height = _map_size()
        max_features = _max_features()
        request = LayerRequest(
            layers=layers,
            bbox=bbox,
            area_text=area.text,
            time_range=time_range,
            crs=crs,
            width=width,
            height=height,
            max_features=max_features,
        )
        payload = provider.load(request)
        return _ok(payload, request=request, area=area, ctx=ctx, provider=provider)
    except LayerInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def _ok(
    payload: dict[str, Any],
    *,
    request: LayerRequest,
    area: ParsedArea,
    ctx: RuntimeContext | None,
    provider: MapLayerProvider,
) -> Observation:
    provider_name = _provider_name(provider)
    crs = str(payload.get("crs") or request.crs)
    bbox_payload = payload.get("bbox")
    bbox = bbox_payload if isinstance(bbox_payload, dict) else (
        _bbox_applied(request.bbox, crs) if request.bbox is not None else None
    )
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise EngineUnavailableError("图层回执缺少 layers")
    numbered: list[dict[str, Any]] = []
    image_ids: list[str] = []
    image_paths: dict[str, str] = {}
    mapping: dict[str, str] = {}
    raw_mapping = payload.get("layer_mapping")
    if isinstance(raw_mapping, dict):
        mapping = {str(key): str(value) for key, value in raw_mapping.items()}
    for item in raw_layers:
        if not isinstance(item, dict):
            continue
        row, image_id, image_path = _normalize_layer(
            item,
            index=len(numbered) + 1,
            crs=crs,
            max_features=request.max_features,
            ctx=ctx,
        )
        numbered.append(row)
        if image_id:
            image_ids.append(image_id)
            if image_path:
                image_paths[image_id] = image_path
        provider_layer = row.get("provider_layer")
        name = row.get("name")
        if isinstance(name, str) and isinstance(provider_layer, str):
            mapping.setdefault(name, provider_layer)
    if not numbered:
        raise EngineUnavailableError("图层回执没有可归一化的图层")
    applied: dict[str, Any] = {
        "provider": provider_name,
        "crs": crs,
        "width": payload.get("width") or request.width,
        "height": payload.get("height") or request.height,
        "max_features": request.max_features,
        "layers": mapping,
    }
    version = payload.get("wms_version")
    if version:
        applied["wms_version"] = version
    time_value = payload.get("time")
    if time_value:
        applied["time"] = time_value
    elif request.time_range:
        applied["time"] = request.time_range
    if area.applied is not None:
        applied["area"] = area.applied
    unsupported: dict[str, Any] = {}
    if payload.get("unsupported_time"):
        unsupported["time_range"] = payload.get("unsupported_time")
    if area.unsupported is not None:
        unsupported["area"] = area.unsupported
    if unsupported:
        applied["unsupported"] = unsupported
    result: dict[str, Any] = {
        "operation": "load_layer",
        "layers": numbered,
        "bbox": bbox,
        "applied": applied,
        "assumptions": _assumptions(has_vector=any(
            row.get("representation") == "vector" for row in numbered
        )),
    }
    artifacts: dict[str, Any] = {}
    if image_ids:
        artifacts["image_ids"] = image_ids
        artifacts["images"] = image_paths
    return Observation(ok=True, result=_strip_forbidden(result), artifacts=artifacts)

def _normalize_layer(
    item: dict[str, Any],
    *,
    index: int,
    crs: str,
    max_features: int,
    ctx: RuntimeContext | None,
) -> tuple[dict[str, Any], str | None, str | None]:
    name = str(item.get("name") or f"layer_{index}")
    representation = str(item.get("representation") or "image")
    provider_layer = item.get("provider_layer")
    row: dict[str, Any] = {
        "result_id": f"layer_{index}",
        "name": name,
        "provider_layer": provider_layer if isinstance(provider_layer, str) else name,
        "representation": representation,
        "usable_for_gis": bool(item.get("usable_for_gis"))
        if "usable_for_gis" in item
        else representation in {"vector", "raster"},
        "crs": str(item.get("crs") or crs),
        "data_date": item.get("data_date"),
    }
    if representation == "image":
        row["usable_for_gis"] = False
        image_id, image_path = _store_layer_image(item, name=name, ctx=ctx)
        row["image_id"] = image_id
        return row, image_id, image_path
    if representation == "vector":
        features, count = _truncate_features(item.get("features"), max_features)
        row["features"] = features
        row["feature_count"] = int(item.get("feature_count") or count)
        row["usable_for_gis"] = True
        return row, None, None
    if representation == "raster":
        row["usable_for_gis"] = True
        image_id, image_path = _store_layer_image(item, name=name, ctx=ctx)
        if image_id:
            row["image_id"] = image_id
        return row, image_id, image_path
    return row, None, None

def _store_layer_image(
    item: dict[str, Any],
    *,
    name: str,
    ctx: RuntimeContext | None,
) -> tuple[str | None, str | None]:
    image = item.get("image")
    png = item.get("image_png")
    pil: Image.Image | None = None
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(png, (bytes, bytearray)):
        pil = Image.open(BytesIO(bytes(png)))
        pil.load()
    if pil is None:
        existing = item.get("image_id")
        if isinstance(existing, str) and existing.strip():
            return existing.strip(), None
        raise EngineUnavailableError(f"图层 {name} 缺少图片数据")
    image_id, path = put_image(pil, source_id=name, suffix="png", ctx=ctx)
    return image_id, str(path)

def _truncate_features(raw: Any, limit: int) -> tuple[Any, int]:
    if not isinstance(raw, dict):
        return _strip_forbidden(raw), 0
    features = raw.get("features")
    if not isinstance(features, list):
        return _strip_forbidden(raw), 0
    clipped = features[: max(0, limit)]
    payload = dict(raw)
    payload["features"] = clipped
    payload.pop("raw_content", None)
    return _strip_forbidden(payload), len(clipped)

def _assumptions(*, has_vector: bool) -> list[str]:
    items = [_IMAGE_ASSUMPTION, _VECTOR_ASSUMPTION, _CRS_ASSUMPTION, _NO_SESSION_ASSUMPTION]
    if not has_vector:
        items[1] = "本步仅为图层可视化；需要可计算几何请用 osm_query 或配置 WFS"
    return items

def _parse_layers(raw: Any) -> tuple[str, ...]:
    if raw is None or raw == "":
        return ()
    values: list[str]
    if isinstance(raw, str):
        values = [part.strip() for part in raw.replace("，", ",").split(",") if part.strip()]
        if len(values) <= 1:
            values = [raw.strip()] if raw.strip() else []
    elif isinstance(raw, (list, tuple)):
        values = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise LayerInputError("layers 必须是字符串或字符串列表", "invalid_layers")
            values.append(item.strip())
    else:
        raise LayerInputError("layers 必须是字符串或字符串列表", "invalid_layers")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        semantic = _LAYER_ALIASES.get(item.lower(), _LAYER_ALIASES.get(item, item.lower()))
        if semantic not in seen:
            seen.add(semantic)
            normalized.append(semantic)
    return tuple(normalized)

def _parse_time_range(raw: Any) -> str | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        return raw.strip() or None
    if isinstance(raw, dict):
        start = _optional_str(raw.get("start") or raw.get("from") or raw.get("begin"))
        end = _optional_str(raw.get("end") or raw.get("to"))
        if start and end:
            return f"{start}/{end}"
        return start or end
    raise LayerInputError("time_range 必须是字符串或对象", "invalid_time_range")

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
        return ParsedArea(text=text, applied=text)
    if isinstance(raw, (list, tuple)):
        polygon = _polygon_from_points(raw)
        if polygon is not None:
            return ParsedArea(polygon=polygon, applied=_polygon_applied(polygon))
        if len(raw) == 4:
            bbox = _bbox_from_values(list(raw))
            if bbox is not None:
                return ParsedArea(bbox=bbox, applied=_bbox_applied(bbox, _DEFAULT_CRS))
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
            return ParsedArea(bbox=bbox, applied=_bbox_applied(bbox, _DEFAULT_CRS))
        center = _center_from_mapping(raw)
        if center is not None:
            radius_raw = raw.get("radius_m", raw.get("radius"))
            radius: int | None = None
            if radius_raw is not None and radius_raw != "":
                parsed = _as_float(radius_raw)
                if parsed is None or parsed < 0:
                    raise LayerInputError("radius_m 必须是非负数", "invalid_area")
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

def _bbox_from_area(area: ParsedArea) -> BBox | None:
    if area.bbox is not None:
        return area.bbox
    if area.polygon is not None:
        lons = [point[0] for point in area.polygon]
        lats = [point[1] for point in area.polygon]
        return BBox(west=min(lons), south=min(lats), east=max(lons), north=max(lats))
    if area.center is not None and area.radius_m is not None:
        return _bbox_from_center(area.center, area.radius_m)
    return None

def _bbox_from_center(center: GeoPoint, radius_m: int) -> BBox:
    lat_delta = radius_m / _METERS_PER_DEG_LAT
    cos_lat = math.cos(math.radians(center.lat))
    lon_delta = radius_m / (_METERS_PER_DEG_LAT * max(abs(cos_lat), 1e-6))
    return BBox(
        west=max(-180.0, center.lon - lon_delta),
        south=max(-90.0, center.lat - lat_delta),
        east=min(180.0, center.lon + lon_delta),
        north=min(90.0, center.lat + lat_delta),
    )

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

def _polygon_from_points(raw: Any) -> tuple[tuple[float, float], ...] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
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
    if len(points) < 2:
        return None
    return tuple(points)

def _reject_unsupported_provider(raw: Any) -> None:
    if raw is None or raw == "":
        return
    if not isinstance(raw, str):
        raise LayerInputError("provider 必须是字符串", "invalid_provider")
    name = raw.strip().lower()
    if name in _PROVIDER_ALIASES:
        return
    raise EngineUnavailableError(f"未知 map_layer provider: {raw}", "unsupported_provider")

def _resolve_provider(ctx: RuntimeContext | None) -> MapLayerProvider:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("map_layer_query_provider")
    if injected is not None:
        if not isinstance(injected, MapLayerProvider):
            raise EngineUnavailableError("map_layer_query_provider 必须提供 load(request)")
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_API=false，禁止调用真实图层 API")
    name = _env_value("MAP_LAYER_PROVIDER", _PROVIDER_WMS).lower()
    if name not in _PROVIDER_ALIASES:
        raise EngineUnavailableError(f"未知 MAP_LAYER_PROVIDER: {name}")
    return _build_wms_provider()

def _build_wms_provider() -> OgcWmsProvider:
    endpoint = os.environ.get("MAP_LAYER_WMS_ENDPOINT", "").strip()
    if not endpoint:
        raise EngineUnavailableError("未配置 MAP_LAYER_WMS_ENDPOINT")
    wfs = os.environ.get("MAP_LAYER_WFS_ENDPOINT", "").strip() or None
    user = os.environ.get("MAP_LAYER_WMS_USER", "").strip() or None
    password = os.environ.get("MAP_LAYER_WMS_PASSWORD", "").strip() or None
    mapping = _parse_layer_mapping(os.environ.get("MAP_LAYER_WMS_LAYERS", "").strip())
    wfs_mapping = _parse_layer_mapping(os.environ.get("MAP_LAYER_WFS_LAYERS", "").strip())
    width, height = _map_size()
    return OgcWmsProvider(
        wms_endpoint=endpoint,
        wfs_endpoint=wfs,
        wms_version=_env_value("MAP_LAYER_WMS_VERSION", _DEFAULT_WMS_VERSION),
        wfs_version=_env_value("MAP_LAYER_WFS_VERSION", _DEFAULT_WFS_VERSION),
        crs=_env_value("MAP_LAYER_WMS_CRS", _DEFAULT_CRS),
        layer_mapping=mapping,
        wfs_mapping=wfs_mapping,
        user=user,
        password=password,
        timeout_sec=_env_timeout("MAP_LAYER_TIMEOUT_SEC"),
        user_agent=_env_value("MAP_LAYER_WMS_USER_AGENT", _DEFAULT_UA),
        width=width,
        height=height,
        max_features=_max_features(),
    )

def _parse_layer_mapping(raw: str) -> dict[str, str]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EngineUnavailableError(f"MAP_LAYER_WMS_LAYERS 不是合法 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EngineUnavailableError("图层映射必须是 JSON 对象")
    mapping: dict[str, str] = {}
    for key, value in payload.items():
        semantic = _LAYER_ALIASES.get(str(key).lower(), str(key).lower())
        if isinstance(value, str) and value.strip():
            mapping[semantic] = value.strip()
    return mapping

def _map_size() -> tuple[int, int]:
    width = _as_int(os.environ.get("MAP_LAYER_WMS_WIDTH", "").strip()) or _DEFAULT_WIDTH
    height = _as_int(os.environ.get("MAP_LAYER_WMS_HEIGHT", "").strip()) or _DEFAULT_HEIGHT
    return max(1, width), max(1, height)

def _max_features() -> int:
    parsed = _as_int(os.environ.get("MAP_LAYER_MAX_FEATURES", "").strip())
    if parsed is None or parsed <= 0:
        return _DEFAULT_MAX_FEATURES
    return parsed

def _is_ogc_provider(provider: MapLayerProvider) -> bool:
    if isinstance(provider, OgcWmsProvider):
        return True
    return _provider_name(provider) in _PROVIDER_ALIASES

def _provider_name(provider: MapLayerProvider) -> str:
    return str(getattr(provider, "name", "injected"))

def _provider_crs(provider: MapLayerProvider) -> str:
    raw = getattr(provider, "crs", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return _DEFAULT_CRS

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

def _parse_capabilities(raw: bytes) -> Capabilities:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise EngineUnavailableError(f"WMS GetCapabilities 不是合法 XML: {exc}") from exc
    caps = Capabilities()
    for layer in root.iter():
        if _local_tag(layer.tag) != "Layer":
            continue
        name_el = next((child for child in list(layer) if _local_tag(child.tag) == "Name"), None)
        title_el = next((child for child in list(layer) if _local_tag(child.tag) == "Title"), None)
        name = (name_el.text or "").strip() if name_el is not None else ""
        title = (title_el.text or "").strip() if title_el is not None else ""
        if not name:
            continue
        semantic = _LAYER_ALIASES.get(name.lower()) or _LAYER_ALIASES.get(title.lower())
        if semantic:
            caps.layers.setdefault(semantic, name)
        caps.layers.setdefault(name, name)
        if title:
            caps.layers.setdefault(title.lower(), name)
        for child in layer:
            if _local_tag(child.tag) in {"Dimension", "Extent"}:
                dim_name = (child.attrib.get("name") or "").lower()
                if dim_name == "time":
                    caps.timed.add(name)
    return caps

def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag

def _wms_bbox(bbox: BBox, crs: str, version: str) -> str:
    geographic = crs.upper() in {"EPSG:4326", "CRS:84"}
    if version.startswith("1.3") and crs.upper() == "EPSG:4326":
        return f"{bbox.south},{bbox.west},{bbox.north},{bbox.east}"
    if geographic:
        return f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}"
    return f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}"

def _looks_like_xml(raw: bytes) -> bool:
    stripped = raw.lstrip()
    return stripped.startswith(b"<?xml") or stripped.startswith(b"<")

def _xml_exception_message(raw: bytes, prefix: str) -> str:
    try:
        root = ET.fromstring(raw)
        text = "".join(root.itertext()).strip()
        if text:
            return f"{prefix} 失败: {text[:200]}"
    except ET.ParseError:
        pass
    return f"{prefix} 返回了 XML 异常"

def _append_query(endpoint: str, query: str) -> str:
    separator = "&" if urllib.parse.urlparse(endpoint).query else "?"
    return f"{endpoint}{separator}{query}"

def _bbox_applied(bbox: BBox, crs: str) -> dict[str, float | str]:
    return {
        "west": bbox.west,
        "south": bbox.south,
        "east": bbox.east,
        "north": bbox.north,
        "crs": crs,
    }

def _polygon_applied(points: tuple[tuple[float, float], ...]) -> list[list[float]]:
    return [[lon, lat] for lon, lat in points]

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
