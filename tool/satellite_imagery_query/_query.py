"""satellite_imagery_query 共享执行器：Copernicus Catalog 检索 + Process API 预览。"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from tool._gate import allow_real_api as _allow_real_api, load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext, declared_inputs
from tool.runtime.image_store import put_image

_PROVIDER_COPERNICUS = "copernicus"
_CRS_WGS84 = "wgs84"
_OP_RETRIEVE = "retrieve"
_OP_CHANGE_TIME = "change_time"
_OP_OBLIQUE = "oblique_view"
_VIEW_NADIR = "nadir"
_VIEW_NADIR_PREVIEW = "nadir_preview"
_DEFAULT_CATALOG = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
_DEFAULT_PROCESS = "https://sh.dataspace.copernicus.eu/api/v1/process"
_DEFAULT_TOKEN = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_UA = "geoagent-dataset/1.0 (satellite_imagery_query; local)"
_DEFAULT_SEARCH_RADIUS_M = 2000
_SEARCH_RADIUS_MIN = 50
_SEARCH_RADIUS_MAX = 50_000
_DEFAULT_PREVIEW_WIDTH = 512
_DEFAULT_PREVIEW_HEIGHT = 512
_DEFAULT_SCENE_LIMIT = 50
_LIMIT_MIN = 1
_LIMIT_MAX = 100
_METERS_PER_DEG_LAT = 111_320.0
_YEAR_MIN = 1000
_YEAR_MAX = 2100
_SENTINEL2_START = date(2015, 6, 23)
_COLLECTION_S2 = "sentinel-2-l2a"
_COLLECTION_LANDSAT_OT = "landsat-ot-l2"
_COLLECTION_LANDSAT_ETM = "landsat-etm-l2"
_COLLECTION_LANDSAT_TM = "landsat-tm-l2"
_COLLECTION_GSD: dict[str, float] = {
    _COLLECTION_S2: 10.0,
    _COLLECTION_LANDSAT_OT: 30.0,
    _COLLECTION_LANDSAT_ETM: 30.0,
    _COLLECTION_LANDSAT_TM: 30.0,
}
_COLLECTION_START: dict[str, date] = {
    _COLLECTION_S2: _SENTINEL2_START,
    _COLLECTION_LANDSAT_OT: date(2013, 2, 11),
    _COLLECTION_LANDSAT_ETM: date(1999, 4, 15),
    _COLLECTION_LANDSAT_TM: date(1982, 8, 22),
}
_LAYER_SATELLITE = "satellite"
_LAYER_ALIASES: dict[str, str] = {
    "satellite": _LAYER_SATELLITE,
    "sentinel": _LAYER_SATELLITE,
    "sentinel-2": _LAYER_SATELLITE,
    "sentinel2": _LAYER_SATELLITE,
    "s2": _LAYER_SATELLITE,
    "optical": _LAYER_SATELLITE,
    "卫星": _LAYER_SATELLITE,
    "遥感": _LAYER_SATELLITE,
    "aerial": "aerial",
    "airphoto": "aerial",
    "ortho": "aerial",
    "航片": "aerial",
    "航空": "aerial",
    "航空影像": "aerial",
    "terrain": "terrain",
    "dem": "terrain",
    "地形": "terrain",
}
_COPERNICUS_ALIASES = frozenset(
    {
        "copernicus",
        "cdse",
        "sentinel",
        "sentinel-hub",
        "sentinelhub",
        "sentinel-2",
        "sentinel2",
    }
)
_PLANET_ALIASES = frozenset({"planet", "planet.com", "planetscope", "skysat"})
_GOOGLE_ALIASES = frozenset(
    {"google", "google earth", "google_earth", "googleearth", "earth"}
)
_EARLIEST_ALIASES = frozenset(
    {"earliest", "earliest available", "最早", "最早可用", "最早的"}
)
_LATEST_ALIASES = frozenset({"latest", "newest", "最晚", "最新", "最晚可用"})
_SESSIONS_KEY = "satellite_sessions"
_SESSION_COUNTER_KEY = "satellite_session_counter"
_YEAR_RE = re.compile(r"^(\d{4})$")
_YEAR_RANGE_RE = re.compile(r"^(\d{4})\s*[-–—至到]\s*(\d{4})$")
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
_DATE_RANGE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s*(?:to|/|[-–—至到])\s*(\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)
_YEAR_IN_TEXT_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")
_COORD_SPLIT_RE = re.compile(r"[,;，、\s]+")
_SEASON_SPECS: tuple[tuple[frozenset[str], int, int], ...] = (
    (frozenset({"wet", "wet season", "monsoon", "wetseason", "湿季", "汛期", "雨季"}), 6, 9),
    (frozenset({"dry", "dry season", "dryseason", "旱季", "枯水期"}), 12, 3),
    (frozenset({"summer", "夏季"}), 6, 8),
    (frozenset({"winter", "冬季"}), 12, 2),
    (frozenset({"spring", "春季"}), 3, 5),
    (frozenset({"autumn", "fall", "秋季"}), 9, 11),
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
_TRUE_COLOR_EVALSCRIPT = """//VERSION=3
function setup() {
  return { input: ["B04", "B03", "B02"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
}
"""
_LANDSAT_TM_EVALSCRIPT = """//VERSION=3
function setup() {
  return { input: ["B03", "B02", "B01"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  return [2.5 * sample.B03, 2.5 * sample.B02, 2.5 * sample.B01];
}
"""
_ASSUMPTIONS = [
    "resolution_m 只筛选原生空间分辨率，不会把 10 m 哨兵影像写成旗杆或小电塔级细节",
    "Sentinel-2 L2A 约 2015 年起、原生约 10 m；更早时相尝试 Landsat（约 30 m）",
    "目录无命中时如实无覆盖，不会用近景顶替所请求的历史年份",
    "当前仅接入 Copernicus Data Space，未接 Planet 或 Google Earth",
    "不下载 SAFE 全产品，只返回 Catalog 元数据与 Process API 裁剪真彩预览",
    "纯地名未在本步 geocode，需要 bbox 或中心点加半径",
]

class SatelliteInputError(Exception):
    """area / coordinates / time_range / layer 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """真实影像服务未配置、被闸门拒绝或调用失败。"""

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
class TimeWindow:
    """闭区间日期窗。"""

    start: date
    end: date

@dataclass(frozen=True)
class TimeSpec:
    """解析后的时间条件。"""

    window: TimeWindow | None = None
    earliest: bool = False
    latest: bool = False
    applied: Any = None
    season: str | None = None

@dataclass(frozen=True)
class SceneSearchRequest:
    """目录检索请求。"""

    bbox: BBox
    collections: tuple[str, ...]
    datetime: str | None = None
    cloud_cover_max: float | None = None
    limit: int = _DEFAULT_SCENE_LIMIT

@dataclass(frozen=True)
class SceneRecord:
    """归一化后的一景目录命中。"""

    scene_id: str
    captured_at: str
    captured_on: date
    collection: str
    resolution_m: float
    cloud_cover: float | None = None
    bbox: BBox | None = None

@dataclass
class SatelliteSession:
    """一次卫星影像会话，供 change_time / source_result 复用。"""

    session_id: str
    provider: str
    bbox: BBox
    coverage: bool = False
    scene_id: str | None = None
    captured_at: str | None = None
    resolution_m: float | None = None
    cloud_cover: float | None = None
    collection: str | None = None
    image_id: str | None = None
    layer: str = _LAYER_SATELLITE

@dataclass(frozen=True)
class PreviewRequest:
    """裁剪真彩预览请求。"""

    bbox: BBox
    scene: SceneRecord
    width: int
    height: int

@runtime_checkable
class SatelliteImageryProvider(Protocol):
    """可注入的影像后端；测试用 extras['satellite_imagery_query_provider'] 替换。"""

    def search_scenes(self, request: SceneSearchRequest) -> list[dict[str, Any]]:
        """按空间/时间/云量检索目录，返回可归一化的 JSON 对象列表。"""

    def fetch_preview(self, request: PreviewRequest) -> bytes:
        """按选中场景与 bbox 返回 PNG/JPEG 预览字节。"""

class CopernicusProvider:
    """Copernicus Data Space Catalog + Process API 适配器。"""

    name = _PROVIDER_COPERNICUS
    crs = _CRS_WGS84

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        catalog_endpoint: str,
        process_endpoint: str,
        token_endpoint: str,
        timeout_sec: float,
        user_agent: str,
        preview_width: int,
        preview_height: int,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._catalog_endpoint = catalog_endpoint
        self._process_endpoint = process_endpoint
        self._token_endpoint = token_endpoint
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent
        self._preview_width = preview_width
        self._preview_height = preview_height
        self._token: str | None = None
        self._token_expires_at: datetime | None = None

    def search_scenes(self, request: SceneSearchRequest) -> list[dict[str, Any]]:
        if not request.collections:
            return []
        body: dict[str, Any] = {
            "collections": list(request.collections),
            "bbox": [request.bbox.west, request.bbox.south, request.bbox.east, request.bbox.north],
            "limit": request.limit,
        }
        if request.datetime:
            body["datetime"] = request.datetime
        if request.cloud_cover_max is not None:
            body["query"] = {"eo:cloud_cover": {"lt": request.cloud_cover_max}}
        payload = self._post_json(
            self._catalog_endpoint,
            body,
            error_prefix="Copernicus Catalog",
        )
        features = payload.get("features")
        if not isinstance(features, list):
            return []
        return [item for item in features if isinstance(item, dict)]

    def fetch_preview(self, request: PreviewRequest) -> bytes:
        bbox = request.bbox
        width = request.width or self._preview_width
        height = request.height or self._preview_height
        start, end = _scene_process_window(request.scene)
        evalscript = (
            _LANDSAT_TM_EVALSCRIPT
            if request.scene.collection == _COLLECTION_LANDSAT_TM
            else _TRUE_COLOR_EVALSCRIPT
        )
        body: dict[str, Any] = {
            "input": {
                "bounds": {
                    "bbox": [bbox.west, bbox.south, bbox.east, bbox.north],
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [
                    {
                        "type": request.scene.collection,
                        "dataFilter": {
                            "timeRange": {"from": start, "to": end},
                            "maxCloudCoverage": 100,
                        },
                    }
                ],
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
            },
            "evalscript": evalscript,
        }
        return self._post_bytes(
            self._process_endpoint,
            body,
            error_prefix="Copernicus Process",
            accept="image/png",
        )

    def _auth_header(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token()}",
            "User-Agent": self._user_agent,
        }

    def _access_token(self) -> str:
        now = datetime.now(timezone.utc)
        if self._token and self._token_expires_at and now < self._token_expires_at:
            return self._token
        form = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            }
        ).encode("utf-8")
        raw = _http_bytes(
            self._token_endpoint,
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "User-Agent": self._user_agent,
            },
            body=form,
            timeout_sec=self._timeout_sec,
            error_prefix="Copernicus OAuth",
        )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise EngineUnavailableError("Copernicus OAuth 回执不是 JSON") from exc
        if not isinstance(payload, dict):
            raise EngineUnavailableError("Copernicus OAuth 回执不是 JSON 对象")
        token = payload.get("access_token")
        if not isinstance(token, str) or not token.strip():
            raise EngineUnavailableError("Copernicus OAuth 未返回 access_token")
        expires = _as_float(payload.get("expires_in")) or 3600.0
        self._token = token.strip()
        self._token_expires_at = now + timedelta(seconds=max(60.0, expires - 60.0))
        return self._token

    def _post_json(self, url: str, body: dict[str, Any], *, error_prefix: str) -> dict[str, Any]:
        raw = self._post_bytes(url, body, error_prefix=error_prefix, accept="application/json")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise EngineUnavailableError(f"{error_prefix} 回执不是 JSON") from exc
        if not isinstance(payload, dict):
            raise EngineUnavailableError(f"{error_prefix} 回执不是 JSON 对象")
        return payload

    def _post_bytes(
        self,
        url: str,
        body: dict[str, Any],
        *,
        error_prefix: str,
        accept: str,
    ) -> bytes:
        headers = self._auth_header()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = accept
        return _http_bytes(
            url,
            method="POST",
            headers=headers,
            body=json.dumps(body).encode("utf-8"),
            timeout_sec=self._timeout_sec,
            error_prefix=error_prefix,
        )

def execute_retrieve(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """获取指定区域和时间的卫星或航片。"""

    del purpose
    try:
        inputs = declared_inputs(
            inputs,
            "area",
            "coordinates",
            "source_result",
            "time_range",
            "provider",
            "layer",
            "cloud_cover_max",
            "resolution_m",
        )
        _reject_unsupported_provider(inputs.get("provider"))
        location = _resolve_search_location(inputs, ctx, require_geometry=True)
        return _run_query(
            _OP_RETRIEVE,
            inputs=inputs,
            ctx=ctx,
            location=location,
            reuse_session=None,
            heading=None,
            tilt=None,
        )
    except SatelliteInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def execute_change_time(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """在同一区域切换历史年份、季节或水期影像。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "area", "time_range", "source_result", "provider")
        _reject_unsupported_provider(inputs.get("provider"))
        if inputs.get("area") in (None, ""):
            raise SatelliteInputError("缺少必填输入 area", "missing_input")
        if inputs.get("time_range") in (None, ""):
            raise SatelliteInputError("缺少必填输入 time_range", "missing_input")
        session, location = _resolve_session_or_location(inputs, ctx, area_required=True)
        return _run_query(
            _OP_CHANGE_TIME,
            inputs=inputs,
            ctx=ctx,
            location=location,
            reuse_session=session,
            heading=None,
            tilt=None,
        )
    except SatelliteInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def execute_oblique_view(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """返回同场景正射预览与相机参数，不是真斜摄也不是 Cesium 三维渲染。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "area", "heading", "tilt", "time_range")
        if inputs.get("area") in (None, ""):
            raise SatelliteInputError("缺少必填输入 area", "missing_input")
        heading = _parse_angle(inputs.get("heading"), "heading", 0.0, 360.0)
        tilt = _parse_angle(inputs.get("tilt"), "tilt", 0.0, 90.0)
        location = _resolve_search_location(inputs, ctx, require_geometry=True)
        return _run_query(
            _OP_OBLIQUE,
            inputs=inputs,
            ctx=ctx,
            location=location,
            reuse_session=None,
            heading=heading,
            tilt=tilt,
        )
    except SatelliteInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def _run_query(
    operation: str,
    *,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    location: "_SearchLocation",
    reuse_session: SatelliteSession | None,
    heading: float | None,
    tilt: float | None,
) -> Observation:
    layer = _parse_layer(inputs.get("layer"))
    time_spec = _optional_time_spec(inputs.get("time_range"), required=operation == _OP_CHANGE_TIME)
    cloud_max = _parse_cloud_cover_max(inputs.get("cloud_cover_max"))
    resolution_m = _parse_resolution_m(inputs.get("resolution_m"))
    collections = _select_collections(layer=layer, time_spec=time_spec, resolution_m=resolution_m)
    provider = _resolve_provider(ctx)
    records: list[SceneRecord] = []
    if collections:
        raw_scenes = provider.search_scenes(
            SceneSearchRequest(
                bbox=location.bbox,
                collections=collections,
                datetime=_catalog_datetime(time_spec),
                cloud_cover_max=cloud_max,
                limit=_scene_limit(),
            )
        )
        records = [
            record
            for record in (_normalize_scene(item) for item in raw_scenes)
            if record is not None and _scene_matches(record, time_spec, cloud_max, resolution_m)
        ]
    chosen = _select_scene(records, time_spec)
    session = reuse_session or _new_session(
        ctx,
        provider_name=_provider_name(provider),
        bbox=location.bbox,
        layer=layer,
    )
    session.bbox = location.bbox
    session.layer = layer
    image_id: str | None = None
    artifacts: dict[str, Any] = {}
    if chosen is None:
        _clear_scene(session)
    else:
        _apply_scene(session, chosen)
        preview = provider.fetch_preview(
            PreviewRequest(
                bbox=location.bbox,
                scene=chosen,
                width=_preview_width(),
                height=_preview_height(),
            )
        )
        if not preview:
            raise EngineUnavailableError("影像预览为空")
        pil = Image.open(BytesIO(preview))
        pil.load()
        image_id, path = put_image(pil, source_id=chosen.scene_id, suffix="png", ctx=ctx)
        session.image_id = image_id
        artifacts = {"image_ids": [image_id], "images": {image_id: str(path)}}
    _store_session(ctx, session)
    extra: dict[str, Any] = {
        "layer": layer,
        "collections": list(collections),
    }
    if time_spec is not None:
        extra["time_range"] = time_spec.applied
        if time_spec.season:
            extra["season"] = time_spec.season
    if cloud_max is not None:
        extra["cloud_cover_max"] = cloud_max
    if resolution_m is not None:
        extra["resolution_m_filter"] = resolution_m
    if heading is not None:
        extra["heading"] = heading
    if tilt is not None:
        extra["tilt"] = tilt
    available_dates = sorted({record.captured_at for record in records})
    return _ok(
        operation,
        session,
        coverage=chosen is not None,
        applied=_applied(provider, location=location, extra=extra),
        ctx=ctx,
        image_id=image_id,
        artifacts=artifacts,
        available_dates=available_dates if operation == _OP_CHANGE_TIME else None,
        heading=heading,
        tilt=tilt,
    )

@dataclass(frozen=True)
class _SearchLocation:
    bbox: BBox
    center: GeoPoint
    radius_m: int
    area: ParsedArea

def _resolve_session_or_location(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    area_required: bool,
) -> tuple[SatelliteSession, _SearchLocation]:
    session = _load_session(inputs, ctx, required=False)
    has_area = _has_location_input(inputs)
    if session is None and not has_area:
        if area_required:
            raise SatelliteInputError("缺少必填输入 area", "missing_input")
        raise SatelliteInputError("需要 area、coordinates 或 source_result", "missing_input")
    if has_area:
        location = _resolve_search_location(inputs, ctx, require_geometry=True)
        if session is None:
            session = _new_session(
                ctx,
                provider_name=_PROVIDER_COPERNICUS,
                bbox=location.bbox,
            )
        return session, location
    assert session is not None
    center = GeoPoint(
        lat=(session.bbox.south + session.bbox.north) / 2.0,
        lon=(session.bbox.west + session.bbox.east) / 2.0,
    )
    location = _SearchLocation(
        bbox=session.bbox,
        center=center,
        radius_m=_search_radius(),
        area=ParsedArea(bbox=session.bbox, applied=_bbox_applied(session.bbox)),
    )
    return session, location

def _has_location_input(inputs: dict[str, Any]) -> bool:
    if inputs.get("coordinates") not in (None, ""):
        return True
    if inputs.get("area") not in (None, "", "$active_area"):
        return True
    return False

def _resolve_search_location(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    require_geometry: bool,
) -> _SearchLocation:
    point = _parse_coordinates(inputs.get("coordinates"))
    area = _parse_area(_resolve_area(inputs.get("area"), ctx))
    source_bbox = _bbox_from_source_result(inputs.get("source_result"), ctx)
    center = point or area.center
    bbox = area.bbox
    if bbox is None and area.polygon is not None:
        bbox = _bbox_from_polygon(area.polygon)
        if center is None:
            center = GeoPoint(
                lat=(bbox.south + bbox.north) / 2.0,
                lon=(bbox.west + bbox.east) / 2.0,
            )
    radius = area.radius_m if area.radius_m is not None else _search_radius()
    if center is not None and bbox is None:
        bbox = _bbox_from_center(center, radius)
    if bbox is None and source_bbox is not None:
        bbox = source_bbox
        if center is None:
            center = GeoPoint(
                lat=(bbox.south + bbox.north) / 2.0,
                lon=(bbox.west + bbox.east) / 2.0,
            )
    if bbox is not None and center is None:
        center = GeoPoint(
            lat=(bbox.south + bbox.north) / 2.0,
            lon=(bbox.west + bbox.east) / 2.0,
        )
    if bbox is None or center is None:
        if require_geometry:
            if area.text:
                raise SatelliteInputError(
                    "需要 bbox 或中心点加半径；纯地名请先 geocode",
                    "missing_input",
                )
            raise SatelliteInputError(
                "缺少必填输入 area、coordinates 或 source_result",
                "missing_input",
            )
        raise SatelliteInputError("无法解析查询位置", "invalid_area")
    return _SearchLocation(bbox=bbox, center=center, radius_m=radius, area=area)

def _resolve_area(raw: Any, ctx: RuntimeContext | None) -> Any:
    if raw is None or raw == "" or raw == "$active_area":
        return ctx.active_area if ctx is not None else None
    return raw

def _bbox_from_source_result(raw: Any, ctx: RuntimeContext | None) -> BBox | None:
    payload = raw
    if payload in (None, "", "$previous_tool_result"):
        payload = ctx.previous_tool_result if ctx is not None else None
    if payload is None or payload == "":
        return None
    session = _session_from_token(payload, ctx)
    if session is not None:
        return session.bbox
    if isinstance(payload, dict):
        nested = payload.get("bbox")
        if isinstance(nested, dict):
            bbox = _bbox_from_mapping(nested)
            if bbox is not None:
                return bbox
        bbox = _bbox_from_mapping(payload)
        if bbox is not None:
            return bbox
        result = payload.get("result")
        if isinstance(result, dict):
            return _bbox_from_source_result(result, ctx)
    return None

def _parse_area(raw: Any) -> ParsedArea:
    if raw is None or raw == "":
        return ParsedArea()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ParsedArea()
        point = _parse_coordinate_text(text)
        if point is not None:
            return ParsedArea(
                center=point,
                applied={"lat": point.lat, "lon": point.lon},
            )
        return ParsedArea(text=text, applied=text)
    if isinstance(raw, (list, tuple)):
        polygon = _polygon_from_points(raw)
        if polygon is not None:
            return ParsedArea(polygon=polygon, applied=_polygon_applied(polygon))
        if len(raw) == 4:
            bbox = _bbox_from_values(list(raw))
            if bbox is not None:
                return ParsedArea(bbox=bbox, applied=_bbox_applied(bbox))
        if len(raw) == 2:
            point = _parse_lonlat_pair(raw[0], raw[1])
            if point is not None:
                return ParsedArea(
                    center=point,
                    applied={"lat": point.lat, "lon": point.lon},
                )
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
            radius_raw = raw.get("radius_m", raw.get("radius"))
            radius: int | None = None
            if radius_raw is not None and radius_raw != "":
                parsed = _as_float(radius_raw)
                if parsed is None or parsed < 0:
                    raise SatelliteInputError("radius_m 必须是非负数", "invalid_area")
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

def _parse_coordinates(raw: Any) -> GeoPoint | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        return _parse_coordinate_text(raw)
    if isinstance(raw, dict):
        return _center_from_mapping(raw)
    if isinstance(raw, (list, tuple)):
        if len(raw) == 2:
            return _parse_lonlat_pair(raw[0], raw[1])
        if len(raw) == 4:
            bbox = _bbox_from_values(list(raw))
            if bbox is not None:
                return GeoPoint(
                    lat=(bbox.south + bbox.north) / 2.0,
                    lon=(bbox.west + bbox.east) / 2.0,
                )
        raise SatelliteInputError("coordinates 必须是两点坐标", "invalid_area")
    raise SatelliteInputError("coordinates 必须是数组或 lon/lat 对象", "invalid_area")

def _parse_coordinate_text(text: str) -> GeoPoint | None:
    parts = [part for part in _COORD_SPLIT_RE.split(text.strip()) if part]
    if len(parts) != 2:
        return None
    return _parse_lonlat_pair(parts[0], parts[1])

def _parse_lonlat_pair(first_raw: Any, second_raw: Any) -> GeoPoint | None:
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
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return GeoPoint(lat=lat, lon=lon)

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

def _bbox_from_polygon(points: tuple[tuple[float, float], ...]) -> BBox:
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    return BBox(west=min(lons), south=min(lats), east=max(lons), north=max(lats))

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

def _bbox_applied(bbox: BBox) -> dict[str, float | str]:
    return {
        "west": bbox.west,
        "south": bbox.south,
        "east": bbox.east,
        "north": bbox.north,
        "crs": _CRS_WGS84,
    }

def _polygon_applied(points: tuple[tuple[float, float], ...]) -> list[list[float]]:
    return [[lon, lat] for lon, lat in points]

def _optional_time_spec(raw: Any, *, required: bool) -> TimeSpec | None:
    if raw is None or raw == "":
        if required:
            raise SatelliteInputError("缺少必填输入 time_range", "missing_input")
        return None
    return _parse_time_spec(raw)

def _parse_time_spec(raw: Any) -> TimeSpec:
    if isinstance(raw, str):
        text = raw.strip()
        lowered = text.lower()
        if lowered in _EARLIEST_ALIASES or text in _EARLIEST_ALIASES:
            return TimeSpec(earliest=True, applied=text)
        if lowered in _LATEST_ALIASES or text in _LATEST_ALIASES:
            return TimeSpec(latest=True, applied=text)
        seasonal = _parse_seasonal(text)
        if seasonal is not None:
            return seasonal
    window, unsupported = _parse_time_window(raw)
    if window is not None:
        return TimeSpec(window=window, applied=_window_applied(window, raw))
    raise SatelliteInputError(
        "time_range 无法解析为年份、日期、季节或 earliest/latest",
        "invalid_time_range",
    )

def _parse_seasonal(text: str) -> TimeSpec | None:
    lowered = text.lower()
    season_name: str | None = None
    start_month = 0
    end_month = 0
    for aliases, first, last in _SEASON_SPECS:
        if any(alias in lowered or alias in text for alias in aliases):
            season_name = next(iter(aliases))
            start_month, end_month = first, last
            break
    if season_name is None:
        return None
    years = [int(match.group(1)) for match in _YEAR_IN_TEXT_RE.finditer(text)]
    if not years:
        raise SatelliteInputError("季节/水期需要同时给出年份", "invalid_time_range")
    year = years[0]
    window = _season_window(year, start_month, end_month)
    return TimeSpec(
        window=window,
        applied=text,
        season=season_name,
    )

def _season_window(year: int, start_month: int, end_month: int) -> TimeWindow:
    start = date(year, start_month, 1)
    if end_month >= start_month:
        end = _month_end(year, end_month)
        return TimeWindow(start, end)
    end = _month_end(year + 1, end_month)
    return TimeWindow(start, end)

def _month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)

def _window_applied(window: TimeWindow, raw: Any) -> Any:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return {"start": window.start.isoformat(), "end": window.end.isoformat()}

def _parse_time_window(raw: Any) -> tuple[TimeWindow | None, Any | None]:
    if raw is None or raw == "":
        return None, None
    if isinstance(raw, bool):
        return None, raw
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
        compact = re.sub(r"\s+", "", text)
        date_range = _DATE_RANGE_RE.fullmatch(compact)
        if date_range:
            start_d = _parse_iso_date(date_range.group(1))
            end_d = _parse_iso_date(date_range.group(2))
            if start_d is not None and end_d is not None:
                lo, hi = (start_d, end_d) if start_d <= end_d else (end_d, start_d)
                return TimeWindow(lo, hi), None
        return None, raw
    if isinstance(raw, dict):
        start = _bound_to_date(raw.get("start") or raw.get("from") or raw.get("begin"), start=True)
        end = _bound_to_date(raw.get("end") or raw.get("to"), start=False)
        if start is not None and end is not None:
            lo, hi = (start, end) if start <= end else (end, start)
            return TimeWindow(lo, hi), None
        return None, raw
    return None, raw

def _bound_to_date(raw: Any, *, start: bool) -> date | None:
    if isinstance(raw, bool) or raw is None:
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
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None

def _catalog_datetime(spec: TimeSpec | None) -> str | None:
    if spec is None or spec.window is None:
        return None
    start = datetime.combine(spec.window.start, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(spec.window.end, datetime.max.time().replace(microsecond=0), tzinfo=timezone.utc)
    return f"{_rfc3339(start)}/{_rfc3339(end)}"

def _scene_process_window(scene: SceneRecord) -> tuple[str, str]:
    start = datetime.combine(scene.captured_on, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(scene.captured_on, datetime.max.time().replace(microsecond=0), tzinfo=timezone.utc)
    return _rfc3339(start), _rfc3339(end)

def _rfc3339(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _parse_layer(raw: Any) -> str:
    if raw is None or raw == "":
        return _LAYER_SATELLITE
    if not isinstance(raw, str):
        raise SatelliteInputError("layer 必须是字符串", "unsupported_layer")
    token = raw.strip()
    semantic = _LAYER_ALIASES.get(token.lower(), _LAYER_ALIASES.get(token, token.lower()))
    if semantic != _LAYER_SATELLITE:
        raise SatelliteInputError(
            f"本阶段不支持 layer={token}；航片/地形请改用对应数据源或 map_layer_query",
            "unsupported_layer",
        )
    return semantic

def _parse_cloud_cover_max(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    value = _as_float(raw)
    if value is None:
        raise SatelliteInputError("cloud_cover_max 必须是数字", "invalid_area")
    if value < 0.0 or value > 100.0:
        raise SatelliteInputError("cloud_cover_max 必须在 0 到 100 之间", "invalid_area")
    return value

def _parse_resolution_m(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    value = _as_float(raw)
    if value is None or value <= 0:
        raise SatelliteInputError("resolution_m 必须是正数", "invalid_area")
    return value

def _parse_angle(raw: Any, name: str, minimum: float, maximum: float) -> float | None:
    if raw is None or raw == "":
        return None
    value = _as_float(raw)
    if value is None:
        raise SatelliteInputError(f"{name} 必须是数字", f"invalid_{name}")
    if value < minimum or value > maximum:
        raise SatelliteInputError(
            f"{name} 必须在 {minimum} 到 {maximum} 之间",
            f"invalid_{name}",
        )
    return value

def _select_collections(
    *,
    layer: str,
    time_spec: TimeSpec | None,
    resolution_m: float | None,
) -> tuple[str, ...]:
    del layer
    ranked = (
        _COLLECTION_S2,
        _COLLECTION_LANDSAT_OT,
        _COLLECTION_LANDSAT_ETM,
        _COLLECTION_LANDSAT_TM,
    )
    window = time_spec.window if time_spec is not None else None
    selected: list[str] = []
    for collection in ranked:
        gsd = _COLLECTION_GSD[collection]
        if resolution_m is not None and gsd > resolution_m:
            continue
        start = _COLLECTION_START[collection]
        if window is not None and window.end < start:
            continue
        selected.append(collection)
    return tuple(selected)

def _normalize_scene(raw: dict[str, Any]) -> SceneRecord | None:
    props = raw.get("properties") if isinstance(raw.get("properties"), dict) else {}
    scene_id = _optional_str(raw.get("id") or raw.get("scene_id") or props.get("id"))
    collection = _optional_str(raw.get("collection") or props.get("collection") or _COLLECTION_S2)
    if scene_id is None or collection is None:
        return None
    captured_text, captured_on = _parse_captured_at(
        raw.get("datetime") or props.get("datetime") or props.get("captured_at")
    )
    if captured_text is None or captured_on is None:
        return None
    cloud = _as_float(
        raw.get("cloud_cover", raw.get("eo:cloud_cover", props.get("eo:cloud_cover", props.get("cloud_cover"))))
    )
    gsd = _as_float(raw.get("gsd") or raw.get("resolution_m") or props.get("gsd") or props.get("resolution_m"))
    resolution = gsd if gsd is not None else _COLLECTION_GSD.get(collection, 10.0)
    bbox = None
    raw_bbox = raw.get("bbox")
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        bbox = _bbox_from_values(list(raw_bbox))
    return SceneRecord(
        scene_id=scene_id,
        captured_at=captured_text,
        captured_on=captured_on,
        collection=collection,
        resolution_m=resolution,
        cloud_cover=cloud,
        bbox=bbox,
    )

def _parse_captured_at(raw: Any) -> tuple[str | None, date | None]:
    if raw is None or raw == "":
        return None, None
    if isinstance(raw, datetime):
        moment = raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
        return moment.date().isoformat(), moment.date()
    if isinstance(raw, date):
        return raw.isoformat(), raw
    if isinstance(raw, bool):
        return None, None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None, None
        iso = text.replace("Z", "+00:00")
        try:
            moment = datetime.fromisoformat(iso)
        except ValueError:
            parsed = _parse_iso_date(text[:10])
            if parsed is None:
                return None, None
            return parsed.isoformat(), parsed
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.date().isoformat(), moment.date()
    return None, None

def _scene_matches(
    record: SceneRecord,
    spec: TimeSpec | None,
    cloud_max: float | None,
    resolution_m: float | None,
) -> bool:
    if resolution_m is not None and record.resolution_m > resolution_m:
        return False
    if cloud_max is not None and record.cloud_cover is not None and record.cloud_cover > cloud_max:
        return False
    if spec is None or spec.window is None:
        return True
    return spec.window.start <= record.captured_on <= spec.window.end

def _select_scene(records: list[SceneRecord], spec: TimeSpec | None) -> SceneRecord | None:
    if not records:
        return None
    if spec is not None and spec.earliest:
        return min(records, key=lambda item: (item.captured_on, item.cloud_cover or 100.0))
    if spec is not None and spec.latest:
        return max(records, key=lambda item: (item.captured_on, -(item.cloud_cover or 100.0)))
    return min(
        records,
        key=lambda item: (item.cloud_cover if item.cloud_cover is not None else 100.0, -item.captured_on.toordinal()),
    )

def _apply_scene(session: SatelliteSession, record: SceneRecord) -> None:
    session.coverage = True
    session.scene_id = record.scene_id
    session.captured_at = record.captured_at
    session.resolution_m = record.resolution_m
    session.cloud_cover = record.cloud_cover
    session.collection = record.collection

def _clear_scene(session: SatelliteSession) -> None:
    session.coverage = False
    session.scene_id = None
    session.captured_at = None
    session.resolution_m = None
    session.cloud_cover = None
    session.collection = None
    session.image_id = None

def _new_session(
    ctx: RuntimeContext | None,
    *,
    provider_name: str,
    bbox: BBox,
    layer: str = _LAYER_SATELLITE,
) -> SatelliteSession:
    extras = _extras(ctx)
    extras[_SESSION_COUNTER_KEY] = int(extras.get(_SESSION_COUNTER_KEY) or 0) + 1
    session_id = f"sat_{int(extras[_SESSION_COUNTER_KEY]):04d}"
    return SatelliteSession(
        session_id=session_id,
        provider=provider_name,
        bbox=bbox,
        layer=layer,
    )

def _store_session(ctx: RuntimeContext | None, session: SatelliteSession) -> None:
    extras = _extras(ctx)
    sessions = extras.setdefault(_SESSIONS_KEY, {})
    if not isinstance(sessions, dict):
        sessions = {}
        extras[_SESSIONS_KEY] = sessions
    sessions[session.session_id] = session

def _load_session(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    required: bool,
) -> SatelliteSession | None:
    token: Any = inputs.get("source_result")
    if token in (None, "", "$previous_tool_result"):
        if ctx is not None and ctx.previous_tool_result is not None:
            token = ctx.previous_tool_result
        elif ctx is not None and ctx.active_session:
            token = ctx.active_session
        else:
            token = None
    session = _session_from_token(token, ctx)
    if session is None and required:
        raise SatelliteInputError("找不到 source_result 对应的影像会话", "missing_input")
    return session

def _session_from_token(token: Any, ctx: RuntimeContext | None) -> SatelliteSession | None:
    extras = _extras(ctx)
    sessions = extras.get(_SESSIONS_KEY)
    if not isinstance(sessions, dict):
        sessions = {}
    if isinstance(token, SatelliteSession):
        return token
    if isinstance(token, str) and token.strip():
        found = sessions.get(token.strip())
        return found if isinstance(found, SatelliteSession) else None
    if isinstance(token, dict):
        for key in ("session", "session_id", "result_id", "source_result"):
            value = token.get(key)
            if isinstance(value, str) and value.strip() in sessions:
                found = sessions.get(value.strip())
                if isinstance(found, SatelliteSession):
                    return found
        nested = token.get("result")
        if isinstance(nested, dict):
            return _session_from_token(nested, ctx)
    return None

def _ok(
    operation: str,
    session: SatelliteSession,
    *,
    coverage: bool,
    applied: dict[str, Any],
    ctx: RuntimeContext | None,
    image_id: str | None,
    artifacts: dict[str, Any],
    available_dates: list[str] | None,
    heading: float | None,
    tilt: float | None,
) -> Observation:
    view_kind = _VIEW_NADIR_PREVIEW if operation == _OP_OBLIQUE else _VIEW_NADIR
    result: dict[str, Any] = {
        "operation": operation,
        "coverage": coverage,
        "session": session.session_id,
        "provider": session.provider,
        "collection": session.collection,
        "scene_id": session.scene_id,
        "captured_at": session.captured_at,
        "resolution_m": session.resolution_m,
        "cloud_cover": session.cloud_cover,
        "bbox": _bbox_applied(session.bbox),
        "view_kind": view_kind,
        "is_true_oblique": False,
        "applied": applied,
        "assumptions": _assumptions(operation),
    }
    if image_id is not None:
        result["image_id"] = image_id
    if available_dates is not None:
        result["available_dates"] = available_dates
    if heading is not None:
        result["heading"] = heading
    if tilt is not None:
        result["tilt"] = tilt
    if ctx is not None:
        ctx.active_session = session.session_id
        ctx.active_area = _bbox_applied(session.bbox)
    return Observation(
        ok=True,
        result=_strip_forbidden(result),
        artifacts=artifacts,
        session=session.session_id,
    )

def _assumptions(operation: str) -> list[str]:
    items = list(_ASSUMPTIONS)
    if operation == _OP_OBLIQUE:
        items.append(
            "oblique_view 返回已有正射预览与 heading/tilt，不是新获得的真斜摄影，也未用 Cesium 渲染三维斜视"
        )
    return items

def _applied(
    provider: SatelliteImageryProvider,
    *,
    location: _SearchLocation,
    extra: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": _provider_name(provider),
        "crs": _CRS_WGS84,
        "search_radius_m": location.radius_m,
        "bbox": _bbox_applied(location.bbox),
    }
    if location.area.applied is not None:
        payload["area"] = location.area.applied
    payload.update(extra)
    unsupported: dict[str, Any] = {}
    if location.area.unsupported is not None:
        unsupported["area"] = location.area.unsupported
    if unsupported:
        payload["unsupported"] = unsupported
    return payload

def _reject_unsupported_provider(raw: Any) -> None:
    if raw is None or raw == "":
        return
    if not isinstance(raw, str):
        raise SatelliteInputError("provider 必须是字符串", "unsupported_provider")
    name = raw.strip().lower()
    if name in _COPERNICUS_ALIASES:
        return
    if name in _PLANET_ALIASES:
        raise EngineUnavailableError(
            "本阶段仅接入 Copernicus，未实现 Planet",
            "unsupported_provider",
        )
    if name in _GOOGLE_ALIASES:
        raise EngineUnavailableError(
            "本阶段仅接入 Copernicus，未实现 Google Earth",
            "unsupported_provider",
        )
    raise EngineUnavailableError(f"未知 satellite provider: {raw}", "unsupported_provider")

def _resolve_provider(ctx: RuntimeContext | None) -> SatelliteImageryProvider:
    extras = _extras(ctx)
    injected = extras.get("satellite_imagery_query_provider")
    if injected is not None:
        if not isinstance(injected, SatelliteImageryProvider):
            raise EngineUnavailableError(
                "satellite_imagery_query_provider 必须提供 search_scenes/fetch_preview",
            )
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_API=false，禁止调用真实卫星影像 API")
    name = _env_value("SATELLITE_PROVIDER", _PROVIDER_COPERNICUS).lower()
    if name in _PLANET_ALIASES:
        raise EngineUnavailableError("本阶段仅接入 Copernicus，未实现 Planet")
    if name not in _COPERNICUS_ALIASES:
        raise EngineUnavailableError(f"未知 SATELLITE_PROVIDER: {name}")
    return _build_copernicus_provider()

def _build_copernicus_provider() -> CopernicusProvider:
    client_id = os.environ.get("COPERNICUS_CLIENT_ID", "").strip()
    client_secret = os.environ.get("COPERNICUS_CLIENT_SECRET", "").strip()
    if not client_id:
        raise EngineUnavailableError("未配置 COPERNICUS_CLIENT_ID")
    if not client_secret:
        raise EngineUnavailableError("未配置 COPERNICUS_CLIENT_SECRET")
    width, height = _preview_size()
    return CopernicusProvider(
        client_id=client_id,
        client_secret=client_secret,
        catalog_endpoint=_env_value("COPERNICUS_CATALOG_ENDPOINT", _DEFAULT_CATALOG),
        process_endpoint=_env_value("COPERNICUS_PROCESS_ENDPOINT", _DEFAULT_PROCESS),
        token_endpoint=_env_value("COPERNICUS_TOKEN_ENDPOINT", _DEFAULT_TOKEN),
        timeout_sec=_env_timeout("COPERNICUS_TIMEOUT_SEC"),
        user_agent=_env_value("COPERNICUS_USER_AGENT", _DEFAULT_UA),
        preview_width=width,
        preview_height=height,
    )

def _provider_name(provider: SatelliteImageryProvider) -> str:
    return str(getattr(provider, "name", "injected"))

def _search_radius() -> int:
    parsed = _as_int(os.environ.get("SATELLITE_SEARCH_RADIUS_M", "").strip())
    if parsed is None:
        return _DEFAULT_SEARCH_RADIUS_M
    return max(_SEARCH_RADIUS_MIN, min(_SEARCH_RADIUS_MAX, parsed))

def _preview_size() -> tuple[int, int]:
    return _preview_width(), _preview_height()

def _preview_width() -> int:
    parsed = _as_int(os.environ.get("SATELLITE_PREVIEW_WIDTH", "").strip())
    if parsed is None or parsed <= 0:
        return _DEFAULT_PREVIEW_WIDTH
    return parsed

def _preview_height() -> int:
    parsed = _as_int(os.environ.get("SATELLITE_PREVIEW_HEIGHT", "").strip())
    if parsed is None or parsed <= 0:
        return _DEFAULT_PREVIEW_HEIGHT
    return parsed

def _scene_limit() -> int:
    parsed = _as_int(os.environ.get("SATELLITE_SCENE_LIMIT", "").strip())
    if parsed is None:
        return _DEFAULT_SCENE_LIMIT
    return max(_LIMIT_MIN, min(_LIMIT_MAX, parsed))

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

def _extras(ctx: RuntimeContext | None) -> dict[str, Any]:
    if ctx is None:
        return {}
    return ctx.extras

def _http_bytes(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout_sec: float,
    error_prefix: str,
) -> bytes:
    if not url:
        raise EngineUnavailableError(f"{error_prefix} 缺少 URL")
    request = urllib.request.Request(url, data=body, method=method)
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise EngineUnavailableError(f"{error_prefix} HTTP {exc.code}: {detail[:200]}") from exc
    except urllib.error.URLError as exc:
        raise EngineUnavailableError(f"{error_prefix} 网络失败: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise EngineUnavailableError(f"{error_prefix} 调用失败: {exc}") from exc

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
