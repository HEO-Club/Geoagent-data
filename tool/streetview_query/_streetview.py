"""streetview_query 共享执行器：默认百度全景 / Google 街景网页取图。"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from tool._crs import in_china
from tool._gate import allow_real_api as _allow_real_api
from tool._gate import load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext, declared_inputs
from tool.runtime.image_store import put_image

_PROVIDER_AUTO = "auto"
_PROVIDER_MAPILLARY = "mapillary"
_PROVIDER_BAIDU = "baidu"
_PROVIDER_GOOGLE = "google"
_CRS_WGS84 = "wgs84"
_OP_OPEN = "open"
_OP_NAVIGATE = "navigate"
_OP_CHANGE_TIME = "change_time"
_OP_CAPTURE = "capture"
_DEFAULT_ENDPOINT = "https://graph.mapillary.com"
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_UA = "geoagent-dataset/1.0 (streetview_query; local)"
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_DEFAULT_SEARCH_RADIUS_M = 50
_SEARCH_RADIUS_MIN = 1
_SEARCH_RADIUS_MAX = 5000
_RADIUS_API_MAX_M = 50
_DEFAULT_IMAGE_LIMIT = 50
_SEARCH_RESULT_MAX = 20
_LIMIT_MIN = 1
_LIMIT_MAX = 200
_DEFAULT_CAPTURE_WIDTH = 640
_DEFAULT_CAPTURE_HEIGHT = 480
_DEFAULT_FOV = 90.0
_EARTH_RADIUS_M = 6_371_000.0
_METERS_PER_DEG_LAT = 111_320.0
_YEAR_MIN = 1000
_YEAR_MAX = 2100
_SESSIONS_KEY = "streetview_sessions"
_SESSION_COUNTER_KEY = "streetview_session_counter"
_SEARCH_FIELDS = "id,captured_at,compass_angle,geometry,sequence,is_pano"
_IMAGE_FIELDS = (
    "id,captured_at,compass_angle,computed_compass_angle,"
    "computed_geometry,geometry,sequence,is_pano,"
    "thumb_1024_url,thumb_2048_url"
)
_YEAR_RE = re.compile(r"^(\d{4})$")
_YEAR_RANGE_RE = re.compile(r"^(\d{4})\s*[-–—至到]\s*(\d{4})$")
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
_DATE_RANGE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s*to\s*(\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)
_COORD_SPLIT_RE = re.compile(r"[,;，、\s]+")
_BEARING_ALIASES: dict[str, float] = {
    "n": 0.0,
    "north": 0.0,
    "北": 0.0,
    "正北": 0.0,
    "ne": 45.0,
    "northeast": 45.0,
    "东北": 45.0,
    "e": 90.0,
    "east": 90.0,
    "东": 90.0,
    "正东": 90.0,
    "se": 135.0,
    "southeast": 135.0,
    "东南": 135.0,
    "s": 180.0,
    "south": 180.0,
    "南": 180.0,
    "正南": 180.0,
    "sw": 225.0,
    "southwest": 225.0,
    "西南": 225.0,
    "w": 270.0,
    "west": 270.0,
    "西": 270.0,
    "正西": 270.0,
    "nw": 315.0,
    "northwest": 315.0,
    "西北": 315.0,
}
_FORWARD_ALIASES = frozenset(
    {"forward", "ahead", "next", "along", "along_road", "along the road", "前", "向前", "沿道路", "沿路"}
)
_BACK_ALIASES = frozenset({"back", "backward", "prev", "previous", "后", "向后", "回头"})
_EARLIEST_ALIASES = frozenset(
    {"earliest", "earliest available", "最早", "最早可用", "最早的", "check_earliest_available_streetview"}
)
_LATEST_ALIASES = frozenset({"latest", "newest", "最晚", "最新", "最晚可用"})
_MAPILLARY_ALIASES = frozenset({"mapillary", "mapillary.com"})
_BAIDU_ALIASES = frozenset(
    {"baidu", "baidu_pano", "baidu_panorama", "baidu panorama", "百度", "百度全景"}
)
_GOOGLE_ALIASES = frozenset(
    {"google", "google_street_view", "google street view", "streetview", "gsw"}
)
_AUTO_ALIASES = frozenset({"auto", "default", "auto_detect"})
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
_ASSUMPTIONS = [
    "拍摄日期来自该点已返回的 captured_at，不是网站上可选的任意年份",
    "指定年代没有覆盖时返回无覆盖，不会用当前街景证明过去",
    "默认国内走百度全景网页接口，国外走 Google 街景网页接口；Mapillary 仅在显式指定时使用",
    "无法唯一判定时，坐标串按 lon,lat 解析",
    "open 遇到纯地名时内部先 geocode，没有坐标再按名称查 POI，不要求模型拆成两步",
    "open 只查当前点最近一张全景，不扫区域网格",
    "接口报错或回执无法解析不是无覆盖",
    "available_dates 仅来自当前点已返回的拍摄，不是全局完整编年史",
    "街景截图只供本次查看，用后删除本地文件",
]

class StreetviewInputError(Exception):
    """area / coordinates / session / time_range 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """真实街景服务未配置、被闸门拒绝或调用失败。"""

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
    """闭区间日期窗，只用于过滤 captured_at。"""

    start: date
    end: date

@dataclass(frozen=True)
class TimeSpec:
    """change_time 的可执行时间条件。"""

    window: TimeWindow | None = None
    earliest: bool = False
    latest: bool = False
    applied: Any = None

@dataclass(frozen=True)
class ImageSearchRequest:
    """组装后的 Mapillary 影像检索。"""

    bbox: BBox
    start_captured_at: str | None = None
    end_captured_at: str | None = None
    limit: int = _DEFAULT_IMAGE_LIMIT

@dataclass(frozen=True)
class ImageRecord:
    """归一化后的街景视点。"""

    image_id: str
    lat: float | None = None
    lon: float | None = None
    captured_at: str | None = None
    captured_date: date | None = None
    compass_angle: float | None = None
    sequence_id: str | None = None
    is_pano: bool = False
    thumb_url: str | None = None
    neighbors: tuple[ImageRecord, ...] = ()
    history: tuple[ImageRecord, ...] = ()

@dataclass
class StreetviewSession:
    """一次街景会话；无覆盖时只保留查询点。"""

    session_id: str
    provider: str
    lat: float | None = None
    lon: float | None = None
    image_id: str | None = None
    sequence_id: str | None = None
    captured_at: str | None = None
    compass_angle: float | None = None
    is_pano: bool = False
    thumb_url: str | None = None
    coverage: bool = False
    neighbors: list[ImageRecord] = field(default_factory=list)
    history: list[ImageRecord] = field(default_factory=list)
    renders_perspective: bool = False

@runtime_checkable
class StreetviewProvider(Protocol):
    """可注入的街景后端；测试用 extras['streetview_query_provider'] 替换。"""

    def search_images(self, request: ImageSearchRequest) -> list[dict[str, Any]]:
        """按 bbox 检索影像，返回可归一化的 JSON 对象列表。"""

    def get_image(self, image_id: str) -> dict[str, Any]:
        """按影像 ID 取元数据。"""

    def list_sequence(self, sequence_id: str) -> list[str]:
        """返回序列中按行驶顺序排列的影像 ID。"""

    def fetch_image_bytes(self, url: str) -> bytes:
        """下载缩略图字节。"""

class MapillaryStreetviewProvider:
    """Mapillary Graph API 适配器；访问令牌与端点只读环境变量。"""

    name = _PROVIDER_MAPILLARY
    crs = _CRS_WGS84

    def __init__(
        self,
        *,
        access_token: str,
        endpoint: str,
        timeout_sec: float,
        user_agent: str,
    ) -> None:
        self._access_token = access_token
        self._endpoint = endpoint.rstrip("/")
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent

    def search_images(self, request: ImageSearchRequest) -> list[dict[str, Any]]:
        """按中心点半径检索。Mapillary 的 bbox 检索会因数据量被拒绝。"""

        center_lat = (request.bbox.south + request.bbox.north) / 2.0
        center_lon = (request.bbox.west + request.bbox.east) / 2.0
        params: dict[str, str] = {
            "access_token": self._access_token,
            "fields": _SEARCH_FIELDS,
            "lat": _coord_param(center_lat),
            "lng": _coord_param(center_lon),
            "radius": str(_radius_m_for_bbox(request.bbox)),
            "limit": str(max(_LIMIT_MIN, min(_SEARCH_RESULT_MAX, request.limit))),
        }
        if request.start_captured_at:
            params["start_captured_at"] = request.start_captured_at
        if request.end_captured_at:
            params["end_captured_at"] = request.end_captured_at
        payload = self._get_json("/images", params, error_prefix="Mapillary 影像检索")
        data = payload.get("data")
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def get_image(self, image_id: str) -> dict[str, Any]:
        token = image_id.strip()
        if not token:
            raise EngineUnavailableError("Mapillary 影像 ID 为空")
        params = {
            "access_token": self._access_token,
            "fields": _IMAGE_FIELDS,
        }
        payload = self._get_json(f"/{urllib.parse.quote(token)}", params, error_prefix="Mapillary 影像")
        if not isinstance(payload, dict):
            raise EngineUnavailableError("Mapillary 影像回执不是 JSON 对象")
        return payload

    def list_sequence(self, sequence_id: str) -> list[str]:
        token = sequence_id.strip()
        if not token:
            return []
        params = {
            "access_token": self._access_token,
            "sequence_id": token,
        }
        payload = self._get_json("/image_ids", params, error_prefix="Mapillary 序列")
        data = payload.get("data")
        if not isinstance(data, list):
            return []
        ids: list[str] = []
        for item in data:
            if isinstance(item, dict):
                value = item.get("id")
                if isinstance(value, str) and value.strip():
                    ids.append(value.strip())
            elif isinstance(item, str) and item.strip():
                ids.append(item.strip())
        return ids

    def fetch_image_bytes(self, url: str) -> bytes:
        return _http_bytes(
            url,
            headers={"User-Agent": self._user_agent},
            timeout_sec=self._timeout_sec,
            error_prefix="Mapillary 缩略图",
        )

    def _get_json(self, path: str, params: dict[str, str], *, error_prefix: str) -> dict[str, Any]:
        url = _append_query(f"{self._endpoint}{path}", urllib.parse.urlencode(params))
        raw = _http_json(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self._user_agent,
            },
            timeout_sec=self._timeout_sec,
            error_prefix=error_prefix,
        )
        if not isinstance(raw, dict):
            raise EngineUnavailableError(f"{error_prefix} 回执不是 JSON 对象")
        return raw

def execute_open(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """打开指定地点或坐标的街景会话。"""

    try:
        inputs = declared_inputs(inputs, "area", "coordinates", "provider")
        requested = inputs.get("provider")
        _reject_unsupported_provider(requested)
        location = _resolve_search_location(
            inputs,
            ctx,
            require_geometry=True,
            geocode_place_names=True,
            purpose=purpose,
        )
        provider = _resolve_provider(ctx, requested=requested, location=location)
        records = _search_records(provider, location.bbox, location.center)
        chosen = _closest_within_radius(records, location.center, location.radius_m)
        session = _new_session(
            ctx,
            provider_name=_provider_name(provider),
            lat=location.center.lat,
            lon=location.center.lon,
            renders_perspective=_provider_renders_perspective(provider),
        )
        _apply_record(session, chosen)
        _store_session(ctx, session)
        return _ok(
            _OP_OPEN,
            session,
            coverage=chosen is not None,
            applied=_applied(
                provider,
                location=location,
                extra={
                    "search_radius_m": location.radius_m,
                    **location.geocode_applied,
                },
            ),
            ctx=ctx,
        )
    except StreetviewInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def execute_navigate(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """沿道路或方向移动街景视点。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "session", "area", "direction", "distance_m")
        direction_raw = inputs.get("direction")
        if direction_raw is None or direction_raw == "":
            raise StreetviewInputError("缺少必填输入 direction", "missing_input")
        bearing, step_sign = _parse_direction(direction_raw)
        distance_m = _parse_distance_m(inputs.get("distance_m"))
        session, location = _resolve_session_or_location(inputs, ctx)
        provider = _resolve_provider(
            ctx,
            location=location,
            session=session,
        )
        session.provider = _provider_name(provider)
        session.renders_perspective = session.renders_perspective or _provider_renders_perspective(provider)
        if session.image_id is None:
            records = _search_records(provider, location.bbox, location.center)
            chosen = _closest_within_radius(records, location.center, location.radius_m)
            _apply_record(session, chosen)
        moved = False
        if session.image_id is not None:
            if session.neighbors:
                target = _step_along_neighbors(
                    session,
                    bearing=bearing,
                    step_sign=step_sign,
                    distance_m=distance_m,
                )
            else:
                target = _step_along_sequence(
                    provider,
                    session,
                    bearing=bearing,
                    step_sign=step_sign,
                    distance_m=distance_m,
                )
            if target is not None and target.image_id != session.image_id:
                _apply_record(session, target, keep_neighbors=True)
                moved = True
        _store_session(ctx, session)
        extra: dict[str, Any] = {"direction": _stringify(direction_raw)}
        if bearing is not None:
            extra["bearing"] = bearing
        if step_sign is not None:
            extra["step"] = "forward" if step_sign > 0 else "back"
        if distance_m is not None:
            extra["distance_m"] = distance_m
        return _ok(
            _OP_NAVIGATE,
            session,
            coverage=session.coverage,
            moved=moved,
            applied=_applied(provider, location=location, extra=extra),
            ctx=ctx,
        )
    except StreetviewInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def execute_change_time(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """切换街景年份：只从实际 captured_at 中选择。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "session", "area", "time_range")
        if inputs.get("time_range") is None or inputs.get("time_range") == "":
            raise StreetviewInputError("缺少必填输入 time_range", "missing_input")
        spec = _parse_time_spec(inputs.get("time_range"))
        session, location = _resolve_session_or_location(inputs, ctx)
        provider = _resolve_provider(ctx, location=location, session=session)
        session.provider = _provider_name(provider)
        session.renders_perspective = session.renders_perspective or _provider_renders_perspective(provider)
        if session.history or session.renders_perspective:
            records = _session_time_records(session)
        else:
            records = _search_records(provider, location.bbox, location.center)
        available_dates = _available_dates(records)
        chosen = _select_by_time(records, spec, location.center)
        if chosen is None:
            _clear_image_evidence(session)
        else:
            _apply_record(session, chosen, keep_neighbors=True)
        _store_session(ctx, session)
        extra: dict[str, Any] = {"time_range": spec.applied}
        return _ok(
            _OP_CHANGE_TIME,
            session,
            coverage=chosen is not None,
            available_dates=available_dates,
            applied=_applied(provider, location=location, extra=extra),
            ctx=ctx,
        )
    except StreetviewInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def execute_capture(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """从街景会话获取指定视角画面。"""

    del purpose
    try:
        inputs = declared_inputs(inputs, "session", "heading", "pitch", "fov")
        session = _load_session(inputs, ctx, required=True)
        if not session.coverage or not session.image_id:
            raise StreetviewInputError("当前会话没有可用街景覆盖，无法截图", "no_coverage")
        heading = _parse_angle(inputs.get("heading"), "heading", 0.0, 360.0)
        pitch = _parse_angle(inputs.get("pitch"), "pitch", -90.0, 90.0)
        fov = _parse_angle(inputs.get("fov"), "fov", 1.0, 180.0)
        provider = _resolve_provider(ctx, session=session)
        applied_heading = heading
        applied_pitch = 0.0 if pitch is None else pitch
        applied_fov = _DEFAULT_FOV if fov is None else fov
        perspective = session.renders_perspective or _provider_renders_perspective(provider)
        record: ImageRecord | None = None
        if not perspective:
            record = _image_record(provider, session.image_id, {})
            if record.thumb_url:
                session.thumb_url = record.thumb_url
        pano = bool(session.is_pano or (record.is_pano if record is not None else False))
        if perspective:
            if applied_heading is None:
                applied_heading = session.compass_angle if session.compass_angle is not None else 0.0
            renderer = getattr(provider, "render_view", None)
            if not callable(renderer):
                raise EngineUnavailableError("街景后端未提供按朝向截图")
            image_bytes = renderer(
                session.image_id,
                heading=applied_heading,
                pitch=applied_pitch,
                fov=applied_fov,
                width=_DEFAULT_CAPTURE_WIDTH,
                height=_DEFAULT_CAPTURE_HEIGHT,
            )
            pano = True
        else:
            image_bytes = provider.fetch_image_bytes(
                (record.thumb_url if record is not None else None) or session.thumb_url or ""
            )
        if not image_bytes:
            raise EngineUnavailableError("街景缩略图为空")
        pil = Image.open(BytesIO(image_bytes))
        pil.load()
        if pano and not perspective:
            if applied_heading is None:
                applied_heading = session.compass_angle if session.compass_angle is not None else 0.0
            pil = _project_pano(
                pil,
                heading=applied_heading,
                pitch=applied_pitch,
                fov=applied_fov,
            )
        elif not pano:
            applied_heading = session.compass_angle
            applied_pitch = None
            applied_fov = None
        extra: dict[str, Any] = {
            "pano": pano,
            "width": pil.width,
            "height": pil.height,
        }
        if applied_heading is not None:
            extra["heading"] = applied_heading
        if applied_pitch is not None:
            extra["pitch"] = applied_pitch
        if applied_fov is not None:
            extra["fov"] = applied_fov
        image_id, path = put_image(pil, source_id=session.image_id, suffix="jpeg", ctx=ctx)
        artifacts = {"image_ids": [image_id]}
        observation = _ok(
            _OP_CAPTURE,
            session,
            coverage=True,
            image_id=image_id,
            applied=_applied(provider, extra=extra),
            artifacts=artifacts,
            ctx=ctx,
            pano=pano,
            perspective=perspective,
        )
        _discard_captured_image(ctx, image_id, path)
        return observation
    except StreetviewInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

def _ok(
    operation: str,
    session: StreetviewSession,
    *,
    coverage: bool,
    applied: dict[str, Any],
    ctx: RuntimeContext | None,
    moved: bool | None = None,
    available_dates: list[str] | None = None,
    image_id: str | None = None,
    artifacts: dict[str, Any] | None = None,
    pano: bool | None = None,
    perspective: bool | None = None,
) -> Observation:
    result: dict[str, Any] = {
        "operation": operation,
        "coverage": coverage,
        "session": session.session_id,
        "viewpoint": _viewpoint(session) if coverage and session.image_id else None,
        "applied": applied,
        "assumptions": _assumptions(pano=pano, perspective=perspective),
    }
    if moved is not None:
        result["moved"] = moved
    if available_dates is not None:
        result["available_dates"] = available_dates
    if image_id is not None:
        result["image_id"] = image_id
    if ctx is not None:
        ctx.active_session = session.session_id
    return Observation(
        ok=True,
        result=_strip_forbidden(result),
        artifacts=artifacts or {},
        session=session.session_id,
    )

def _assumptions(*, pano: bool | None, perspective: bool | None = None) -> list[str]:
    items = list(_ASSUMPTIONS)
    if pano is False:
        items.append("当前影像不是全景，capture 返回原图朝向，不能按任意 heading/pitch 重投影")
    elif pano is True and perspective:
        items.append("全景由服务端按 heading/pitch/fov 截透视图片")
    elif pano is True:
        items.append("全景按 heading/pitch/fov 从等距柱状图投影，不是服务端任意年份截图")
    return items

def _applied(
    provider: StreetviewProvider,
    *,
    location: _SearchLocation | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": _provider_name(provider),
        "crs": _CRS_WGS84,
    }
    if location is not None:
        payload["bbox"] = _bbox_applied(location.bbox)
        payload["center"] = {"lat": location.center.lat, "lon": location.center.lon}
        if location.area.applied is not None:
            payload["area"] = location.area.applied
        if location.area.unsupported is not None:
            payload["unsupported"] = {"area": location.area.unsupported}
    if extra:
        payload.update(extra)
    return payload

def _viewpoint(session: StreetviewSession) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_id": session.image_id,
        "sequence_id": session.sequence_id,
        "lat": session.lat,
        "lon": session.lon,
        "captured_at": session.captured_at,
        "compass_angle": session.compass_angle,
        "is_pano": session.is_pano,
    }
    return payload

def _search_records(
    provider: StreetviewProvider,
    bbox: BBox,
    center: GeoPoint,
) -> list[ImageRecord]:
    del center
    payload = provider.search_images(
        ImageSearchRequest(bbox=bbox, limit=_image_limit())
    )
    records: list[ImageRecord] = []
    for item in payload:
        record = _normalize_image(item)
        if record is not None:
            records.append(record)
    return records

def _closest_record(records: list[ImageRecord], center: GeoPoint) -> ImageRecord | None:
    best: ImageRecord | None = None
    best_dist = math.inf
    for record in records:
        point = _record_point(record)
        if point is None:
            if best is None:
                best = record
            continue
        dist = _haversine_m(center, point)
        if dist < best_dist:
            best = record
            best_dist = dist
    return best


def _closest_within_radius(
    records: list[ImageRecord],
    center: GeoPoint,
    radius_m: int,
) -> ImageRecord | None:
    chosen = _closest_record(records, center)
    if chosen is None:
        return None
    point = _record_point(chosen)
    if point is not None and _haversine_m(center, point) > float(max(radius_m, 1)):
        return None
    return chosen


def _session_time_records(session: StreetviewSession) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    if session.image_id:
        captured_at, captured_date = _parse_captured_at(session.captured_at)
        records.append(
            ImageRecord(
                image_id=session.image_id,
                lat=session.lat,
                lon=session.lon,
                captured_at=captured_at,
                captured_date=captured_date,
                compass_angle=session.compass_angle,
                sequence_id=session.sequence_id,
                is_pano=session.is_pano,
                thumb_url=session.thumb_url,
                neighbors=tuple(session.neighbors),
                history=tuple(session.history),
            )
        )
    records.extend(session.history)
    return records


def _step_along_neighbors(
    session: StreetviewSession,
    *,
    bearing: float | None,
    step_sign: int | None,
    distance_m: float | None,
) -> ImageRecord | None:
    """在 open 已保存的相邻点中走一步，不再重新检索。"""

    origin = (
        GeoPoint(lat=session.lat, lon=session.lon)
        if session.lat is not None and session.lon is not None
        else None
    )
    wanted = bearing
    if step_sign is not None:
        base = wanted if wanted is not None else session.compass_angle
        if base is None:
            base = 0.0
        wanted = base if step_sign > 0 else (base + 180.0) % 360.0
    remaining = [item for item in session.neighbors if item.image_id != session.image_id]
    if not remaining:
        return None

    def pick(from_point: GeoPoint | None, options: list[ImageRecord]) -> ImageRecord | None:
        if wanted is None:
            return options[0]
        best: ImageRecord | None = None
        best_diff = math.inf
        for item in options:
            dest = _record_point(item)
            if from_point is None or dest is None:
                diff = 90.0
            else:
                diff = _angle_diff(_bearing_deg(from_point, dest), wanted)
            if diff < best_diff:
                best_diff = diff
                best = item
        return best

    if distance_m is None or distance_m <= 0:
        return pick(origin, remaining)
    traveled = 0.0
    last: ImageRecord | None = None
    cursor = origin
    unused = list(remaining)
    while unused:
        nxt = pick(cursor, unused)
        if nxt is None:
            break
        dest = _record_point(nxt)
        if cursor is not None and dest is not None:
            traveled += _haversine_m(cursor, dest)
        else:
            traveled += 1.0
        last = nxt
        unused = [item for item in unused if item.image_id != nxt.image_id]
        cursor = dest
        if traveled >= distance_m:
            break
    return last

def _select_by_time(
    records: list[ImageRecord],
    spec: TimeSpec,
    center: GeoPoint,
) -> ImageRecord | None:
    dated = [item for item in records if item.captured_date is not None]
    if not dated:
        return None
    if spec.earliest:
        oldest = min(dated, key=lambda item: item.captured_date or date.max)
        return _closest_record(
            [item for item in dated if item.captured_date == oldest.captured_date],
            center,
        )
    if spec.latest:
        newest = max(dated, key=lambda item: item.captured_date or date.min)
        return _closest_record(
            [item for item in dated if item.captured_date == newest.captured_date],
            center,
        )
    window = spec.window
    if window is None:
        return None
    matched = [
        item
        for item in dated
        if item.captured_date is not None and window.start <= item.captured_date <= window.end
    ]
    return _closest_record(matched, center)

def _available_dates(records: list[ImageRecord]) -> list[str]:
    dates = sorted({item.captured_at[:10] for item in records if item.captured_at})
    return dates

def _step_along_sequence(
    provider: StreetviewProvider,
    session: StreetviewSession,
    *,
    bearing: float | None,
    step_sign: int | None,
    distance_m: float | None,
) -> ImageRecord | None:
    current_id = session.image_id
    if current_id is None:
        return None
    cache: dict[str, ImageRecord] = {}
    current = _image_record(provider, current_id, cache)
    sequence_id = current.sequence_id or session.sequence_id
    if not sequence_id:
        return None
    sequence = provider.list_sequence(sequence_id)
    if current_id not in sequence:
        sequence = [current_id, *sequence]
    index = sequence.index(current_id)
    direction = _sequence_direction(
        provider,
        sequence,
        index,
        current,
        cache,
        bearing=bearing,
        step_sign=step_sign,
    )
    if direction is None:
        return None
    if distance_m is None or distance_m <= 0:
        nxt = index + direction
        if not 0 <= nxt < len(sequence):
            return None
        return _image_record(provider, sequence[nxt], cache)
    traveled = 0.0
    cursor = index
    last: ImageRecord | None = None
    while True:
        nxt = cursor + direction
        if not 0 <= nxt < len(sequence):
            break
        here = _image_record(provider, sequence[cursor], cache)
        there = _image_record(provider, sequence[nxt], cache)
        a = _record_point(here)
        b = _record_point(there)
        if a is not None and b is not None:
            traveled += _haversine_m(a, b)
        else:
            traveled += 1.0
        last = there
        cursor = nxt
        if traveled >= distance_m:
            break
    return last

def _sequence_direction(
    provider: StreetviewProvider,
    sequence: list[str],
    index: int,
    current: ImageRecord,
    cache: dict[str, ImageRecord],
    *,
    bearing: float | None,
    step_sign: int | None,
) -> int | None:
    if step_sign is not None:
        nxt = index + step_sign
        if 0 <= nxt < len(sequence):
            return step_sign
        return None
    wanted = bearing
    if wanted is None:
        wanted = current.compass_angle
    if wanted is None:
        if index + 1 < len(sequence):
            return 1
        if index - 1 >= 0:
            return -1
        return None
    best_sign: int | None = None
    best_diff = math.inf
    origin = _record_point(current)
    for sign in (1, -1):
        nxt = index + sign
        if not 0 <= nxt < len(sequence):
            continue
        neighbor = _image_record(provider, sequence[nxt], cache)
        dest = _record_point(neighbor)
        if origin is None or dest is None:
            diff = 90.0
        else:
            diff = _angle_diff(_bearing_deg(origin, dest), wanted)
        if diff < best_diff:
            best_diff = diff
            best_sign = sign
    return best_sign

def _image_record(
    provider: StreetviewProvider,
    image_id: str,
    cache: dict[str, ImageRecord],
) -> ImageRecord:
    cached = cache.get(image_id)
    if cached is not None:
        return cached
    raw = provider.get_image(image_id)
    record = _normalize_image(raw, image_id=image_id)
    if record is None:
        raise EngineUnavailableError(f"无法解析街景影像: {image_id}")
    cache[image_id] = record
    return record

def _normalize_image(
    raw: dict[str, Any],
    *,
    image_id: str | None = None,
    nested: bool = False,
) -> ImageRecord | None:
    token = image_id or _optional_str(raw.get("id", raw.get("image_id")))
    if not token:
        return None
    lat, lon = _geometry_point(raw)
    captured_at, captured_date = _parse_captured_at(
        raw.get("captured_at", raw.get("captured_date"))
    )
    compass = _as_float(raw.get("computed_compass_angle", raw.get("compass_angle")))
    sequence = _optional_str(raw.get("sequence", raw.get("sequence_id")))
    thumb = _optional_str(raw.get("thumb_2048_url") or raw.get("thumb_1024_url") or raw.get("thumb_url"))
    is_pano = bool(raw.get("is_pano", False))
    neighbors: tuple[ImageRecord, ...] = ()
    history: tuple[ImageRecord, ...] = ()
    if not nested:
        neighbor_rows: list[ImageRecord] = []
        for item in raw.get("neighbors") or []:
            if isinstance(item, dict):
                parsed = _normalize_image(item, nested=True)
                if parsed is not None:
                    neighbor_rows.append(parsed)
        neighbors = tuple(neighbor_rows)
        history_rows: list[ImageRecord] = []
        for item in raw.get("history") or []:
            if isinstance(item, dict):
                parsed = _normalize_image(item, nested=True)
                if parsed is not None:
                    history_rows.append(parsed)
        history = tuple(history_rows)
    return ImageRecord(
        image_id=token,
        lat=lat,
        lon=lon,
        captured_at=captured_at,
        captured_date=captured_date,
        compass_angle=compass,
        sequence_id=sequence,
        is_pano=is_pano,
        thumb_url=thumb,
        neighbors=neighbors,
        history=history,
    )

def _geometry_point(raw: dict[str, Any]) -> tuple[float | None, float | None]:
    lat = _as_float(raw.get("lat", raw.get("latitude")))
    lon = _as_float(raw.get("lon", raw.get("lng", raw.get("longitude"))))
    if lat is not None and lon is not None:
        return lat, lon
    for key in ("computed_geometry", "geometry"):
        geom = raw.get(key)
        if not isinstance(geom, dict):
            continue
        coords = geom.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            parsed_lon = _as_float(coords[0])
            parsed_lat = _as_float(coords[1])
            if parsed_lat is not None and parsed_lon is not None:
                return parsed_lat, parsed_lon
    return None, None

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
    if isinstance(raw, (int, float)):
        millis = float(raw)
        seconds = millis / 1000.0 if millis > 10_000_000_000 else millis
        moment = datetime.fromtimestamp(seconds, tz=timezone.utc)
        return moment.date().isoformat(), moment.date()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None, None
        if text.isdigit():
            if len(text) == 8:
                parsed = _parse_iso_date(f"{text[0:4]}-{text[4:6]}-{text[6:8]}")
                if parsed is not None:
                    return parsed.isoformat(), parsed
            return _parse_captured_at(int(text))
        month_only = re.fullmatch(r"^(\d{4})-(\d{2})$", text)
        if month_only:
            year = int(month_only.group(1))
            month = int(month_only.group(2))
            if _YEAR_MIN <= year <= _YEAR_MAX and 1 <= month <= 12:
                parsed = date(year, month, 1)
                return parsed.isoformat(), parsed
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

def _record_point(record: ImageRecord) -> GeoPoint | None:
    if record.lat is None or record.lon is None:
        return None
    return GeoPoint(lat=record.lat, lon=record.lon)

def _apply_record(
    session: StreetviewSession,
    record: ImageRecord | None,
    *,
    keep_neighbors: bool = False,
) -> None:
    if record is None:
        _clear_image_evidence(session)
        return
    session.image_id = record.image_id
    session.sequence_id = record.sequence_id
    if record.lat is not None:
        session.lat = record.lat
    if record.lon is not None:
        session.lon = record.lon
    session.captured_at = record.captured_at
    session.compass_angle = record.compass_angle
    session.is_pano = record.is_pano
    session.thumb_url = record.thumb_url
    session.coverage = True
    if record.neighbors:
        session.neighbors = list(record.neighbors)
    elif not keep_neighbors:
        session.neighbors = []
    if record.history:
        session.history = list(record.history)
    elif not keep_neighbors:
        session.history = []

def _clear_image_evidence(session: StreetviewSession) -> None:
    session.image_id = None
    session.sequence_id = None
    session.captured_at = None
    session.compass_angle = None
    session.is_pano = False
    session.thumb_url = None
    session.coverage = False
    session.neighbors = []
    session.history = []

def _new_session(
    ctx: RuntimeContext | None,
    *,
    provider_name: str,
    lat: float | None,
    lon: float | None,
    renders_perspective: bool = False,
) -> StreetviewSession:
    extras = _extras(ctx)
    extras[_SESSION_COUNTER_KEY] = int(extras.get(_SESSION_COUNTER_KEY) or 0) + 1
    session_id = f"sv_{int(extras[_SESSION_COUNTER_KEY]):04d}"
    return StreetviewSession(
        session_id=session_id,
        provider=provider_name,
        lat=lat,
        lon=lon,
        renders_perspective=renders_perspective,
    )

def _store_session(ctx: RuntimeContext | None, session: StreetviewSession) -> None:
    extras = _extras(ctx)
    sessions = extras.setdefault(_SESSIONS_KEY, {})
    if not isinstance(sessions, dict):
        sessions = {}
        extras[_SESSIONS_KEY] = sessions
    sessions[session.session_id] = session

def restore_runtime_context(result: dict[str, Any], ctx: RuntimeContext) -> None:
    """Rebuild a street-view session from its durable public result."""

    session_id = _optional_str(result.get("session"))
    if session_id is None:
        return
    viewpoint = result.get("viewpoint") if isinstance(result.get("viewpoint"), dict) else {}
    applied = result.get("applied") if isinstance(result.get("applied"), dict) else {}
    provider_name = str(applied.get("provider") or _PROVIDER_AUTO)
    session = StreetviewSession(
        session_id=session_id,
        provider=provider_name,
        lat=_as_float(viewpoint.get("lat")),
        lon=_as_float(viewpoint.get("lon")),
        image_id=_optional_str(viewpoint.get("image_id")),
        sequence_id=_optional_str(viewpoint.get("sequence_id")),
        captured_at=_optional_str(viewpoint.get("captured_at")),
        compass_angle=_as_float(viewpoint.get("compass_angle")),
        is_pano=bool(viewpoint.get("is_pano", False)),
        coverage=bool(result.get("coverage", False)),
        renders_perspective=provider_name in {_PROVIDER_BAIDU, _PROVIDER_GOOGLE},
    )
    _store_session(ctx, session)
    suffix = session_id.rsplit("_", 1)[-1]
    if suffix.isdigit():
        ctx.extras[_SESSION_COUNTER_KEY] = max(
            int(ctx.extras.get(_SESSION_COUNTER_KEY) or 0),
            int(suffix),
        )

def _load_session(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    required: bool,
) -> StreetviewSession:
    token = _resolve_session_id(inputs.get("session"), ctx)
    if not token:
        if required:
            raise StreetviewInputError("缺少必填输入 session；没有会话时先执行 open", "missing_input")
        raise StreetviewInputError("缺少 session 或 area", "missing_input")
    extras = _extras(ctx)
    sessions = extras.get(_SESSIONS_KEY)
    if isinstance(sessions, dict):
        found = sessions.get(token)
        if isinstance(found, StreetviewSession):
            return found
    raise StreetviewInputError("找不到街景会话，请先 open", "missing_input")

def _resolve_session_id(raw: Any, ctx: RuntimeContext | None) -> str | None:
    if raw is None or raw == "" or raw == "$active_session":
        if ctx is not None and ctx.active_session:
            return str(ctx.active_session).strip() or None
        return None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None

@dataclass(frozen=True)
class _SearchLocation:
    bbox: BBox
    center: GeoPoint
    radius_m: int
    area: ParsedArea
    geocode_applied: dict[str, Any] = field(default_factory=dict)

def _resolve_session_or_location(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> tuple[StreetviewSession, _SearchLocation]:
    session_token = _resolve_session_id(inputs.get("session"), ctx)
    has_area = _has_location_input(inputs)
    if not session_token and not has_area:
        raise StreetviewInputError("需要 session 或 area", "missing_input")
    session: StreetviewSession | None = None
    if session_token:
        try:
            session = _load_session(inputs, ctx, required=False)
        except StreetviewInputError:
            if not has_area:
                raise
    if session is not None and session.lat is not None and session.lon is not None:
        center = GeoPoint(lat=session.lat, lon=session.lon)
        radius = _search_radius()
        bbox = _bbox_from_center(center, radius)
        location = _SearchLocation(
            bbox=bbox,
            center=center,
            radius_m=radius,
            area=ParsedArea(center=center, radius_m=radius, applied={"lat": center.lat, "lon": center.lon}),
        )
        return session, location
    location = _resolve_search_location(inputs, ctx, require_geometry=True)
    if session is None:
        session = _new_session(
            ctx,
            provider_name=_PROVIDER_AUTO,
            lat=location.center.lat,
            lon=location.center.lon,
        )
    return session, location

def _has_location_input(inputs: dict[str, Any]) -> bool:
    return inputs.get("area") not in (None, "")

def _resolve_search_location(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
    *,
    require_geometry: bool,
    geocode_place_names: bool = False,
    purpose: str = "",
) -> _SearchLocation:
    point = _parse_coordinates(inputs.get("coordinates"))
    area = _parse_area(_resolve_area(inputs.get("area"), ctx))
    center = point or area.center
    bbox = area.bbox
    radius = area.radius_m if area.radius_m is not None else _search_radius()
    geocode_applied: dict[str, Any] = {}
    if bbox is None and area.polygon is not None:
        bbox = _bbox_from_polygon(area.polygon)
        if center is None:
            center = GeoPoint(
                lat=(bbox.south + bbox.north) / 2.0,
                lon=(bbox.west + bbox.east) / 2.0,
            )
    if center is None and bbox is None and area.text and geocode_place_names:
        center, geocode_applied = _geocode_place_name(
            area.text,
            ctx=ctx,
            purpose=purpose,
        )
    if center is not None and bbox is None:
        bbox = _bbox_from_center(center, radius)
    if bbox is not None and center is None:
        center = GeoPoint(
            lat=(bbox.south + bbox.north) / 2.0,
            lon=(bbox.west + bbox.east) / 2.0,
        )
    if bbox is None or center is None:
        if require_geometry:
            if area.text:
                raise StreetviewInputError(
                    "需要 bbox、中心点加半径或可解析地名",
                    "missing_input",
                )
            raise StreetviewInputError("缺少必填输入 area 或 coordinates", "missing_input")
        raise StreetviewInputError("无法解析查询位置", "invalid_area")
    return _SearchLocation(
        bbox=bbox,
        center=center,
        radius_m=radius,
        area=area,
        geocode_applied=geocode_applied,
    )


def _geocode_place_name(
    query: str,
    *,
    ctx: RuntimeContext | None,
    purpose: str,
) -> tuple[GeoPoint, dict[str, Any]]:
    """把 open 的纯地名交给 geocode；没有点时再按同一名称查 POI。"""

    from tool.geocode.geocode import execute as execute_geocode

    observation = execute_geocode(
        purpose=purpose or "解析街景打开地点",
        inputs={"query": query, "top_k": 1},
        ctx=ctx,
    )
    applied: dict[str, Any] = {"geocode_query": query}
    if not observation.ok:
        code = observation.error_code or "engine_unavailable"
        message = observation.error or "内部 geocode 失败"
        if code in {"missing_input", "invalid_input"}:
            raise StreetviewInputError(message, code)
        if code != "no_result":
            raise EngineUnavailableError(message, code)
        point = None
    else:
        result = observation.result if isinstance(observation.result, dict) else {}
        provider = result.get("provider")
        if provider is None and isinstance(result.get("applied"), dict):
            provider = result["applied"].get("provider")
        if provider:
            applied["geocode_provider"] = provider
        point = _point_from_rows(result.get("candidates") or result.get("results") or [])
    if point is not None:
        return point, applied
    poi_point, poi_applied = _poi_place_name(query, ctx=ctx, purpose=purpose)
    if poi_point is None:
        raise StreetviewInputError(f"无法解析地名: {query}", "no_result")
    applied.update(poi_applied)
    return poi_point, applied


def _poi_place_name(
    query: str,
    *,
    ctx: RuntimeContext | None,
    purpose: str,
) -> tuple[GeoPoint | None, dict[str, Any]]:
    """地理编码没有点时，按名称取第一个 POI 坐标。"""

    from tool.poi_search.poi_search import execute as execute_poi_search

    observation = execute_poi_search(
        purpose=purpose or "解析街景打开地点",
        inputs={"query": query, "top_k": 1},
        ctx=ctx,
    )
    if not observation.ok:
        code = observation.error_code or "engine_unavailable"
        message = observation.error or "内部 POI 检索失败"
        if code in {"missing_input", "invalid_input", "no_result", "needs_acquisition"}:
            return None, {}
        raise EngineUnavailableError(message, code)
    result = observation.result if isinstance(observation.result, dict) else {}
    point = _point_from_rows(result.get("candidates") or result.get("results") or [])
    if point is None:
        return None, {}
    applied: dict[str, Any] = {"poi_query": query}
    provider = result.get("provider")
    if provider is None and isinstance(result.get("applied"), dict):
        provider = result["applied"].get("provider")
    if provider:
        applied["poi_provider"] = provider
    return point, applied


def _point_from_rows(rows: Any) -> GeoPoint | None:
    if not isinstance(rows, list):
        return None
    for item in rows:
        if not isinstance(item, dict):
            continue
        point = _point_from_mapping(item)
        if point is not None:
            return point
    return None


def _point_from_mapping(item: dict[str, Any]) -> GeoPoint | None:
    lat = _as_float(item.get("latitude", item.get("lat")))
    lon = _as_float(item.get("longitude", item.get("lon", item.get("lng"))))
    if lat is None or lon is None:
        location = item.get("location")
        if isinstance(location, dict):
            lat = _as_float(location.get("latitude", location.get("lat")))
            lon = _as_float(location.get("longitude", location.get("lon", location.get("lng"))))
        elif isinstance(location, str):
            lat, lon = _lat_lon_from_text(location)
    if lat is None or lon is None:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return GeoPoint(lat=lat, lon=lon)


def _lat_lon_from_text(text: str) -> tuple[float | None, float | None]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        return None, None
    first = _as_float(parts[0])
    second = _as_float(parts[1])
    if first is None or second is None:
        return None, None
    if abs(first) <= 90.0 and abs(second) > 90.0:
        return first, second
    return second, first

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
                    raise StreetviewInputError("radius_m 必须是非负数", "invalid_area")
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
        raise StreetviewInputError("coordinates 必须是两点坐标", "invalid_coordinates")
    raise StreetviewInputError("coordinates 必须是数组或 lon/lat 对象", "invalid_coordinates")

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

def _coord_param(value: float) -> str:
    return format(value, ".7f").rstrip("0").rstrip(".")


def _radius_m_for_bbox(bbox: BBox) -> int:
    """把查询矩形收成 Mapillary 半径检索，上限 50 米。"""

    center = GeoPoint(
        lat=(bbox.south + bbox.north) / 2.0,
        lon=(bbox.west + bbox.east) / 2.0,
    )
    corner = GeoPoint(lat=bbox.north, lon=bbox.east)
    radius_m = int(math.ceil(_haversine_m(center, corner)))
    return max(_SEARCH_RADIUS_MIN, min(_RADIUS_API_MAX_M, radius_m))


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

def _parse_time_spec(raw: Any) -> TimeSpec:
    if isinstance(raw, str):
        text = raw.strip()
        lowered = text.lower()
        if lowered in _EARLIEST_ALIASES or text in _EARLIEST_ALIASES:
            return TimeSpec(earliest=True, applied=text)
        if lowered in _LATEST_ALIASES or text in _LATEST_ALIASES:
            return TimeSpec(latest=True, applied=text)
    window, unsupported = _parse_time_window(raw)
    if window is not None:
        return TimeSpec(window=window, applied=_window_applied(window, raw))
    raise StreetviewInputError("time_range 无法解析为年份、日期或 earliest/latest", "invalid_time_range")

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
        date_range = _DATE_RANGE_RE.fullmatch(re.sub(r"\s+", "", text))
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

def _parse_direction(raw: Any) -> tuple[float | None, int | None]:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw) % 360.0, None
    if not isinstance(raw, str) or not raw.strip():
        raise StreetviewInputError("direction 必须是方位、沿道路方向或度数", "invalid_direction")
    text = raw.strip()
    lowered = text.lower()
    if lowered in _FORWARD_ALIASES or text in _FORWARD_ALIASES:
        return None, 1
    if lowered in _BACK_ALIASES or text in _BACK_ALIASES:
        return None, -1
    if lowered in _BEARING_ALIASES:
        return _BEARING_ALIASES[lowered], None
    if text in _BEARING_ALIASES:
        return _BEARING_ALIASES[text], None
    bearing = _as_float(text)
    if bearing is not None:
        return bearing % 360.0, None
    raise StreetviewInputError("direction 必须是方位、沿道路方向或度数", "invalid_direction")

def _parse_distance_m(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    value = _as_float(raw)
    if value is None or value < 0:
        raise StreetviewInputError("distance_m 必须是非负数", "invalid_distance")
    return value

def _parse_angle(raw: Any, name: str, minimum: float, maximum: float) -> float | None:
    if raw is None or raw == "":
        return None
    value = _as_float(raw)
    if value is None:
        raise StreetviewInputError(f"{name} 必须是数字", f"invalid_{name}")
    if value < minimum or value > maximum:
        raise StreetviewInputError(
            f"{name} 必须在 {minimum} 到 {maximum} 之间",
            f"invalid_{name}",
        )
    return value

def _reject_unsupported_provider(raw: Any) -> None:
    if raw is None or raw == "":
        return
    if not isinstance(raw, str):
        raise StreetviewInputError("provider 必须是字符串", "invalid_provider")
    name = raw.strip().lower()
    if name in _MAPILLARY_ALIASES or name in _BAIDU_ALIASES or name in _GOOGLE_ALIASES or name in _AUTO_ALIASES:
        return
    raise EngineUnavailableError(f"未知 streetview provider: {raw}")


def _canonical_provider_name(raw: str) -> str:
    name = raw.strip().lower()
    if name in _MAPILLARY_ALIASES:
        return _PROVIDER_MAPILLARY
    if name in _BAIDU_ALIASES:
        return _PROVIDER_BAIDU
    if name in _GOOGLE_ALIASES:
        return _PROVIDER_GOOGLE
    if name in _AUTO_ALIASES:
        return _PROVIDER_AUTO
    return name


def _in_baidu_coverage(point: GeoPoint) -> bool:
    return in_china(point.lon, point.lat)


def _provider_renders_perspective(provider: StreetviewProvider) -> bool:
    return bool(getattr(provider, "renders_perspective", False))


def _resolve_provider(
    ctx: RuntimeContext | None,
    *,
    requested: Any = None,
    location: _SearchLocation | None = None,
    session: StreetviewSession | None = None,
) -> StreetviewProvider:
    extras = _extras(ctx)
    injected = extras.get("streetview_query_provider")
    if injected is not None:
        if not isinstance(injected, StreetviewProvider):
            raise EngineUnavailableError(
                "streetview_query_provider 必须提供 search_images/get_image/list_sequence/fetch_image_bytes",
            )
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_TOOL_API=false，禁止调用真实街景 API")
    if isinstance(requested, str) and requested.strip():
        name = _canonical_provider_name(requested)
    else:
        name = _canonical_provider_name(_env_value("STREETVIEW_PROVIDER", _PROVIDER_AUTO))
    if name == _PROVIDER_AUTO:
        if session is not None and session.provider in {_PROVIDER_BAIDU, _PROVIDER_GOOGLE, _PROVIDER_MAPILLARY}:
            name = session.provider
        elif location is not None and _in_baidu_coverage(location.center):
            name = _PROVIDER_BAIDU
        else:
            name = _PROVIDER_GOOGLE
    if name == _PROVIDER_MAPILLARY:
        token = _mapillary_token()
        if not token:
            raise EngineUnavailableError("未配置 MAPILLARY_ACCESS_TOKEN 或 MAPILLARY_TOKEN")
        return MapillaryStreetviewProvider(
            access_token=token,
            endpoint=_env_value("MAPILLARY_ENDPOINT", _DEFAULT_ENDPOINT),
            timeout_sec=_env_timeout("MAPILLARY_TIMEOUT_SEC"),
            user_agent=_env_value("MAPILLARY_USER_AGENT", _DEFAULT_UA),
        )
    timeout_sec = _env_timeout("STREETVIEW_TIMEOUT_SEC")
    user_agent = _env_value("STREETVIEW_USER_AGENT", _BROWSER_UA)
    if name == _PROVIDER_BAIDU:
        from tool.streetview_query._baidu import BaiduPanoramaProvider

        return BaiduPanoramaProvider(
            endpoint=_env_value("STREETVIEW_BAIDU_ENDPOINT", "https://mapsv0.bdimg.com/"),
            timeout_sec=timeout_sec,
            user_agent=user_agent,
        )
    if name == _PROVIDER_GOOGLE:
        from tool.streetview_query._google import GoogleStreetviewProvider

        return GoogleStreetviewProvider(
            meta_endpoint=_env_value(
                "STREETVIEW_GOOGLE_META_ENDPOINT",
                "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/SingleImageSearch",
            ),
            thumb_endpoint=_env_value(
                "STREETVIEW_GOOGLE_THUMB_ENDPOINT",
                "https://streetviewpixels-pa.googleapis.com/v1/thumbnail",
            ),
            timeout_sec=timeout_sec,
            user_agent=user_agent,
        )
    raise EngineUnavailableError(f"未知 STREETVIEW_PROVIDER: {name}")

def _mapillary_token() -> str:
    return (
        os.environ.get("MAPILLARY_ACCESS_TOKEN", "").strip()
        or os.environ.get("MAPILLARY_TOKEN", "").strip()
    )

def _search_radius() -> int:
    parsed = _as_int(
        os.environ.get("STREETVIEW_SEARCH_RADIUS_M", "").strip()
        or os.environ.get("MAPILLARY_SEARCH_RADIUS_M", "").strip()
    )
    if parsed is None:
        return _DEFAULT_SEARCH_RADIUS_M
    return max(_SEARCH_RADIUS_MIN, min(_SEARCH_RADIUS_MAX, parsed))

def _image_limit() -> int:
    parsed = _as_int(os.environ.get("MAPILLARY_IMAGE_LIMIT", "").strip())
    if parsed is None:
        return _DEFAULT_IMAGE_LIMIT
    return max(_LIMIT_MIN, min(_LIMIT_MAX, parsed))

def _provider_name(provider: StreetviewProvider) -> str:
    return str(getattr(provider, "name", "injected"))

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

def _haversine_m(a: GeoPoint, b: GeoPoint) -> float:
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = lat2 - lat1
    dlon = math.radians(b.lon - a.lon)
    chord = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(chord)))

def _bearing_deg(a: GeoPoint, b: GeoPoint) -> float:
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlon = math.radians(b.lon - a.lon)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

def _angle_diff(a: float, b: float) -> float:
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)

def _project_pano(
    image: Image.Image,
    *,
    heading: float,
    pitch: float,
    fov: float,
    width: int = _DEFAULT_CAPTURE_WIDTH,
    height: int = _DEFAULT_CAPTURE_HEIGHT,
) -> Image.Image:
    src = image.convert("RGB")
    src_w, src_h = src.size
    pixels = src.load()
    out = Image.new("RGB", (width, height))
    out_px = out.load()
    if pixels is None or out_px is None:
        return src.resize((width, height))
    heading_rad = math.radians(heading)
    pitch_rad = math.radians(pitch)
    fov_rad = math.radians(max(1.0, min(179.0, fov)))
    focal = 0.5 * width / math.tan(0.5 * fov_rad)
    sin_h, cos_h = math.sin(heading_rad), math.cos(heading_rad)
    sin_p, cos_p = math.sin(pitch_rad), math.cos(pitch_rad)
    for y in range(height):
        for x in range(width):
            vx = (x + 0.5) - width / 2.0
            vy = -((y + 0.5) - height / 2.0)
            vz = focal
            length = math.sqrt(vx * vx + vy * vy + vz * vz)
            vx, vy, vz = vx / length, vy / length, vz / length
            vy2 = vy * cos_p - vz * sin_p
            vz2 = vy * sin_p + vz * cos_p
            vx3 = vx * cos_h + vz2 * sin_h
            vz3 = -vx * sin_h + vz2 * cos_h
            lon = math.atan2(vx3, vz3)
            lat = math.asin(max(-1.0, min(1.0, vy2)))
            u = int((lon / (2.0 * math.pi) + 0.5) * src_w) % src_w
            v = min(max(int((0.5 - lat / math.pi) * src_h), 0), src_h - 1)
            out_px[x, y] = pixels[u, v]
    return out

def _http_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout_sec: float,
    error_prefix: str,
) -> Any:
    raw = _http_bytes(url, headers=headers, timeout_sec=timeout_sec, error_prefix=error_prefix)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise EngineUnavailableError(f"{error_prefix} 回执不是 JSON") from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise EngineUnavailableError(f"{error_prefix} 失败: {payload['error']}")
    return payload

def _http_bytes(
    url: str,
    *,
    headers: dict[str, str],
    timeout_sec: float,
    error_prefix: str,
    data: bytes | None = None,
) -> bytes:
    if not url:
        raise EngineUnavailableError(f"{error_prefix} 缺少 URL")
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise EngineUnavailableError(f"{error_prefix} HTTP {exc.code}: {detail[:200]}") from exc
    except urllib.error.URLError as exc:
        raise EngineUnavailableError(f"{error_prefix} 网络失败: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise EngineUnavailableError(f"{error_prefix} 调用失败: {exc}") from exc

def _append_query(endpoint: str, query: str) -> str:
    separator = "&" if urllib.parse.urlparse(endpoint).query else "?"
    return f"{endpoint}{separator}{query}"

def _stringify(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    return str(raw)

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

def _discard_captured_image(
    ctx: RuntimeContext | None,
    image_id: str,
    path: Path,
) -> None:
    """街景截图用后删除本地文件，不留在会话目录。"""

    store = ctx.image_store if ctx is not None else None
    deleter = getattr(store, "delete", None)
    if callable(deleter):
        deleter(image_id)
        return
    try:
        path.unlink(missing_ok=True)
        parent = path.parent
        if parent.name.startswith("image_edit_") and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        return


def _fail(message: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=message, error_code=error_code)
