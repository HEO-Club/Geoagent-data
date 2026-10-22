"""Google 街景网页内部接口适配器；不申请官方 Street View Key。"""

from __future__ import annotations

import json
import math
from typing import Any
from urllib.parse import urlencode

from tool.streetview_query._streetview import (
    EngineUnavailableError,
    ImageSearchRequest,
    _http_bytes,
)

_PROVIDER_GOOGLE = "google"
_DEFAULT_META = (
    "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1"
    ".MapsJsInternalService/SingleImageSearch"
)
_DEFAULT_THUMB = "https://streetviewpixels-pa.googleapis.com/v1/thumbnail"
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 480
_SEARCH_RADIUS_MAX = 5000.0
_METERS_PER_DEG_LAT = 111_320.0


class GoogleStreetviewProvider:
    """SingleImageSearch + streetviewpixels 缩略图；只取当前点。"""

    name = _PROVIDER_GOOGLE
    crs = "wgs84"
    renders_perspective = True

    def __init__(
        self,
        *,
        meta_endpoint: str = _DEFAULT_META,
        thumb_endpoint: str = _DEFAULT_THUMB,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
        user_agent: str = _DEFAULT_UA,
    ) -> None:
        self._meta_endpoint = meta_endpoint
        self._thumb_endpoint = thumb_endpoint.rstrip("/")
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent
        self._cache: dict[str, dict[str, Any]] = {}

    def search_images(self, request: ImageSearchRequest) -> list[dict[str, Any]]:
        """只查 bbox 中心最近点，不扫网格。"""

        lat = (request.bbox.south + request.bbox.north) / 2.0
        lon = (request.bbox.west + request.bbox.east) / 2.0
        radius = _bbox_radius_m(request.bbox)
        record = self._lookup(lat, lon, radius)
        return [record] if record is not None else []

    def get_image(self, image_id: str) -> dict[str, Any]:
        token = image_id.strip()
        if not token:
            raise EngineUnavailableError("Google 街景 ID 为空")
        cached = self._cache.get(token)
        if cached is None:
            raise EngineUnavailableError(f"Google 街景会话中没有该全景: {token}")
        return dict(cached)

    def list_sequence(self, sequence_id: str) -> list[str]:
        record = self._cache.get(sequence_id.strip())
        if record is None:
            return []
        ids = [str(record.get("id") or sequence_id)]
        for item in record.get("neighbors") or []:
            if isinstance(item, dict):
                token = str(item.get("id") or "").strip()
                if token and token not in ids:
                    ids.append(token)
        return ids

    def fetch_image_bytes(self, url: str) -> bytes:
        return _http_bytes(
            url,
            headers={"User-Agent": self._user_agent},
            timeout_sec=self._timeout_sec,
            error_prefix="Google 街景截图",
        )

    def render_view(
        self,
        image_id: str,
        *,
        heading: float,
        pitch: float,
        fov: float,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
    ) -> bytes:
        """网页缩略图已是透视画面；pitch 按页面接口取反。"""

        params = {
            "panoid": image_id.strip(),
            "cb_client": "maps_sv.tactile",
            "w": str(width),
            "h": str(height),
            "yaw": f"{heading % 360.0:.1f}",
            "pitch": f"{-pitch:.1f}",
            "thumbfov": f"{max(1.0, min(179.0, fov)):.0f}",
        }
        url = f"{self._thumb_endpoint}?{urlencode(params)}"
        data = self.fetch_image_bytes(url)
        if not data:
            raise EngineUnavailableError("Google 街景截图为空")
        return data

    def _lookup(self, lat: float, lon: float, radius: float) -> dict[str, Any] | None:
        body = json.dumps(_meta_query(lat, lon, radius), ensure_ascii=False).encode("utf-8")
        raw = _http_bytes(
            self._meta_endpoint,
            headers={
                "Content-Type": "application/json+protobuf",
                "x-user-agent": "grpc-web-javascript/0.1",
                "User-Agent": self._user_agent,
            },
            timeout_sec=self._timeout_sec,
            error_prefix="Google 街景查询",
            data=body,
        )
        text = raw.decode("utf-8", errors="replace")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EngineUnavailableError("Google 街景查询回执不是 JSON") from exc
        return self._parse_lookup(payload, text)

    def _parse_lookup(self, payload: Any, text: str) -> dict[str, Any] | None:
        if not isinstance(payload, list) or not payload:
            raise EngineUnavailableError("Google 街景查询回执无法解析")
        try:
            code = payload[0][0]
        except (IndexError, TypeError) as exc:
            raise EngineUnavailableError("Google 街景查询回执无法解析") from exc
        body = payload[1] if len(payload) > 1 else None
        if body is None:
            if code == 5 and "no images" in text.lower():
                return None
            raise EngineUnavailableError(f"Google 街景查询失败: {text[:160]}")
        try:
            panoid = str(body[1][1])
            loc = body[5][0][1]
            lat = float(loc[0][2])
            lon = float(loc[0][3])
        except (IndexError, TypeError, ValueError) as exc:
            raise EngineUnavailableError("Google 街景查询回执无法解析") from None
        heading = None
        try:
            heading = round(float(loc[2][0]), 1) if loc[2] else None
        except (IndexError, TypeError, ValueError):
            heading = None
        captured_at = None
        try:
            captured_at = _year_month(body[6][7])
        except (IndexError, TypeError, ValueError):
            captured_at = None
        neighbors: list[dict[str, Any]] = []
        try:
            for item in body[5][0][3][0]:
                nid = str(item[0][1])
                nlat = float(item[2][0][2])
                nlon = float(item[2][0][3])
                neighbor = {
                    "id": nid,
                    "lat": nlat,
                    "lon": nlon,
                    "is_pano": True,
                    "sequence": panoid,
                }
                neighbors.append(neighbor)
                self._cache.setdefault(nid, dict(neighbor))
        except (IndexError, TypeError, ValueError):
            neighbors = []
        history: list[dict[str, Any]] = []
        try:
            for row in body[5][0][8] or []:
                idx = int(row[0])
                hid = neighbors[idx]["id"] if 0 <= idx < len(neighbors) else ""
                if not hid or len(hid) != 22 or hid.startswith(("CIHM", "CIAB", "CAoS")):
                    continue
                hist = {
                    "id": hid,
                    "lat": neighbors[idx].get("lat"),
                    "lon": neighbors[idx].get("lon"),
                    "captured_at": _year_month(row[1]),
                    "is_pano": True,
                }
                history.append(hist)
                self._cache[hid] = dict(hist)
        except (IndexError, TypeError, ValueError):
            history = []
        record = {
            "id": panoid,
            "lat": lat,
            "lon": lon,
            "captured_at": captured_at,
            "compass_angle": heading,
            "sequence": panoid,
            "is_pano": True,
            "neighbors": neighbors,
            "history": history,
        }
        self._cache[panoid] = record
        return record


def _meta_query(lat: float, lon: float, radius: float) -> list[Any]:
    """与地图页 SingleImageSearch 相同的查询体；只取街景车官方覆盖。"""

    return [
        ["apiv3", None, None, None, "US", None, None, None, None, None, [[0]]],
        [[None, None, lat, lon], radius],
        [None, ["en", "US"], None, None, None, None, None, None, [2], None, [[[2, True, 2]]]],
        [[1, 2, 3, 4, 8, 6]],
    ]


def _year_month(value: Any) -> str | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        year = int(value[0])
        month = int(value[1])
    except (TypeError, ValueError):
        return None
    if not 1000 <= year <= 2100 or not 1 <= month <= 12:
        return None
    return f"{year:04d}-{month:02d}-01"


def _bbox_radius_m(bbox: Any) -> float:
    lat = (bbox.south + bbox.north) / 2.0
    lon = (bbox.west + bbox.east) / 2.0
    dlat = (bbox.north - lat) * _METERS_PER_DEG_LAT
    dlon = (bbox.east - lon) * _METERS_PER_DEG_LAT * max(abs(math.cos(math.radians(lat))), 1e-6)
    radius = math.hypot(dlat, dlon)
    return max(1.0, min(_SEARCH_RADIUS_MAX, radius))
