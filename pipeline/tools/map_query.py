"""地图查询 adapter；支持 Nominatim 与高德 AMap，外部客户端可注入/mock。"""

from __future__ import annotations

import time
from typing import Any, Optional, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pipeline.config import get_settings

# Nominatim 公共实例使用政策：约 1 次/秒
_MIN_REQUEST_INTERVAL_SEC = 1.05
_last_request_monotonic: float = 0.0


class MapQueryClient(Protocol):
    def query(
        self,
        query: str | None,
        latlng: list[float] | None,
    ) -> dict[str, Any]: ...


_client: Optional[MapQueryClient] = None


def set_client(client: Optional[MapQueryClient]) -> None:
    """测试用：注入或清除客户端。"""
    global _client
    _client = client


def _empty_obs() -> dict[str, Any]:
    return {
        "status": "empty",
        "error_message": None,
        "formatted_address": None,
        "resolved_latlng": None,
        "place_type": None,
    }


def _error_obs(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "error_message": message,
        "formatted_address": None,
        "resolved_latlng": None,
        "place_type": None,
    }


def _throttle() -> None:
    """遵守 Nominatim 公共实例速率限制。"""
    global _last_request_monotonic
    now = time.monotonic()
    wait = _MIN_REQUEST_INTERVAL_SEC - (now - _last_request_monotonic)
    if wait > 0:
        time.sleep(wait)
    _last_request_monotonic = time.monotonic()


def _place_type_from_raw(raw: dict[str, Any] | None) -> str | None:
    """从 Nominatim raw 提取简要 place_type。"""
    if not raw:
        return None
    osm_type = raw.get("type")
    if isinstance(osm_type, str) and osm_type and osm_type != "yes":
        return osm_type
    osm_class = raw.get("class")
    if isinstance(osm_class, str) and osm_class:
        return osm_class
    address = raw.get("address")
    if isinstance(address, dict):
        for key in (
            "tourism",
            "amenity",
            "building",
            "historic",
            "leisure",
            "shop",
            "highway",
            "place",
        ):
            val = address.get(key)
            if val:
                return str(val)
    return None


def _obs_from_location(location: Any) -> dict[str, Any]:
    if location is None:
        return _empty_obs()
    lat = getattr(location, "latitude", None)
    lng = getattr(location, "longitude", None)
    if lat is None or lng is None:
        return _empty_obs()
    raw = getattr(location, "raw", None)
    raw_dict = raw if isinstance(raw, dict) else None
    address = getattr(location, "address", None) or (
        raw_dict.get("display_name") if raw_dict else None
    )
    return {
        "status": "success",
        "error_message": None,
        "formatted_address": str(address) if address else None,
        "resolved_latlng": [float(lat), float(lng)],
        "place_type": _place_type_from_raw(raw_dict),
    }


def _parse_amap_location(location: str) -> list[float] | None:
    """高德 location 字符串「经度,纬度」→ [lat, lng]。"""
    parts = [p.strip() for p in location.split(",")]
    if len(parts) != 2:
        return None
    try:
        lng, lat = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return [lat, lng]


def _obs_from_amap_geocode(payload: dict[str, Any]) -> dict[str, Any]:
    """解析高德地理编码 JSON → Observation。"""
    if str(payload.get("status")) != "1":
        info = str(payload.get("info") or payload.get("infocode") or "amap geo failed")
        return _error_obs(info)
    geocodes = payload.get("geocodes")
    if not isinstance(geocodes, list) or not geocodes:
        return _empty_obs()
    item = geocodes[0]
    if not isinstance(item, dict):
        return _empty_obs()
    resolved = _parse_amap_location(str(item.get("location") or ""))
    if resolved is None:
        return _empty_obs()
    place_type = item.get("level") or item.get("type")
    return {
        "status": "success",
        "error_message": None,
        "formatted_address": (
            str(item["formatted_address"])
            if item.get("formatted_address")
            else None
        ),
        "resolved_latlng": resolved,
        "place_type": str(place_type) if place_type else None,
    }


def _obs_from_amap_regeo(payload: dict[str, Any]) -> dict[str, Any]:
    """解析高德逆地理编码 JSON → Observation。"""
    if str(payload.get("status")) != "1":
        info = str(payload.get("info") or payload.get("infocode") or "amap regeo failed")
        return _error_obs(info)
    regeo = payload.get("regeocode")
    if not isinstance(regeo, dict):
        return _empty_obs()
    addr = regeo.get("formatted_address")
    # 逆地理通常用请求坐标；若无 location 字段则由调用方补上
    component = regeo.get("addressComponent")
    place_type = None
    if isinstance(component, dict):
        place_type = component.get("township") or component.get("district")
    return {
        "status": "success" if addr or component else "empty",
        "error_message": None,
        "formatted_address": str(addr) if addr else None,
        "resolved_latlng": None,  # 由 _AmapAdapter 填入请求坐标
        "place_type": str(place_type) if place_type else None,
    }


def _http_get_json(url: str, *, timeout: float) -> dict[str, Any]:
    """轻量 HTTP GET JSON（避免为地图单独引入新依赖）。"""
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — URL 来自配置 base
        import json

        data = json.loads(resp.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("AMap 响应不是 JSON object")
    return data


def _nominatim_client() -> MapQueryClient:
    settings = get_settings()
    user_agent = (settings.NOMINATIM_USER_AGENT or "").strip()
    if not user_agent:
        raise ValueError("NOMINATIM_USER_AGENT 未配置（Nominatim 要求可识别的 User-Agent）")

    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent=user_agent, timeout=settings.NOMINATIM_TIMEOUT_SEC)

    class _NominatimAdapter:
        def query(self, query: str | None, latlng: list[float] | None) -> dict[str, Any]:
            if query:
                _throttle()
                location = geolocator.geocode(query, exactly_one=True, language="en")
                return _obs_from_location(location)
            assert latlng is not None
            _throttle()
            location = geolocator.reverse(
                (float(latlng[0]), float(latlng[1])),
                exactly_one=True,
                language="en",
            )
            return _obs_from_location(location)

    return _NominatimAdapter()


def _amap_client() -> MapQueryClient:
    settings = get_settings()
    api_key = (settings.AMAP_API_KEY or "").strip()
    if not api_key:
        raise ValueError("未配置 AMAP_API_KEY（MAP_PROVIDER=amap 时必填）")
    base = (settings.AMAP_BASE_URL or "").rstrip("/")
    if not base:
        raise ValueError("AMAP_BASE_URL 未配置")
    timeout = float(settings.AMAP_TIMEOUT_SEC)

    class _AmapAdapter:
        def query(self, query: str | None, latlng: list[float] | None) -> dict[str, Any]:
            if query:
                params = {
                    "key": api_key,
                    "address": query,
                    "output": "JSON",
                }
                url = f"{base}/v3/geocode/geo?{urlencode(params)}"
                payload = _http_get_json(url, timeout=timeout)
                return _obs_from_amap_geocode(payload)

            assert latlng is not None
            lat, lng = float(latlng[0]), float(latlng[1])
            params = {
                "key": api_key,
                "location": f"{lng:.6f},{lat:.6f}",
                "extensions": "base",
                "output": "JSON",
            }
            url = f"{base}/v3/geocode/regeo?{urlencode(params)}"
            payload = _http_get_json(url, timeout=timeout)
            obs = _obs_from_amap_regeo(payload)
            if obs["status"] == "success":
                obs["resolved_latlng"] = [lat, lng]
            elif obs["status"] == "empty":
                pass
            return obs

    return _AmapAdapter()


def _default_client() -> MapQueryClient:
    settings = get_settings()
    if settings.APP_ENV == "test" and not settings.ALLOW_REAL_API:
        raise RuntimeError("test 环境禁止真实 map_query 调用；请 mock set_client")
    provider = settings.MAP_PROVIDER.strip().lower()
    if provider in {"amap", "gaode", "高德"}:
        return _amap_client()
    if provider in {"nominatim", "osm"}:
        return _nominatim_client()
    raise ValueError(
        f"不支持的 MAP_PROVIDER={settings.MAP_PROVIDER!r}；可选: amap / nominatim"
    )


def execute(params: dict[str, Any], image_path: str) -> dict[str, Any]:
    """执行 map_query；输出坐标字段为 resolved_latlng。"""
    _ = image_path
    query = params.get("query")
    latlng = params.get("latlng")
    try:
        client = _client if _client is not None else _default_client()
        return client.query(query, latlng)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": str(exc),
            "formatted_address": None,
            "resolved_latlng": None,
            "place_type": None,
        }
