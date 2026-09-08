"""Nominatim 地理编码适配器与严格结果归一。"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol, runtime_checkable

from tool.contract import Observation, RuntimeContext
from tool.runtime.result_store import store_result

_DEFAULT_ENDPOINT = "https://nominatim.openstreetmap.org"
_DEFAULT_LIMIT = 5
_MAX_LIMIT = 10
_MAX_QUERIES = 5
_TIMEOUT_SEC = 30.0
_RATE_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


class GeocodeInputError(Exception):
    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class GeocodeProviderError(Exception):
    def __init__(self, message: str, error_code: str = "provider_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code


@runtime_checkable
class GeocodeClient(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        limit: int,
        language: str | None,
        viewbox: tuple[float, float, float, float] | None,
    ) -> list[dict[str, Any]]:
        """正向地理编码。"""

    def reverse(self, lat: float, lon: float, *, language: str | None) -> dict[str, Any]:
        """反向地理编码。"""


class NominatimClient:
    """低频研发用公共 Nominatim 客户端；生产批量任务应换自建或商业实例。"""

    name = "nominatim"

    def __init__(self, *, endpoint: str, user_agent: str, timeout_sec: float) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._user_agent = user_agent
        self._timeout_sec = timeout_sec

    def search(
        self,
        query: str,
        *,
        limit: int,
        language: str | None,
        viewbox: tuple[float, float, float, float] | None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "q": query,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": limit,
        }
        if language:
            params["accept-language"] = language
        if viewbox is not None:
            west, south, east, north = viewbox
            params["viewbox"] = f"{west:g},{north:g},{east:g},{south:g}"
            params["bounded"] = 1
        raw = self._get("search", params)
        if not isinstance(raw, list):
            raise GeocodeProviderError("Nominatim search 回执不是数组", "provider_error")
        return [item for item in raw if isinstance(item, dict)]

    def reverse(self, lat: float, lon: float, *, language: str | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "addressdetails": 1,
            "zoom": 18,
        }
        if language:
            params["accept-language"] = language
        raw = self._get("reverse", params)
        if not isinstance(raw, dict):
            raise GeocodeProviderError("Nominatim reverse 回执不是对象", "provider_error")
        return raw

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        global _LAST_REQUEST_AT
        url = f"{self._endpoint}/{path}?{urllib.parse.urlencode(params, doseq=True)}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self._user_agent, "Accept": "application/json"},
        )
        with _RATE_LOCK:
            wait = 1.05 - (time.monotonic() - _LAST_REQUEST_AT)
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_sec) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
                raise GeocodeProviderError(
                    f"Nominatim HTTP {exc.code}: {detail[:200]}",
                    "provider_error",
                ) from exc
            except urllib.error.URLError as exc:
                raise GeocodeProviderError(f"Nominatim 网络失败: {exc.reason}") from exc
            except (json.JSONDecodeError, TimeoutError, OSError) as exc:
                raise GeocodeProviderError(f"Nominatim 调用失败: {exc}") from exc
            finally:
                _LAST_REQUEST_AT = time.monotonic()
        return payload


def execute_geocode(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    """执行正向/反向地理编码并返回带来源的候选，不替 Agent 选最终地点。"""

    del purpose
    try:
        direction = _direction(inputs.get("direction"), inputs.get("query"))
        limit = _limit(inputs.get("top_k"))
        language = _language(inputs.get("language"))
        area_text = ""
        viewbox: tuple[float, float, float, float] | None = None
        if direction == "reverse":
            lat, lon = _coordinates(inputs.get("query"))
            request_value: Any = {"lat": lat, "lon": lon}
        else:
            queries = _queries(inputs.get("query"))
            area_text, viewbox = _area_scope(inputs.get("area"))
            request_value = queries
        client = _client(ctx)
        if direction == "reverse":
            raw_rows = [client.reverse(lat, lon, language=language)]
        else:
            raw_rows = []
            for index, query in enumerate(queries):
                submitted = f"{query}, {area_text}" if area_text and area_text not in query else query
                for item in client.search(
                    submitted,
                    limit=limit,
                    language=language,
                    viewbox=viewbox,
                ):
                    row = dict(item)
                    row["_input_index"] = index
                    raw_rows.append(row)
    except GeocodeInputError as exc:
        return _fail(str(exc), exc.error_code)
    except GeocodeProviderError as exc:
        return _fail(str(exc), exc.error_code)

    candidates = [
        row
        for row in (_normalize_candidate(item, client.name) for item in raw_rows)
        if row is not None
    ]
    result: dict[str, Any] = {
        "provider": client.name,
        "direction": direction,
        "request": request_value,
        "area": area_text if direction == "forward" and area_text else None,
        "viewbox": list(viewbox) if direction == "forward" and viewbox else None,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "crs": "EPSG:4326",
        "attribution": "© OpenStreetMap contributors" if client.name == "nominatim" else client.name,
        "assumptions": [
            "地理编码返回候选而非最终地点，重名地点必须结合 working_scope 和其他证据消歧",
            "坐标统一输出为 WGS84（EPSG:4326）",
        ],
    }
    result_id = store_result(result, namespace="geocode", ctx=ctx)
    if result_id:
        result["result_id"] = result_id
    return Observation(ok=True, result=result)


def _client(ctx: RuntimeContext | None) -> GeocodeClient:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("geocode_client")
    if injected is not None:
        if not isinstance(injected, GeocodeClient):
            raise GeocodeProviderError(
                "geocode_client 必须实现 search 与 reverse",
                "invalid_provider",
            )
        return injected
    _load_dotenv()
    if not _enabled():
        raise GeocodeProviderError("ALLOW_REAL_TOOL_API=false，禁止调用真实地理编码服务")
    user_agent = os.environ.get("GEOAGENT_USER_AGENT", "").strip()
    if not user_agent:
        raise GeocodeProviderError(
            "公共 Nominatim 要求设置可识别的 GEOAGENT_USER_AGENT（建议包含项目名和联系邮箱）",
        )
    return NominatimClient(
        endpoint=_endpoint(),
        user_agent=user_agent,
        timeout_sec=_timeout(),
    )


def _endpoint() -> str:
    endpoint = os.environ.get("NOMINATIM_ENDPOINT", _DEFAULT_ENDPOINT).strip()
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise GeocodeProviderError("NOMINATIM_ENDPOINT 必须是有效 HTTPS 地址")
    custom = os.environ.get("ALLOW_CUSTOM_TOOL_ENDPOINTS", "false").strip().lower()
    if parsed.hostname.lower() != "nominatim.openstreetmap.org" and custom not in {
        "1", "true", "yes", "on",
    }:
        raise GeocodeProviderError("自定义 Nominatim 主机需显式开启 ALLOW_CUSTOM_TOOL_ENDPOINTS")
    return endpoint


def _direction(raw: Any, query: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return "reverse" if _looks_like_coordinates(query) else "forward"
    if not isinstance(raw, str) or raw.strip().lower() not in {"forward", "reverse", "auto"}:
        raise GeocodeInputError("direction 必须是 forward、reverse 或 auto", "invalid_direction")
    value = raw.strip().lower()
    return ("reverse" if _looks_like_coordinates(query) else "forward") if value == "auto" else value


def _queries(raw: Any) -> list[str]:
    value = [raw] if isinstance(raw, str) else raw
    if not isinstance(value, list) or not value:
        raise GeocodeInputError("正向地理编码需要 query 字符串或字符串数组", "missing_input")
    queries = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(queries) != len(value):
        raise GeocodeInputError("query 数组只能包含非空字符串", "invalid_query")
    if len(queries) > _MAX_QUERIES:
        raise GeocodeInputError(f"单次最多地理编码 {_MAX_QUERIES} 个查询", "too_many_queries")
    return queries


def _looks_like_coordinates(raw: Any) -> bool:
    if isinstance(raw, dict):
        return {"lat", "lon"} <= set(raw) or {"latitude", "longitude"} <= set(raw)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in raw)
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) == 2:
            try:
                float(parts[0]); float(parts[1])
                return True
            except ValueError:
                return False
    return False


def _coordinates(raw: Any) -> tuple[float, float]:
    lat: Any = None
    lon: Any = None
    try:
        if isinstance(raw, dict):
            lat = raw.get("lat", raw.get("latitude"))
            lon = raw.get("lon", raw.get("longitude"))
        elif isinstance(raw, (list, tuple)) and len(raw) == 2:
            lat, lon = raw
        elif isinstance(raw, str):
            lat, lon = [part.strip() for part in raw.split(",")]
        else:
            raise ValueError
        latitude, longitude = float(lat), float(lon)
    except (TypeError, ValueError) as exc:
        raise GeocodeInputError("反向地理编码需要 [lat,lon] 或坐标对象", "invalid_coordinates") from exc
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise GeocodeInputError("坐标超出 WGS84 有效范围", "invalid_coordinates")
    return latitude, longitude


def _area_scope(raw: Any) -> tuple[str, tuple[float, float, float, float] | None]:
    if raw is None:
        return "", None
    if isinstance(raw, str) and raw.strip():
        return raw.strip(), None
    name = ""
    bbox_raw = raw
    if isinstance(raw, dict):
        raw_name = raw.get("name")
        if raw_name is not None:
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise GeocodeInputError("area.name 必须是非空字符串", "invalid_area")
            name = raw_name.strip()
        bbox_raw = raw.get("bbox")
    if bbox_raw is None:
        if name:
            return name, None
        raise GeocodeInputError("area 对象需要 name 或 bbox", "invalid_area")
    if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
        raise GeocodeInputError(
            "area bbox 必须是 [west,south,east,north]",
            "invalid_area",
        )
    try:
        west, south, east, north = [float(value) for value in bbox_raw]
    except (TypeError, ValueError) as exc:
        raise GeocodeInputError("area bbox 必须由四个数字组成", "invalid_area") from exc
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise GeocodeInputError("area bbox 超出 WGS84 范围或顺序错误", "invalid_area")
    return name, (west, south, east, north)


def _limit(raw: Any) -> int:
    if raw is None:
        return _DEFAULT_LIMIT
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise GeocodeInputError("top_k 必须是整数", "invalid_top_k") from exc
    if value < 1:
        raise GeocodeInputError("top_k 必须 >= 1", "invalid_top_k")
    return min(value, _MAX_LIMIT)


def _language(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise GeocodeInputError("language 必须是非空语言代码", "invalid_language")
    return raw.strip()


def _normalize_candidate(raw: dict[str, Any], provider: str) -> dict[str, Any] | None:
    try:
        lat = float(raw["lat"])
        lon = float(raw["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    bbox = raw.get("boundingbox")
    normalized_bbox: list[float] | None = None
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            south, north, west, east = [float(item) for item in bbox]
            normalized_bbox = [west, south, east, north]
        except (TypeError, ValueError):
            normalized_bbox = None
    row: dict[str, Any] = {
        "display_name": str(raw.get("display_name") or raw.get("name") or ""),
        "latitude": lat,
        "longitude": lon,
        "osm_type": str(raw.get("osm_type") or ""),
        "osm_id": raw.get("osm_id"),
        "category": str(raw.get("category") or raw.get("class") or ""),
        "type": str(raw.get("type") or ""),
        "address": raw.get("address") if isinstance(raw.get("address"), dict) else {},
        "provider": provider,
    }
    if normalized_bbox is not None:
        row["bbox"] = normalized_bbox
    if "_input_index" in raw:
        row["input_index"] = int(raw["_input_index"])
    return row


def _enabled() -> bool:
    return os.environ.get("ALLOW_REAL_TOOL_API", "false").strip().lower() in {"1", "true", "yes", "on"}


def _timeout() -> float:
    try:
        value = float(os.environ.get("GEOCODE_TIMEOUT_SEC", _TIMEOUT_SEC))
    except ValueError:
        value = _TIMEOUT_SEC
    return value if value > 0 else _TIMEOUT_SEC


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _fail(message: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=message, error_code=error_code)


__all__ = ["GeocodeClient", "NominatimClient", "execute_geocode"]
