"""地图查询 adapter（Google Maps）；外部客户端可注入/mock。"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from pipeline.config import get_settings


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


def _default_client() -> MapQueryClient:
    settings = get_settings()
    if settings.APP_ENV == "test" and not settings.ALLOW_REAL_API:
        raise RuntimeError("test 环境禁止真实 map_query 调用；请 mock set_client")
    if not settings.GOOGLE_MAPS_KEY:
        raise ValueError("GOOGLE_MAPS_KEY 未配置")

    import googlemaps

    gmaps = googlemaps.Client(key=settings.GOOGLE_MAPS_KEY)

    class _GMapsAdapter:
        def query(self, query: str | None, latlng: list[float] | None) -> dict[str, Any]:
            if query:
                results = gmaps.geocode(query)
                if not results:
                    return {
                        "status": "empty",
                        "error_message": None,
                        "formatted_address": None,
                        "resolved_latlng": None,
                        "place_type": None,
                    }
                top = results[0]
                loc = top["geometry"]["location"]
                types = top.get("types") or []
                return {
                    "status": "success",
                    "error_message": None,
                    "formatted_address": top.get("formatted_address"),
                    "resolved_latlng": [float(loc["lat"]), float(loc["lng"])],
                    "place_type": types[0] if types else None,
                }
            assert latlng is not None
            results = gmaps.reverse_geocode((latlng[0], latlng[1]))
            if not results:
                return {
                    "status": "empty",
                    "error_message": None,
                    "formatted_address": None,
                    "resolved_latlng": None,
                    "place_type": None,
                }
            top = results[0]
            loc = top["geometry"]["location"]
            types = top.get("types") or []
            return {
                "status": "success",
                "error_message": None,
                "formatted_address": top.get("formatted_address"),
                "resolved_latlng": [float(loc["lat"]), float(loc["lng"])],
                "place_type": types[0] if types else None,
            }

    return _GMapsAdapter()


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
