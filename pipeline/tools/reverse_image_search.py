"""以图搜图 adapter（SerpAPI）；外部客户端可注入/mock。"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from pipeline.config import get_settings


class ReverseImageSearchClient(Protocol):
    def search(self, image_path: str, bbox: list[float] | None) -> list[dict[str, str]]: ...


_client: Optional[ReverseImageSearchClient] = None


def set_client(client: Optional[ReverseImageSearchClient]) -> None:
    """测试用：注入或清除客户端。"""
    global _client
    _client = client


def _default_client() -> ReverseImageSearchClient:
    settings = get_settings()
    if settings.APP_ENV == "test" and not settings.ALLOW_REAL_API:
        raise RuntimeError("test 环境禁止真实 reverse_image_search 调用；请 mock set_client")
    if not settings.SERPAPI_KEY:
        raise ValueError("SERPAPI_KEY 未配置")

    from serpapi import GoogleSearch

    class _SerpAdapter:
        def search(self, image_path: str, bbox: list[float] | None) -> list[dict[str, str]]:
            _ = bbox
            result = GoogleSearch(
                {
                    "engine": "google_reverse_image",
                    "image_path": image_path,
                    "api_key": settings.SERPAPI_KEY,
                }
            ).get_dict()
            matches: list[dict[str, str]] = []
            for item in result.get("image_results", []) or result.get("visual_matches", []) or []:
                matches.append(
                    {
                        "title": str(item.get("title") or ""),
                        "snippet": str(item.get("snippet") or item.get("source") or ""),
                        "url": str(item.get("link") or item.get("url") or ""),
                    }
                )
            return matches

    return _SerpAdapter()


def execute(params: dict[str, Any], image_path: str) -> dict[str, Any]:
    """执行 reverse_image_search。"""
    bbox = params.get("bbox")
    try:
        client = _client if _client is not None else _default_client()
        matches = client.search(image_path, bbox)
        if not matches:
            return {"status": "empty", "error_message": None, "matches": None}
        return {"status": "success", "error_message": None, "matches": matches}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_message": str(exc), "matches": None}
