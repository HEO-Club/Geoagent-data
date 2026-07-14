"""网页检索 adapter（Tavily）；外部客户端可注入/mock。"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from pipeline.config import get_settings


class WebSearchClient(Protocol):
    def search(self, query: str, max_results: int) -> list[dict[str, str]]: ...


_client: Optional[WebSearchClient] = None


def set_client(client: Optional[WebSearchClient]) -> None:
    """测试用：注入或清除 WebSearch 客户端。"""
    global _client
    _client = client


def _default_client() -> WebSearchClient:
    settings = get_settings()
    if settings.APP_ENV == "test" and not settings.ALLOW_REAL_API:
        raise RuntimeError("test 环境禁止真实 web_search 调用；请 mock set_client")
    if not settings.TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY 未配置")

    from tavily import TavilyClient

    raw = TavilyClient(api_key=settings.TAVILY_API_KEY)

    class _TavilyAdapter:
        def search(self, query: str, max_results: int) -> list[dict[str, str]]:
            resp = raw.search(query=query, max_results=max_results)
            results = []
            for item in resp.get("results", []):
                results.append(
                    {
                        "title": str(item.get("title") or ""),
                        "snippet": str(item.get("content") or item.get("snippet") or ""),
                        "url": str(item.get("url") or ""),
                    }
                )
            return results

    return _TavilyAdapter()


def execute(params: dict[str, Any], image_path: str) -> dict[str, Any]:
    """执行 web_search；image_path 由分发器统一传入，本 tool 不使用。"""
    _ = image_path
    query = str(params["query"])
    top_k = int(params.get("top_k", 3))
    try:
        client = _client if _client is not None else _default_client()
        hits = client.search(query, top_k)
        if not hits:
            return {"status": "empty", "error_message": None, "results": None}
        return {
            "status": "success",
            "error_message": None,
            "results": hits[:top_k],
        }
    except Exception as exc:  # noqa: BLE001 — 转为结构化 error observation
        return {
            "status": "error",
            "error_message": str(exc),
            "results": None,
        }
