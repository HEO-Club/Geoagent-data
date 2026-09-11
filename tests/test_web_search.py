"""web_search 执行器测试；禁止真实付费 API。"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.web_search import _search as search_mod
from tool.web_search._search import WebSearchRequest


class FakeSearchEngine:
    """测试替身：记录请求并返回 Brave 风格载荷。"""

    name = "fake"
    max_top_k = 20

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload if payload is not None else _sample_brave()
        self.calls: list[WebSearchRequest] = []

    def search(self, request: WebSearchRequest) -> dict[str, Any]:
        self.calls.append(request)
        return self.payload


def _sample_brave() -> dict[str, Any]:
    return {
        "web": {
            "results": [
                {
                    "title": "Example Bridge Page",
                    "url": "https://news.example.com/bridge",
                    "description": "A historic stone railing railway bridge.",
                    "extra_snippets": ["Photographed in winter."],
                    "content": "FULL PAGE TEXT MUST NOT LEAK",
                    "raw_content": "<html>secret</html>",
                },
                {
                    "title": "Forum thread",
                    "url": "https://forum.example.com/topic/12",
                    "description": "Discussion of the same span.",
                },
            ]
        },
        "answer": "Must not appear in observation",
        "raw_content": "Must not appear either",
    }


def _nested_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(value)
        for item in value.values():
            found.update(_nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_nested_keys(item))
    return found


def _search(
    *,
    operation: str = "keyword_search",
    inputs: dict[str, object] | None = None,
    engine: FakeSearchEngine | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, object] = {"query": "铁路桥 石栏"}
    if inputs:
        payload.update(inputs)
    runtime = ctx
    if runtime is None:
        runtime = RuntimeContext(
            extras={
                "web_search_engine": engine if engine is not None else FakeSearchEngine(),
            }
        )
    elif engine is not None:
        runtime.extras["web_search_engine"] = engine
    return execute(
        "web_search",
        operation,
        purpose="检索网页线索",
        inputs=payload,
        ctx=runtime,
    )


def test_keyword_search_returns_title_url_snippet_and_result_id() -> None:
    engine = FakeSearchEngine()
    observation = _search(engine=engine)
    assert observation.ok is True
    assert observation.result is not None
    assert engine.calls and engine.calls[0].query == "铁路桥 石栏"
    assert engine.calls[0].top_k == 10
    results = observation.result["results"]
    assert results[0]["result_id"] == "ws_1"
    assert results[0]["title"] == "Example Bridge Page"
    assert results[0]["url"] == "https://news.example.com/bridge"
    assert "historic stone railing" in results[0]["snippet"]
    assert "Photographed in winter." in results[0]["snippet"]
    assert results[1]["result_id"] == "ws_2"
    assert observation.result["applied"]["engine"] == "fake"
    assert observation.result["applied"]["top_k"] == 10
    dumped = str(observation.result)
    assert "FULL PAGE TEXT MUST NOT LEAK" not in dumped
    assert "Must not appear" not in dumped
    keys = _nested_keys(observation.result)
    assert "raw_content" not in keys
    assert "answer" not in keys
    assert "content" not in keys
    assert "confirmed_location" not in keys
    assert any("全文" in item for item in observation.result["assumptions"])
    assert any("web_page_read" in item for item in observation.result["assumptions"])


def test_site_search_requires_site() -> None:
    observation = execute(
        "web_search",
        "site_search",
        purpose="站点检索",
        inputs={"query": "铁路桥"},
        ctx=RuntimeContext(extras={"web_search_engine": FakeSearchEngine()}),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "site" in observation.error


def test_site_search_adds_site_constraint() -> None:
    engine = FakeSearchEngine()
    observation = _search(
        operation="site_search",
        inputs={"site": "https://www.wikipedia.org/wiki/Bridge"},
        engine=engine,
    )
    assert observation.ok is True
    assert engine.calls[0].query == "铁路桥 石栏 site:wikipedia.org"
    assert observation.result is not None
    assert observation.result["applied"]["site"] == "wikipedia.org"
    assert "domains" not in observation.result["applied"]


def test_keyword_search_domains_become_site_operators() -> None:
    engine = FakeSearchEngine()
    observation = _search(
        inputs={"domains": ["https://news.example.com/a", "forum.example.com"]},
        engine=engine,
    )
    assert observation.ok is True
    assert engine.calls[0].query == (
        "铁路桥 石栏 (site:news.example.com OR site:forum.example.com)"
    )
    assert observation.result is not None
    assert observation.result["applied"]["domains"] == [
        "news.example.com",
        "forum.example.com",
    ]


def test_query_array_is_joined() -> None:
    engine = FakeSearchEngine()
    observation = _search(
        inputs={"query": ["铁路桥", "石栏", "历史照片"]},
        engine=engine,
    )
    assert observation.ok is True
    assert engine.calls[0].query == "铁路桥 石栏 历史照片"


def test_unmapped_time_range_is_reported_and_not_sent() -> None:
    engine = FakeSearchEngine()
    observation = _search(
        inputs={"time_range": "最早可用"},
        engine=engine,
    )
    assert observation.ok is True
    assert engine.calls[0].freshness is None
    assert observation.result is not None
    assert "time_range" not in observation.result["applied"]
    assert observation.result["applied"]["unsupported"]["time_range"] == "最早可用"


def test_mapped_year_range_is_sent_as_freshness() -> None:
    engine = FakeSearchEngine()
    observation = _search(
        inputs={"time_range": "1980-1995"},
        engine=engine,
    )
    assert observation.ok is True
    assert engine.calls[0].freshness == "1980-01-01to1995-12-31"
    assert observation.result is not None
    assert observation.result["applied"]["time_range"] == "1980-01-01to1995-12-31"
    assert "unsupported" not in observation.result["applied"]


def test_allow_real_api_false_without_engine_is_unavailable() -> None:
    observation = execute(
        "web_search",
        "keyword_search",
        purpose="闸门",
        inputs={"query": "铁路桥"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_API" in observation.error


def test_allow_real_api_true_missing_key_is_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "")
    monkeypatch.setenv("BRAVE_API_KEY", "")
    observation = execute(
        "web_search",
        "keyword_search",
        purpose="缺钥匙",
        inputs={"query": "铁路桥"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "BRAVE" in observation.error


def test_brave_engine_sends_token_and_strips_full_text(monkeypatch: Any) -> None:
    calls: list[Any] = []

    class _FakeHttpResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._body = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> _FakeHttpResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        parsed = urlparse(str(request.full_url))
        params = parse_qs(parsed.query)
        assert params["q"] == ["铁路桥 site:news.example.com"]
        assert params["count"] == ["10"]
        assert params["search_lang"] == ["zh-hans"]
        assert params["freshness"] == ["2020-01-01to2020-12-31"]
        headers = {key.lower(): value for key, value in request.header_items()}
        assert headers["x-subscription-token"] == "test-brave-key"
        return _FakeHttpResponse(_sample_brave())

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-brave-key")
    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setattr(search_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "web_search",
        "keyword_search",
        purpose="mock brave",
        inputs={
            "query": "铁路桥",
            "domains": ["news.example.com"],
            "language": "zh-CN",
            "time_range": "2020",
        },
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert calls, "应发起一次 Brave HTTP 请求"
    assert observation.result["applied"]["engine"] == "brave"
    assert observation.result["applied"]["language"] == "zh-hans"
    assert observation.result["results"][0]["url"] == "https://news.example.com/bridge"
    keys = _nested_keys(observation.result)
    assert "raw_content" not in keys
    assert "answer" not in keys
    dumped = str(observation.result)
    assert "FULL PAGE TEXT" not in dumped


def test_missing_query_is_missing_input() -> None:
    observation = execute(
        "web_search",
        "keyword_search",
        purpose="缺查询",
        inputs={},
        ctx=RuntimeContext(extras={"web_search_engine": FakeSearchEngine()}),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_invalid_top_k_is_rejected() -> None:
    observation = _search(inputs={"top_k": "many"})
    assert observation.ok is False
    assert observation.error_code == "invalid_top_k"


def test_top_k_is_capped_by_engine_limit() -> None:
    engine = FakeSearchEngine()
    observation = _search(inputs={"top_k": 50}, engine=engine)
    assert observation.ok is True
    assert engine.calls[0].top_k == 20
    assert observation.result is not None
    assert observation.result["applied"]["top_k"] == 20
