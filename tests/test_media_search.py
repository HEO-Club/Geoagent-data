"""media_search 执行器测试；禁止真实外网。"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.media_search import _search as search_mod
from tool.media_search._search import MediaSearchRequest


class FakeMediaProvider:
    """测试替身：记录请求并返回归一化或 Commons 风格载荷。"""

    name = "fake"
    max_top_k = 50

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload if payload is not None else _sample_hits()
        self.calls: list[MediaSearchRequest] = []

    def search(self, request: MediaSearchRequest) -> dict[str, Any]:
        self.calls.append(request)
        return self.payload


def _sample_hits() -> dict[str, Any]:
    return {
        "results": [
            {
                "media_id": "File:Yellow_River_bridge.jpg",
                "title": "Yellow River bridge",
                "url": "https://commons.wikimedia.org/wiki/File:Yellow_River_bridge.jpg",
                "source_url": "https://upload.wikimedia.org/wikipedia/commons/b/b1/bridge.jpg",
                "media_kind": "photo",
                "author": "<b>Jane Doe</b>",
                "license": "CC BY-SA 4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "uploaded_at": "2020-03-15T12:00:00Z",
                "captured_at": "2018-07-01 10:00:00",
                "snippet": "A historic stone railing railway bridge.",
                "taken_at": "MUST NOT LEAK",
                "confirmed_location": "Zhengzhou",
                "content": "FULL FILE BYTES MUST NOT LEAK",
            },
            {
                "media_id": "File:Bridge_winter.jpg",
                "title": "Bridge in winter",
                "url": "https://commons.wikimedia.org/wiki/File:Bridge_winter.jpg",
                "source_url": "https://upload.wikimedia.org/wikipedia/commons/w/w1/winter.jpg",
                "media_kind": "photo",
                "author": "Uploader",
                "license": "CC BY 4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "uploaded_at": "2021-01-02T08:00:00Z",
                "captured_at": "2020-06-15",
                "snippet": "Photographed in 2020.",
            },
            {
                "media_id": "File:Bridge_upload_only.jpg",
                "title": "Upload year only",
                "url": "https://commons.wikimedia.org/wiki/File:Bridge_upload_only.jpg",
                "source_url": "https://upload.wikimedia.org/wikipedia/commons/u/u1/upload.jpg",
                "media_kind": "photo",
                "author": "Unknown",
                "license": "Public domain",
                "license_url": "https://creativecommons.org/publicdomain/mark/1.0/",
                "uploaded_at": "2020-11-01T00:00:00Z",
                "captured_at": None,
                "snippet": "No capture date in metadata.",
            },
        ]
    }


def _sample_commons_payload() -> dict[str, Any]:
    return {
        "query": {
            "pages": [
                {
                    "pageid": 1,
                    "title": "File:Yellow_River_bridge.jpg",
                    "imageinfo": [
                        {
                            "timestamp": "2020-03-15T12:00:00Z",
                            "user": "Uploader",
                            "url": "https://upload.wikimedia.org/wikipedia/commons/b/b1/bridge.jpg",
                            "descriptionurl": (
                                "https://commons.wikimedia.org/wiki/File:Yellow_River_bridge.jpg"
                            ),
                            "mime": "image/jpeg",
                            "extmetadata": {
                                "DateTimeOriginal": {
                                    "value": "<time>2018-07-01 10:00:00</time>",
                                },
                                "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                "LicenseUrl": {
                                    "value": "https://creativecommons.org/licenses/by-sa/4.0/",
                                },
                                "Artist": {"value": "<a href=\"https://example.com\">Jane Doe</a>"},
                                "ImageDescription": {
                                    "value": "<p>A historic stone railing railway bridge.</p>",
                                },
                            },
                        }
                    ],
                }
            ]
        },
        "taken_at": "MUST NOT LEAK FROM PAYLOAD",
    }


def _sample_videos() -> dict[str, Any]:
    return {
        "results": [
            {
                "media_id": "File:Bridge_drone.webm",
                "title": "Bridge drone",
                "url": "https://commons.wikimedia.org/wiki/File:Bridge_drone.webm",
                "source_url": "https://upload.wikimedia.org/wikipedia/commons/v/v1/drone.webm",
                "media_kind": "video",
                "author": "Pilot",
                "license": "CC BY-SA 4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "uploaded_at": "2021-05-01T00:00:00Z",
                "captured_at": "2019-08-20",
                "snippet": "Aerial footage of the span.",
            }
        ]
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
    operation: str = "photo_search",
    inputs: dict[str, object] | None = None,
    provider: FakeMediaProvider | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, object] = {"query": "铁路桥 石栏"}
    if inputs:
        payload.update(inputs)
    runtime = ctx
    if runtime is None:
        runtime = RuntimeContext(
            extras={
                "media_search_provider": (
                    provider if provider is not None else FakeMediaProvider()
                ),
            }
        )
    elif provider is not None:
        runtime.extras["media_search_provider"] = provider
    return execute(
        "media_search",
        operation,
        purpose="检索公开影像",
        inputs=payload,
        ctx=runtime,
    )


def test_photo_search_returns_metadata_and_separates_upload_from_capture() -> None:
    provider = FakeMediaProvider()
    observation = _search(provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    assert provider.calls and provider.calls[0].query == "铁路桥 石栏"
    assert provider.calls[0].media_kind == "photo"
    assert provider.calls[0].top_k == 10
    results = observation.result["results"]
    assert results[0]["result_id"] == "ms_1"
    assert results[0]["media_id"] == "File:Yellow_River_bridge.jpg"
    assert results[0]["title"] == "Yellow River bridge"
    assert results[0]["url"] == "https://commons.wikimedia.org/wiki/File:Yellow_River_bridge.jpg"
    assert results[0]["license"] == "CC BY-SA 4.0"
    assert results[0]["author"] == "Jane Doe"
    assert results[0]["uploaded_at"] == "2020-03-15T12:00:00Z"
    assert results[0]["captured_at"] == "2018-07-01 10:00:00"
    assert results[0]["uploaded_at"] != results[0]["captured_at"]
    assert observation.result["applied"]["provider"] == "fake"
    keys = _nested_keys(observation.result)
    assert "taken_at" not in keys
    assert "confirmed_location" not in keys
    assert "content" not in keys
    dumped = str(observation.result)
    assert "MUST NOT LEAK" not in dumped
    assert "FULL FILE BYTES" not in dumped
    assumptions = observation.result["assumptions"]
    assert any("上传时间" in item for item in assumptions)
    assert any("使用权" in item for item in assumptions)
    assert any("未下载" in item for item in assumptions)
    assert any("Commons" in item for item in assumptions)


def test_time_range_filters_on_captured_at_not_upload() -> None:
    provider = FakeMediaProvider()
    observation = _search(inputs={"time_range": "2020"}, provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    results = observation.result["results"]
    assert [item["media_id"] for item in results] == ["File:Bridge_winter.jpg"]
    assert results[0]["captured_at"] == "2020-06-15"
    assert results[0]["uploaded_at"] != "2020-06-15"
    assert observation.result["applied"]["time_range"] == "2020-01-01to2020-12-31"
    assert "unsupported" not in observation.result["applied"]
    assert provider.calls[0].top_k == 50


def test_unparsed_time_range_is_recorded_not_fatal() -> None:
    provider = FakeMediaProvider()
    observation = _search(inputs={"time_range": "最早可用"}, provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    assert "time_range" not in observation.result["applied"]
    assert observation.result["applied"]["unsupported"]["time_range"] == "最早可用"
    assert len(observation.result["results"]) == 3


def test_video_search_uses_video_kind() -> None:
    provider = FakeMediaProvider(_sample_videos())
    observation = _search(operation="video_search", provider=provider)
    assert observation.ok is True
    assert provider.calls[0].media_kind == "video"
    assert observation.result is not None
    assert observation.result["operation"] == "video_search"
    assert observation.result["results"][0]["media_kind"] == "video"
    assert observation.result["results"][0]["media_id"] == "File:Bridge_drone.webm"


def test_youtube_only_platform_is_unavailable() -> None:
    observation = _search(
        operation="video_search",
        inputs={"platforms": ["youtube"]},
        provider=FakeMediaProvider(_sample_videos()),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "YouTube" in observation.error or "youtube" in observation.error.lower()


def test_mixed_sources_keep_commons_and_record_unsupported() -> None:
    observation = _search(inputs={"sources": ["commons", "europeana"]})
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["unsupported"]["sources"] == ["europeana"]


def test_photo_search_requires_query_or_area() -> None:
    observation = execute(
        "media_search",
        "photo_search",
        purpose="缺输入",
        inputs={},
        ctx=RuntimeContext(extras={"media_search_provider": FakeMediaProvider()}),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_photo_search_area_only_appends_place_name() -> None:
    provider = FakeMediaProvider()
    observation = execute(
        "media_search",
        "photo_search",
        purpose="地名检索",
        inputs={"area": "郑州附近黄河沿线"},
        ctx=RuntimeContext(extras={"media_search_provider": provider}),
    )
    assert observation.ok is True
    assert provider.calls[0].query == "郑州附近黄河沿线"
    assert observation.result is not None
    assert observation.result["applied"]["area"] == "郑州附近黄河沿线"


def test_video_search_requires_query() -> None:
    observation = execute(
        "media_search",
        "video_search",
        purpose="缺查询",
        inputs={"area": "郑州"},
        ctx=RuntimeContext(extras={"media_search_provider": FakeMediaProvider()}),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_allow_real_api_false_without_provider_is_unavailable() -> None:
    observation = execute(
        "media_search",
        "photo_search",
        purpose="闸门",
        inputs={"query": "铁路桥"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_API" in observation.error


def test_commons_http_sends_user_agent_and_strips_html(monkeypatch: Any) -> None:
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
        assert params["action"] == ["query"]
        assert params["generator"] == ["search"]
        search = params["gsrsearch"][0]
        assert "filetype:bitmap|drawing" in search or "filetype:bitmap%7Cdrawing" in search
        assert "铁路桥" in search
        assert "DateTimeOriginal" in params["iiextmetadatafilter"][0]
        assert "LicenseShortName" in params["iiextmetadatafilter"][0]
        headers = {key.lower(): value for key, value in request.header_items()}
        assert "geoagent-dataset" in headers["user-agent"]
        assert "media_search" in headers["user-agent"]
        return _FakeHttpResponse(_sample_commons_payload())

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setattr(search_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "media_search",
        "photo_search",
        purpose="mock commons",
        inputs={"query": "铁路桥"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert calls, "应发起一次 Commons HTTP 请求"
    assert observation.result["applied"]["provider"] == "wikimedia_commons"
    hit = observation.result["results"][0]
    assert hit["author"] == "Jane Doe"
    assert hit["captured_at"] == "2018-07-01 10:00:00"
    assert hit["uploaded_at"] == "2020-03-15T12:00:00Z"
    assert "<" not in hit["author"]
    assert "<" not in hit["snippet"]
    keys = _nested_keys(observation.result)
    assert "taken_at" not in keys
    dumped = str(observation.result)
    assert "MUST NOT LEAK" not in dumped


def test_commons_video_search_uses_video_filetype(monkeypatch: Any) -> None:
    calls: list[Any] = []

    class _FakeHttpResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._body = json.dumps({"query": {"pages": []}}).encode("utf-8")

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> _FakeHttpResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        params = parse_qs(urlparse(str(request.full_url)).query)
        assert "filetype:video" in params["gsrsearch"][0]
        return _FakeHttpResponse({"query": {"pages": []}})

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setattr(search_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "media_search",
        "video_search",
        purpose="mock video",
        inputs={"query": "铁路桥"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert calls
    assert observation.result is not None
    assert observation.result["results"] == []


def test_geo_area_uses_geosearch(monkeypatch: Any) -> None:
    calls: list[Any] = []

    class _FakeHttpResponse:
        def read(self) -> bytes:
            return json.dumps({"query": {"pages": []}}).encode("utf-8")

        def __enter__(self) -> _FakeHttpResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        params = parse_qs(urlparse(str(request.full_url)).query)
        assert params["generator"] == ["geosearch"]
        assert params["ggscoord"] == ["34.75|113.65"]
        assert params["ggsradius"] == ["5000"]
        return _FakeHttpResponse()

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setattr(search_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "media_search",
        "photo_search",
        purpose="geo",
        inputs={"area": {"lat": 34.75, "lon": 113.65, "radius_m": 5000}},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert calls
    assert observation.result is not None
    assert observation.result["applied"]["area"] == {
        "lat": 34.75,
        "lon": 113.65,
        "radius_m": 5000,
    }


def test_invalid_top_k_is_rejected() -> None:
    observation = _search(inputs={"top_k": "many"})
    assert observation.ok is False
    assert observation.error_code == "invalid_top_k"


def test_top_k_is_capped_by_provider_limit() -> None:
    provider = FakeMediaProvider()
    observation = _search(inputs={"top_k": 80}, provider=provider)
    assert observation.ok is True
    assert provider.calls[0].top_k == 50
    assert observation.result is not None
    assert observation.result["applied"]["top_k"] == 50
