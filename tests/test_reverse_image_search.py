"""reverse_image_search 执行器测试；禁止真实付费 API。"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.reverse_image_search import _search as search_mod

_JPEG_QUALITY = 95


class FakeSearchEngine:
    """测试替身：记录提交字节并返回预设 Web Detection 载荷。"""

    name = "fake"

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload if payload is not None else _sample_payload()
        self.calls: list[dict[str, Any]] = []

    def search(self, image_bytes: bytes, *, top_k: int) -> dict[str, Any]:
        self.calls.append({"image_bytes": image_bytes, "top_k": top_k})
        return self.payload


def _sample_payload() -> dict[str, Any]:
    return {
        "fullMatchingImages": [{"url": "https://cdn.example.com/full.jpg", "score": 0.9}],
        "partialMatchingImages": [{"url": "https://cdn.example.com/part.jpg"}],
        "pagesWithMatchingImages": [
            {
                "url": "https://news.example.com/bridge",
                "pageTitle": "Example Bridge Page",
                "fullMatchingImages": [{"url": "https://cdn.example.com/page-full.jpg"}],
            }
        ],
        "visuallySimilarImages": [{"url": "https://cdn.example.com/similar.jpg"}],
        "webEntities": [
            {"entityId": "/m/bridge", "description": "bridge", "score": 1.2},
        ],
        "bestGuessLabels": [{"label": "cable-stayed bridge", "languageCode": "en"}],
    }


def _gradient(path: Path, width: int = 40, height: int = 24) -> Path:
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    assert pixels is not None
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (x * 6, y * 10, 80)
    image.save(path)
    return path


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


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=_JPEG_QUALITY)
    return buffer.getvalue()


def _search(
    tmp_path: Path,
    *,
    source: Path | None = None,
    operation: str = "search",
    inputs: dict[str, object] | None = None,
    engine: FakeSearchEngine | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    image = source if source is not None else _gradient(tmp_path / "src.png")
    payload: dict[str, object] = {"image": str(image)}
    if inputs:
        payload.update(inputs)
    runtime = ctx
    if runtime is None:
        runtime = RuntimeContext(
            extras={
                "artifact_dir": str(tmp_path / "out"),
                "reverse_image_search_engine": engine if engine is not None else FakeSearchEngine(),
            }
        )
    elif engine is not None:
        runtime.extras.setdefault("artifact_dir", str(tmp_path / "out"))
        runtime.extras["reverse_image_search_engine"] = engine
    return execute(
        "reverse_image_search",
        operation,
        purpose="以图搜图",
        inputs=payload,
        ctx=runtime,
    )


def test_search_returns_urls_and_match_types_without_location_claim(tmp_path: Path) -> None:
    engine = FakeSearchEngine()
    source = _gradient(tmp_path / "full.png")
    observation = _search(tmp_path, source=source, engine=engine)
    assert observation.ok is True
    assert observation.result is not None
    assert engine.calls and engine.calls[0]["top_k"] == 10
    with Image.open(source) as opened:
        expected = _jpeg_bytes(opened.convert("RGB"))
    assert engine.calls[0]["image_bytes"] == expected

    types = [item["match_type"] for item in observation.result["matches"]]
    assert types == ["full", "partial", "page", "similar"]
    assert observation.result["matches"][0]["image_url"] == "https://cdn.example.com/full.jpg"
    assert observation.result["matches"][2]["page_url"] == "https://news.example.com/bridge"
    assert observation.result["matches"][2]["title"] == "Example Bridge Page"
    assert observation.result["pages"][0]["url"] == "https://news.example.com/bridge"
    assert observation.result["entities"][0]["description"] == "bridge"
    assert observation.result["best_guess_labels"][0]["label"] == "cable-stayed bridge"
    assert observation.result["applied"]["engines"] == ["serpapi"]
    assert observation.result["applied"]["top_k"] == 10
    dumped = str(observation.result)
    assert "确认拍摄于" not in dumped
    assert "confirmed_location" not in _nested_keys(observation.result)
    assert any("地点" in item for item in observation.result["assumptions"])


def test_search_crop_submits_only_cropped_jpeg(tmp_path: Path) -> None:
    engine = FakeSearchEngine()
    source = _gradient(tmp_path / "src.png")
    observation = _search(
        tmp_path,
        source=source,
        operation="search_crop",
        inputs={"region": [10, 4, 20, 14]},
        engine=engine,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["region"] == [10, 4, 20, 14]
    assert observation.result["query_image_id"]
    crop_path = Path(observation.artifacts["image_path"])
    assert crop_path.is_file()
    cropped = Image.open(crop_path)
    assert cropped.size == (10, 10)

    with Image.open(source) as opened:
        expected_crop = _jpeg_bytes(opened.convert("RGB").crop((10, 4, 20, 14)))
        full_jpeg = _jpeg_bytes(opened.convert("RGB"))
    assert engine.calls[0]["image_bytes"] == expected_crop
    assert engine.calls[0]["image_bytes"] != full_jpeg


def test_current_image_named_region_and_default_top_k(tmp_path: Path) -> None:
    source = _gradient(tmp_path / "named.png")
    engine = FakeSearchEngine()
    current = execute(
        "reverse_image_search",
        "search",
        purpose="当前图",
        inputs={"image": "$current_image"},
        ctx=RuntimeContext(
            current_image=str(source),
            extras={
                "artifact_dir": str(tmp_path / "cur"),
                "reverse_image_search_engine": engine,
            },
        ),
    )
    assert current.ok is True
    assert current.result is not None
    assert engine.calls[0]["top_k"] == 10

    named_engine = FakeSearchEngine()
    named = execute(
        "reverse_image_search",
        "search_crop",
        purpose="命名区域",
        inputs={"image": str(source), "region": "bridge_tower_top"},
        ctx=RuntimeContext(
            extras={
                "artifact_dir": str(tmp_path / "named"),
                "named_regions": {"bridge_tower_top": [2, 2, 20, 12]},
                "reverse_image_search_engine": named_engine,
            },
        ),
    )
    assert named.ok is True
    assert named.result is not None
    assert named.result["applied"]["region"] == [2, 2, 20, 12]


def test_top_k_truncates_and_dedupes_urls(tmp_path: Path) -> None:
    engine = FakeSearchEngine(
        {
            "fullMatchingImages": [
                {"url": "https://cdn.example.com/a.jpg"},
                {"url": "https://cdn.example.com/b.jpg"},
            ],
            "partialMatchingImages": [
                {"url": "https://cdn.example.com/a.jpg"},
                {"url": "https://cdn.example.com/c.jpg"},
            ],
            "pagesWithMatchingImages": [{"url": "https://news.example.com/d"}],
            "visuallySimilarImages": [
                {"url": "https://cdn.example.com/e.jpg"},
                {"url": "https://cdn.example.com/f.jpg"},
            ],
        }
    )
    observation = _search(tmp_path, inputs={"top_k": 3}, engine=engine)
    assert observation.ok is True
    assert observation.result is not None
    urls = [item["url"] for item in observation.result["matches"]]
    assert urls == [
        "https://cdn.example.com/a.jpg",
        "https://cdn.example.com/b.jpg",
        "https://cdn.example.com/c.jpg",
    ]
    assert engine.calls[0]["top_k"] == 3


def test_strips_confirmed_location_from_engine_payload(tmp_path: Path) -> None:
    engine = FakeSearchEngine(
        {
            "fullMatchingImages": [
                {
                    "url": "https://cdn.example.com/full.jpg",
                    "confirmed_location": "Paris",
                    "taken_at": "Eiffel Tower",
                }
            ],
            "confirmed_location": "Paris",
            "taken_at": "Eiffel Tower",
        }
    )
    observation = _search(tmp_path, engine=engine)
    assert observation.ok is True
    assert observation.result is not None
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert "taken_at" not in keys
    evidence = (
        str(observation.result["matches"])
        + str(observation.result["pages"])
        + str(observation.result["entities"])
        + str(observation.result["best_guess_labels"])
    )
    assert "Paris" not in evidence
    assert "Eiffel Tower" not in evidence
    assert observation.result["matches"][0]["url"] == "https://cdn.example.com/full.jpg"


def test_missing_unknown_engine_and_api_gate(tmp_path: Path) -> None:
    missing = execute("reverse_image_search", "search", purpose="缺图", inputs={})
    assert missing.ok is False
    assert missing.error_code == "missing_input"

    missing_region = execute(
        "reverse_image_search",
        "search_crop",
        purpose="缺区域",
        inputs={"image": str(_gradient(tmp_path / "need_region.png"))},
        ctx=RuntimeContext(
            extras={
                "artifact_dir": str(tmp_path / "miss_region"),
                "reverse_image_search_engine": FakeSearchEngine(),
            }
        ),
    )
    assert missing_region.ok is False
    assert missing_region.error_code == "missing_input"

    missing_file = _search(tmp_path, source=tmp_path / "nope.png")
    assert missing_file.ok is False
    assert missing_file.error_code == "image_not_found"

    unknown = _search(tmp_path, inputs={"engines": ["yandex"]})
    assert unknown.ok is False
    assert unknown.error_code == "unsupported_engine"

    mixed = _search(tmp_path, inputs={"engines": ["vision", "serpapi"]})
    assert mixed.ok is True
    assert mixed.result is not None
    assert mixed.result["applied"]["engines"] == ["google_cloud_vision", "serpapi"]
    assert "ignored" not in mixed.result["applied"]

    alias = _search(tmp_path, inputs={"engines": ["serpai", "google_lens"]})
    assert alias.ok is True
    assert alias.result is not None
    assert alias.result["applied"]["engines"] == ["serpapi"]

    gated = execute(
        "reverse_image_search",
        "search",
        purpose="禁网",
        inputs={"image": str(_gradient(tmp_path / "gate.png"))},
        ctx=RuntimeContext(extras={"artifact_dir": str(tmp_path / "gate")}),
    )
    assert gated.ok is False
    assert gated.error_code == "engine_unavailable"


def test_unresolved_named_region(tmp_path: Path) -> None:
    observation = _search(
        tmp_path,
        operation="search_crop",
        inputs={"region": "not_a_region"},
    )
    assert observation.ok is False
    assert observation.error_code == "unresolved_region"


class _FakeHttpResponse:
    """urlopen 替身：返回预设 JSON 字节。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_serpapi_lens_payload_maps_to_urls_and_labels(tmp_path: Path) -> None:
    lens = {
        "exact_matches": [
            {
                "title": "Example Bridge",
                "link": "https://news.example.com/bridge",
                "thumbnail": "https://cdn.example.com/full.jpg",
            }
        ],
        "visual_matches": [
            {
                "title": "Similar span",
                "link": "https://photos.example.com/span",
                "image": "https://cdn.example.com/similar.jpg",
            }
        ],
        "knowledge_graph": {"title": "cable-stayed bridge", "kgmid": "/m/bridge"},
    }
    engine = FakeSearchEngine(search_mod._lens_to_web_detection(lens))
    observation = _search(tmp_path, engine=engine)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["matches"][0]["match_type"] == "full"
    assert observation.result["matches"][0]["url"] == "https://cdn.example.com/full.jpg"
    types = [item["match_type"] for item in observation.result["matches"]]
    assert "page" in types
    assert "similar" in types
    assert observation.result["pages"][0]["url"] == "https://news.example.com/bridge"
    assert observation.result["best_guess_labels"][0]["label"] == "cable-stayed bridge"
    assert observation.result["entities"][0]["entity_id"] == "/m/bridge"


def test_serpapi_engine_uploads_then_searches(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[str] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        url = str(request.full_url)
        calls.append(url)
        if "search.json" in url:
            assert "engine=google_lens" in url
            assert "image_id=img_test_1" in url
            assert "api_key=test-serpapi-key" in url
            return _FakeHttpResponse(
                {
                    "visual_matches": [
                        {
                            "title": "Source page",
                            "link": "https://news.example.com/found",
                            "image": "https://cdn.example.com/hit.jpg",
                        }
                    ],
                    "knowledge_graph": {"title": "found object"},
                }
            )
        body = request.data or b""
        assert b'name="api_key"' in body
        assert b"test-serpapi-key" in body
        assert b"query.jpg" in body
        return _FakeHttpResponse({"image_id": "img_test_1"})

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-serpapi-key")
    monkeypatch.setenv("SERPAPI_KEY", "")
    monkeypatch.setattr(search_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "reverse_image_search",
        "search",
        purpose="mock lens",
        inputs={"image": str(_gradient(tmp_path / "lens.png"))},
        ctx=RuntimeContext(extras={"artifact_dir": str(tmp_path / "lens_out")}),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["engine"] == "serpapi"
    assert observation.result["pages"][0]["url"] == "https://news.example.com/found"
    assert observation.result["best_guess_labels"][0]["label"] == "found object"
    assert len(calls) == 2


def test_serpapi_missing_key_is_engine_unavailable(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("SERPAPI_API_KEY", "")
    monkeypatch.setenv("SERPAPI_KEY", "")
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "")
    observation = execute(
        "reverse_image_search",
        "search",
        purpose="缺钥匙",
        inputs={"image": str(_gradient(tmp_path / "nokey.png")), "engines": ["serpapi"]},
        ctx=RuntimeContext(extras={"artifact_dir": str(tmp_path / "nokey")}),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "SERPAPI" in observation.error
