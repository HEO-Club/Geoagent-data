"""streetview_query 执行器测试；禁止真实付费 API。"""

from __future__ import annotations

import importlib
import json

import pytest
from io import BytesIO
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.runtime import FilesystemImageStore
from tool.streetview_query._streetview import (
    BBox,
    ImageSearchRequest,
    MapillaryStreetviewProvider,
)


class FakeStreetviewProvider:
    """测试替身：固定影像/序列/PNG，不访问网络。"""

    name = "fake"
    crs = "wgs84"

    def __init__(
        self,
        images: list[dict[str, Any]] | None = None,
        *,
        sequences: dict[str, list[str]] | None = None,
        png: bytes | None = None,
    ) -> None:
        self.images = {str(item["id"]): item for item in (images if images is not None else _sample_images())}
        self.sequences = sequences if sequences is not None else {"seq_east": ["img_a", "img_b", "img_c"]}
        self.png = png if png is not None else _png_bytes()
        self.search_calls: list[ImageSearchRequest] = []
        self.get_calls: list[str] = []
        self.sequence_calls: list[str] = []
        self.fetch_calls: list[str] = []

    def search_images(self, request: ImageSearchRequest) -> list[dict[str, Any]]:
        self.search_calls.append(request)
        hits: list[dict[str, Any]] = []
        for item in self.images.values():
            lon, lat = _image_lonlat(item)
            if lon is None or lat is None:
                continue
            if (
                request.bbox.west <= lon <= request.bbox.east
                and request.bbox.south <= lat <= request.bbox.north
            ):
                hits.append(dict(item))
        return hits

    def get_image(self, image_id: str) -> dict[str, Any]:
        self.get_calls.append(image_id)
        item = self.images.get(image_id)
        if item is None:
            return {"id": image_id}
        return dict(item)

    def list_sequence(self, sequence_id: str) -> list[str]:
        self.sequence_calls.append(sequence_id)
        return list(self.sequences.get(sequence_id, []))

    def fetch_image_bytes(self, url: str) -> bytes:
        self.fetch_calls.append(url)
        return self.png


def _png_bytes(color: tuple[int, int, int] = (30, 120, 80), size: tuple[int, int] = (16, 8)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_lonlat(item: dict[str, Any]) -> tuple[float | None, float | None]:
    geom = item.get("geometry") or item.get("computed_geometry")
    if isinstance(geom, dict):
        coords = geom.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return float(coords[0]), float(coords[1])
    return None, None


def _sample_images() -> list[dict[str, Any]]:
    return [
        {
            "id": "img_a",
            "captured_at": 1_557_619_200_000,
            "compass_angle": 90.0,
            "geometry": {"type": "Point", "coordinates": [113.6700, 34.8900]},
            "sequence": "seq_east",
            "is_pano": False,
            "thumb_1024_url": "https://example.test/a.jpg",
            "confirmed_location": "MUST NOT LEAK",
            "raw_content": "FULL MAPILLARY JSON MUST NOT LEAK",
        },
        {
            "id": "img_b",
            "captured_at": "2019-05-12T12:00:00Z",
            "compass_angle": 90.0,
            "geometry": {"type": "Point", "coordinates": [113.6704, 34.8900]},
            "sequence": "seq_east",
            "is_pano": False,
            "thumb_1024_url": "https://example.test/b.jpg",
        },
        {
            "id": "img_c",
            "captured_at": "2019-05-12T12:01:00Z",
            "compass_angle": 88.0,
            "geometry": {"type": "Point", "coordinates": [113.6708, 34.8900]},
            "sequence": "seq_east",
            "is_pano": True,
            "thumb_2048_url": "https://example.test/c.jpg",
        },
        {
            "id": "img_2023",
            "captured_at": "2023-01-08T08:00:00Z",
            "compass_angle": 12.0,
            "geometry": {"type": "Point", "coordinates": [113.67005, 34.89002]},
            "sequence": "seq_2023",
            "is_pano": False,
            "thumb_1024_url": "https://example.test/2023.jpg",
        },
        {
            "id": "img_old",
            "captured_at": "2015-03-01",
            "compass_angle": 10.0,
            "geometry": {"type": "Point", "coordinates": [113.6701, 34.8901]},
            "sequence": "seq_old",
            "is_pano": False,
            "thumb_1024_url": "https://example.test/old.jpg",
        },
    ]


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


def _call(
    operation: str,
    *,
    inputs: dict[str, Any],
    provider: FakeStreetviewProvider | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    runtime = ctx
    engine = provider if provider is not None else FakeStreetviewProvider()
    if runtime is None:
        runtime = RuntimeContext(extras={"streetview_query_provider": engine})
    elif provider is not None:
        runtime.extras["streetview_query_provider"] = engine
    return execute(
        "streetview_query",
        operation,
        purpose="街景",
        inputs=inputs,
        ctx=runtime,
    )


def test_empty_inputs_are_missing_input() -> None:
    for operation in ("open", "navigate", "change_time", "capture"):
        observation = execute(
            "streetview_query",
            operation,
            purpose="scaffold",
            inputs={},
        )
        assert observation.ok is False
        assert observation.error_code == "missing_input", operation


def test_open_with_coordinates_creates_session() -> None:
    provider = FakeStreetviewProvider()
    observation = _call("open", inputs={"coordinates": [113.67, 34.89]}, provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.session == "sv_0001"
    assert observation.result["coverage"] is True
    assert observation.result["viewpoint"]["image_id"] == "img_a"
    assert observation.result["viewpoint"]["captured_at"] == "2019-05-12"
    assert observation.result["applied"]["provider"] == "fake"
    assert observation.result["applied"]["crs"] == "wgs84"
    assert provider.search_calls
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert "raw_content" not in keys


class _PlaceNameGeocoder:
    name = "fake_geocoder"

    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        limit: int,
        language: str | None,
        viewbox: tuple[float, float, float, float] | None,
    ) -> list[dict[str, Any]]:
        self.search_calls.append(
            {"query": query, "limit": limit, "language": language, "viewbox": viewbox}
        )
        return [
            {
                "place_id": 1,
                "lat": "34.8900",
                "lon": "113.6700",
                "display_name": query,
            }
        ]

    def reverse(self, lat: float, lon: float, *, language: str | None) -> dict[str, Any]:
        del language
        return {"lat": str(lat), "lon": str(lon), "display_name": "reverse"}


def test_open_place_name_uses_internal_geocode() -> None:
    geocoder = _PlaceNameGeocoder()
    provider = FakeStreetviewProvider()
    ctx = RuntimeContext(
        extras={
            "streetview_query_provider": provider,
            "geocode_client": geocoder,
        }
    )
    observation = _call("open", inputs={"area": "郑州市"}, provider=provider, ctx=ctx)
    assert observation.ok is True
    assert observation.result is not None
    assert geocoder.search_calls
    assert geocoder.search_calls[0]["query"] == "郑州市"
    assert observation.result["applied"]["geocode_query"] == "郑州市"
    assert observation.result["applied"]["center"] == {"lat": 34.89, "lon": 113.67}


def test_open_reads_nested_geocode_location(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_geocode(*, purpose: str, inputs: dict[str, Any], ctx: object = None) -> Observation:
        del purpose, inputs, ctx
        return Observation(
            ok=True,
            result={
                "results": [
                    {"location": {"lat": 34.89, "lon": 113.67, "crs": "EPSG:4326"}}
                ],
                "applied": {"provider": "amap"},
            },
        )

    monkeypatch.setattr(importlib.import_module("tool.geocode.geocode"), "execute", fake_geocode)
    provider = FakeStreetviewProvider()
    observation = _call("open", inputs={"area": "郑州市"}, provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["center"] == {"lat": 34.89, "lon": 113.67}
    assert observation.result["applied"]["geocode_provider"] == "amap"
    assert "poi_query" not in observation.result["applied"]


def test_open_place_name_falls_back_to_poi(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_geocode(*, purpose: str, inputs: dict[str, Any], ctx: object = None) -> Observation:
        del purpose, inputs, ctx
        return Observation(ok=True, result={"results": [], "applied": {"provider": "amap"}})

    def fake_poi(*, purpose: str, inputs: dict[str, Any], ctx: object = None) -> Observation:
        del purpose, ctx
        assert inputs["query"]
        return Observation(
            ok=True,
            result={
                "candidates": [{"location": {"lat": 25.86, "lon": 115.01}}],
                "provider": "amap",
            },
        )

    monkeypatch.setattr(importlib.import_module("tool.geocode.geocode"), "execute", fake_geocode)
    monkeypatch.setattr(importlib.import_module("tool.poi_search.poi_search"), "execute", fake_poi)
    provider = FakeStreetviewProvider()
    observation = _call("open", inputs={"area": "某地桥梁"}, provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["poi_query"] == "某地桥梁"
    assert observation.result["applied"]["center"] == {"lat": 25.86, "lon": 115.01}


def test_open_place_name_without_geocode_backend_is_unavailable() -> None:
    observation = _call("open", inputs={"area": "郑州市"})
    assert observation.ok is False
    assert observation.error_code in {"engine_unavailable", "provider_unavailable"}
    assert observation.error is not None


def test_open_with_no_nearby_images_is_no_coverage() -> None:
    provider = FakeStreetviewProvider()
    observation = _call(
        "open",
        inputs={"coordinates": [10.0, 50.0]},
        provider=provider,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["coverage"] is False
    assert observation.result["viewpoint"] is None
    assert observation.session == "sv_0001"


def test_navigate_moves_along_sequence() -> None:
    provider = FakeStreetviewProvider()
    ctx = RuntimeContext(extras={"streetview_query_provider": provider})
    opened = _call("open", inputs={"coordinates": [113.67, 34.89]}, provider=provider, ctx=ctx)
    assert opened.result is not None
    assert opened.result["viewpoint"]["image_id"] == "img_a"
    moved = _call(
        "navigate",
        inputs={"session": opened.session, "direction": "东"},
        provider=provider,
        ctx=ctx,
    )
    assert moved.ok is True
    assert moved.result is not None
    assert moved.result["moved"] is True
    assert moved.result["viewpoint"]["image_id"] == "img_b"
    assert moved.session == opened.session


def test_change_time_hits_year_and_miss_does_not_keep_current() -> None:
    provider = FakeStreetviewProvider()
    ctx = RuntimeContext(extras={"streetview_query_provider": provider})
    opened = _call("open", inputs={"coordinates": [113.67, 34.89]}, provider=provider, ctx=ctx)
    assert opened.result is not None
    assert opened.result["viewpoint"]["image_id"] == "img_a"
    hit = _call(
        "change_time",
        inputs={"session": opened.session, "time_range": "2023"},
        provider=provider,
        ctx=ctx,
    )
    assert hit.ok is True
    assert hit.result is not None
    assert hit.result["coverage"] is True
    assert hit.result["viewpoint"]["image_id"] == "img_2023"
    assert hit.result["viewpoint"]["captured_at"] == "2023-01-08"
    assert "2015-03-01" in hit.result["available_dates"]
    miss = _call(
        "change_time",
        inputs={"session": opened.session, "time_range": "2010"},
        provider=provider,
        ctx=ctx,
    )
    assert miss.ok is True
    assert miss.result is not None
    assert miss.result["coverage"] is False
    assert miss.result["viewpoint"] is None
    assert "2010-01-01" not in miss.result["available_dates"]


def test_change_time_earliest_picks_oldest_captured_at() -> None:
    provider = FakeStreetviewProvider()
    observation = _call(
        "change_time",
        inputs={"area": {"lat": 34.89, "lon": 113.67}, "time_range": "earliest"},
        provider=provider,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["coverage"] is True
    assert observation.result["viewpoint"]["image_id"] == "img_old"
    assert observation.result["viewpoint"]["captured_at"] == "2015-03-01"


def test_capture_writes_image_store_and_pano_heading(tmp_path: Any) -> None:
    provider = FakeStreetviewProvider()
    store = FilesystemImageStore(tmp_path / "store")
    ctx = RuntimeContext(
        extras={"streetview_query_provider": provider},
        image_store=store,
    )
    opened = _call("open", inputs={"coordinates": [113.67, 34.89]}, provider=provider, ctx=ctx)
    plain = _call(
        "capture",
        inputs={"session": opened.session},
        provider=provider,
        ctx=ctx,
    )
    assert plain.ok is True
    assert plain.result is not None
    assert plain.result["image_id"]
    assert "image_ids" in plain.artifacts
    assert any("不是全景" in item for item in plain.result["assumptions"])
    assert plain.result["applied"]["heading"] == 90.0
    assert provider.fetch_calls

    east = _call(
        "navigate",
        inputs={"session": opened.session, "direction": "forward", "distance_m": 200},
        provider=provider,
        ctx=ctx,
    )
    assert east.result is not None
    assert east.result["viewpoint"]["image_id"] == "img_c"
    pano = _call(
        "capture",
        inputs={"session": opened.session, "heading": 45, "pitch": 0, "fov": 80},
        provider=provider,
        ctx=ctx,
    )
    assert pano.ok is True
    assert pano.result is not None
    assert pano.result["applied"]["heading"] == 45
    assert pano.result["applied"]["pano"] is True
    assert pano.result["applied"]["width"] == 640
    assert pano.result["applied"]["height"] == 480
    leftover = list((tmp_path / "store").glob("img_*.*"))
    assert leftover == []


def test_allow_real_api_false_without_provider_is_unavailable() -> None:
    observation = execute(
        "streetview_query",
        "open",
        purpose="闸门",
        inputs={"coordinates": [113.67, 34.89]},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_TOOL_API" in observation.error


def test_allow_real_api_true_mapillary_missing_token_is_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_TOOL_API", "true")
    monkeypatch.setenv("STREETVIEW_PROVIDER", "mapillary")
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", "")
    monkeypatch.setenv("MAPILLARY_TOKEN", "")
    observation = execute(
        "streetview_query",
        "open",
        purpose="缺令牌",
        inputs={"coordinates": [113.67, 34.89]},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "MAPILLARY" in observation.error


def test_mapillary_search_sends_radius_and_token(monkeypatch: Any) -> None:
    calls: list[str] = []

    class _FakeHttpResponse:
        def __init__(self, payload: Any) -> None:
            self._body = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> _FakeHttpResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: Any, timeout: float = 0) -> _FakeHttpResponse:
        del timeout
        calls.append(request.full_url)
        return _FakeHttpResponse(
            {
                "data": [
                    {
                        "id": "mly_1",
                        "captured_at": 1_557_619_200_000,
                        "compass_angle": 12,
                        "geometry": {"type": "Point", "coordinates": [113.67, 34.89]},
                        "sequence": "seq",
                        "is_pano": False,
                        "thumb_1024_url": "https://example.test/m.jpg",
                    }
                ]
            }
        )

    monkeypatch.setattr("tool.streetview_query._streetview.urllib.request.urlopen", fake_urlopen)
    provider = MapillaryStreetviewProvider(
        access_token="test-token",
        endpoint="https://graph.mapillary.com",
        timeout_sec=5.0,
        user_agent="geoagent-test",
    )
    rows = provider.search_images(
        ImageSearchRequest(
            bbox=BBox(west=113.6, south=34.8, east=113.7, north=34.9),
            limit=20,
        )
    )
    assert rows and rows[0]["id"] == "mly_1"
    assert calls
    parsed = urlparse(calls[0])
    query = parse_qs(parsed.query)
    assert query["access_token"] == ["test-token"]
    assert "bbox" not in query
    assert query["lat"] == ["34.85"]
    assert query["lng"] == ["113.65"]
    assert query["radius"] == ["50"]
    assert query["limit"] == ["20"]
    fields = query["fields"][0]
    assert "id" in fields
    assert "geometry" in fields
    assert "thumb_1024_url" not in fields


def _jpeg_bytes(color: tuple[int, int, int] = (40, 80, 120)) -> bytes:
    image = Image.new("RGB", (32, 24), color=color)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class _FakeHttpResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _google_ok_payload() -> list[Any]:
    panoid = "ABCDEFGHIJKLMNOPQRSTUV"
    hist_id = "HISTPANO1234567890123A"
    return [
        [0],
        [
            None,
            [None, panoid],
            None,
            [None, None, [["Shibuya"]]],
            None,
            [
                [
                    None,
                    [
                        [None, None, 35.6595, 139.7005],
                        None,
                        [47.0],
                    ],
                    None,
                    [
                        [
                            [
                                [None, hist_id],
                                None,
                                [[None, None, 35.6600, 139.7010]],
                            ]
                        ]
                    ],
                    None,
                    None,
                    None,
                    None,
                    [[0, [2018, 7]]],
                ]
            ],
            [None, None, None, None, None, None, None, [2019, 5]],
        ],
    ]


def test_zhengzhou_open_uses_baidu_and_navigate_does_not_rescan(monkeypatch: Any) -> None:
    calls: list[str] = []
    jpeg = _jpeg_bytes()

    def fake_urlopen(request: Any, timeout: float = 0) -> _FakeHttpResponse:
        del timeout
        url = request.full_url
        calls.append(url)
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        qt = (query.get("qt") or [""])[0]
        if qt == "qsdata":
            x = int(query["x"][0])
            y = int(query["y"][0])
            return _FakeHttpResponse(
                json.dumps({"content": {"id": "bd_1", "x": x * 100, "y": y * 100}}).encode("utf-8")
            )
        if qt == "sdata":
            base_x = 1_269_000_000
            base_y = 356_000_000
            for prior in calls:
                prior_q = parse_qs(urlparse(prior).query)
                if (prior_q.get("qt") or [""])[0] == "qsdata":
                    base_x = int(prior_q["x"][0]) * 100
                    base_y = int(prior_q["y"][0]) * 100
                    break
            payload = {
                "content": [
                    {
                        "Date": "20170920",
                        "X": base_x,
                        "Y": base_y,
                        "MoveDir": 90,
                        "Roads": [
                            {
                                "Panos": [
                                    {"PID": "bd_1", "X": base_x, "Y": base_y},
                                    {"PID": "bd_2", "X": base_x + 50_000, "Y": base_y},
                                ]
                            }
                        ],
                        "TimeLine": [],
                    }
                ]
            }
            return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))
        if qt == "pr3d":
            assert "heading=45.0" in url or "heading=45" in url
            return _FakeHttpResponse(jpeg)
        raise AssertionError(url)

    monkeypatch.setenv("ALLOW_REAL_TOOL_API", "true")
    monkeypatch.setenv("STREETVIEW_PROVIDER", "auto")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ctx = RuntimeContext()
    opened = execute(
        "streetview_query",
        "open",
        purpose="郑州",
        inputs={"coordinates": [113.67, 34.89]},
        ctx=ctx,
    )
    assert opened.ok is True
    assert opened.result is not None
    assert opened.result["coverage"] is True
    assert opened.result["applied"]["provider"] == "baidu"
    assert opened.result["viewpoint"]["image_id"] == "bd_1"
    assert opened.result["viewpoint"]["captured_at"] == "2017-09-20"
    assert all("mapsv0.bdimg.com" in item or "qt=qsdata" in item or "qt=sdata" in item for item in calls)
    assert any("qt=qsdata" in item for item in calls)
    assert any("qt=sdata" in item for item in calls)
    open_calls = len(calls)
    moved = execute(
        "streetview_query",
        "navigate",
        purpose="转向",
        inputs={"session": opened.session, "direction": "东"},
        ctx=ctx,
    )
    assert moved.ok is True
    assert moved.result is not None
    assert moved.result["moved"] is True
    assert moved.result["viewpoint"]["image_id"] == "bd_2"
    assert len(calls) == open_calls
    captured = execute(
        "streetview_query",
        "capture",
        purpose="截图",
        inputs={"session": opened.session, "heading": 45, "pitch": 0, "fov": 80},
        ctx=ctx,
    )
    assert captured.ok is True
    assert captured.result is not None
    assert captured.result["applied"]["heading"] == 45
    assert captured.result["applied"]["pano"] is True
    assert "images" not in captured.artifacts
    assert any("服务端按 heading" in item for item in captured.result["assumptions"])
    assert any("用后删除" in item for item in captured.result["assumptions"])
    assert any("qt=pr3d" in item for item in calls)


def test_tokyo_open_uses_google_and_capture_sends_heading(monkeypatch: Any) -> None:
    calls: list[tuple[str, str]] = []
    jpeg = _jpeg_bytes((90, 20, 20))

    def fake_urlopen(request: Any, timeout: float = 0) -> _FakeHttpResponse:
        del timeout
        url = request.full_url
        method = request.get_method()
        calls.append((method, url))
        if "SingleImageSearch" in url:
            body = json.dumps(_google_ok_payload()).encode("utf-8")
            return _FakeHttpResponse(body)
        if "streetviewpixels-pa.googleapis.com" in url:
            query = parse_qs(urlparse(url).query)
            assert query["yaw"] == ["90.0"]
            assert query["panoid"] == ["ABCDEFGHIJKLMNOPQRSTUV"]
            return _FakeHttpResponse(jpeg)
        raise AssertionError(url)

    monkeypatch.setenv("ALLOW_REAL_TOOL_API", "true")
    monkeypatch.setenv("STREETVIEW_PROVIDER", "auto")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ctx = RuntimeContext()
    opened = execute(
        "streetview_query",
        "open",
        purpose="东京",
        inputs={"coordinates": [139.7005, 35.6595]},
        ctx=ctx,
    )
    assert opened.ok is True
    assert opened.result is not None
    assert opened.result["applied"]["provider"] == "google"
    assert opened.result["viewpoint"]["image_id"] == "ABCDEFGHIJKLMNOPQRSTUV"
    assert opened.result["viewpoint"]["captured_at"] == "2019-05-01"
    assert calls[0][0] == "POST"
    open_calls = len(calls)
    captured = execute(
        "streetview_query",
        "capture",
        purpose="截图",
        inputs={"session": opened.session, "heading": 90, "pitch": 10, "fov": 90},
        ctx=ctx,
    )
    assert captured.ok is True
    assert captured.result is not None
    assert captured.result["applied"]["heading"] == 90
    assert "images" not in captured.artifacts
    assert len(calls) == open_calls + 1
    assert calls[-1][0] == "GET"
    assert "streetviewpixels-pa.googleapis.com" in calls[-1][1]


def test_google_no_images_is_no_coverage(monkeypatch: Any) -> None:
    def fake_urlopen(request: Any, timeout: float = 0) -> _FakeHttpResponse:
        del timeout
        payload = [[5, "generic", "Search returned no images."]]
        return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setenv("ALLOW_REAL_TOOL_API", "true")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    observation = execute(
        "streetview_query",
        "open",
        purpose="无覆盖",
        inputs={"coordinates": [139.7, 35.6], "provider": "google"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["coverage"] is False
    assert observation.result["viewpoint"] is None


def test_google_unreadable_reply_is_engine_unavailable(monkeypatch: Any) -> None:
    def fake_urlopen(request: Any, timeout: float = 0) -> _FakeHttpResponse:
        del timeout
        return _FakeHttpResponse(b"<<<not-json>>>")

    monkeypatch.setenv("ALLOW_REAL_TOOL_API", "true")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    observation = execute(
        "streetview_query",
        "open",
        purpose="坏回执",
        inputs={"coordinates": [139.7, 35.6], "provider": "google"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "JSON" in observation.error


def test_baidu_empty_content_is_no_coverage(monkeypatch: Any) -> None:
    def fake_urlopen(request: Any, timeout: float = 0) -> _FakeHttpResponse:
        del timeout
        return _FakeHttpResponse(json.dumps({"content": None}).encode("utf-8"))

    monkeypatch.setenv("ALLOW_REAL_TOOL_API", "true")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    observation = execute(
        "streetview_query",
        "open",
        purpose="无覆盖",
        inputs={"coordinates": [113.67, 34.89], "provider": "baidu"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["coverage"] is False


def test_provider_google_override_in_china(monkeypatch: Any) -> None:
    calls: list[str] = []

    def fake_urlopen(request: Any, timeout: float = 0) -> _FakeHttpResponse:
        del timeout
        calls.append(request.full_url)
        return _FakeHttpResponse(json.dumps(_google_ok_payload()).encode("utf-8"))

    monkeypatch.setenv("ALLOW_REAL_TOOL_API", "true")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    observation = execute(
        "streetview_query",
        "open",
        purpose="强制谷歌",
        inputs={"coordinates": [113.67, 34.89], "provider": "google"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["applied"]["provider"] == "google"
    assert any("SingleImageSearch" in item for item in calls)
    assert all("mapsv0.bdimg.com" not in item for item in calls)
