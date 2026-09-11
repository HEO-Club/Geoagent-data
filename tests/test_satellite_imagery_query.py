"""satellite_imagery_query 执行器测试；禁止真实付费 API。"""

from __future__ import annotations

import json
from datetime import date
from io import BytesIO
from typing import Any
from urllib.parse import parse_qs

from PIL import Image

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.runtime import FilesystemImageStore
from tool.satellite_imagery_query._query import (
    BBox,
    CopernicusProvider,
    PreviewRequest,
    SceneSearchRequest,
)


class FakeSatelliteProvider:
    """测试替身：固定目录命中与 PNG，不访问网络。"""

    name = "fake"
    crs = "wgs84"

    def __init__(
        self,
        scenes: list[dict[str, Any]] | None = None,
        *,
        png: bytes | None = None,
    ) -> None:
        self.scenes = list(scenes if scenes is not None else _sample_scenes())
        self.png = png if png is not None else _png_bytes()
        self.search_calls: list[SceneSearchRequest] = []
        self.preview_calls: list[PreviewRequest] = []

    def search_scenes(self, request: SceneSearchRequest) -> list[dict[str, Any]]:
        self.search_calls.append(request)
        hits: list[dict[str, Any]] = []
        for item in self.scenes:
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            lon = (float(bbox[0]) + float(bbox[2])) / 2.0
            lat = (float(bbox[1]) + float(bbox[3])) / 2.0
            if not (
                request.bbox.west <= lon <= request.bbox.east
                and request.bbox.south <= lat <= request.bbox.north
            ):
                continue
            collection = str(item.get("collection") or "")
            if request.collections and collection not in request.collections:
                continue
            captured = _scene_date(item)
            if request.datetime and captured is not None and not _in_datetime(request.datetime, captured):
                continue
            hits.append(dict(item))
        return hits

    def fetch_preview(self, request: PreviewRequest) -> bytes:
        self.preview_calls.append(request)
        return self.png


def _png_bytes(color: tuple[int, int, int] = (30, 90, 40), size: tuple[int, int] = (8, 8)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _sample_scenes() -> list[dict[str, Any]]:
    return [
        _scene("S2_LOW_CLOUD", "2024-06-12T03:11:00Z", 3.2, 10.0),
        _scene("S2_2023", "2023-01-08T02:40:00Z", 12.0, 10.0),
        _scene("S2_2016", "2016-03-01T03:05:00Z", 8.0, 10.0),
        _scene(
            "S2_ELSEWHERE",
            "2024-06-12T03:11:00Z",
            1.0,
            10.0,
            bbox=[10.0, 50.0, 10.2, 50.2],
        ),
    ]


def _scene(
    scene_id: str,
    datetime_text: str,
    cloud_cover: float,
    gsd: float,
    *,
    bbox: list[float] | None = None,
    collection: str = "sentinel-2-l2a",
) -> dict[str, Any]:
    return {
        "id": scene_id,
        "collection": collection,
        "bbox": bbox or [113.5, 34.7, 113.8, 34.9],
        "properties": {
            "datetime": datetime_text,
            "eo:cloud_cover": cloud_cover,
            "gsd": gsd,
            "confirmed_location": "MUST NOT LEAK",
            "raw_content": "FULL STAC JSON MUST NOT LEAK",
        },
    }


def _scene_date(item: dict[str, Any]) -> date | None:
    props = item.get("properties")
    if not isinstance(props, dict):
        return None
    raw = str(props.get("datetime") or "")
    if len(raw) < 10:
        return None
    return date.fromisoformat(raw[:10])


def _in_datetime(spec: str, captured: date) -> bool:
    if "/" not in spec:
        return True
    start_raw, end_raw = spec.split("/", 1)
    start = date.fromisoformat(start_raw[:10])
    end = date.fromisoformat(end_raw[:10])
    return start <= captured <= end


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
    inputs: dict[str, Any] | None = None,
    provider: FakeSatelliteProvider | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, Any] = dict(inputs or {})
    engine = provider if provider is not None else FakeSatelliteProvider()
    runtime = ctx
    if runtime is None:
        runtime = RuntimeContext(extras={"satellite_imagery_query_provider": engine})
    elif provider is not None:
        runtime.extras["satellite_imagery_query_provider"] = engine
    return execute(
        "satellite_imagery_query",
        operation,
        purpose="测试",
        inputs=payload,
        ctx=runtime,
    )


def test_missing_area_coordinates_source_result_is_missing_input() -> None:
    observation = execute(
        "satellite_imagery_query",
        "retrieve",
        purpose="缺输入",
        inputs={},
        ctx=RuntimeContext(extras={"satellite_imagery_query_provider": FakeSatelliteProvider()}),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_place_name_without_geometry_is_missing_input() -> None:
    observation = _call("retrieve", inputs={"area": "郑州市"})
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "geocode" in observation.error


def test_retrieve_picks_lowest_cloud_and_reports_native_gsd() -> None:
    provider = FakeSatelliteProvider()
    observation = _call(
        "retrieve",
        inputs={"area": {"bbox": [113.5, 34.7, 113.8, 34.9]}},
        provider=provider,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["coverage"] is True
    assert observation.result["scene_id"] == "S2_LOW_CLOUD"
    assert observation.result["captured_at"] == "2024-06-12"
    assert observation.result["resolution_m"] == 10.0
    assert observation.result["cloud_cover"] == 3.2
    assert observation.result["collection"] == "sentinel-2-l2a"
    assert observation.result["view_kind"] == "nadir"
    assert observation.result["is_true_oblique"] is False
    assert observation.result["image_id"]
    assert "image_ids" in observation.artifacts
    assert provider.search_calls
    assert provider.preview_calls
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert "raw_content" not in keys
    assert any("resolution_m" in item for item in observation.result["assumptions"])


def test_resolution_m_one_meter_has_no_coverage() -> None:
    provider = FakeSatelliteProvider()
    observation = _call(
        "retrieve",
        inputs={
            "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
            "resolution_m": 1,
        },
        provider=provider,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["coverage"] is False
    assert observation.result["scene_id"] is None
    assert observation.result["resolution_m"] is None
    assert "image_id" not in observation.result
    assert provider.search_calls == []
    assert provider.preview_calls == []
    assert observation.session == "sat_0001"


def test_cloud_cover_filter_excludes_high_cloud() -> None:
    provider = FakeSatelliteProvider()
    observation = _call(
        "retrieve",
        inputs={
            "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
            "time_range": "2023",
            "cloud_cover_max": 10,
        },
        provider=provider,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["coverage"] is False
    assert provider.search_calls
    assert provider.search_calls[0].cloud_cover_max == 10


def test_change_time_hits_year_and_miss_keeps_session() -> None:
    provider = FakeSatelliteProvider()
    ctx = RuntimeContext(extras={"satellite_imagery_query_provider": provider})
    opened = _call(
        "retrieve",
        inputs={"area": {"bbox": [113.5, 34.7, 113.8, 34.9]}},
        provider=provider,
        ctx=ctx,
    )
    assert opened.result is not None
    assert opened.result["scene_id"] == "S2_LOW_CLOUD"
    hit = _call(
        "change_time",
        inputs={
            "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
            "time_range": "2023",
            "source_result": opened.session,
        },
        provider=provider,
        ctx=ctx,
    )
    assert hit.ok is True
    assert hit.result is not None
    assert hit.result["coverage"] is True
    assert hit.result["scene_id"] == "S2_2023"
    assert hit.result["captured_at"] == "2023-01-08"
    assert hit.session == opened.session
    assert "2023-01-08" in hit.result["available_dates"]
    miss = _call(
        "change_time",
        inputs={
            "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
            "time_range": "2010",
            "source_result": opened.session,
        },
        provider=provider,
        ctx=ctx,
    )
    assert miss.ok is True
    assert miss.result is not None
    assert miss.result["coverage"] is False
    assert miss.result["scene_id"] is None
    assert miss.session == opened.session
    assert "image_id" not in miss.result


def test_change_time_missing_time_range_is_missing_input() -> None:
    observation = _call(
        "change_time",
        inputs={"area": {"bbox": [113.5, 34.7, 113.8, 34.9]}},
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_oblique_view_is_nadir_preview_not_true_oblique(tmp_path: Any) -> None:
    provider = FakeSatelliteProvider()
    store = FilesystemImageStore(tmp_path / "store")
    ctx = RuntimeContext(
        extras={"satellite_imagery_query_provider": provider},
        image_store=store,
    )
    observation = _call(
        "oblique_view",
        inputs={
            "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
            "heading": 45,
            "tilt": 30,
        },
        provider=provider,
        ctx=ctx,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["coverage"] is True
    assert observation.result["view_kind"] == "nadir_preview"
    assert observation.result["is_true_oblique"] is False
    assert observation.result["heading"] == 45
    assert observation.result["tilt"] == 30
    assert observation.result["applied"]["heading"] == 45
    assert observation.result["applied"]["tilt"] == 30
    assert any("真斜摄" in item or "Cesium" in item for item in observation.result["assumptions"])
    stored = store.resolve(observation.result["image_id"])
    assert Image.open(stored).size == (8, 8)


def test_planet_provider_is_unsupported() -> None:
    observation = _call(
        "retrieve",
        inputs={"area": {"bbox": [113.5, 34.7, 113.8, 34.9]}, "provider": "planet"},
    )
    assert observation.ok is False
    assert observation.error_code == "unsupported_provider"
    assert observation.error is not None
    assert "Planet" in observation.error


def test_google_provider_is_unsupported() -> None:
    observation = _call(
        "retrieve",
        inputs={"area": {"bbox": [113.5, 34.7, 113.8, 34.9]}, "provider": "google"},
    )
    assert observation.ok is False
    assert observation.error_code == "unsupported_provider"
    assert observation.error is not None
    assert "Google" in observation.error


def test_aerial_layer_is_unsupported() -> None:
    observation = _call(
        "retrieve",
        inputs={"area": {"bbox": [113.5, 34.7, 113.8, 34.9]}, "layer": "aerial"},
    )
    assert observation.ok is False
    assert observation.error_code == "unsupported_layer"


def test_allow_real_api_false_without_provider_is_unavailable() -> None:
    observation = execute(
        "satellite_imagery_query",
        "retrieve",
        purpose="闸门",
        inputs={"area": {"bbox": [113.5, 34.7, 113.8, 34.9]}},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_API" in observation.error


def test_allow_real_api_true_missing_client_is_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("COPERNICUS_CLIENT_ID", "")
    monkeypatch.setenv("COPERNICUS_CLIENT_SECRET", "")
    observation = execute(
        "satellite_imagery_query",
        "retrieve",
        purpose="缺凭证",
        inputs={"area": {"bbox": [113.5, 34.7, 113.8, 34.9]}},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "COPERNICUS_CLIENT_ID" in observation.error


def test_center_radius_builds_bbox() -> None:
    provider = FakeSatelliteProvider()
    observation = _call(
        "retrieve",
        inputs={"area": {"lat": 34.8, "lon": 113.65, "radius_m": 5000}},
        provider=provider,
    )
    assert observation.ok is True
    assert provider.search_calls
    bbox = provider.search_calls[0].bbox
    assert bbox.west < 113.65 < bbox.east
    assert bbox.south < 34.8 < bbox.north


def test_copernicus_catalog_sends_bbox_and_bearer(monkeypatch: Any) -> None:
    calls: list[tuple[str, bytes | None, dict[str, str]]] = []

    class _FakeHttpResponse:
        def __init__(self, payload: bytes) -> None:
            self._body = payload

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> _FakeHttpResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: Any, timeout: float = 0) -> _FakeHttpResponse:
        del timeout
        body = request.data if isinstance(request.data, (bytes, bytearray)) else None
        headers = {str(key): str(value) for key, value in request.header_items()}
        calls.append((request.full_url, bytes(body) if body else None, headers))
        if "openid-connect/token" in request.full_url:
            return _FakeHttpResponse(json.dumps({"access_token": "tok-1", "expires_in": 3600}).encode("utf-8"))
        return _FakeHttpResponse(
            json.dumps(
                {
                    "features": [
                        {
                            "id": "S2_HTTP",
                            "collection": "sentinel-2-l2a",
                            "bbox": [113.6, 34.8, 113.7, 34.9],
                            "properties": {
                                "datetime": "2024-06-12T03:11:00Z",
                                "eo:cloud_cover": 4.0,
                                "gsd": 10,
                            },
                        }
                    ]
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("tool.satellite_imagery_query._query.urllib.request.urlopen", fake_urlopen)
    provider = CopernicusProvider(
        client_id="id-1",
        client_secret="secret-1",
        catalog_endpoint="https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search",
        process_endpoint="https://sh.dataspace.copernicus.eu/api/v1/process",
        token_endpoint="https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        timeout_sec=5.0,
        user_agent="geoagent-test",
        preview_width=64,
        preview_height=64,
    )
    rows = provider.search_scenes(
        SceneSearchRequest(
            bbox=BBox(west=113.6, south=34.8, east=113.7, north=34.9),
            collections=("sentinel-2-l2a",),
            cloud_cover_max=20,
            limit=10,
        )
    )
    assert rows and rows[0]["id"] == "S2_HTTP"
    assert len(calls) == 2
    token_url, token_body, _token_headers = calls[0]
    assert "openid-connect/token" in token_url
    assert token_body is not None
    form = parse_qs(token_body.decode("utf-8"))
    assert form["client_id"] == ["id-1"]
    catalog_url, catalog_body, catalog_headers = calls[1]
    assert catalog_url.endswith("/search")
    auth = catalog_headers.get("Authorization") or catalog_headers.get("authorization")
    assert auth == "Bearer tok-1"
    assert catalog_body is not None
    payload = json.loads(catalog_body.decode("utf-8"))
    assert payload["bbox"] == [113.6, 34.8, 113.7, 34.9]
    assert payload["collections"] == ["sentinel-2-l2a"]
    assert payload["query"]["eo:cloud_cover"]["lt"] == 20
