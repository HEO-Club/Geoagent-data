"""map_layer_query 执行器测试；禁止真实付费 API。"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.map_layer_query import _layers as layers_mod
from tool.map_layer_query._layers import LayerInputError, LayerRequest


class FakeMapLayerProvider:
    """测试替身：记录请求并返回图片/矢量图层。"""

    name = "fake"
    crs = "EPSG:4326"

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        supported: frozenset[str] | None = None,
        include_vector: bool = False,
        extra_features: int = 0,
    ) -> None:
        self.payload = payload
        self.supported = supported or frozenset(
            {"hydrology", "terrain", "administrative", "roads"}
        )
        self.include_vector = include_vector
        self.extra_features = extra_features
        self.calls: list[LayerRequest] = []

    def load(self, request: LayerRequest) -> dict[str, Any]:
        self.calls.append(request)
        for name in request.layers:
            if name not in self.supported:
                raise LayerInputError(f"不支持图层: {name}", "unsupported_layer")
        if self.payload is not None:
            return self.payload
        return _sample_payload(
            request,
            include_vector=self.include_vector,
            extra_features=self.extra_features,
        )


def _png_bytes(color: tuple[int, int, int] = (0, 90, 160)) -> bytes:
    image = Image.new("RGB", (8, 8), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _sample_payload(
    request: LayerRequest,
    *,
    include_vector: bool,
    extra_features: int,
) -> dict[str, Any]:
    bbox = request.bbox
    bbox_obj = (
        {
            "west": bbox.west,
            "south": bbox.south,
            "east": bbox.east,
            "north": bbox.north,
            "crs": request.crs,
        }
        if bbox is not None
        else {"west": 113.5, "south": 34.7, "east": 113.8, "north": 34.9, "crs": request.crs}
    )
    layers: list[dict[str, Any]] = []
    for name in request.layers:
        layers.append(
            {
                "name": name,
                "provider_layer": f"workspace:{name}",
                "representation": "image",
                "image_png": _png_bytes(),
                "data_date": request.time_range,
                "crs": request.crs,
                "confirmed_location": "MUST NOT LEAK",
                "raw_content": "FULL WMS XML MUST NOT LEAK",
            }
        )
        if include_vector:
            features = [
                {
                    "type": "Feature",
                    "properties": {
                        "name": f"river_{index}",
                        "confirmed_location": "MUST NOT LEAK",
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[113.5, 34.8], [113.7, 34.85]],
                    },
                }
                for index in range(max(1, extra_features))
            ]
            layers.append(
                {
                    "name": name,
                    "provider_layer": f"workspace:{name}",
                    "representation": "vector",
                    "features": {
                        "type": "FeatureCollection",
                        "features": features,
                        "raw_content": "FULL WFS XML MUST NOT LEAK",
                    },
                    "crs": request.crs,
                }
            )
    return {
        "layers": layers,
        "bbox": bbox_obj,
        "crs": request.crs,
        "wms_version": "1.3.0",
        "layer_mapping": {name: f"workspace:{name}" for name in request.layers},
        "width": request.width,
        "height": request.height,
        "time": request.time_range,
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


def _load(
    *,
    inputs: dict[str, Any] | None = None,
    provider: FakeMapLayerProvider | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, Any] = {
        "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
        "layers": ["hydrology"],
    }
    if inputs:
        payload.update(inputs)
    runtime = ctx
    engine = provider if provider is not None else FakeMapLayerProvider()
    if runtime is None:
        runtime = RuntimeContext(extras={"map_layer_query_provider": engine})
    elif provider is not None:
        runtime.extras["map_layer_query_provider"] = engine
    return execute(
        "map_layer_query",
        "load_layer",
        purpose="加载图层",
        inputs=payload,
        ctx=runtime,
    )


class _FakeHttpResponse:
    def __init__(self, payload: bytes) -> None:
        self._body = payload

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_missing_area_is_missing_input() -> None:
    observation = _load(inputs={"area": None})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_missing_layers_is_missing_input() -> None:
    observation = _load(inputs={"layers": None})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_empty_inputs_are_missing_input() -> None:
    observation = execute(
        "map_layer_query",
        "load_layer",
        purpose="scaffold",
        inputs={},
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_hydrology_image_is_not_usable_for_gis() -> None:
    provider = FakeMapLayerProvider()
    observation = _load(provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    assert provider.calls and provider.calls[0].layers == ("hydrology",)
    layer = observation.result["layers"][0]
    assert layer["result_id"] == "layer_1"
    assert layer["name"] == "hydrology"
    assert layer["representation"] == "image"
    assert layer["usable_for_gis"] is False
    assert layer["image_id"]
    assert observation.result["bbox"]["west"] == 113.5
    assert observation.result["bbox"]["crs"] == "EPSG:4326"
    assert "image_ids" in observation.artifacts
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert "raw_content" not in keys


def test_wfs_vector_is_usable_for_gis_and_truncated(monkeypatch: Any) -> None:
    monkeypatch.setenv("MAP_LAYER_MAX_FEATURES", "2")
    provider = FakeMapLayerProvider(include_vector=True, extra_features=5)
    observation = _load(provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    layers = observation.result["layers"]
    assert layers[0]["representation"] == "image"
    assert layers[0]["usable_for_gis"] is False
    vector = layers[1]
    assert vector["representation"] == "vector"
    assert vector["usable_for_gis"] is True
    assert vector["feature_count"] == 2
    assert len(vector["features"]["features"]) == 2
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert "raw_content" not in keys


def test_place_name_without_bbox_on_builtin_wms_is_missing_input(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("MAP_LAYER_PROVIDER", "wms")
    monkeypatch.setenv("MAP_LAYER_WMS_ENDPOINT", "http://localhost/geoserver/wms")
    monkeypatch.setenv("MAP_LAYER_WMS_LAYERS", '{"hydrology":"workspace:rivers"}')
    observation = execute(
        "map_layer_query",
        "load_layer",
        purpose="纯地名",
        inputs={"area": "郑州附近黄河沿线", "layers": ["hydrology"]},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "bbox" in observation.error


def test_historical_map_without_layer_is_unsupported() -> None:
    observation = _load(inputs={"layers": ["historical_map"]})
    assert observation.ok is False
    assert observation.error_code == "unsupported_layer"


def test_undeclared_or_unknown_provider_is_rejected() -> None:
    observation = _load(inputs={"provider": "xyz"})
    assert observation.ok is False
    assert observation.error_code == "unsupported_provider"


def test_allow_real_api_false_without_provider_is_unavailable() -> None:
    observation = execute(
        "map_layer_query",
        "load_layer",
        purpose="闸门",
        inputs={"area": {"bbox": [113.5, 34.7, 113.8, 34.9]}, "layers": ["hydrology"]},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_API" in observation.error


def test_allow_real_api_true_missing_endpoint_is_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("MAP_LAYER_PROVIDER", "wms")
    monkeypatch.setenv("MAP_LAYER_WMS_ENDPOINT", "")
    observation = execute(
        "map_layer_query",
        "load_layer",
        purpose="缺端点",
        inputs={"area": {"bbox": [113.5, 34.7, 113.8, 34.9]}, "layers": ["hydrology"]},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "MAP_LAYER_WMS_ENDPOINT" in observation.error


def test_center_radius_builds_bbox() -> None:
    provider = FakeMapLayerProvider()
    observation = _load(
        inputs={"area": {"lat": 34.8, "lon": 113.65, "radius_m": 5000}},
        provider=provider,
    )
    assert observation.ok is True
    assert provider.calls
    bbox = provider.calls[0].bbox
    assert bbox is not None
    assert bbox.west < 113.65 < bbox.east
    assert bbox.south < 34.8 < bbox.north


def test_layer_alias_hydrology_from_chinese() -> None:
    provider = FakeMapLayerProvider()
    observation = _load(inputs={"layers": "水系"}, provider=provider)
    assert observation.ok is True
    assert provider.calls[0].layers == ("hydrology",)


def test_wms_getmap_uses_13_axis_order(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        parsed = urlparse(str(request.full_url))
        params = parse_qs(parsed.query)
        assert params["REQUEST"] == ["GetMap"]
        assert params["LAYERS"] == ["workspace:rivers"]
        assert params["CRS"] == ["EPSG:4326"]
        assert params["BBOX"] == ["34.7,113.5,34.9,113.8"]
        return _FakeHttpResponse(_png_bytes())

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("MAP_LAYER_PROVIDER", "wms")
    monkeypatch.setenv("MAP_LAYER_WMS_ENDPOINT", "http://localhost/geoserver/wms")
    monkeypatch.setenv("MAP_LAYER_WFS_ENDPOINT", "")
    monkeypatch.setenv(
        "MAP_LAYER_WMS_LAYERS",
        json.dumps({"hydrology": "workspace:rivers"}),
    )
    monkeypatch.setattr(layers_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "map_layer_query",
        "load_layer",
        purpose="WMS",
        inputs={"area": [113.5, 34.7, 113.8, 34.9], "layers": ["hydrology"]},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert calls
    layer = observation.result["layers"][0]
    assert layer["representation"] == "image"
    assert layer["usable_for_gis"] is False
    assert layer["provider_layer"] == "workspace:rivers"


def test_wms_with_wfs_returns_vector(monkeypatch: Any) -> None:
    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        parsed = urlparse(str(request.full_url))
        params = parse_qs(parsed.query)
        if params.get("REQUEST") == ["GetFeature"]:
            payload = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "name": "黄河",
                            "confirmed_location": "MUST NOT LEAK",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[113.5, 34.8], [113.7, 34.85]],
                        },
                    }
                ],
                "raw_content": "MUST NOT LEAK",
            }
            return _FakeHttpResponse(json.dumps(payload).encode("utf-8"))
        return _FakeHttpResponse(_png_bytes())

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("MAP_LAYER_PROVIDER", "wms")
    monkeypatch.setenv("MAP_LAYER_WMS_ENDPOINT", "http://localhost/geoserver/wms")
    monkeypatch.setenv("MAP_LAYER_WFS_ENDPOINT", "http://localhost/geoserver/wfs")
    monkeypatch.setenv(
        "MAP_LAYER_WMS_LAYERS",
        json.dumps({"hydrology": "workspace:rivers"}),
    )
    monkeypatch.setattr(layers_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "map_layer_query",
        "load_layer",
        purpose="WFS",
        inputs={"area": [113.5, 34.7, 113.8, 34.9], "layers": ["hydrology"]},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert observation.result is not None
    names = [row["representation"] for row in observation.result["layers"]]
    assert names == ["image", "vector"]
    vector = observation.result["layers"][1]
    assert vector["usable_for_gis"] is True
    assert vector["feature_count"] == 1
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert "raw_content" not in keys
