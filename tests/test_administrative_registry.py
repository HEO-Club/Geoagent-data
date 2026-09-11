"""administrative_registry 执行器测试；禁止真实付费 API。"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.administrative_registry import _registry as registry_mod
from tool.administrative_registry._registry import OfficialDumpProvider, RegistryRequest


class FakeGazetteerProvider:
    """测试替身：记录 search/hierarchy 并返回 GeoNames 风格载荷。"""

    name = "geonames"

    def __init__(
        self,
        hits: list[dict[str, Any]] | None = None,
        hierarchies: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.hits = hits if hits is not None else _sample_geonames()
        self.hierarchies = hierarchies if hierarchies is not None else _sample_hierarchy()
        self.search_calls: list[RegistryRequest] = []
        self.hierarchy_calls: list[str] = []

    def search(self, request: RegistryRequest) -> list[dict[str, Any]]:
        self.search_calls.append(request)
        return self.hits

    def hierarchy(self, provider_id: str) -> list[dict[str, Any]]:
        self.hierarchy_calls.append(provider_id)
        return list(self.hierarchies.get(provider_id, []))


def _sample_geonames() -> list[dict[str, Any]]:
    return [
        {
            "geonameId": 1808722,
            "name": "Huiji",
            "toponymName": "惠济区",
            "lat": "34.867000",
            "lng": "113.617000",
            "countryName": "China",
            "fcl": "A",
            "fcode": "ADM3",
            "adminName1": "Henan",
            "adminName2": "Zhengzhou",
            "alternateNames": [
                {"name": "惠济", "lang": "zh"},
                {"name": "Huiji District", "lang": "en"},
            ],
            "confirmed_location": "MUST NOT LEAK",
            "raw_content": "FULL GEONAMES JSON MUST NOT LEAK",
        }
    ]


def _sample_hierarchy() -> dict[str, list[dict[str, Any]]]:
    return {
        "1808722": [
            {"geonameId": 6295630, "name": "Earth", "fcode": "AREA"},
            {"geonameId": 1814991, "name": "China", "fcode": "PCLI"},
            {"geonameId": 1808520, "name": "Henan", "fcode": "ADM1"},
            {"geonameId": 1784658, "name": "Zhengzhou", "fcode": "ADM2"},
            {"geonameId": 1808722, "name": "惠济区", "fcode": "ADM3"},
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


def _lookup(
    operation: str = "administrative",
    *,
    inputs: dict[str, object] | None = None,
    provider: FakeGazetteerProvider | None = None,
    official: OfficialDumpProvider | FakeGazetteerProvider | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, object] = {"query": "惠济区"}
    if inputs:
        payload.update(inputs)
    engine = provider if provider is not None else FakeGazetteerProvider()
    runtime = ctx if ctx is not None else RuntimeContext()
    runtime.extras["administrative_registry_provider"] = engine
    if official is not None:
        runtime.extras["official_gazetteer_provider"] = official
    return execute(
        "administrative_registry",
        operation,
        purpose="查行政归属",
        inputs=payload,
        ctx=runtime,
    )


class _FakeHttpResponse:
    def __init__(self, payload: Any) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_empty_inputs_are_missing_input() -> None:
    admin = execute(
        "administrative_registry",
        "administrative",
        purpose="scaffold",
        inputs={},
        ctx=RuntimeContext(extras={"administrative_registry_provider": FakeGazetteerProvider()}),
    )
    directory = execute(
        "administrative_registry",
        "directory",
        purpose="scaffold",
        inputs={},
        ctx=RuntimeContext(extras={"administrative_registry_provider": FakeGazetteerProvider()}),
    )
    assert admin.ok is False
    assert admin.error_code == "missing_input"
    assert directory.ok is False
    assert directory.error_code == "missing_input"


def test_allow_real_api_false_without_provider_is_unavailable() -> None:
    observation = execute(
        "administrative_registry",
        "administrative",
        purpose="闸门",
        inputs={"query": "惠济区"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_API" in observation.error


def test_administrative_returns_name_hierarchy_aliases_and_wgs84() -> None:
    provider = FakeGazetteerProvider()
    observation = _lookup(provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    assert provider.search_calls
    assert provider.hierarchy_calls == ["1808722"]
    hit = observation.result["results"][0]
    assert hit["result_id"] == "admin_1"
    assert hit["standard_name"] == "惠济区"
    assert "惠济" in hit["aliases"]
    assert hit["admin_level"] == "ADM3"
    assert hit["location"]["lon"] == 113.617
    assert hit["location"]["lat"] == 34.867
    assert hit["location"]["crs"] == "wgs84"
    names = [node["name"] for node in hit["hierarchy"]]
    assert names == ["Earth", "China", "Henan", "Zhengzhou", "惠济区"]
    assert hit["evidence"]["source"] == "geonames"
    assert hit["evidence"]["provider_id"] == "1808722"
    assert hit["evidence"]["url"] == "https://www.geonames.org/1808722"
    assert hit["validity"]["scope"] == "current_source"
    assert hit["validity"]["historical_affiliation_stated"] is False
    keys = _nested_keys(observation.result)
    assert "raw_content" not in keys
    assert "confirmed_location" not in keys
    assert "MUST NOT LEAK" not in str(observation.result)


def test_directory_does_not_call_hierarchy() -> None:
    provider = FakeGazetteerProvider()
    observation = _lookup("directory", provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    assert provider.search_calls
    assert provider.hierarchy_calls == []
    hit = observation.result["results"][0]
    assert hit["result_id"] == "directory_1"
    assert hit["standard_name"] == "惠济区"
    assert "hierarchy" not in hit or hit["hierarchy"]


def test_area_bbox_reaches_request() -> None:
    provider = FakeGazetteerProvider()
    observation = _lookup(
        provider=provider,
        inputs={"query": "bridge", "area": [113.5, 34.7, 113.8, 35.0]},
    )
    assert observation.ok is True
    assert provider.search_calls
    bbox = provider.search_calls[0].bbox
    assert bbox is not None
    assert bbox.west == 113.5
    assert bbox.south == 34.7
    assert bbox.east == 113.8
    assert bbox.north == 35.0
    assert observation.result["applied"]["area"]["west"] == 113.5


def test_active_area_context_is_used() -> None:
    provider = FakeGazetteerProvider()
    observation = _lookup(
        provider=provider,
        inputs={"area": "$active_area"},
        ctx=RuntimeContext(active_area="郑州市"),
    )
    assert observation.ok is True
    assert provider.search_calls[0].area_text == "郑州市"
    assert observation.result["applied"]["area"] == "郑州市"


def test_time_range_marks_current_scope_not_historical_admin() -> None:
    provider = FakeGazetteerProvider()
    observation = _lookup(
        provider=provider,
        inputs={"time_range": "1980-1995"},
    )
    assert observation.ok is True
    hit = observation.result["results"][0]
    assert hit["standard_name"] == "惠济区"
    assert hit["admin_level"] == "ADM3"
    assert hit["validity"]["scope"] == "current_source"
    assert hit["validity"]["historical_affiliation_stated"] is False
    assert hit["validity"]["time_range_requested"] == {
        "start": "1980-01-01",
        "end": "1995-12-31",
    }
    assert observation.result["applied"]["time_range"]["start"] == "1980-01-01"
    assert any("不能自行向过去延伸" in item for item in observation.result["assumptions"])


def test_official_interval_overlap_states_historical_affiliation() -> None:
    official = OfficialDumpProvider(
        [
            {
                "standard_name": "惠济区",
                "aliases": ["惠济"],
                "admin_level": "county",
                "parents": ["郑州市", "河南省", "中国"],
                "adcode": "410103",
                "valid_from": "2014-01-01",
                "valid_to": None,
                "lat": 34.86,
                "lon": 113.61,
            }
        ]
    )
    observation = _lookup(
        inputs={"query": "惠济区", "registry": "official", "time_range": "2015"},
        official=official,
    )
    assert observation.ok is True
    hit = observation.result["results"][0]
    assert hit["standard_name"] == "惠济区"
    assert hit["validity"]["historical_affiliation_stated"] is True
    assert hit["validity"]["scope"] == "stated_interval"
    assert hit["validity"]["valid_from"] == "2014-01-01"
    assert hit["evidence"]["source"] == "official"
    assert hit["evidence"]["provider_id"] == "410103"
    assert [node["name"] for node in hit["hierarchy"]] == ["郑州市", "河南省", "中国"]
    assert not any("不能自行向过去延伸" in item for item in observation.result["assumptions"])


def test_fields_keep_validity_and_evidence() -> None:
    observation = _lookup(inputs={"fields": ["standard_name"]})
    assert observation.ok is True
    hit = observation.result["results"][0]
    assert set(hit) >= {"result_id", "standard_name", "validity", "evidence"}
    assert "aliases" not in hit
    assert "location" not in hit
    assert hit["validity"]["scope"] == "current_source"
    assert hit["evidence"]["source"] == "geonames"


def test_official_registry_without_config_is_unavailable() -> None:
    observation = execute(
        "administrative_registry",
        "administrative",
        purpose="缺官方",
        inputs={"query": "惠济区", "registry": "official"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "官方" in observation.error


def test_unknown_registry_is_unavailable() -> None:
    observation = _lookup(inputs={"registry": "not-a-source"})
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"


def test_geonames_http_sends_username_bbox_and_hierarchy(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        parsed = urlparse(str(request.full_url))
        params = parse_qs(parsed.query)
        if parsed.path.endswith("/searchJSON"):
            assert params["username"] == ["demo-user"]
            assert params["q"] == ["惠济区"]
            assert params["lang"] == ["zh"]
            assert params["style"] == ["FULL"]
            assert params["west"] == ["113.500000"]
            assert params["south"] == ["34.700000"]
            assert params["east"] == ["113.800000"]
            assert params["north"] == ["35.000000"]
            assert request.get_header("User-agent") == "geoagent-dataset/1.0 (administrative_registry; test)"
            return _FakeHttpResponse({"geonames": _sample_geonames()})
        assert parsed.path.endswith("/hierarchyJSON")
        assert params["geonameId"] == ["1808722"]
        return _FakeHttpResponse({"geonames": _sample_hierarchy()["1808722"]})

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("GEONAMES_USERNAME", "demo-user")
    monkeypatch.setenv("GEONAMES_USER_AGENT", "geoagent-dataset/1.0 (administrative_registry; test)")
    monkeypatch.setattr(registry_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "administrative_registry",
        "administrative",
        purpose="mock geonames",
        inputs={"query": "惠济区", "area": [113.5, 34.7, 113.8, 35.0], "registry": "geonames"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert len(calls) == 2
    hit = observation.result["results"][0]
    assert hit["standard_name"] == "惠济区"
    assert hit["location"]["crs"] == "wgs84"
    assert observation.result["applied"]["provider"] == "geonames"
    keys = _nested_keys(observation.result)
    assert "raw_content" not in keys


def test_missing_geonames_username_is_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("GEONAMES_USERNAME", "")
    observation = execute(
        "administrative_registry",
        "administrative",
        purpose="缺用户名",
        inputs={"query": "惠济区", "registry": "geonames"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "GEONAMES_USERNAME" in observation.error


def test_query_array_is_joined() -> None:
    provider = FakeGazetteerProvider()
    observation = _lookup(provider=provider, inputs={"query": ["惠济", "区"]})
    assert observation.ok is True
    assert provider.search_calls[0].query == "惠济 区"
