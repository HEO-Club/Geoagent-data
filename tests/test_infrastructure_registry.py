"""infrastructure_registry 执行器测试；禁止真实付费 API。"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.infrastructure_registry import _registry as registry_mod
from tool.infrastructure_registry._registry import AuthorizedDumpProvider, RegistryRequest


class FakeRegistryProvider:
    """测试替身：记录 search 并返回档案风格载荷。"""

    name = "dump"

    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.hits = hits if hits is not None else _sample_construction()
        self.search_calls: list[RegistryRequest] = []

    def search(self, request: RegistryRequest) -> list[dict[str, Any]]:
        self.search_calls.append(request)
        return self.hits


def _sample_construction() -> list[dict[str, Any]]:
    return [
        {
            "project_name": "郑州黄河铁路桥",
            "record_type": "construction",
            "reference": "YR-BRIDGE-01",
            "permit_date": "1904-06-01",
            "started_at": "1903-01-01",
            "completed_at": "1905-11-01",
            "opened_at": "1906-04-01",
            "lat": 34.90,
            "lon": 113.65,
            "source_url": "https://archive.example.com/yellow-river-bridge",
            "file": "authorized/bridge.json",
            "construction_year": "MUST NOT LEAK",
            "raw_content": "FULL REGISTRY JSON MUST NOT LEAK",
        }
    ]


def _sample_permit() -> list[dict[str, Any]]:
    return [
        {
            "project_name": "郑州黄河铁路桥改建",
            "record_type": "permit",
            "reference": "建许字2018-01",
            "permit_date": "2018-03-15",
            "started_at": "2010-01-01",
            "source_url": "https://archive.example.com/permit-2018",
        }
    ]


def _sample_planning_entities() -> list[dict[str, Any]]:
    return [
        {
            "entity": 123456,
            "name": "Listed Railway Bridge",
            "dataset": "listed-building",
            "reference": "LB-001",
            "start-date": "1985-02-01",
            "point": "POINT(-0.127800 51.507400)",
        }
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


def _lookup(
    operation: str = "construction",
    *,
    inputs: dict[str, object] | None = None,
    provider: FakeRegistryProvider | None = None,
    dump: AuthorizedDumpProvider | FakeRegistryProvider | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, object] = {"query": "黄河铁路桥"}
    if inputs:
        payload.update(inputs)
    engine = provider if provider is not None else FakeRegistryProvider()
    runtime = ctx if ctx is not None else RuntimeContext()
    runtime.extras["infrastructure_registry_provider"] = engine
    if dump is not None:
        runtime.extras["infrastructure_dump_provider"] = dump
    return execute(
        "infrastructure_registry",
        operation,
        purpose="查建设档案",
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
    construction = execute(
        "infrastructure_registry",
        "construction",
        purpose="scaffold",
        inputs={},
    )
    permit = execute(
        "infrastructure_registry",
        "permit",
        purpose="scaffold",
        inputs={},
    )
    assert construction.ok is False
    assert construction.error_code == "missing_input"
    assert permit.ok is False
    assert permit.error_code == "missing_input"


def test_allow_real_api_false_without_provider_is_unavailable() -> None:
    observation = execute(
        "infrastructure_registry",
        "construction",
        purpose="闸门",
        inputs={"query": "黄河铁路桥"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_API" in observation.error


def test_construction_returns_project_split_dates_and_evidence() -> None:
    provider = FakeRegistryProvider()
    observation = _lookup(provider=provider)
    assert observation.ok is True
    assert observation.result is not None
    assert provider.search_calls
    hit = observation.result["results"][0]
    assert hit["result_id"] == "construction_1"
    assert hit["project_name"] == "郑州黄河铁路桥"
    assert hit["record_type"] == "construction"
    assert hit["reference"] == "YR-BRIDGE-01"
    assert hit["dates"]["permit"] == "1904-06-01"
    assert hit["dates"]["started"] == "1903-01-01"
    assert hit["dates"]["completed"] == "1905-11-01"
    assert hit["dates"]["opened"] == "1906-04-01"
    assert hit["evidence"]["url"] == "https://archive.example.com/yellow-river-bridge"
    assert hit["evidence"]["file"] == "authorized/bridge.json"
    assert "construction_year" not in _nested_keys(observation.result)
    assert "raw_content" not in _nested_keys(observation.result)
    assert any("不能混成修建年份" in item for item in observation.result["assumptions"])


def test_permit_uses_permit_date_not_start_date() -> None:
    provider = FakeRegistryProvider(hits=_sample_permit())
    observation = _lookup("permit", provider=provider)
    assert observation.ok is True
    hit = observation.result["results"][0]
    assert hit["result_id"] == "permit_1"
    assert hit["reference"] == "建许字2018-01"
    assert hit["dates"]["permit"] == "2018-03-15"
    assert hit["dates"]["started"] == "2010-01-01"
    assert hit["dates"]["recorded_kind"] == "permit"


def test_fields_keep_validity_and_evidence() -> None:
    observation = _lookup(inputs={"fields": ["project_name"]})
    assert observation.ok is True
    hit = observation.result["results"][0]
    assert set(hit) >= {"result_id", "project_name", "validity", "evidence"}
    assert "dates" not in hit
    assert "reference" not in hit
    assert hit["evidence"]["source"] == "dump"


def test_area_bbox_reaches_request() -> None:
    provider = FakeRegistryProvider()
    observation = _lookup(
        provider=provider,
        inputs={"query": "bridge", "area": [113.5, 34.7, 113.8, 35.0]},
    )
    assert observation.ok is True
    bbox = provider.search_calls[0].bbox
    assert bbox is not None
    assert bbox.west == 113.5
    assert bbox.south == 34.7
    assert bbox.east == 113.8
    assert bbox.north == 35.0
    assert observation.result["applied"]["area"]["west"] == 113.5


def test_active_area_context_is_used() -> None:
    provider = FakeRegistryProvider()
    observation = _lookup(
        provider=provider,
        inputs={"area": "$active_area"},
        ctx=RuntimeContext(active_area="郑州市"),
    )
    assert observation.ok is True
    assert provider.search_calls[0].area_text == "郑州市"
    assert observation.result["applied"]["area"] == "郑州市"


def test_time_range_filters_by_permit_date_not_start() -> None:
    dump = AuthorizedDumpProvider(
        [
            {
                "project_name": "改建许可",
                "record_type": "permit",
                "reference": "P-2018",
                "permit_date": "2018-03-15",
                "started_at": "2010-01-01",
            },
            {
                "project_name": "旧开工记录",
                "record_type": "permit",
                "reference": "P-2010",
                "permit_date": "2009-01-01",
                "started_at": "2018-06-01",
            },
        ]
    )
    observation = _lookup(
        "permit",
        inputs={"query": "P-", "registry": "dump", "time_range": "2018"},
        dump=dump,
    )
    # injected primary Fake still takes auto/dump? registry=dump uses dump provider
    # _lookup also injects infrastructure_registry_provider as Fake.
    # For registry=dump, code uses injected_dump or injected_primary.
    # injected_dump is set so dump should win... wait:
    # kind == "dump": dump = injected_dump or injected_primary or _build_dump
    # Yes injected_dump wins.
    assert observation.ok is True
    names = [item["project_name"] for item in observation.result["results"]]
    assert names == ["改建许可"]
    hit = observation.result["results"][0]
    assert hit["dates"]["permit"] == "2018-03-15"
    assert hit["dates"]["started"] == "2010-01-01"


def test_empty_hits_mark_coverage_insufficient() -> None:
    provider = FakeRegistryProvider(hits=[])
    observation = _lookup(provider=provider)
    assert observation.ok is True
    assert observation.result["results"] == []
    assert observation.result["coverage"]["insufficient"] is True
    assert any("不代表建筑从未存在" in item for item in observation.result["assumptions"])


def test_dump_keeps_opened_and_permit_dates_separate() -> None:
    dump = AuthorizedDumpProvider(
        [
            {
                "project_name": "某桥",
                "record_type": "construction",
                "reference": "BR-1",
                "permit_date": "2018-03-01",
                "opened_at": "2020-10-01",
                "source_url": "https://archive.example.com/br-1",
            }
        ]
    )
    observation = execute(
        "infrastructure_registry",
        "construction",
        purpose="dump 分日期",
        inputs={"query": "某桥", "registry": "dump"},
        ctx=RuntimeContext(extras={"infrastructure_dump_provider": dump}),
    )
    assert observation.ok is True
    hit = observation.result["results"][0]
    assert hit["dates"]["permit"] == "2018-03-01"
    assert hit["dates"]["opened"] == "2020-10-01"
    assert "construction_year" not in hit["dates"]
    assert hit["evidence"]["url"] == "https://archive.example.com/br-1"


def test_jzsc_portal_without_dump_is_empty_coverage() -> None:
    observation = execute(
        "infrastructure_registry",
        "construction",
        purpose="门户",
        inputs={"query": "黄河铁路桥", "registry": "jzsc"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert observation.result["results"] == []
    assert observation.result["coverage"]["insufficient"] is True
    assert observation.result["coverage"]["official_portal"] == "https://jzsc.mohurd.gov.cn/"
    assert observation.result["applied"]["official_portal"] == "https://jzsc.mohurd.gov.cn/"
    assert any("web_page_read" in item for item in observation.result["assumptions"])


def test_unknown_registry_is_unavailable() -> None:
    observation = _lookup(inputs={"registry": "not-a-source"})
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"


def test_dump_registry_without_config_is_unavailable() -> None:
    observation = execute(
        "infrastructure_registry",
        "construction",
        purpose="缺 dump",
        inputs={"query": "黄河铁路桥", "registry": "dump"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "INFRASTRUCTURE_REGISTRY_DUMP_PATH" in observation.error


def test_planning_data_http_sends_dataset_geometry_and_user_agent(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        parsed = urlparse(str(request.full_url))
        params = parse_qs(parsed.query)
        assert "listed-building" in params.get("dataset", [])
        assert params["geometry_relation"] == ["intersects"]
        assert "POLYGON" in params["geometry"][0]
        assert "-0.15" in params["geometry"][0]
        assert "q" not in params
        assert request.get_header("User-agent") == (
            "geoagent-dataset/1.0 (infrastructure_registry; test)"
        )
        return _FakeHttpResponse({"entities": _sample_planning_entities()})

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv(
        "PLANNING_DATA_USER_AGENT",
        "geoagent-dataset/1.0 (infrastructure_registry; test)",
    )
    monkeypatch.setattr(registry_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "infrastructure_registry",
        "construction",
        purpose="mock planning data",
        inputs={
            "query": "Railway Bridge",
            "area": [-0.15, 51.50, -0.10, 51.52],
            "registry": "planning_data",
        },
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert len(calls) == 1
    hit = observation.result["results"][0]
    assert hit["project_name"] == "Listed Railway Bridge"
    assert hit["record_type"] == "listed-building"
    assert hit["dates"]["recorded"] == "1985-02-01"
    assert hit["dates"]["recorded_kind"] == "entry"
    assert "started" not in hit["dates"]
    assert hit["location"]["crs"] == "wgs84"
    assert hit["evidence"]["url"] == "https://www.planning.data.gov.uk/entity/123456"
    assert observation.result["applied"]["provider"] == "planning_data"
    assert observation.result["coverage"]["insufficient"] is True
    assert "construction_year" not in _nested_keys(observation.result)


def test_planning_data_q_only_for_postcode(monkeypatch: Any) -> None:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        calls.append(request)
        params = parse_qs(urlparse(str(request.full_url)).query)
        assert params["q"] == ["SW1A 1AA"]
        assert params["dataset"] == ["planning-application"]
        return _FakeHttpResponse(
            {
                "entities": [
                    {
                        "entity": 9,
                        "name": "Palace permit",
                        "dataset": "planning-application",
                        "reference": "APP/2020/1",
                        "start-date": "2020-04-01",
                    }
                ]
            }
        )

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setattr(registry_mod.urllib.request, "urlopen", fake_urlopen)

    observation = execute(
        "infrastructure_registry",
        "permit",
        purpose="postcode",
        inputs={"query": "SW1A 1AA", "registry": "england"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert calls
    hit = observation.result["results"][0]
    assert hit["reference"] == "APP/2020/1"
    assert hit["dates"]["recorded_kind"] == "permit"


def test_query_array_is_joined() -> None:
    provider = FakeRegistryProvider()
    observation = _lookup(provider=provider, inputs={"query": ["黄河", "铁路桥"]})
    assert observation.ok is True
    assert provider.search_calls[0].query == "黄河 铁路桥"
