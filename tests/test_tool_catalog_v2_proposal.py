from __future__ import annotations

from pipeline.tool_catalog_v2_proposal import build_tool_catalog_v2


def test_v2_catalog_splits_broad_tools_and_maps_every_operation_once() -> None:
    proposal = build_tool_catalog_v2()
    stats = proposal["stats"]
    assert stats["source_tools"] == 17
    assert stats["proposed_tools"] == 31
    assert stats["operations"] == 57
    assert stats["parameter_fields"] == 223
    assert stats["missing_mappings"] == 0
    assert stats["duplicate_mappings"] == 0


def test_v2_operations_have_compact_and_full_parameter_contracts() -> None:
    proposal = build_tool_catalog_v2()
    for tool in proposal["tools"]:
        assert tool["description"]
        assert tool["executor"]
        assert tool["operations"]
        for operation in tool["operations"]:
            assert operation["description"]
            assert set(operation["compact_params"]) == {
                "required",
                "optional",
                "context",
                "one_of",
            }
            schema = operation["input_schema"]
            assert schema is not None
            for field in schema["fields"]:
                assert field["description"]
                assert field["acquisition_hint"]


def test_osm_features_stay_one_query_tool_but_result_processing_is_separate() -> None:
    proposal = build_tool_catalog_v2()
    by_name = {tool["name"]: tool for tool in proposal["tools"]}
    assert {op["name"] for op in by_name["osm_query"]["operations"]} == {
        "query",
        "count",
    }
    assert {op["name"] for op in by_name["osm_result_process"]["operations"]} == {
        "filter",
        "export",
    }
    assert {op["name"] for op in by_name["satellite_imagery_compare"]["operations"]} == {
        "compare_time",
        "compare_candidates",
    }
