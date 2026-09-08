from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import clear_settings_cache
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.tools import MatchDecision
from pipeline.stage3_normalize_format.format_jsonl import run_stage3
from pipeline.stage3_normalize_format.params import (
    attach_operation_input_schemas,
    normalize_and_validate_tool_inputs,
)
from pipeline.stage3_normalize_format.trees import load_forest


def _forest():
    return attach_operation_input_schemas(
        load_forest(Path("canonical_tool_catalog.json"))
    )


def test_every_catalog_operation_has_explained_input_schema() -> None:
    forest = _forest()
    assert len(forest.trees) == 17
    for tree in forest.trees:
        for operation in tree.canonical.operations:
            assert operation.input_schema is not None, (
                tree.canonical.name,
                operation.name,
            )
            assert operation.input_schema.description.strip()
            for field in operation.input_schema.fields:
                assert field.name.strip()
                assert field.description.strip()


def test_osm_structured_query_does_not_require_raw_code() -> None:
    audit = normalize_and_validate_tool_inputs(
        _forest(),
        tool="osm_query",
        operation="query",
        inputs={
            "区域": "郑州市",
            "标签": {"bridge": "yes"},
            "返回几何": True,
        },
        step_index=1,
    )
    assert audit.valid is True
    assert audit.normalized_inputs == {
        "area": "郑州市",
        "tags": {"bridge": "yes"},
        "return_geometry": True,
    }
    assert "overpass_ql" not in audit.normalized_inputs


def test_osm_raw_overpass_query_is_optional_advanced_input() -> None:
    audit = normalize_and_validate_tool_inputs(
        _forest(),
        tool="osm_query",
        operation="overpass",
        inputs={"查询代码": '[out:json];nwr["bridge"](area.a);out geom;'},
        step_index=1,
    )
    assert audit.valid is True
    assert audit.operation == "query"
    assert audit.normalized_inputs["overpass_ql"].startswith("[out:json]")


def test_satellite_compare_alias_means_candidate_comparison_not_time_comparison() -> None:
    audit = normalize_and_validate_tool_inputs(
        _forest(),
        tool="satellite_imagery_query",
        operation="compare",
        inputs={
            "候选点": ["兰州近水广场", "兰州水车园"],
            "比对模板": "台阶-植被-台阶结构",
        },
        step_index=6,
    )
    assert audit.valid is True
    assert audit.operation == "compare_candidates"
    assert audit.normalized_inputs["candidates"] == [
        "兰州近水广场",
        "兰州水车园",
    ]
    assert audit.normalized_inputs["template"] == "台阶-植被-台阶结构"


def test_missing_candidate_values_is_reported_without_discarding_extra_context() -> None:
    audit = normalize_and_validate_tool_inputs(
        _forest(),
        tool="satellite_imagery_query",
        operation="compare",
        inputs={"候选点数量": 13, "比对模板": "台阶-植被-台阶结构"},
        step_index=6,
    )
    assert audit.valid is False
    assert audit.readiness == "repairable"
    assert audit.normalized_inputs["template"] == "台阶-植被-台阶结构"
    assert audit.normalized_inputs["extensions"] == {"候选点数量": 13}
    codes = {issue.code for issue in audit.issues}
    assert "required_input_missing" in codes
    assert "extra_inputs_preserved" in codes
    assert any(
        action.field == "candidates"
        and action.strategy == "call_prerequisite_tool"
        and "候选列表" in action.guidance
        for action in audit.repair_actions
    )


def test_missing_image_uses_current_image_context_instead_of_review() -> None:
    audit = normalize_and_validate_tool_inputs(
        _forest(),
        tool="image_process",
        operation="enhance",
        inputs={"目标区域": "桥梁索塔"},
        step_index=2,
        available_context={"current_image": "$current_image"},
    )
    assert audit.valid is True
    assert audit.readiness == "context_resolvable"
    assert audit.normalized_inputs["image"] == "$current_image"
    assert any(
        action.strategy == "use_context" and "crop/zoom" in action.guidance
        for action in audit.repair_actions
    )


def test_missing_previous_result_gets_explicit_prerequisite_guidance() -> None:
    audit = normalize_and_validate_tool_inputs(
        _forest(),
        tool="osm_query",
        operation="export",
        inputs={"格式": "geojson"},
        step_index=3,
    )
    assert audit.readiness == "repairable"
    assert any(
        action.field == "source_result"
        and action.strategy == "call_prerequisite_tool"
        and "前置 Tool" in action.guidance
        for action in audit.repair_actions
    )


def test_instance_specific_field_names_stay_in_extensions() -> None:
    """个别轨迹里的对象名不得升格成合同别名。"""

    satellite = normalize_and_validate_tool_inputs(
        _forest(),
        tool="satellite_imagery_query",
        operation="retrieve",
        inputs={"电塔坐标": [113.6, 34.7], "区域": "河南省"},
        step_index=4,
    )
    assert satellite.normalized_inputs["area"] == "河南省"
    assert "coordinates" not in satellite.normalized_inputs
    assert satellite.normalized_inputs["extensions"] == {"电塔坐标": [113.6, 34.7]}

    sightline = normalize_and_validate_tool_inputs(
        _forest(),
        tool="geospatial_analysis",
        operation="sightline",
        inputs={"参照物1": "塔基", "参照物2": "山顶"},
        step_index=5,
    )
    assert "observer" not in sightline.normalized_inputs
    assert "target" not in sightline.normalized_inputs
    assert sightline.normalized_inputs["extensions"] == {
        "参照物1": "塔基",
        "参照物2": "山顶",
    }


def test_unknown_extra_inputs_are_preserved_as_extensions() -> None:
    audit = normalize_and_validate_tool_inputs(
        _forest(),
        tool="web_search",
        operation="keyword_search",
        inputs={"检索对象": "铁路桥", "自定义检索策略": "优先历史照片"},
        step_index=2,
    )
    assert audit.valid is True
    assert audit.normalized_inputs["query"] == "铁路桥"
    assert audit.normalized_inputs["extensions"] == {
        "自定义检索策略": "优先历史照片"
    }


def test_run_stage3_writes_parameter_audit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TOOL_CATALOG_PATH", "canonical_tool_catalog.json")
    monkeypatch.setenv("TOOL_TREES_PATH", str(tmp_path / "runtime_tools.json"))
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    clear_settings_cache()
    freeform = FreeFormTrajectory.model_validate(
        {
            "source_video": "parameter_demo",
            "steps": [
                {
                    "event_type": "tool_call",
                    "thought": "查询郑州桥梁。",
                    "tool": "custom_overpass_bridge_query",
                    "params": {"区域": "郑州市", "标签": {"bridge": "yes"}},
                    "observation": {"result": "返回桥梁要素"},
                },
                {
                    "event_type": "final",
                    "thought": "提交答案。",
                    "tool": "final_answer",
                    "params": {"location": "郑州市"},
                    "observation": None,
                },
            ],
        }
    )

    def matcher(name, _forest):
        assert name == "custom_overpass_bridge_query"
        return MatchDecision(
            raw_tool=name,
            action="map",
            canonical_name="osm_query",
            operation="query",
            operation_description="按区域和标签查询桥梁",
            confidence=0.99,
        )

    trajectory_path = tmp_path / "stage3_trajectory.json"
    entry = run_stage3(
        freeform,
        trees_path=tmp_path / "runtime_tools.json",
        out_trajectory_path=str(trajectory_path),
        out_jsonl_path=str(tmp_path / "demo.jsonl"),
        image_paths=["input.jpg"],
        matcher=matcher,
        compile_params=False,
    )
    parameter_audit = json.loads(
        (tmp_path / "stage3_parameter_audit.json").read_text(encoding="utf-8")
    )
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    params = trajectory["steps"][0]["action"]["params"]
    assert params["operation"] == "query"
    assert params["inputs"] == {
        "area": "郑州市",
        "tags": {"bridge": "yes"},
    }
    assert parameter_audit["valid_calls"] == 1
    assert parameter_audit["total_calls"] == 1
    assert entry.quality_score is None
    assert not (tmp_path / "stage3_quality_report.json").exists()
    clear_settings_cache()
