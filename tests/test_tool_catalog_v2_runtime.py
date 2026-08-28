"""Production Tool v2 catalog and Stage 2/3 contract tests."""

from __future__ import annotations

from pathlib import Path

from pipeline.config import clear_settings_cache, get_settings
from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.stage3_normalize_format.format_jsonl import remap_trajectory
from pipeline.stage3_normalize_format.map_tools import ensure_tool_trees
from pipeline.stage3_normalize_format.params import normalize_and_validate_tool_inputs
from pipeline.tool_catalog_v2 import (
    build_tool_forest_v2,
    render_tool_contract_guidance,
)


def test_v2_catalog_has_granular_tools_and_contracts() -> None:
    forest = build_tool_forest_v2()
    assert len(forest.trees) == 31
    assert sum(len(tree.canonical.operations) for tree in forest.trees) == 57
    names = {tree.canonical.name for tree in forest.trees}
    assert {"osm_query", "osm_result_process"}.issubset(names)
    assert {"satellite_imagery_query", "satellite_imagery_compare"}.issubset(
        names
    )
    for tree in forest.trees:
        if tree.canonical.is_terminal:
            continue
        assert tree.canonical.operations
        assert all(op.input_schema is not None for op in tree.canonical.operations)


def test_guidance_explains_fixed_outer_contract_and_acquisition() -> None:
    guidance = render_tool_contract_guidance(build_tool_forest_v2())
    assert "params.operation" in guidance
    assert "params.purpose" in guidance
    assert "params.inputs" in guidance
    assert "acquisition_hint" not in guidance
    assert "获取:" in guidance
    assert "osm_result_process" in guidance
    assert "source_result" in guidance


def test_stage3_accepts_v2_nested_params_and_preserves_purpose() -> None:
    forest = build_tool_forest_v2()
    freeform = FreeFormTrajectory(
        source_video="demo",
        steps=[
            FreeFormStep(
                event_type="tool_call",
                thought="需要统计候选区域内桥梁数量。",
                tool="osm_query",
                params={
                    "operation": "count",
                    "purpose": "比较候选区域的桥梁密度",
                    "inputs": {"area": "郑州市", "tags": {"bridge": "yes"}},
                },
                observation={"count": 4},
            ),
            FreeFormStep(
                event_type="final",
                thought="提交位置。",
                tool="final_answer",
                params={"location": "郑州市"},
                observation=None,
            ),
        ],
    )
    audits = []
    trajectory = remap_trajectory(
        freeform,
        forest,
        system_prompt="system",
        user_query="query",
        parameter_audits=audits,
        compile_params=False,
    )
    action = trajectory.steps[0].action
    assert action is not None
    assert action.tool == "osm_query"
    assert action.params["operation"] == "count"
    assert action.params["purpose"] == "比较候选区域的桥梁密度"
    assert action.params["inputs"]["area"] == "郑州市"
    assert audits[0].operation == "count"


def test_v2_catalog_loads_without_legacy_roots(
    tmp_path: Path, monkeypatch,
) -> None:
    """Official v2 catalog is the only source; no runtime dump merge."""
    monkeypatch.setenv("TOOL_CATALOG_PATH", "canonical_tool_catalog_v2.json")
    clear_settings_cache()
    try:
        freeform = FreeFormTrajectory(
            source_video="demo",
            steps=[
                FreeFormStep(
                    event_type="final",
                    thought="提交位置。",
                    tool="final_answer",
                    params={"location": "郑州市"},
                    observation=None,
                )
            ],
        )
        merged = ensure_tool_trees(freeform, Path(get_settings().TOOL_CATALOG_PATH))
    finally:
        clear_settings_cache()
    names = {tree.canonical.name for tree in merged.trees}
    assert "map_query" not in names
    assert "poi_search" in names
    assert len(names) == 31


def test_narrative_step_reference_is_not_marked_executable() -> None:
    audit = normalize_and_validate_tool_inputs(
        build_tool_forest_v2(),
        tool="video_frame_extract",
        operation="frame_retrieve",
        inputs={
            "video": "步骤3检索到的匹配飞行记录",
            "timestamps": "03:11",
        },
        step_index=4,
        available_context={},
    )
    assert audit.readiness == "repairable"
    assert any(issue.code == "input_unresolved_reference" for issue in audit.issues)


def test_grounded_values_are_coerced_without_requiring_full_params() -> None:
    forest = build_tool_forest_v2()
    heading = normalize_and_validate_tool_inputs(
        forest,
        tool="satellite_imagery_query",
        operation="oblique_view",
        inputs={"area": "荔浦市", "heading": "朝西"},
        step_index=1,
    )
    assert heading.readiness == "ready"
    assert heading.normalized_inputs["heading"] == 270.0

    measurement = normalize_and_validate_tool_inputs(
        forest,
        tool="image_measure",
        operation="measure",
        inputs={
            "image": "$current_image",
            "measurement": "人物身高与投影长度的比例",
        },
        step_index=2,
    )
    assert measurement.readiness == "ready"
    assert measurement.normalized_inputs["measurement"] == "ratio"

    streetview = normalize_and_validate_tool_inputs(
        forest,
        tool="streetview_query",
        operation="open",
        inputs={"coordinates": "$step_1_tool_result.coordinates"},
        step_index=3,
    )
    assert streetview.readiness == "ready"
