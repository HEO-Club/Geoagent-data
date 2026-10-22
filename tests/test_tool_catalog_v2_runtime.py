"""Production Tool v2 catalog and Stage 2/3 contract tests."""

from __future__ import annotations

from pathlib import Path

from pipeline.config import clear_settings_cache, get_settings
from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.stage3_normalize_format.format_jsonl import remap_trajectory
from pipeline.stage3_normalize_format.map_tools import (
    _catalog_payload,
    ensure_tool_trees,
)
from pipeline.stage3_normalize_format.trees import load_forest
from pipeline.stage3_normalize_format.params import normalize_and_validate_tool_inputs
from pipeline.tool_catalog_v2 import (
    build_tool_forest_v2,
    render_tool_contract_guidance,
)


def test_v2_catalog_has_granular_tools_and_contracts() -> None:
    forest = build_tool_forest_v2()
    assert len(forest.trees) == 31
    assert sum(len(tree.canonical.operations) for tree in forest.trees) == 56
    names = {tree.canonical.name for tree in forest.trees}
    assert {"osm_query", "osm_result_process"}.issubset(names)
    assert {"satellite_imagery_query", "satellite_imagery_compare"}.issubset(
        names
    )
    infra = next(
        tree for tree in forest.trees if tree.canonical.name == "infrastructure_registry"
    )
    assert infra.canonical.available is False
    route = next(tree for tree in forest.trees if tree.canonical.name == "route_query")
    assert route.canonical.available is False
    admin = next(
        tree for tree in forest.trees if tree.canonical.name == "administrative_registry"
    )
    assert admin.canonical.available is False
    for tree in forest.trees:
        if tree.canonical.is_terminal:
            continue
        assert tree.canonical.operations
        assert all(op.input_schema is not None for op in tree.canonical.operations)


def test_image_runtime_contract_matches_executor_inputs() -> None:
    forest = build_tool_forest_v2()
    image_tools = {tree.canonical.name: tree for tree in forest.trees}

    edit_ops = {
        operation.name: operation.input_schema
        for operation in image_tools["image_edit"].canonical.operations
    }
    crop_fields = {field.name: field for field in edit_ops["crop"].fields}
    zoom_fields = {field.name: field for field in edit_ops["zoom"].fields}
    assert crop_fields["region"].type == "region"
    assert {"padding", "padding_mode", "output_format"} <= set(crop_fields)
    assert zoom_fields["scale"].maximum == 16
    assert "output_format" in zoom_fields

    measure = image_tools["image_measure"].canonical.operations[0]
    measure_fields = {field.name: field for field in measure.input_schema.fields}
    assert measure_fields["axis"].allowed_values == [
        "horizontal",
        "vertical",
        "diagonal",
    ]


def test_guidance_explains_fixed_outer_contract_and_acquisition() -> None:
    forest = build_tool_forest_v2()
    guidance = render_tool_contract_guidance(forest)
    assert "params.operation" in guidance
    assert "params.purpose" in guidance
    assert "params.inputs" in guidance
    assert "acquisition_hint" not in guidance
    assert "获取:" in guidance
    assert "osm_result_process" in guidance
    assert "source_result" in guidance
    assert "infrastructure_registry" not in guidance
    assert "route_query" not in guidance
    assert "administrative_registry" not in guidance
    assert "web_search" in guidance
    offered = {item["canonical"] for item in _catalog_payload(forest)}
    assert "infrastructure_registry" not in offered
    assert "route_query" not in offered
    assert "administrative_registry" not in offered
    assert "web_search" in offered
    official = load_forest(Path("canonical_tool_catalog_v2.json"))
    official_infra = next(
        tree
        for tree in official.trees
        if tree.canonical.name == "infrastructure_registry"
    )
    assert official_infra.canonical.available is False
    official_route = next(
        tree for tree in official.trees if tree.canonical.name == "route_query"
    )
    assert official_route.canonical.available is False
    official_admin = next(
        tree
        for tree in official.trees
        if tree.canonical.name == "administrative_registry"
    )
    assert official_admin.canonical.available is False
    official_guidance = render_tool_contract_guidance(official)
    assert "infrastructure_registry" not in official_guidance
    assert "route_query" not in official_guidance
    assert "administrative_registry" not in official_guidance


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
    assert "geo_feature_search" not in names
    assert "osm_query" in names
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


def test_osm_query_guidance_shows_feature_examples() -> None:
    """阶段2 只渲染工具简介，地物例子必须出现在 osm_query 简介里。"""
    forest = load_forest(Path("canonical_tool_catalog_v2.json"))
    tree = next(item for item in forest.trees if item.canonical.name == "osm_query")
    assert "桥梁" in tree.canonical.description
    assert "feature_types" in tree.canonical.description
    guidance = render_tool_contract_guidance(forest)
    line = next(item for item in guidance.splitlines() if item.startswith("- osm_query:"))
    assert "桥梁" in line
    assert "河流" in line
    assert "feature_types" in line


def test_osm_near_guidance_is_visible() -> None:
    """阶段2 只渲染工具简介，相邻地物检索必须出现在 osm_query 简介里。"""
    forest = load_forest(Path("canonical_tool_catalog_v2.json"))
    tree = next(item for item in forest.trees if item.canonical.name == "osm_query")
    assert "near" in tree.canonical.description
    guidance = render_tool_contract_guidance(forest)
    line = next(item for item in guidance.splitlines() if item.startswith("- osm_query:"))
    assert "near" in line
    assert "subject" in line
    assert "reference" in line


def test_poi_search_contract_does_not_claim_osm_feature_mapping() -> None:
    """poi_search 默认是名称/类型码检索，地物标签属于 osm_query。"""
    forest = load_forest(Path("canonical_tool_catalog_v2.json"))
    tree = next(item for item in forest.trees if item.canonical.name == "poi_search")
    assert "osm_query" in tree.canonical.description
    by_operation = {operation.name: operation for operation in tree.canonical.operations}
    search = by_operation["poi_search"].input_schema
    browse = by_operation["browse"].input_schema
    assert search is not None and browse is not None
    search_fields = {field.name: field.description for field in search.fields}
    browse_fields = {field.name: field.description for field in browse.fields}
    assert "安全映射" not in search_fields["categories"]
    assert "高德" in search_fields["categories"]
    assert "osm_query" in search_fields["query"]
    assert "不是 OSM tag" in browse_fields["filters"]
    assert "osm_query" in (search.description or "")


def test_center_scoped_geo_queries_require_radius() -> None:
    forest = build_tool_forest_v2()
    missing_radius = normalize_and_validate_tool_inputs(
        forest,
        tool="osm_query",
        operation="query",
        inputs={
            "center": {"lat": 34.95, "lon": 113.5},
            "feature_types": ["桥梁"],
        },
        step_index=1,
    )
    assert missing_radius.readiness == "repairable"
    assert any(
        issue.code == "required_input_missing" and issue.field == "radius_m"
        for issue in missing_radius.issues
    )

    complete = normalize_and_validate_tool_inputs(
        forest,
        tool="osm_query",
        operation="query",
        inputs={
            "center": {"lat": 34.95, "lon": 113.5},
            "radius_m": 1000,
            "feature_types": ["桥梁"],
        },
        step_index=2,
    )
    assert complete.readiness == "ready"
