"""Stage 3 Thought→schema 参数编译测试（全部 mock，禁止真 API）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.config import clear_settings_cache
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.tools import MatchDecision
from pipeline.stage3_normalize_format.compile_params import (
    CompileFill,
    CompileRequest,
    apply_compile_and_revalidate,
    build_compile_request,
    filter_grounded_fills,
)
from pipeline.stage3_normalize_format.format_jsonl import remap_trajectory, run_stage3
from pipeline.stage3_normalize_format.params import (
    attach_operation_input_schemas,
    normalize_and_validate_tool_inputs,
)
from pipeline.stage3_normalize_format.trees import load_forest


def _forest():
    return attach_operation_input_schemas(
        load_forest(Path("canonical_tool_catalog.json"))
    )


def test_geometry_filter_compiles_near_distance_from_thought() -> None:
    forest = _forest()
    context = {"previous_tool_result": "$step_1_tool_result"}
    first = normalize_and_validate_tool_inputs(
        forest,
        tool="geospatial_analysis",
        operation="geometry_filter",
        inputs={
            "region": "山东山区",
            "nearby_condition": "5公里内有风力发电机",
        },
        step_index=2,
        available_context=context,
    )
    assert first.readiness == "repairable"
    assert first.normalized_inputs.get("source_result") == "$step_1_tool_result"

    req = build_compile_request(
        forest=forest,
        audit=first,
        thought="筛选电塔：只保留 5 公里内有风力发电机的结果。",
        available_context=context,
    )
    assert req is not None
    fill = CompileFill(
        step_index=2,
        filled_inputs={
            "relation": "near",
            "geometry": "风力发电机",
            "distance_m": 5000,
        },
        reason="从 nearby_condition 编译",
    )
    second = apply_compile_and_revalidate(forest, req, fill)
    assert second.readiness in {"ready", "context_resolvable"}
    assert second.normalized_inputs["relation"] == "near"
    assert "风力" in str(second.normalized_inputs["geometry"])
    assert second.normalized_inputs["distance_m"] == 5000
    assert second.normalized_inputs["source_result"] == "$step_1_tool_result"
    assert any(
        issue.code == "input_compiled_from_thought" for issue in second.issues
    )


def test_load_layer_compiles_layers_from_historical_thought() -> None:
    forest = _forest()
    first = normalize_and_validate_tool_inputs(
        forest,
        tool="map_query",
        operation="load_layer",
        inputs={
            "area": "郑州附近黄河段",
            "time": "90年代",
            "features": "高地、桥梁、平原",
        },
        step_index=1,
    )
    assert first.readiness == "repairable"
    assert "layers" not in first.normalized_inputs

    req = build_compile_request(
        forest=forest,
        audit=first,
        thought="打开郑州黄河段历史地图，查看90年代高地、桥梁与平原。",
        available_context={},
    )
    assert req is not None
    fill = CompileFill(
        step_index=1,
        filled_inputs={
            "layers": ["historical_map"],
            "time_range": "90年代",
        },
        reason="历史地图 → historical_map",
    )
    second = apply_compile_and_revalidate(forest, req, fill)
    assert second.readiness in {"ready", "context_resolvable"}
    layers = second.normalized_inputs["layers"]
    assert layers == ["historical_map"] or layers == "historical_map"
    assert any(
        issue.code == "input_compiled_from_thought" for issue in second.issues
    )


def test_ready_tool_call_still_invokes_param_compiler() -> None:
    """方案 A：有 schema 的 tool_call 都编译，不因第一轮 ready 而跳过。"""
    forest = _forest()
    freeform = FreeFormTrajectory.model_validate(
        {
            "source_video": "ready_demo",
            "steps": [
                {
                    "event_type": "tool_call",
                    "thought": "查询郑州桥梁。",
                    "tool": "osm_query",
                    "params": {"area": "郑州市", "tags": {"bridge": "yes"}},
                    "observation": {"result": "ok"},
                },
                {
                    "event_type": "final",
                    "thought": "提交。",
                    "tool": "final_answer",
                    "params": {"location": "郑州市"},
                    "observation": None,
                },
            ],
        }
    )
    calls: list[list[CompileRequest]] = []

    def compiler(requests: list[CompileRequest]) -> dict[int, CompileFill]:
        calls.append(requests)
        return {}

    audits: list = []
    remap_trajectory(
        freeform,
        forest,
        system_prompt="sys",
        user_query="Locate.",
        image_paths=["a.jpg"],
        parameter_audits=audits,
        param_compiler=compiler,
        compile_params=True,
    )
    assert len(calls) == 1
    assert len(calls[0]) == 1
    assert calls[0][0].operation == "query"
    assert audits[0].readiness == "ready"


def test_fabricated_coordinates_and_overpass_are_rejected() -> None:
    forest = _forest()
    first = normalize_and_validate_tool_inputs(
        forest,
        tool="osm_query",
        operation="query",
        inputs={"features": "桥梁"},
        step_index=1,
    )
    assert first.readiness == "repairable"
    req = build_compile_request(
        forest=forest,
        audit=first,
        thought="查一下附近有没有桥。",
        available_context={},
    )
    assert req is not None
    # 来源中没有坐标与 Overpass 代码
    grounded = filter_grounded_fills(
        req,
        {
            "area": "编造市",
            "coordinates": [113.6, 34.7],
            "overpass_ql": '[out:json];node["bridge"](area);out;',
        },
    )
    assert "coordinates" not in grounded
    assert "overpass_ql" not in grounded
    assert "area" not in grounded  # 「编造市」不在来源中

    second = apply_compile_and_revalidate(
        forest,
        req,
        CompileFill(step_index=1, filled_inputs=grounded or {"area": "编造市"}),
    )
    assert second.readiness == "repairable"


def test_compiler_failure_keeps_repairable_and_writes_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOOL_CATALOG_PATH", "canonical_tool_catalog.json")
    monkeypatch.setenv("TOOL_TREES_PATH", str(tmp_path / "runtime_tools.json"))
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    clear_settings_cache()

    freeform = FreeFormTrajectory.model_validate(
        {
            "source_video": "fail_open_demo",
            "steps": [
                {
                    "event_type": "tool_call",
                    "thought": "打开郑州黄河段历史地图查看90年代地貌。",
                    "tool": "load_hydrology_map",
                    "params": {
                        "area": "郑州附近黄河段",
                        "features": "高地、桥梁",
                    },
                    "observation": {"result": "图层已开"},
                },
                {
                    "event_type": "final",
                    "thought": "提交。",
                    "tool": "final_answer",
                    "params": {"location": "郑州"},
                    "observation": None,
                },
            ],
        }
    )

    def matcher(name: str, _forest):
        return MatchDecision(
            raw_tool=name,
            action="map",
            canonical_name="map_query",
            operation="load_layer",
            confidence=0.99,
        )

    def boom(_requests: list[CompileRequest]) -> dict[int, CompileFill]:
        raise RuntimeError("compiler down")

    traj_path = tmp_path / "stage3_trajectory.json"
    entry = run_stage3(
        freeform,
        trees_path=tmp_path / "runtime_tools.json",
        out_trajectory_path=str(traj_path),
        out_jsonl_path=str(tmp_path / "demo.jsonl"),
        image_paths=["input.jpg"],
        matcher=matcher,
        param_compiler=boom,
        compile_params=True,
    )
    assert traj_path.exists()
    assert entry.id
    audit = json.loads(
        (tmp_path / "stage3_parameter_audit.json").read_text(encoding="utf-8")
    )
    assert audit["total_calls"] == 1
    assert audit["calls"][0]["readiness"] == "repairable"
    clear_settings_cache()


def test_run_stage3_writes_compiled_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOOL_CATALOG_PATH", "canonical_tool_catalog.json")
    monkeypatch.setenv("TOOL_TREES_PATH", str(tmp_path / "runtime_tools.json"))
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    clear_settings_cache()

    freeform = FreeFormTrajectory.model_validate(
        {
            "source_video": "compile_demo",
            "steps": [
                {
                    "event_type": "tool_call",
                    "thought": "先查山东山区电塔。",
                    "tool": "gis_tower_query",
                    "params": {"area": "山东山区", "features": "电塔"},
                    "observation": {"result": "电塔列表"},
                },
                {
                    "event_type": "tool_call",
                    "thought": "筛选：只保留 5 公里内有风力发电机的电塔。",
                    "tool": "gis_tower_filter",
                    "params": {
                        "nearby_condition": "5公里内有风力发电机",
                    },
                    "observation": {"result": "过滤后电塔"},
                },
                {
                    "event_type": "final",
                    "thought": "提交。",
                    "tool": "final_answer",
                    "params": {"location": "山东"},
                    "observation": None,
                },
            ],
        }
    )

    def matcher(name: str, _forest):
        mapping = {
            "gis_tower_query": ("osm_query", "query"),
            "gis_tower_filter": ("geospatial_analysis", "geometry_filter"),
        }
        canonical, operation = mapping[name]
        return MatchDecision(
            raw_tool=name,
            action="map",
            canonical_name=canonical,
            operation=operation,
            confidence=0.99,
        )

    def compiler(requests: list[CompileRequest]) -> dict[int, CompileFill]:
        ops = {item.operation for item in requests}
        assert "geometry_filter" in ops
        assert "query" in ops
        fills: dict[int, CompileFill] = {}
        for item in requests:
            if item.operation != "geometry_filter":
                continue
            fills[item.step_index] = CompileFill(
                step_index=item.step_index,
                filled_inputs={
                    "relation": "near",
                    "geometry": "风力发电机",
                    "distance_m": 5000,
                },
                reason="mock compile",
            )
        return fills

    traj_path = tmp_path / "stage3_trajectory.json"
    run_stage3(
        freeform,
        trees_path=tmp_path / "runtime_tools.json",
        out_trajectory_path=str(traj_path),
        out_jsonl_path=str(tmp_path / "demo.jsonl"),
        image_paths=["input.jpg"],
        matcher=matcher,
        param_compiler=compiler,
        compile_params=True,
    )
    trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
    filter_inputs = trajectory["steps"][1]["action"]["params"]["inputs"]
    assert filter_inputs["relation"] == "near"
    assert filter_inputs["distance_m"] == 5000
    assert "风力" in str(filter_inputs["geometry"])
    assert filter_inputs["source_result"].startswith("$step_")

    audit = json.loads(
        (tmp_path / "stage3_parameter_audit.json").read_text(encoding="utf-8")
    )
    filter_audit = audit["calls"][1]
    assert filter_audit["readiness"] in {"ready", "context_resolvable"}
    assert any(
        issue["code"] == "input_compiled_from_thought"
        for issue in filter_audit["issues"]
    )
    clear_settings_cache()
