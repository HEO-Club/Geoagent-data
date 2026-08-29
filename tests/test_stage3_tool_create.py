"""阶段3 create 程序门、提案队列与临时名（禁止真实 API）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.config import clear_settings_cache
from pipeline.schemas.confidence import ConfidenceJudgeDraft
from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.schemas.tools import (
    InputFieldSpec,
    MatchDecision,
    ToolDefinition,
    ToolInputSchema,
    ToolOperation,
)
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage3_normalize_format import map_tools, trees
from pipeline.stage3_normalize_format.format_jsonl import run_stage3
from pipeline.stage4_confidence import merge_confidence, run_stage4
from pipeline.tool_catalog_v2 import build_tool_forest_v2
from tests.test_stage4_confidence import _entry, _freeform, _task, _traj_ok

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CATALOG = REPO_ROOT / "canonical_tool_catalog_v2.json"


def _complete_lidar_definition() -> ToolDefinition:
    return ToolDefinition(
        name="lidar_tile_query",
        description="按范围查询激光雷达切片",
        executor="lidar_tile_api",
        usage="需要点云或高程切片证据时调用",
        operations=[
            ToolOperation(
                name="fetch_tile",
                description="按包围盒取切片",
                input_schema=ToolInputSchema(
                    description="切片查询",
                    fields=[
                        InputFieldSpec(
                            name="bbox",
                            type="array",
                            requirement_level="execution",
                            description="查询范围 [min_lon, min_lat, max_lon, max_lat]",
                            acquisition_hint="从当前工作区或前置定位框获取",
                        )
                    ],
                ),
            )
        ],
    )


def _incomplete_definition(name: str = "crop_helper") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="裁剪并观察",
        executor=name,
        usage="看图细节时调用",
        operations=[
            ToolOperation(name="execute", description="执行裁剪观察")
        ],
    )


def _unknown_crop_freeform(source_video: str) -> FreeFormTrajectory:
    return FreeFormTrajectory(
        source_video=source_video,
        steps=[
            FreeFormStep(
                event_type="tool_call",
                thought="放大看屋顶纹理",
                tool="crop_and_look",
                params={"bbox": [0, 0, 10, 10]},
                observation={"detail": "红瓦"},
            ),
            FreeFormStep(
                event_type="final",
                thought="提交",
                tool="final_answer",
                params={"location": "某市"},
                observation=None,
            ),
        ],
    )


def _write_v2_catalog(path: Path) -> Path:
    trees.save_forest(build_tool_forest_v2(), path)
    return path


def test_match_decision_coerces_new_operation_action() -> None:
    decision = MatchDecision.model_validate(
        {
            "raw_tool": "extra_filter",
            "action": "new_operation",
            "canonical_name": "osm_query",
            "operation": "filter_local",
            "confidence": 0.9,
        }
    )
    assert decision.action == "create"
    assert decision.create_kind == "new_operation"


def test_unknown_tool_maps_to_catalog_without_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STAGE3_COMPILE_PARAMS", "false")
    monkeypatch.setenv("TOOL_CATALOG_PATH", str(OFFICIAL_CATALOG))
    clear_settings_cache()
    catalog_path = _write_v2_catalog(tmp_path / "catalog.json")
    before = OFFICIAL_CATALOG.read_bytes()

    def matcher(name: str, _forest: object) -> MatchDecision | None:
        if name == "crop_and_look":
            return MatchDecision(
                raw_tool=name,
                action="map",
                canonical_name="web_search",
                operation="keyword_search",
                confidence=0.99,
                reason="同一检索执行器",
            )
        return None

    entry = run_stage3(
        _unknown_crop_freeform("map_ok"),
        trees_path=catalog_path,
        out_trajectory_path=str(tmp_path / "stage3_trajectory.json"),
        out_jsonl_path=str(tmp_path / "shard.jsonl"),
        image_paths=["scene.jpg"],
        matcher=matcher,
        compile_params=False,
    )
    assert '"tool": "web_search"' in entry.messages[2].content
    mapping = json.loads(
        (tmp_path / "stage3_tool_mapping.json").read_text(encoding="utf-8")
    )
    crop = next(
        item for item in mapping["mappings"] if item.get("raw_tool") == "crop_and_look"
    )
    assert crop["disposition"] == "mapped"
    assert crop["temporary"] is False
    assert mapping["temporary_tools"] == []
    proposals = json.loads(
        (tmp_path / "stage3_tool_proposals.json").read_text(encoding="utf-8")
    )
    assert proposals["tools"] == []
    assert OFFICIAL_CATALOG.read_bytes() == before

    report = merge_confidence(
        task_id="map_ok",
        format_score=1.0,
        format_reason="ok",
        programmatic_gates=[],
        draft=ConfidenceJudgeDraft(
            evidence_grounding=0.95,
            final_answer_support=0.95,
            logical_consistency=0.95,
            input_quality_alignment=0.95,
            notes="主链正确",
        ),
        judge_call_failed=False,
        param_score=1.0,
        param_reason="全部 ready",
        audit_coverage=1.0,
        temporary_tools=mapping["temporary_tools"],
    )
    assert "临时工具" not in report.notes


def test_create_without_schema_is_temporary_and_notes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STAGE3_COMPILE_PARAMS", "false")
    monkeypatch.setenv("TOOL_CATALOG_PATH", str(OFFICIAL_CATALOG))
    clear_settings_cache()
    catalog_path = _write_v2_catalog(tmp_path / "catalog.json")
    before = OFFICIAL_CATALOG.read_bytes()

    def matcher(name: str, _forest: object) -> MatchDecision | None:
        if name == "crop_and_look":
            return MatchDecision(
                raw_tool=name,
                action="create",
                create_kind="new_executor",
                canonical_name="crop_helper",
                operation="execute",
                confidence=0.95,
                proposed_definition=_incomplete_definition(),
                not_catalog_reason="不是 zoom_inspect / web_search",
                reason="需要新执行器",
            )
        return None

    entry = run_stage3(
        _unknown_crop_freeform("temp_ok"),
        trees_path=catalog_path,
        out_trajectory_path=str(tmp_path / "stage3_trajectory.json"),
        out_jsonl_path=str(tmp_path / "shard.jsonl"),
        image_paths=["scene.jpg"],
        matcher=matcher,
        compile_params=False,
    )
    blob = json.dumps([m.content for m in entry.messages], ensure_ascii=False)
    assert "Action:" in blob
    assert "crop_and_look" in blob or "custom_tool_" in blob
    mapping = json.loads(
        (tmp_path / "stage3_tool_mapping.json").read_text(encoding="utf-8")
    )
    crop = next(
        item for item in mapping["mappings"] if item.get("raw_tool") == "crop_and_look"
    )
    assert crop["disposition"] == "temporary"
    assert crop["temporary"] is True
    assert mapping["temporary_tools"]
    assert mapping["temporary_tools"][0]["raw_tool"] == "crop_and_look"
    assert "缺 input_schema" in mapping["temporary_tools"][0]["reason"]
    proposals = json.loads(
        (tmp_path / "stage3_tool_proposals.json").read_text(encoding="utf-8")
    )
    names = [item.get("canonical_name") for item in proposals["tools"]]
    assert "crop_helper" not in names
    assert "crop_and_look" not in names
    assert OFFICIAL_CATALOG.read_bytes() == before

    temp_tools = mapping["temporary_tools"]
    high = dict(
        task_id="temp_ok",
        format_score=1.0,
        format_reason="ok",
        programmatic_gates=[],
        draft=ConfidenceJudgeDraft(
            evidence_grounding=0.95,
            final_answer_support=0.95,
            logical_consistency=0.95,
            input_quality_alignment=0.95,
            notes="主链正确",
        ),
        judge_call_failed=False,
        param_score=1.0,
        param_reason="全部 ready",
        audit_coverage=1.0,
    )
    without = merge_confidence(**high)
    with_temp = merge_confidence(**high, temporary_tools=temp_tools)
    assert without.decision == with_temp.decision
    assert without.quality_score == pytest.approx(with_temp.quality_score)
    assert "临时工具:" in with_temp.notes
    assert "crop_and_look→" in with_temp.notes
    assert "缺 input_schema" in with_temp.notes
    assert "临时工具" not in without.notes

    def fake_judge(**_k: Any) -> ConfidenceJudgeDraft:
        return ConfidenceJudgeDraft(
            evidence_grounding=0.95,
            final_answer_support=0.95,
            logical_consistency=0.95,
            input_quality_alignment=0.95,
            notes="主链正确",
        )

    report = run_stage4(
        task=_task(),
        transcript=[TranscriptSegment(start=0, end=1, text="旁白")],
        freeform=_freeform(),
        trajectory=_traj_ok(),
        entry=_entry(),
        tool_mapping=mapping,
        out_report_path=str(tmp_path / "stage4.json"),
        out_jsonl_path=str(tmp_path / "out.jsonl"),
        judge=fake_judge,
    )
    assert "临时工具:" in report.notes
    assert report.soft_flags == []
    assert "temporary" not in report.applied_soft_caps


def test_create_with_complete_schema_writes_proposal_not_official_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STAGE3_COMPILE_PARAMS", "false")
    monkeypatch.setenv("TOOL_CATALOG_PATH", str(OFFICIAL_CATALOG))
    clear_settings_cache()
    catalog_path = _write_v2_catalog(tmp_path / "catalog.json")
    before = OFFICIAL_CATALOG.read_bytes()

    def matcher(name: str, _forest: object) -> MatchDecision | None:
        if name == "crop_and_look":
            return MatchDecision(
                raw_tool=name,
                action="create",
                create_kind="new_executor",
                canonical_name="lidar_tile_query",
                operation="fetch_tile",
                confidence=0.96,
                proposed_definition=_complete_lidar_definition(),
                not_catalog_reason="不是 satellite_imagery_query / map_layer_query",
                reason="独立点云切片执行器",
            )
        return None

    run_stage3(
        _unknown_crop_freeform("proposal_ok"),
        trees_path=catalog_path,
        out_trajectory_path=str(tmp_path / "stage3_trajectory.json"),
        out_jsonl_path=str(tmp_path / "shard.jsonl"),
        image_paths=["scene.jpg"],
        matcher=matcher,
        compile_params=False,
    )
    mapping = json.loads(
        (tmp_path / "stage3_tool_mapping.json").read_text(encoding="utf-8")
    )
    crop = next(
        item for item in mapping["mappings"] if item.get("raw_tool") == "crop_and_look"
    )
    assert crop["disposition"] == "created_proposal"
    assert mapping["temporary_tools"] == []
    proposals = json.loads(
        (tmp_path / "stage3_tool_proposals.json").read_text(encoding="utf-8")
    )
    assert len(proposals["tools"]) == 1
    assert proposals["tools"][0]["canonical_name"] == "lidar_tile_query"
    assert proposals["tools"][0]["kind"] == "new_executor"
    fields = proposals["tools"][0]["definition"]["operations"][0]["input_schema"][
        "fields"
    ]
    assert fields[0]["acquisition_hint"]
    assert OFFICIAL_CATALOG.read_bytes() == before
    official_forest = trees.load_forest(OFFICIAL_CATALOG)
    assert trees.find_tree_for_name(official_forest, "lidar_tile_query") is None


def test_high_confidence_reasoning_still_demoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STAGE3_COMPILE_PARAMS", "false")
    monkeypatch.setenv("TOOL_CATALOG_PATH", str(OFFICIAL_CATALOG))
    clear_settings_cache()
    catalog_path = _write_v2_catalog(tmp_path / "catalog.json")
    freeform = FreeFormTrajectory(
        source_video="demote2",
        steps=[
            FreeFormStep(
                thought="根据上一轮日落时间排除江苏。",
                tool="apply_time_consistency_filter",
                params={"capture_time": "18:54"},
                observation={"eliminated": "江苏"},
            )
        ],
    )

    def matcher(name: str, _forest: object) -> MatchDecision | None:
        return MatchDecision(
            raw_tool=name,
            action="reasoning",
            confidence=0.98,
            reason="没有访问外部执行器，只在合并已有证据",
        )

    records: list[map_tools.ToolResolutionRecord] = []
    forest = map_tools.ensure_tool_trees(
        freeform, catalog_path, matcher=matcher, resolution_records=records
    )
    assert freeform.steps[0].event_type == "reasoning"
    assert freeform.steps[0].tool is None
    assert trees.find_tree_for_name(forest, "apply_time_consistency_filter") is None
    assert records[0].disposition == "demoted"


def test_reclassify_still_rejects_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STAGE3_COMPILE_PARAMS", "false")
    monkeypatch.setenv("TOOL_CATALOG_PATH", str(OFFICIAL_CATALOG))
    clear_settings_cache()
    catalog_path = _write_v2_catalog(tmp_path / "catalog.json")
    freeform = FreeFormTrajectory(
        source_video="reclassify_create",
        steps=[
            FreeFormStep(
                event_type="tool_call",
                thought="检索1999年之前建成的超100米建筑名单",
                tool="web_search",
                params={
                    "operation": "keyword_search",
                    "purpose": "列高楼候选",
                    "inputs": {"query": ["1999年前 超100米 建筑"]},
                },
                observation={"result": "列出若干候选大楼"},
            )
        ],
    )

    def matcher(_name: str, _forest: object) -> MatchDecision:
        return MatchDecision(
            raw_tool="web_search",
            action="create",
            create_kind="new_executor",
            canonical_name="lidar_tile_query",
            operation="fetch_tile",
            confidence=0.99,
            proposed_definition=_complete_lidar_definition(),
            reason="重分类不得 create",
        )

    forest = map_tools.ensure_tool_trees(freeform, catalog_path, matcher=matcher)
    assert freeform.steps[0].tool == "web_search"
    assert trees.find_tree_for_name(forest, "lidar_tile_query") is None
