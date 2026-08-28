"""Stage 2 tool-interval soft prior and action-coverage review."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.config import clear_settings_cache
from pipeline.schemas.audit import (
    GeoTaskSpec,
    ProcessInterval,
    ProcessRole,
    TargetKind,
)
from pipeline.schemas.clues import ClueExtractionResult
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage2_freeform_tao import run as stage2
from pipeline.stage2_freeform_tao.action_review import (
    ActionCoverageItem,
    ActionCoverageResult,
    action_coverage_retry_warning,
    missed_actions,
)


def _task_with_intervals() -> GeoTaskSpec:
    return GeoTaskSpec(
        task_id="demo__t01",
        time_start=0.0,
        time_end=100.0,
        target_kind=TargetKind.still_image,
        visual_evidence_brief="河边老人与对岸高楼",
        process_intervals=[
            ProcessInterval(start=0.0, end=20.0, role=ProcessRole.show_source),
            ProcessInterval(start=30.0, end=50.0, role=ProcessRole.tool),
            ProcessInterval(start=55.0, end=70.0, role=ProcessRole.tool),
            ProcessInterval(start=80.0, end=95.0, role=ProcessRole.reveal),
        ],
    )


def test_tool_interval_hint_only_includes_tool_windows() -> None:
    hint = stage2._format_tool_interval_hint(_task_with_intervals())
    assert "[30.0, 50.0)" in hint
    assert "[55.0, 70.0)" in hint
    assert "[0.0, 20.0)" not in hint
    assert "[80.0, 95.0)" not in hint
    assert "不含 show_source/reveal" in hint
    assert stage2._format_tool_interval_hint(None) == ""


def test_stage2_prompt_contains_map_boundary_and_tool_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STAGE2_OBSERVATION_REVIEW", "false")
    monkeypatch.setenv("STAGE2_ACTION_COVERAGE_REVIEW", "false")
    clear_settings_cache()
    monkeypatch.setattr(stage2, "extract_working_scope", lambda _: ClueExtractionResult())
    prompts: list[str] = []

    def generate(prompt: str, schema: type, **kwargs: object) -> object:
        prompts.append(prompt)
        return stage2._LLMFreeFormResult.model_validate(
            {
                "steps": [
                    {
                        "event_type": "reasoning",
                        "thought": "直接看图排除湖泊",
                        "tool": None,
                        "params": {},
                        "observation": None,
                    },
                    {
                        "event_type": "final",
                        "thought": "提交",
                        "tool": "final_answer",
                        "params": {"location": "示例地"},
                        "observation": None,
                    },
                ]
            }
        )

    monkeypatch.setattr(stage2, "call_structured", generate)
    stage2.run_stage2(
        "v.mp4",
        [TranscriptSegment(start=0, end=10, text="细看照片排除湖泊")],
        out_path=str(tmp_path / "stage2_freeform_tao.json"),
        image_paths=["frame.jpg"],
        task=_task_with_intervals(),
    )
    assert prompts
    assert "satellite_imagery_query" in prompts[0]
    assert "streetview_query" in prompts[0]
    assert "禁止把后三类默认写成 web_search" in prompts[0]
    assert "[30.0, 50.0)" in prompts[0]
    assert "[80.0, 95.0)" not in prompts[0]


def test_missed_map_action_triggers_regeneration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STAGE2_OBSERVATION_REVIEW", "false")
    monkeypatch.setenv("STAGE2_ACTION_COVERAGE_REVIEW", "true")
    monkeypatch.setenv("STAGE2_MAX_GENERATIONS", "3")
    clear_settings_cache()
    monkeypatch.setattr(stage2, "extract_working_scope", lambda _: ClueExtractionResult())
    prompts: list[str] = []
    coverage_calls = {"n": 0}

    def generate(prompt: str, schema: type, **kwargs: object) -> object:
        prompts.append(prompt)
        if len(prompts) == 1:
            steps = [
                {
                    "event_type": "reasoning",
                    "thought": "许昌附近多为平原，应转向黄河格局",
                    "tool": None,
                    "params": {},
                    "observation": None,
                },
                {
                    "event_type": "final",
                    "thought": "提交",
                    "tool": "final_answer",
                    "params": {"location": "黄河文化公园"},
                    "observation": None,
                },
            ]
        else:
            steps = [
                {
                    "event_type": "tool_call",
                    "thought": "打开地图并调到最早卫星排查",
                    "tool": "satellite_imagery_query",
                    "params": {
                        "operation": "change_time",
                        "purpose": "查看最早卫星",
                        "inputs": {"area": "候选河段", "time_range": "最早可用"},
                    },
                    "observation": {"result": "未见符合两山夹桥的河段"},
                },
                {
                    "event_type": "final",
                    "thought": "提交",
                    "tool": "final_answer",
                    "params": {"location": "黄河文化公园"},
                    "observation": None,
                },
            ]
        return stage2._LLMFreeFormResult.model_validate({"steps": steps})

    def coverage_reviewer(**_kwargs: object) -> ActionCoverageResult:
        coverage_calls["n"] += 1
        if coverage_calls["n"] == 1:
            return ActionCoverageResult(
                items=[
                    ActionCoverageItem(
                        action_summary="打开地图并调到最早卫星排查",
                        evidence="字幕：我就打开地图，把时间调到最早的卫星遥感",
                        covered_by_trajectory=False,
                        confidence=0.95,
                        suggested_tool="satellite_imagery_query",
                        correction="补写卫星调时相 tool_call",
                        kind="map_or_satellite",
                    )
                ]
            )
        return ActionCoverageResult(
            items=[
                ActionCoverageItem(
                    action_summary="打开地图并调到最早卫星排查",
                    evidence="字幕：我就打开地图，把时间调到最早的卫星遥感",
                    covered_by_trajectory=True,
                    confidence=0.95,
                    kind="map_or_satellite",
                )
            ]
        )

    monkeypatch.setattr(stage2, "call_structured", generate)
    dest = tmp_path / "stage2_freeform_tao.json"
    traj = stage2.run_stage2(
        "v.mp4",
        [
            TranscriptSegment(
                start=0,
                end=20,
                text="我就打开地图，把时间调到最早的卫星遥感进行排查",
            )
        ],
        out_path=str(dest),
        image_paths=["frame.jpg"],
        action_coverage_reviewer=coverage_reviewer,
    )
    assert len(prompts) == 2
    assert "动作覆盖提醒" in prompts[1]
    assert any(step.event_type == "tool_call" for step in traj.steps)
    trace = json.loads(
        dest.with_name("stage2_observation_audit.json").read_text(encoding="utf-8")
    )
    assert trace["generation_count"] == 2
    assert trace["accepted"] is True


def test_look_only_reasoning_does_not_trigger_action_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STAGE2_OBSERVATION_REVIEW", "false")
    monkeypatch.setenv("STAGE2_ACTION_COVERAGE_REVIEW", "true")
    clear_settings_cache()
    monkeypatch.setattr(stage2, "extract_working_scope", lambda _: ClueExtractionResult())
    prompts: list[str] = []

    def generate(prompt: str, schema: type, **kwargs: object) -> object:
        prompts.append(prompt)
        return stage2._LLMFreeFormResult.model_validate(
            {
                "steps": [
                    {
                        "event_type": "reasoning",
                        "thought": "细看照片水面均匀，排除湖泊",
                        "tool": None,
                        "params": {},
                        "observation": None,
                    },
                    {
                        "event_type": "final",
                        "thought": "提交",
                        "tool": "final_answer",
                        "params": {"location": "示例河段"},
                        "observation": None,
                    },
                ]
            }
        )

    monkeypatch.setattr(stage2, "call_structured", generate)
    stage2.run_stage2(
        "v.mp4",
        [TranscriptSegment(start=0, end=8, text="细看照片，水面均匀，排除湖泊")],
        out_path=str(tmp_path / "stage2_freeform_tao.json"),
        image_paths=["frame.jpg"],
        action_coverage_reviewer=lambda **_: ActionCoverageResult(items=[]),
    )
    assert len(prompts) == 1


def test_low_confidence_miss_does_not_regenerate() -> None:
    result = ActionCoverageResult(
        items=[
            ActionCoverageItem(
                action_summary="可能打开过地图",
                evidence="不明确",
                covered_by_trajectory=False,
                confidence=0.5,
                kind="map_or_satellite",
            )
        ]
    )
    assert missed_actions(result) == []
    assert action_coverage_retry_warning([]) == ""
