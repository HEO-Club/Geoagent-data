"""阶段2 测试（mock LLM）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.schemas.clues import BoundKind, ClueExtractionResult, WorkingScope
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage2_freeform_tao import run as stage2


def test_run_stage2_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    monkeypatch.setattr(stage2, "video_duration_sec", lambda _p: 10.0)
    monkeypatch.setattr(stage2, "extract_keyframes", lambda *_a, **_k: [])
    monkeypatch.setattr(
        stage2,
        "extract_working_scope",
        lambda _t: ClueExtractionResult(working_scope=None),
    )

    class _Reasoning:
        event_type = "reasoning"
        thought = "图中直接可见竹林与湿热植被，初始范围偏向华南或西南。"
        tool = None
        params = {}
        observation = None

    class _Step:
        event_type = "tool_call"
        thought = (
            "仅凭植被仍无法区分华南与西南，缺少外部分布资料，"
            "因此检索该植被组合的典型分布区。"
        )
        tool = "web_search"
        params = {"query": "竹林 湿热植被 典型分布"}
        observation = {"regions": ["华南", "西南"]}

    class _FinalStep:
        event_type = "final"
        thought = "植被与气候证据已经收敛，因此提交最终地点。"
        tool = "final_answer"
        params = {"location": "广东省广州市"}
        observation = None

    class _Result:
        steps = [_Reasoning(), _Step(), _FinalStep()]
        notes = None

    captured: dict[str, str] = {}

    def _fake_call(prompt: str, *_a, **_k):  # type: ignore[no-untyped-def]
        captured["prompt"] = prompt
        return _Result()

    monkeypatch.setattr(stage2, "call_structured", _fake_call)

    transcript = [
        TranscriptSegment(start=0, end=5, text="这里有竹子"),
        TranscriptSegment(start=5, end=10, text="可能在南方"),
    ]
    traj = stage2.run_stage2(str(video), transcript)
    assert isinstance(traj, FreeFormTrajectory)
    assert traj.source_video == "vid"
    assert len(traj.steps) == 3
    assert traj.steps[0].event_type == "reasoning"
    assert traj.steps[0].tool is None
    assert traj.steps[1].event_type == "tool_call"
    assert traj.steps[1].tool == "web_search"
    assert traj.steps[-1].tool == "final_answer"
    assert traj.steps[-1].params == {"location": "广东省广州市"}
    assert "字幕" not in traj.steps[0].thought
    assert traj.notes is None
    assert traj.working_scope is None

    prompt = captured["prompt"]
    assert "禁止写入产物" in prompt
    assert "reasoning、tool_call、final" in prompt
    assert "Thought→Thought→Action" in prompt
    assert "形成目标签名或时间一致性判断" in prompt
    assert "不得发明 inspect/filter/build/assess 类伪工具" in prompt
    assert "讲解内容参考" in prompt
    assert "无外部工作范围" in prompt
    assert "河南许昌附近" not in prompt
    assert "notes 可简述删除了哪些无用部分" not in prompt
    assert "字幕（带时间戳）" not in prompt
    assert "不得自行补写材料中没有的坐标" in prompt
    assert '"event_type":"final"' in prompt
    assert '"tool":"final_answer"' in prompt
    assert '"params":{"location":"最终地点"}' in prompt
    assert "求助者" in prompt
    assert "待定位图" in prompt

    path = tmp_path / "intermediate" / "vid" / "stage2_freeform_tao.json"
    assert path.is_file()
    loaded = FreeFormTrajectory.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded.notes is None
    assert loaded.working_scope is None


def test_run_stage2_injects_working_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "scope_vid.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    monkeypatch.setattr(stage2, "video_duration_sec", lambda _p: 10.0)
    monkeypatch.setattr(stage2, "extract_keyframes", lambda *_a, **_k: [])
    scope = WorkingScope(region="河南许昌附近", bound_kind=BoundKind.near)
    monkeypatch.setattr(
        stage2,
        "extract_working_scope",
        lambda _t: ClueExtractionResult(
            working_scope=scope,
            candidate_hypotheses=[],
        ),
    )

    class _Step:
        thought = "已知工作范围在河南许昌附近；缺口是画面地标，因此检索沿岸公园。"
        tool = "search_parks"
        params = {"q": "park"}
        observation = {"hits": []}

    class _FinalStep:
        thought = "已完成范围内地标核验，因此提交最终地点。"
        tool = "final_answer"
        params = {"location": "河南省许昌市某公园"}
        observation = None

    class _Result:
        steps = [_Step(), _FinalStep()]
        notes = None

    captured: dict[str, str] = {}

    def _fake_call(prompt: str, *_a, **_k):  # type: ignore[no-untyped-def]
        captured["prompt"] = prompt
        return _Result()

    monkeypatch.setattr(stage2, "call_structured", _fake_call)

    traj = stage2.run_stage2(
        str(video),
        [TranscriptSegment(start=0, end=1, text="拍摄地在河南许昌附近")],
    )
    prompt = captured["prompt"]
    assert "Agent 已知工作范围" in prompt
    assert "河南许昌附近" in prompt
    assert "无外部工作范围" not in prompt
    assert "博主猜" not in prompt
    assert traj.working_scope is not None
    assert traj.working_scope.region == "河南许昌附近"

    path = tmp_path / "intermediate" / "scope_vid" / "stage2_freeform_tao.json"
    loaded = FreeFormTrajectory.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded.working_scope is not None
    assert loaded.working_scope.region == "河南许昌附近"


@pytest.mark.parametrize(
    ("tool", "params", "observation"),
    [
        ("finalize_location", {"location": "甲地"}, None),
        ("final_answer", {"result": "甲地"}, None),
        ("final_answer", {"site": "甲地"}, None),
        ("final_answer", {"location": ""}, None),
        ("final_answer", {"location": "甲地", "confidence": "高"}, None),
        ("final_answer", {"location": "甲地"}, {"result": "甲地"}),
    ],
)
def test_llm_result_rejects_noncanonical_final_answer(
    tool: str,
    params: dict,
    observation: dict | None,
) -> None:
    with pytest.raises(ValueError):
        stage2._LLMFreeFormResult.model_validate(
            {
                "steps": [
                    {
                        "thought": "提交答案",
                        "tool": tool,
                        "params": params,
                        "observation": observation,
                    }
                ]
            }
        )


def test_llm_result_accepts_single_or_multiple_locations() -> None:
    for location in ["甲地", ["甲地", "乙地"]]:
        result = stage2._LLMFreeFormResult.model_validate(
            {
                "steps": [
                    {
                        "thought": "提交答案",
                        "tool": "final_answer",
                        "params": {"location": location},
                        "observation": None,
                    }
                ]
            }
        )
        assert result.steps[-1].params["location"] == location


def test_reasoning_chain_and_real_tool_call_contract() -> None:
    result = stage2._LLMFreeFormResult.model_validate(
        {
            "steps": [
                {
                    "event_type": "reasoning",
                    "thought": "先根据画面方向排除北向候选。",
                    "tool": None,
                    "params": {},
                    "observation": None,
                },
                {
                    "event_type": "reasoning",
                    "thought": "剩余两地只能通过外部日落时间继续区分。",
                    "tool": None,
                    "params": {},
                    "observation": None,
                },
                {
                    "event_type": "tool_call",
                    "thought": "需要新证据，因此查询日落时间。",
                    "tool": "sunset_time_lookup",
                    "params": {"date": "2023-11-13"},
                    "observation": {"result": "甲地日落更晚"},
                },
                {
                    "event_type": "final",
                    "thought": "证据闭合。",
                    "tool": "final_answer",
                    "params": {"location": "甲地"},
                    "observation": None,
                },
            ]
        }
    )
    assert [step.event_type for step in result.steps[:3]] == [
        "reasoning",
        "reasoning",
        "tool_call",
    ]


def test_tool_call_requires_real_observation() -> None:
    with pytest.raises(ValueError, match="必须提供 observation"):
        stage2._LLMFreeFormResult.model_validate(
            {
                "steps": [
                    {
                        "event_type": "tool_call",
                        "thought": "查询外部地图。",
                        "tool": "map_query",
                        "params": {"q": "甲地"},
                        "observation": None,
                    },
                    {
                        "event_type": "final",
                        "thought": "提交。",
                        "tool": "final_answer",
                        "params": {"location": "甲地"},
                        "observation": None,
                    },
                ]
            }
        )


def test_legacy_freeform_step_infers_event_type() -> None:
    legacy = FreeFormTrajectory.model_validate(
        {
            "source_video": "legacy",
            "steps": [
                {
                    "thought": "查询地图",
                    "tool": "map_lookup",
                    "params": {},
                    "observation": {"hits": []},
                },
                {
                    "thought": "提交",
                    "tool": "final_answer",
                    "params": {"location": "甲地"},
                    "observation": None,
                },
            ],
        }
    )
    assert legacy.steps[0].event_type == "tool_call"
    assert legacy.steps[1].event_type == "final"


def test_stage2_rewrites_meta_leak_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "leak.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    monkeypatch.setattr(stage2, "video_duration_sec", lambda _p: 10.0)
    monkeypatch.setattr(stage2, "extract_keyframes", lambda *_a, **_k: [])
    monkeypatch.setattr(
        stage2,
        "extract_working_scope",
        lambda _t: ClueExtractionResult(working_scope=None),
    )

    class _LeakStep:
        thought = "对比求助者发来的两张图"
        tool = "compare"
        params = {"image_1": "第一张求助图"}
        observation = {"ok": True}

    class _FinalLeak:
        thought = "提交"
        tool = "final_answer"
        params = {"location": "甲地"}
        observation = None

    class _CleanStep:
        thought = "对比图1与图2的路旁特征"
        tool = "compare"
        params = {"image_1": "图1"}
        observation = {"ok": True}

    class _FinalClean:
        thought = "提交"
        tool = "final_answer"
        params = {"location": "甲地"}
        observation = None

    calls = {"n": 0}

    def _fake(prompt: str, schema: Any, **_k: Any):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:

            class _R1:
                steps = [_LeakStep(), _FinalLeak()]
                notes = None

            return _R1()

        class _R2:
            steps = [_CleanStep(), _FinalClean()]
            notes = None

        return _R2()

    monkeypatch.setattr(stage2, "call_structured", _fake)
    traj = stage2.run_stage2(
        str(video),
        [TranscriptSegment(start=0, end=1, text="两张图")],
    )
    assert calls["n"] == 2
    assert "求助" not in traj.steps[0].thought
    assert "图1" in traj.steps[0].thought
