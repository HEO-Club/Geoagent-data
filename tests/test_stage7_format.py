"""stage7：DatasetEntry 转换、分片写入与单 writer 合并测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.schemas import (
    Action,
    AgentRole,
    LocationHypothesis,
    ObservationSource,
    SubmitAnswerResult,
    Trajectory,
    TrajectoryStep,
    VerificationResult,
)
from pipeline.stage7_format import (
    format_all_and_save,
    merge_jsonl_shards,
    to_dataset_entry,
    trajectory_to_messages,
)


def _hyp() -> LocationHypothesis:
    return LocationHypothesis(
        possible_countries=["France"],
        possible_regions=["Île-de-France"],
        reasoning_summary="Tower silhouette.",
        confidence=0.8,
        key_clues_remaining=["plaza"],
    )


def _submit() -> SubmitAnswerResult:
    return SubmitAnswerResult(
        latitude=48.8584,
        longitude=2.2945,
        location_name="Eiffel Tower",
        confidence=0.9,
        reasoning="Matched landmark.",
    )


def _coarse_traj() -> Trajectory:
    return Trajectory(
        id="traj-coarse-1",
        agent_role=AgentRole.COARSE,
        system_prompt="sys-coarse",
        user_query="粗定位",
        image_path="frame.jpg",
        steps=[
            TrajectoryStep(
                thought="看铁塔结构。",
                action=Action(tool="ocr", params={}),
                observation={
                    "status": "success",
                    "error_message": None,
                    "texts": ["Tour"],
                },
                observation_source=ObservationSource.LLM_SYNTHESIZED,
            )
        ],
        coarse_output=_hyp(),
    )


def _fine_traj() -> Trajectory:
    submit = _submit()
    return Trajectory(
        id="traj-fine-1",
        agent_role=AgentRole.FINE,
        system_prompt="sys-fine",
        user_query="精定位",
        image_path="frame.jpg",
        steps=[
            TrajectoryStep(
                thought="查地图。",
                action=Action(
                    tool="map_query",
                    params={"query": "Eiffel Tower"},
                ),
                observation={
                    "status": "success",
                    "error_message": None,
                    "resolved_latlng": [48.8584, 2.2945],
                    "formatted_address": "Paris",
                    "place_type": "tourist_attraction",
                    "viewport": None,
                    "place_id": None,
                },
                observation_source=ObservationSource.LLM_SYNTHESIZED,
            ),
            TrajectoryStep(
                thought="提交答案。",
                action=Action(tool="submit_answer", params=submit.model_dump()),
                observation=None,
                observation_source=None,
            ),
        ],
        coarse_handoff=_hyp(),
        fine_output=submit,
    )


def _verifier_traj() -> Trajectory:
    return Trajectory(
        id="traj-ver-1",
        agent_role=AgentRole.VERIFIER,
        system_prompt="sys-ver",
        user_query="验证",
        image_path="frame.jpg",
        steps=[
            TrajectoryStep(
                thought="核对坐标。",
                action=Action(
                    tool="map_query",
                    params={"latlng": [48.8584, 2.2945]},
                ),
                observation={
                    "status": "success",
                    "error_message": None,
                    "resolved_latlng": [48.8584, 2.2945],
                    "formatted_address": "Paris",
                    "place_type": "tourist_attraction",
                    "viewport": None,
                    "place_id": None,
                },
                observation_source=ObservationSource.LLM_SYNTHESIZED,
            )
        ],
        fine_handoff=_submit(),
        coarse_handoff=_hyp(),
        verifier_output=VerificationResult(
            verdict="pass",
            failed_checks=[],
            suggested_recheck="",
            return_to_agent=None,
        ),
    )


def test_trajectory_to_messages_roles_and_terminal() -> None:
    traj = _fine_traj()
    msgs = trajectory_to_messages(traj)
    roles = [m.role for m in msgs]
    assert roles[0] == "system"
    assert roles[1] == "user"
    # step0: assistant + tool; step1 terminal: assistant only
    assert roles[2] == "assistant"
    assert roles[3] == "tool"
    assert roles[4] == "assistant"
    assert "resolved_latlng" in msgs[3].content
    assert "submit_answer" in msgs[4].content
    # 无额外 tool message 在 terminal 后
    assert roles.count("tool") == 1


def test_to_dataset_entry_fields() -> None:
    traj = _coarse_traj()
    entry = to_dataset_entry(
        traj,
        {
            "source_video": "vid001",
            "groundtruth": (48.8584, 2.2945),
            "quality_score": 0.85,
            "verified": True,
            "distance_error_km": None,
        },
    )
    assert entry.id == "traj-coarse-1"
    assert entry.agent_role == AgentRole.COARSE
    assert entry.source_video == "vid001"
    assert entry.verified is True
    assert not hasattr(entry, "contains_draft_tools")
    assert not hasattr(entry, "draft_tool_names")
    assert entry.messages[0].role == "system"
    assert any("LocationHypothesis" in m.content for m in entry.messages)


def test_to_dataset_entry_missing_meta() -> None:
    with pytest.raises(KeyError, match="缺少字段"):
        to_dataset_entry(_coarse_traj(), {"source_video": "x"})


def test_format_all_and_save_shards_only_verified(tmp_path: Path) -> None:
    coarse = _coarse_traj()
    fine = _fine_traj()
    ver = _verifier_traj()
    # fine 标记为未通过 → 不进分片
    meta = {
        "source_video": "vidA",
        "groundtruth": (48.8584, 2.2945),
        "reports": {
            coarse.id: {"quality_score": 0.9, "verified": True, "distance_error_km": None},
            fine.id: {"quality_score": 0.1, "verified": False, "distance_error_km": 100.0},
            ver.id: {"quality_score": 0.8, "verified": True, "distance_error_km": 0.5},
        },
    }
    out = tmp_path / "output"
    entries = format_all_and_save([coarse, fine, ver], meta, str(out), "vidA")
    assert len(entries) == 3
    assert sum(1 for e in entries if e.verified) == 2

    shard1 = (out / "shards" / "vidA_agent1.jsonl").read_text(encoding="utf-8").strip()
    shard2 = (out / "shards" / "vidA_agent2.jsonl").read_text(encoding="utf-8").strip()
    shard3 = (out / "shards" / "vidA_agent3.jsonl").read_text(encoding="utf-8").strip()
    assert shard1  # coarse verified
    assert not shard2  # fine rejected
    assert shard3  # verifier verified
    row = json.loads(shard1.splitlines()[0])
    assert row["agent_role"] == AgentRole.COARSE.value
    assert "resolved_latlng" not in shard1  # coarse 无 map_query


def test_merge_jsonl_shards_single_writer(tmp_path: Path) -> None:
    out = tmp_path / "output"
    # 每视频一次写齐分片，避免后写覆盖先写的角色分片
    format_all_and_save(
        [_coarse_traj(), _fine_traj()],
        {
            "source_video": "v1",
            "groundtruth": (48.0, 2.0),
            "reports": {
                "traj-coarse-1": {"quality_score": 0.9, "verified": True},
                "traj-fine-1": {"quality_score": 0.9, "verified": True},
            },
        },
        str(out),
        "v1",
    )
    t2 = _coarse_traj().model_copy(update={"id": "traj-coarse-2"})
    format_all_and_save(
        [t2],
        {
            "source_video": "v2",
            "groundtruth": (48.0, 2.0),
            "reports": {"traj-coarse-2": {"quality_score": 0.8, "verified": True}},
        },
        str(out),
        "v2",
    )

    counts = merge_jsonl_shards(out)
    assert counts["agent1_coarse.jsonl"] == 2
    assert counts["agent2_fine.jsonl"] == 1
    assert counts["agent3_verifier.jsonl"] == 0

    coarse_lines = (out / "agent1_coarse.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(coarse_lines) == 2
    ids = {json.loads(line)["id"] for line in coarse_lines}
    assert ids == {"traj-coarse-1", "traj-coarse-2"}


def test_agent1_revision_input_strips_fine_place_text() -> None:
    """Agent1 shard 不携带 FINE 地点文本的 revision_input。"""
    traj = _coarse_traj().model_copy(
        update={
            "is_revision": True,
            "revision_round": 1,
            "revision_source": __import__(
                "pipeline.schemas", fromlist=["RevisionSource"]
            ).RevisionSource.SYSTEM_FEEDBACK,
            "parent_trajectory_id": "parent-1",
            "revision_input": VerificationResult(
                verdict="fail",
                failed_checks=["郑州黄河文化公园 visual mismatch"],
                suggested_recheck="核对郑州黄河文化公园大门",
                return_to_agent=1,
            ),
        }
    )
    entry = to_dataset_entry(
        traj,
        {
            "source_video": "vid001",
            "groundtruth": (34.9, 113.5),
            "quality_score": 0.2,
            "verified": True,
        },
    )
    assert entry.revision_input is not None
    blob = " ".join(entry.revision_input.failed_checks)
    assert "郑州黄河文化公园" not in blob
    assert entry.revision_input.suggested_recheck == ""
    assert entry.revision_input.verdict == "fail"
    assert entry.revision_input.return_to_agent == 1
