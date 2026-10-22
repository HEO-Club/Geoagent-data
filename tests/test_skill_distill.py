"""技能蒸馏：过滤、合并与离线运行。不调用真实 API。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.schemas.trajectory import Action, Trajectory, TrajectoryStep
from pipeline.skill_distill.catalog import load_operation_index
from pipeline.skill_distill.load import admit_trajectories
from pipeline.skill_distill.merge import apply_patches
from pipeline.skill_distill.run import run_skill_distill
from pipeline.skill_distill.schemas import (
    AnalystResult,
    AtomizeResult,
    DraftSentence,
    SkillBank,
    SkillPatch,
    SkillSentence,
)


def _trajectory(trajectory_id: str) -> Trajectory:
    return Trajectory(
        id=trajectory_id,
        system_prompt="geolocate",
        user_query="定位画面",
        steps=[
            TrajectoryStep(
                event_type="reasoning",
                thought="画面是高处看宽河",
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="沿工作范围查历史卫星",
                action=Action(
                    tool="satellite_imagery_query",
                    params={"operation": "change_time"},
                ),
                observation={"result": "高地接平原"},
            ),
            TrajectoryStep(
                event_type="final",
                thought="给出地点",
                action=Action(tool="final_answer", params={"location": "园子"}),
            ),
        ],
    )


def _write_task(
    root: Path,
    trajectory_id: str,
    *,
    score: float | None,
    decision: str | None,
) -> None:
    task = root / "video" / "tasks" / trajectory_id
    task.mkdir(parents=True)
    (task / "stage3_trajectory.json").write_text(
        _trajectory(trajectory_id).model_dump_json(),
        encoding="utf-8",
    )
    if score is None and decision is None:
        return
    payload: dict[str, Any] = {"task_id": trajectory_id}
    if score is not None:
        payload["quality_score"] = score
    if decision is not None:
        payload["decision"] = decision
    (task / "stage4_confidence.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _catalog(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "trees": [
                    {
                        "canonical": {
                            "name": "satellite_imagery_query",
                            "operations": [{"name": "change_time"}],
                        }
                    },
                    {
                        "canonical": {
                            "name": "final_answer",
                            "operations": [{"name": "submit"}],
                        }
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _sentence(*, tool: str = "satellite_imagery_query") -> DraftSentence:
    if tool == "satellite_imagery_query":
        return DraftSentence(
            kind="tool",
            text="沿已收窄范围查询历史卫星，只用回执中的地形结论",
            applicability="工作范围已收到区域，且画面有宽河或高地",
            tool="satellite_imagery_query",
            operation="change_time",
        )
    return DraftSentence(
        kind="planning",
        text="高处俯视宽河时先核对高地是否贴着平原",
        applicability="画面同时有高处机位和宽河",
        level="region",
    )


def test_admit_uses_score_and_reject(tmp_path: Path) -> None:
    _write_task(tmp_path, "keep", score=0.9, decision="accept")
    _write_task(tmp_path, "low", score=0.5, decision="needs_review")
    _write_task(tmp_path, "rejected", score=0.95, decision="reject")
    _write_task(tmp_path, "unscored", score=None, decision=None)
    (tmp_path / "groundtruth.json").write_text('{"location":"secret"}', encoding="utf-8")

    admitted, skipped = admit_trajectories(tmp_path, min_quality=0.85)
    assert [item.trajectory_id for item in admitted] == ["keep"]
    reasons = {item.trajectory_id: item.reason for item in skipped}
    assert "低于" in reasons["low"]
    assert reasons["rejected"] == "stage4 decision=reject"
    assert reasons["unscored"] == "缺少 stage4_confidence.json"

    admitted_unscored, _ = admit_trajectories(tmp_path, allow_unscored=True)
    assert {item.trajectory_id for item in admitted_unscored} == {"keep", "unscored"}


def test_single_support_stays_example_and_unknown_tool_is_reviewed() -> None:
    bank = SkillBank()
    good = SkillPatch(op="add", sentence=_sentence())
    foreign = SkillPatch(
        op="add",
        sentence=DraftSentence(
            kind="tool",
            text="用未登记工具扫一遍",
            applicability="任何画面",
            tool="made_up_tool",
            operation="execute",
        ),
    )
    operations = {("satellite_imagery_query", "change_time")}
    merged = apply_patches(
        bank,
        [("t1", good), ("t1", foreign)],
        operations=operations,
    )
    assert bank.sentences == []
    assert len(merged.sentences) == 1
    assert merged.sentences[0].status == "example"
    assert merged.sentences[0].support_count == 1
    assert merged.review_queue[0].reason.startswith("工具句不在正式目录中")


def test_two_trajectories_promote_skill_and_conflict_stays_split() -> None:
    sentence = _sentence(tool="planning")
    other = DraftSentence(
        kind="planning",
        text=sentence.text,
        applicability="工作范围位于中原",
        level="region",
    )
    merged = apply_patches(
        SkillBank(),
        [
            ("t1", SkillPatch(op="add", sentence=sentence)),
            ("t2", SkillPatch(op="add", sentence=sentence)),
            ("t3", SkillPatch(op="add", sentence=other)),
        ],
        operations=set(),
    )
    assert len(merged.sentences) == 2
    by_condition = {item.applicability: item for item in merged.sentences}
    assert by_condition[sentence.applicability].status == "skill"
    assert by_condition[sentence.applicability].source_trajectory_ids == ["t1", "t2"]
    assert by_condition[other.applicability].status == "example"


def test_narrow_does_not_overwrite_other_support() -> None:
    bank = SkillBank(
        sentences=[
            SkillSentence(
                id="sent-0001",
                kind="planning",
                text="看到宽河就沿大河核对",
                applicability="任何宽河",
                level="region",
                source_trajectory_ids=["t1"],
                support_count=1,
                status="example",
            )
        ]
    )
    merged = apply_patches(
        bank,
        [
            (
                "t2",
                SkillPatch(
                    op="narrow",
                    target_id="sent-0001",
                    applicability="工作范围位于中原",
                ),
            )
        ],
        operations=set(),
    )
    assert bank.sentences[0].applicability == "任何宽河"
    assert merged.sentences[0].applicability == "任何宽河"
    assert merged.sentences[1].applicability == "工作范围位于中原"
    assert merged.sentences[1].source_trajectory_ids == ["t2"]


def test_run_writes_bank_without_real_api(tmp_path: Path) -> None:
    _write_task(tmp_path, "t1", score=0.91, decision="accept")
    _write_task(tmp_path, "t2", score=0.9, decision=None)
    catalog = tmp_path / "catalog.json"
    _catalog(catalog)
    sentence = _sentence()

    def fake_call(prompt: str, response_model: type[Any]) -> Any:
        if response_model is AtomizeResult:
            return AtomizeResult(sentences=[sentence])
        assert "冻结技能库" in prompt
        return AnalystResult(patches=[SkillPatch(op="add", sentence=sentence)])

    output = tmp_path / "out"
    report = run_skill_distill(
        tmp_path,
        output,
        catalog_path=catalog,
        caller=fake_call,
    )
    assert report.admitted == 2
    assert report.skills == 1
    bank = SkillBank.model_validate_json(
        (output / "skill_bank.json").read_text(encoding="utf-8")
    )
    assert bank.sentences[0].status == "skill"
    assert (output / "patches.jsonl").read_text(encoding="utf-8").count("t1") == 1
    assert load_operation_index(catalog) == {
        ("satellite_imagery_query", "change_time")
    }
