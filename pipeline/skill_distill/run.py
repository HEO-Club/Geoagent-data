"""离线技能蒸馏：拆句、对冻结库提补丁、再合并。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from pipeline.llm import call_structured
from pipeline.skill_distill.catalog import load_operation_index, render_operation_index
from pipeline.skill_distill.load import AdmittedTrajectory, admit_trajectories, render_trajectory
from pipeline.skill_distill.merge import apply_patches
from pipeline.skill_distill.prompts import analyst_prompt, atomize_prompt, frozen_bank_json
from pipeline.skill_distill.schemas import (
    AnalystResult,
    AtomizeResult,
    DraftSentence,
    MergeReport,
    ReviewItem,
    SkillBank,
    SkillPatch,
)

T = TypeVar("T", bound=BaseModel)
StructuredCaller = Callable[[str, type[T]], T]


def run_skill_distill(
    input_dir: Path,
    output_dir: Path,
    *,
    catalog_path: Path,
    min_quality: float = 0.85,
    allow_unscored: bool = False,
    bank_path: Path | None = None,
    caller: StructuredCaller | None = None,
) -> MergeReport:
    """从轨迹目录写出技能库。

    ``caller`` 缺省时走 ``pipeline.llm.call_structured``。测试注入替身，
    因此本函数在测试中不会访问网络。已有技能库可通过 ``bank_path`` 加深。
    """
    admitted, skipped = admit_trajectories(
        input_dir,
        min_quality=min_quality,
        allow_unscored=allow_unscored,
    )
    operations = load_operation_index(catalog_path)
    bank = _load_bank(bank_path)
    complete = caller or _default_caller
    patches: list[tuple[str, SkillPatch]] = []
    reviews: list[ReviewItem] = []
    index_text = render_operation_index(operations)
    for item in admitted:
        try:
            drafts = _atomize(item, index_text, complete)
        except ValidationError as exc:
            reviews.append(
                ReviewItem(
                    trajectory_id=item.trajectory_id,
                    reason=f"拆句结果不符合句子合同：{exc}",
                )
            )
            continue
        snapshot = bank.model_copy(deep=True)
        try:
            result = complete(
                analyst_prompt(
                    trajectory_text=render_trajectory(item.trajectory),
                    drafts_json=json.dumps(
                        [sentence.model_dump() for sentence in drafts],
                        ensure_ascii=False,
                    ),
                    frozen_bank_json=frozen_bank_json(snapshot),
                ),
                AnalystResult,
            )
        except ValidationError as exc:
            reviews.append(
                ReviewItem(
                    trajectory_id=item.trajectory_id,
                    reason=f"补丁不符合合同：{exc}",
                )
            )
            continue
        if bank.model_dump() != snapshot.model_dump():
            raise RuntimeError("分析员修改了冻结技能库")
        for patch in result.patches:
            patches.append((item.trajectory_id, patch))
    merged = apply_patches(bank, patches, operations=operations)
    merged.review_queue = [*reviews, *merged.review_queue]
    report = MergeReport(
        input_trajectories=len(admitted) + len(skipped),
        admitted=len(admitted),
        skipped=skipped,
        sentences=len(merged.sentences),
        skills=sum(1 for sentence in merged.sentences if sentence.status == "skill"),
        examples=sum(1 for sentence in merged.sentences if sentence.status == "example"),
        review_queue=len(merged.review_queue),
    )
    _write(output_dir, merged, patches, report)
    return report


def _atomize(
    item: AdmittedTrajectory,
    operation_index: str,
    complete: StructuredCaller,
) -> list[DraftSentence]:
    result = complete(
        atomize_prompt(
            trajectory_text=render_trajectory(item.trajectory),
            operation_index=operation_index,
        ),
        AtomizeResult,
    )
    return list(result.sentences)


def _default_caller(prompt: str, response_model: type[T]) -> T:
    return call_structured(prompt, response_model)


def _load_bank(path: Path | None) -> SkillBank:
    if path is None:
        return SkillBank()
    return SkillBank.model_validate_json(path.read_text(encoding="utf-8"))


def _write(
    output_dir: Path,
    bank: SkillBank,
    patches: list[tuple[str, SkillPatch]],
    report: MergeReport,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "skill_bank.json").write_text(
        bank.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output_dir / "merge_report.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    lines = [
        json.dumps(
            {"trajectory_id": trajectory_id, "patch": patch.model_dump()},
            ensure_ascii=False,
        )
        for trajectory_id, patch in patches
    ]
    (output_dir / "patches.jsonl").write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
