"""阶段4 CLI：多产物置信度评分并回写 JSONL.quality_score。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pipeline.config import get_settings
from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.dataset import DatasetEntry
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage2_freeform_tao.run import load_freeform
from pipeline.stage4_confidence.run import run_stage4


def _load_transcript(path: Path) -> list[TranscriptSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "segments" in data:
        return [TranscriptSegment.model_validate(x) for x in data["segments"]]
    if isinstance(data, list):
        return [TranscriptSegment.model_validate(x) for x in data]
    raise ValueError(f"无法解析字幕: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="阶段4：样本置信度评分（回写 quality_score，不拦入库）"
    )
    parser.add_argument("--task-json", required=True, help="GeoTaskSpec JSON")
    parser.add_argument("--transcript", required=True, help="字幕切片 JSON")
    parser.add_argument("--freeform", required=True, help="stage2_freeform_tao.json")
    parser.add_argument("--trajectory", required=True, help="stage3_trajectory.json")
    parser.add_argument("--entry-jsonl", required=True, help="shards/*.jsonl")
    parser.add_argument("--tool-mapping", default=None, help="stage3_tool_mapping.json")
    parser.add_argument("--out-report", default=None, help="stage4_confidence.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    get_settings()

    task = GeoTaskSpec.model_validate_json(
        Path(args.task_json).read_text(encoding="utf-8")
    )
    transcript = _load_transcript(Path(args.transcript))
    freeform = load_freeform(args.freeform)
    trajectory = Trajectory.model_validate_json(
        Path(args.trajectory).read_text(encoding="utf-8")
    )
    entry = DatasetEntry.model_validate_json(
        Path(args.entry_jsonl).read_text(encoding="utf-8").splitlines()[0]
    )
    report = run_stage4(
        task=task,
        transcript=transcript,
        freeform=freeform,
        trajectory=trajectory,
        entry=entry,
        tool_mapping_path=args.tool_mapping,
        out_report_path=args.out_report,
        out_jsonl_path=args.entry_jsonl,
    )
    print(
        json.dumps(
            {
                "task_id": report.task_id,
                "quality_score": report.quality_score,
                "base_score": report.base_score,
                "review_priority": report.review_priority,
                "hard_gates": [g.code for g in report.hard_gates],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
