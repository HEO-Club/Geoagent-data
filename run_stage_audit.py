"""阶段1.5 CLI：字幕 → 审核切分。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pipeline.config import get_settings
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage_audit_split.run import run_audit_split


def _load_transcript(path: str) -> list[TranscriptSegment]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "segments" in raw:
        items = raw["segments"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"无法解析字幕: {path}")
    return [TranscriptSegment.model_validate(x) for x in items]


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段1.5：审核切分定位任务")
    parser.add_argument("--video", required=True)
    parser.add_argument("--transcript", required=True, help="阶段1 字幕 JSON")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    get_settings()
    result = run_audit_split(
        args.video,
        _load_transcript(args.transcript),
        out_path=args.out,
    )
    print(
        json.dumps(
            {
                "video_id": result.video_id,
                "decision": result.decision.value,
                "reason": result.reason,
                "tasks": len(result.tasks),
                "task_ids": [t.task_id for t in result.tasks],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
