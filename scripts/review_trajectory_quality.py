"""批量调用独立审核 Agent，生成语义质量 sidecar；支持断点续跑。"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.quality.reviewer import review_trajectory_semantics
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment


def _load_transcript(path: Path) -> list[TranscriptSegment]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("segments", []) if isinstance(raw, dict) else raw
    return [TranscriptSegment.model_validate(item) for item in items]


def _review_one(
    trajectory_path: Path,
    transcript_root: Path,
    *,
    force: bool,
) -> dict:
    out = trajectory_path.with_name("stage3_semantic_quality_review.json")
    if out.is_file() and not force:
        return {"trajectory": trajectory_path.parent.name, "ok": True, "resumed": True}
    freeform = FreeFormTrajectory.model_validate_json(
        trajectory_path.with_name("stage2_freeform_tao.json").read_text(encoding="utf-8")
    )
    trajectory = Trajectory.model_validate_json(
        trajectory_path.read_text(encoding="utf-8")
    )
    transcript_path = transcript_root / freeform.source_video / "stage1_transcript.json"
    transcript = _load_transcript(transcript_path)
    review = review_trajectory_semantics(
        freeform,
        trajectory,
        transcript=transcript,
        image_paths=trajectory.image_paths,
    )
    out.write_text(review.model_dump_json(indent=2), encoding="utf-8")
    return {
        "trajectory": trajectory.id,
        "source_video": freeform.source_video,
        "ok": True,
        "resumed": False,
        "review": review.model_dump(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--transcript-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("data/runs/semantic_quality_review_batch.json"))
    args = parser.parse_args()

    paths = sorted(args.root.rglob("stage3_trajectory.json"))
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {
            pool.submit(
                _review_one,
                path,
                args.transcript_root,
                force=args.force,
            ): path
            for path in paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                item = future.result()
            except Exception as exc:  # noqa: BLE001
                item = {
                    "trajectory": path.parent.name,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            results.append(item)
            print(json.dumps({k: v for k, v in item.items() if k not in {"review", "traceback"}}, ensure_ascii=False), flush=True)

    report = {
        "root": str(args.root.resolve()),
        "total": len(paths),
        "ok": sum(item.get("ok", False) for item in results),
        "failed": sum(not item.get("ok", False) for item in results),
        "results": sorted(results, key=lambda item: str(item.get("trajectory"))),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
