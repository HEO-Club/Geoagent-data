"""Rebuild Stage 4 human-review JSON/Markdown without calling a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.confidence import ConfidenceReport
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment
from pipeline.stage4_confidence.review_cards import build_review_packet, write_review_packet
from pipeline.stage4_confidence.run import load_parameter_audits


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _transcript(path: Path) -> list[TranscriptSegment]:
    value = _json(path)
    items = value.get("segments", []) if isinstance(value, dict) else value
    return [TranscriptSegment.model_validate(item) for item in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    completed = 0
    errors: list[dict[str, str]] = []
    for report_path in sorted(args.root.rglob("stage4_confidence.json")):
        task_dir = report_path.parent
        try:
            report = ConfidenceReport.model_validate_json(report_path.read_text(encoding="utf-8"))
            task = GeoTaskSpec.model_validate_json((task_dir / "stage15_task.json").read_text(encoding="utf-8"))
            trajectory = Trajectory.model_validate_json((task_dir / "stage3_trajectory.json").read_text(encoding="utf-8"))
            transcript = _transcript(task_dir / "transcript_slice.json")
            mapping_value = _json(task_dir / "stage3_tool_mapping.json")
            mapping = mapping_value if isinstance(mapping_value, dict) else {}
            observation_path = task_dir / "stage2_observation_audit.json"
            observation_value = _json(observation_path) if observation_path.is_file() else None
            observation = observation_value if isinstance(observation_value, dict) else None
            audits, _ = load_parameter_audits(task_dir / "stage3_parameter_audit.json")
            packet = build_review_packet(
                report=report,
                task=task,
                trajectory=trajectory,
                transcript=transcript,
                parameter_audits=audits,
                tool_mapping=mapping,
                observation_audit=observation,
            )
            write_review_packet(packet, report_path)
            completed += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"task_dir": str(task_dir), "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps({"completed": completed, "errors": errors}, ensure_ascii=False))


if __name__ == "__main__":
    main()
