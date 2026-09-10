"""run_tool_harness.py 命令行入口测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pipeline.schemas.trajectory import Action, Trajectory, TrajectoryStep
from run_tool_harness import main


def test_cli_runs_terminal_only_trajectory(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    trajectory = Trajectory(
        id="cli-demo",
        system_prompt="system",
        user_query="query",
        steps=[
            TrajectoryStep(event_type="reasoning", thought="已有证据足够"),
            TrajectoryStep(
                event_type="final",
                thought="提交",
                action=Action(tool="final_answer", params={"location": "郑州市"}),
            ),
        ],
    )
    path = tmp_path / "stage3_trajectory.json"
    path.write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")
    catalog = Path(__file__).resolve().parents[1] / "canonical_tool_catalog_v2.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tool_harness.py",
            "--trajectory",
            str(path),
            "--workspace",
            str(tmp_path / "runs"),
            "--catalog",
            str(catalog),
        ],
    )
    assert main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["status"] == "completed"
    assert output["final_location"] == "郑州市"
    assert Path(output["report"]).is_file()


def test_cli_provider_revision_invalidates_old_checkpoint(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    trajectory = Trajectory(
        id="cli-provider-revision",
        system_prompt="system",
        user_query="query",
        steps=[
            TrajectoryStep(
                event_type="final",
                thought="提交",
                action=Action(tool="final_answer", params={"location": "郑州市"}),
            )
        ],
    )
    path = tmp_path / "trajectory.json"
    path.write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")
    catalog = Path(__file__).resolve().parents[1] / "canonical_tool_catalog_v2.json"
    base_args = [
        "run_tool_harness.py",
        "--trajectory",
        str(path),
        "--workspace",
        str(tmp_path / "runs"),
        "--catalog",
        str(catalog),
        "--provider-revision",
    ]
    monkeypatch.setattr(sys, "argv", [*base_args, "v1"])
    assert main() == 0
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", [*base_args, "v2"])
    assert main() == 2
    output = json.loads(capsys.readouterr().out)
    assert output["error_code"] == "runtime_fingerprint_mismatch"
