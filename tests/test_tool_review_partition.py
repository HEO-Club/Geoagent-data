from __future__ import annotations

import json
from pathlib import Path

from scripts import partition_tool_review_pool


def test_partition_separates_canonical_and_temporary(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "run"
    canonical = root / "canonical_task"
    temporary = root / "temporary_task"
    canonical.mkdir(parents=True)
    temporary.mkdir(parents=True)
    (canonical / "stage3_tool_mapping.json").write_text(
        json.dumps(
            {
                "tool_routing": {
                    "pool": "canonical_only",
                    "requires_tool_review": False,
                },
                "temporary_tools": [],
                "temporary_operations": [],
            }
        ),
        encoding="utf-8",
    )
    (temporary / "stage3_tool_mapping.json").write_text(
        json.dumps(
            {
                "tool_routing": {
                    "pool": "temporary_review",
                    "requires_tool_review": True,
                },
                "temporary_tools": [{"raw_tool": "unknown_lookup"}],
                "temporary_operations": [],
                "tool_review_candidates": [{"raw_tool": "unknown_lookup"}],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "partition.json"
    monkeypatch.setattr(
        "sys.argv",
        ["partition", "--root", str(root), "--out", str(out)],
    )
    partition_tool_review_pool.main()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["counts"] == {
        "total": 2,
        "canonical_only": 1,
        "temporary_review": 1,
    }
    assert payload["temporary_review"][0]["candidates"][0]["raw_tool"] == "unknown_lookup"
