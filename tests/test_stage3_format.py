"""阶段3：tool 树与 JSONL。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.schemas.tools import ToolDefinition, ToolForest, ToolTree
from pipeline.stage3_normalize_format import format_jsonl, map_tools, trees


def test_exact_match_and_variant(tmp_path: Path) -> None:
    path = tmp_path / "tool_trees.json"
    forest = ToolForest(
        trees=[
            ToolTree(
                canonical=ToolDefinition(
                    name="zoom_inspect",
                    description="zoom",
                    is_terminal=False,
                ),
                variants=[],
            )
        ]
    )
    trees.save_forest(forest, path)

    freeform = FreeFormTrajectory(
        source_video="v1",
        steps=[
            FreeFormStep(
                thought="zoom in",
                tool="crop_and_look",
                params={"bbox": [0, 0, 10, 10]},
                observation={"detail": "roof"},
            ),
            FreeFormStep(
                thought="done",
                tool="submit_answer",
                params={"lat": 1.0, "lng": 2.0},
                observation=None,
            ),
        ],
    )

    def matcher(name: str, forest: ToolForest) -> str | None:
        if name == "crop_and_look":
            return "zoom_inspect"
        return None

    out = map_tools.ensure_tool_trees(freeform, path, matcher=matcher)
    names = {t.canonical.name for t in out.trees}
    assert "zoom_inspect" in names
    assert "submit_answer" in names
    zoom = next(t for t in out.trees if t.canonical.name == "zoom_inspect")
    assert "crop_and_look" in zoom.variants


def test_remap_and_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("TOOL_TREES_PATH", str(tmp_path / "tool_trees.json"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    freeform = FreeFormTrajectory(
        source_video="clip",
        steps=[
            FreeFormStep(
                thought="search web",
                tool="web_lookup",
                params={"q": "tower"},
                observation={"hits": ["a"]},
            )
        ],
    )
    entry = format_jsonl.run_stage3(
        freeform,
        trees_path=tmp_path / "tool_trees.json",
        image_path="scene.jpg",
        matcher=lambda _n, _f: None,
    )
    assert entry.source_video == "clip"
    roles = [m.role for m in entry.messages]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert roles[2] == "assistant"
    assert roles[3] == "tool"
    assert "Thought:" in entry.messages[2].content
    blob = json.dumps([m.content for m in entry.messages], ensure_ascii=False)
    assert "web_lookup" in blob
    shard = tmp_path / "output" / "shards" / "clip.jsonl"
    assert shard.is_file()
