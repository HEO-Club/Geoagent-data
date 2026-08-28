"""Stage 3 executor reclassification for existing tool_calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.config import clear_settings_cache
from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.schemas.tools import MatchDecision
from pipeline.stage3_normalize_format import map_tools, trees
from pipeline.stage3_normalize_format.format_jsonl import run_stage3
from pipeline.tool_catalog_v2 import build_tool_forest_v2


def test_web_search_measure_reclassifies_to_satellite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STAGE3_COMPILE_PARAMS", "false")
    monkeypatch.setenv("TOOL_CATALOG_PATH", "canonical_tool_catalog_v2.json")
    clear_settings_cache()
    catalog = build_tool_forest_v2()
    path = tmp_path / "tool_trees.json"
    trees.save_forest(catalog, path)

    freeform = FreeFormTrajectory(
        source_video="demo",
        steps=[
            FreeFormStep(
                event_type="tool_call",
                thought="把时间线拉到2000年左右，测量市区河道宽度",
                tool="web_search",
                params={
                    "operation": "keyword_search",
                    "purpose": "查河宽",
                    "inputs": {"query": ["市区 河道 宽度"]},
                },
                observation={"result": "市区段宽度约80米"},
            ),
            FreeFormStep(
                event_type="final",
                thought="提交",
                tool="final_answer",
                params={"location": "海河"},
                observation=None,
            ),
        ],
    )

    def matcher(name: str, forest: object) -> MatchDecision | None:
        if name == "web_search":
            return MatchDecision(
                raw_tool="web_search",
                action="map",
                canonical_name="satellite_imagery_query",
                operation="change_time",
                confidence=0.95,
                reason="调时相并量测河宽",
            )
        return None

    forest = map_tools.ensure_tool_trees(freeform, path, matcher=matcher)
    assert freeform.steps[0].tool == "satellite_imagery_query"
    assert freeform.steps[0].params.get("operation") == "change_time"
    assert trees.find_tree_for_name(forest, "satellite_imagery_query") is not None


def test_keyword_building_search_stays_web_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STAGE3_COMPILE_PARAMS", "false")
    monkeypatch.setenv("TOOL_CATALOG_PATH", "canonical_tool_catalog_v2.json")
    clear_settings_cache()
    path = tmp_path / "tool_trees.json"
    trees.save_forest(build_tool_forest_v2(), path)
    freeform = FreeFormTrajectory(
        source_video="demo",
        steps=[
            FreeFormStep(
                event_type="tool_call",
                thought="检索1999年之前建成的超100米建筑名单",
                tool="web_search",
                params={
                    "operation": "keyword_search",
                    "purpose": "列高楼候选",
                    "inputs": {"query": ["1999年前 超100米 建筑"]},
                },
                observation={"result": "列出若干候选大楼"},
            ),
            FreeFormStep(
                event_type="final",
                thought="提交",
                tool="final_answer",
                params={"location": "候选岸边"},
                observation=None,
            ),
        ],
    )

    def matcher(name: str, forest: object) -> MatchDecision | None:
        if name == "web_search":
            return MatchDecision(
                raw_tool="web_search",
                action="map",
                canonical_name="web_search",
                operation="keyword_search",
                confidence=0.99,
                reason="网页关键词检索",
            )
        return None

    map_tools.ensure_tool_trees(freeform, path, matcher=matcher)
    assert freeform.steps[0].tool == "web_search"


def test_reasoning_is_not_promoted_to_tool_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STAGE3_COMPILE_PARAMS", "false")
    monkeypatch.setenv("TOOL_CATALOG_PATH", "canonical_tool_catalog_v2.json")
    clear_settings_cache()
    path = tmp_path / "tool_trees.json"
    trees.save_forest(build_tool_forest_v2(), path)
    freeform = FreeFormTrajectory(
        source_video="demo",
        steps=[
            FreeFormStep(
                event_type="reasoning",
                thought="许昌附近多为平原，应改用黄河高地桥平原格局",
                tool=None,
                params={},
                observation=None,
            ),
            FreeFormStep(
                event_type="final",
                thought="提交",
                tool="final_answer",
                params={"location": "黄河文化公园"},
                observation=None,
            ),
        ],
    )

    def matcher(name: str, forest: object) -> MatchDecision | None:
        return MatchDecision(
            raw_tool=name,
            action="map",
            canonical_name="satellite_imagery_query",
            operation="retrieve",
            confidence=1.0,
            reason="不应被调用",
        )

    map_tools.ensure_tool_trees(freeform, path, matcher=matcher)
    assert freeform.steps[0].event_type == "reasoning"
    assert freeform.steps[0].tool is None

    entry = run_stage3(
        freeform,
        trees_path=path,
        out_trajectory_path=str(tmp_path / "stage3_trajectory.json"),
        out_jsonl_path=str(tmp_path / "shard.jsonl"),
        image_paths=[],
        matcher=matcher,
        compile_params=False,
    )
    assert entry.id
    assert freeform.steps[0].event_type == "reasoning"
