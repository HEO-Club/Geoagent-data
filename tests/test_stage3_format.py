"""阶段3：tool 树与 JSONL。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.schemas.clues import BoundKind, WorkingScope
from pipeline.schemas.freeform import FreeFormStep, FreeFormTrajectory
from pipeline.schemas.tools import MatchDecision, ToolDefinition, ToolForest, ToolTree
from pipeline.stage3_normalize_format import format_jsonl, map_tools, trees


def test_non_ascii_tool_slugs_are_stable_and_distinct() -> None:
    first = map_tools._slug_tool_name("观察待定位图像")
    second = map_tools._slug_tool_name("分析索塔外形")
    assert first.startswith("custom_tool_")
    assert first == map_tools._slug_tool_name("观察待定位图像")
    assert first != second


def test_build_user_query_with_and_without_scope() -> None:
    assert format_jsonl.build_user_query(None) == format_jsonl.DEFAULT_USER_QUERY
    q = format_jsonl.build_user_query(
        WorkingScope(region="河南许昌附近", bound_kind=BoundKind.near)
    )
    assert q.startswith(format_jsonl.DEFAULT_USER_QUERY)
    assert "Working scope: 河南许昌附近" in q


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
        image_paths=["scene.jpg", "scene2.jpg"],
        matcher=lambda _n, _f: None,
        compile_params=False,
    )
    assert entry.source_video == "clip"
    roles = [m.role for m in entry.messages]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert roles[2] == "assistant"
    assert roles[3] == "tool"
    assert "Thought:" in entry.messages[2].content
    assert "[Image: scene.jpg]" in entry.messages[1].content
    assert "[Image: scene2.jpg]" in entry.messages[1].content
    blob = json.dumps([m.content for m in entry.messages], ensure_ascii=False)
    assert "web_lookup" in blob
    traj_data = json.loads(
        (tmp_path / "intermediate" / "clip" / "stage3_trajectory.json").read_text(
            encoding="utf-8"
        )
    )
    params = traj_data["steps"][0]["action"]["params"]
    assert set(params) == {"operation", "purpose", "inputs"}
    assert params["inputs"] == {"q": "tower"}
    assert entry.messages[1].content.startswith(format_jsonl.DEFAULT_USER_QUERY)
    assert "Working scope:" not in entry.messages[1].content
    shard = tmp_path / "output" / "shards" / "clip.jsonl"
    assert shard.is_file()


def test_run_stage3_injects_working_scope_into_user_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("TOOL_TREES_PATH", str(tmp_path / "tool_trees.json"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    freeform = FreeFormTrajectory(
        source_video="scoped",
        working_scope=WorkingScope(region="河南许昌附近", bound_kind=BoundKind.near),
        steps=[
            FreeFormStep(
                thought="search",
                tool="web_lookup",
                params={"q": "x"},
                observation={"hits": []},
            )
        ],
    )
    entry = format_jsonl.run_stage3(
        freeform,
        trees_path=tmp_path / "tool_trees.json",
        image_paths=["scene.jpg"],
        matcher=lambda _n, _f: None,
        compile_params=False,
    )
    user = entry.messages[1].content
    assert "Working scope: 河南许昌附近" in user
    assert "Locate the place shown in the image." in user


def test_run_stage3_explicit_user_query_overrides_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("TOOL_TREES_PATH", str(tmp_path / "tool_trees.json"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()

    freeform = FreeFormTrajectory(
        source_video="override",
        working_scope=WorkingScope(region="河南许昌附近", bound_kind=BoundKind.near),
        steps=[
            FreeFormStep(
                thought="t",
                tool="web_lookup",
                params={},
                observation={"ok": True},
            )
        ],
    )
    entry = format_jsonl.run_stage3(
        freeform,
        trees_path=tmp_path / "tool_trees.json",
        image_paths=["scene.jpg"],
        user_query="Custom query only.",
        matcher=lambda _n, _f: None,
        compile_params=False,
    )
    assert entry.messages[1].content.startswith("Custom query only.")
    assert "Working scope:" not in entry.messages[1].content


def test_final_answer_is_reserved_terminal_and_keeps_location(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_trees.json"
    trees.save_forest(
        ToolForest(
            trees=[
                ToolTree(
                    canonical=ToolDefinition(
                        name="location_synthesizer",
                        description="ordinary legacy tool",
                        is_terminal=False,
                    ),
                    variants=[],
                )
            ]
        ),
        path,
    )
    freeform = FreeFormTrajectory(
        source_video="terminal",
        steps=[
            FreeFormStep(
                thought="证据已经闭合，提交最终答案。",
                tool="final_answer",
                params={"location": "山东省淄博市淄川区马棚村"},
                observation=None,
            )
        ],
    )

    def bad_matcher(_name: str, _forest: ToolForest) -> str | None:
        return "location_synthesizer"

    forest = map_tools.ensure_tool_trees(freeform, path, matcher=bad_matcher)
    final_tree = next(t for t in forest.trees if t.canonical.name == "final_answer")
    assert final_tree.canonical.is_terminal is True

    traj = format_jsonl.remap_trajectory(
        freeform,
        forest,
        system_prompt="system",
        user_query="query",
        image_paths=["scene.jpg"],
        compile_params=False,
    )
    assert traj.image_paths == ["scene.jpg"]
    assert traj.steps[-1].action.tool == "final_answer"
    assert traj.steps[-1].action.params == {"location": "山东省淄博市淄川区马棚村"}
    assert traj.steps[-1].observation is None


def test_reasoning_events_do_not_create_fake_tools_and_keep_thought_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOOL_CATALOG_PATH", "canonical_tool_catalog.json")
    from pipeline.config import clear_settings_cache

    clear_settings_cache()
    freeform = FreeFormTrajectory.model_validate(
        {
            "source_video": "events",
            "steps": [
                {
                    "event_type": "reasoning",
                    "thought": "根据已有余晖证据排除较早天黑的候选。",
                    "tool": None,
                    "params": {},
                    "observation": None,
                },
                {
                    "event_type": "reasoning",
                    "thought": "剩余范围仍过大，需要查询 OSM 水面。",
                    "tool": None,
                    "params": {},
                    "observation": None,
                },
                {
                    "event_type": "tool_call",
                    "thought": "查询云南全部水面要素。",
                    "tool": "custom_overpass_water_query",
                    "params": {"area": "云南", "tags": ["water"]},
                    "observation": {"count": 12},
                },
                {
                    "event_type": "final",
                    "thought": "证据闭合。",
                    "tool": "final_answer",
                    "params": {"location": "甲地"},
                    "observation": None,
                },
            ],
        }
    )

    def matcher(name: str, _forest: ToolForest) -> MatchDecision | None:
        if name == "custom_overpass_water_query":
            return MatchDecision(
                raw_tool=name,
                action="map",
                canonical_name="osm_query",
                operation="query",
                operation_description="按区域和标签查询水面要素",
                confidence=0.99,
            )
        return None

    entry = format_jsonl.run_stage3(
        freeform,
        trees_path=tmp_path / "runtime_tools.json",
        out_trajectory_path=str(tmp_path / "trajectory.json"),
        out_jsonl_path=str(tmp_path / "sample.jsonl"),
        image_paths=["scene.jpg"],
        matcher=matcher,
        compile_params=False,
    )
    roles = [message.role for message in entry.messages]
    assert roles == [
        "system",
        "user",
        "assistant",
        "assistant",
        "assistant",
        "tool",
        "assistant",
    ]
    assert "Action:" not in entry.messages[2].content
    assert "Action:" not in entry.messages[3].content
    assert '"tool": "osm_query"' in entry.messages[4].content
    # Mapping is in-memory for this task only; no runtime dump to reload.


def test_high_confidence_pseudo_tool_is_demoted_to_reasoning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOOL_CATALOG_PATH", "canonical_tool_catalog.json")
    from pipeline.config import clear_settings_cache

    clear_settings_cache()
    freeform = FreeFormTrajectory(
        source_video="demote",
        steps=[
            FreeFormStep(
                thought="根据上一轮日落时间排除江苏。",
                tool="apply_time_consistency_filter",
                params={"capture_time": "18:54"},
                observation={"eliminated": "江苏"},
            )
        ],
    )

    def matcher(name: str, _forest: ToolForest) -> MatchDecision | None:
        return MatchDecision(
            raw_tool=name,
            action="reasoning",
            confidence=0.98,
            reason="没有访问外部执行器，只在合并已有证据",
        )

    forest = map_tools.ensure_tool_trees(
        freeform, tmp_path / "runtime_tools.json", matcher=matcher
    )
    assert freeform.steps[0].event_type == "reasoning"
    assert freeform.steps[0].tool is None
    assert "江苏" in freeform.steps[0].thought
    assert trees.find_tree_for_name(forest, "apply_time_consistency_filter") is None


def test_explicit_external_llm_call_maps_to_catalog_tool_without_demotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOOL_CATALOG_PATH", "canonical_tool_catalog.json")
    from pipeline.config import clear_settings_cache

    clear_settings_cache()
    freeform = FreeFormTrajectory(
        source_video="llm_call",
        steps=[
            FreeFormStep(
                thought="调用高级推理模型补充百米建筑候选。",
                tool="AI推理咨询",
                params={"query": "北京天津百米建筑"},
                observation={"result": ["天津国际大厦", "首都大厦"]},
            )
        ],
    )

    def fail_matcher(*_args: object, **_kwargs: object) -> dict[str, MatchDecision]:
        raise AssertionError("目录精确命中时不应再调用语义 matcher")

    monkeypatch.setattr(map_tools, "llm_semantic_match_batch", fail_matcher)
    forest = map_tools.ensure_tool_trees(
        freeform, tmp_path / "runtime_tools.json", matcher=None
    )

    matched = trees.find_tree_for_name(forest, "AI推理咨询")
    assert matched is not None
    assert matched.canonical.name == "llm_query"
    assert trees.resolve_operation(forest, "AI推理咨询") == "consult"
    assert freeform.steps[0].event_type == "tool_call"
