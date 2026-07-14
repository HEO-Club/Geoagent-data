"""registry 注册 / 文件锁 / 升档备份回滚测试。"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from pipeline.schemas import AgentRole, ObservationField, ParamField, ToolDefinition, ToolTier
from pipeline.tools.registry import (
    get_tools_for_agent,
    load_registry,
    promote_tool,
    register_tool,
)


def _status_fields() -> list[ObservationField]:
    return [
        ObservationField(
            name="status",
            type="string",
            nullable=False,
            description="执行状态，取值 success/empty/error",
        ),
        ObservationField(
            name="error_message",
            type="string",
            nullable=True,
            description="status=error 时填写，否则为 null",
        ),
    ]


def _draft_tool(name: str, description: str) -> ToolDefinition:
    return ToolDefinition.model_validate(
        {
            "name": name,
            "description": description,
            "tier": ToolTier.DRAFT,
            "params": [
                ParamField(
                    name="query",
                    type="string",
                    required=True,
                    description="检索查询语句，如[landmark]",
                    example="landmark",
                )
            ],
            "observation_fields": [
                *_status_fields(),
                ObservationField(
                    name="summary",
                    type="string",
                    nullable=True,
                    description="摘要文本；无信息时为 null",
                ),
            ],
            "allowed_agents": [AgentRole.COARSE],
            "is_terminal": False,
            "executor_ref": None,
            "created_at": "2026-07-14T00:00:00Z",
        }
    )


@pytest.fixture()
def tmp_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "tool_registry.json"
    dst = tmp_path / "tool_registry.json"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("TOOL_REGISTRY_PATH", str(dst))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / ".cache"))
    monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    from pipeline.config import clear_settings_cache

    clear_settings_cache()
    (tmp_path / "intermediate").mkdir()
    (tmp_path / "output").mkdir()
    yield dst
    clear_settings_cache()


def test_load_registry_indexes_seeds(tmp_registry: Path) -> None:
    reg = load_registry(tmp_registry)
    assert "web_search" in reg
    assert reg["submit_answer"].is_terminal is True


def test_register_tool_and_reject_duplicate_name(tmp_registry: Path) -> None:
    tool = _draft_tool("detect_signage", "检测路牌文字线索供粗定位使用。")
    register_tool(tool, path=tmp_registry)
    reg = load_registry(tmp_registry)
    assert "detect_signage" in reg
    assert reg["detect_signage"].tier is ToolTier.DRAFT
    with pytest.raises(ValueError, match="已存在|编辑距离|语义"):
        register_tool(tool, path=tmp_registry)


def test_register_rejects_near_duplicate_name(tmp_registry: Path) -> None:
    # 与 web_search 编辑距离很小
    tool = _draft_tool("web_searx", "按用途检索网页用于地理线索发现验证。")
    with pytest.raises(ValueError):
        register_tool(tool, path=tmp_registry)


def test_derived_from_must_exist(tmp_registry: Path) -> None:
    tool = _draft_tool("lookup_plaza", "查询广场周边地理线索供定位使用。")
    tool = tool.model_copy(update={"derived_from_existing_tools": ["not_exists_tool"]})
    with pytest.raises(ValueError, match="derived_from"):
        register_tool(tool, path=tmp_registry)


def test_get_tools_for_agent_filters(tmp_registry: Path) -> None:
    coarse = get_tools_for_agent(AgentRole.COARSE, path=tmp_registry)
    names = {t.name for t in coarse}
    assert "web_search" in names
    assert "sun_position_calc" in names
    assert "map_query" not in names
    assert "reverse_image_search" not in names


def test_concurrent_register_with_file_lock(tmp_registry: Path) -> None:
    errors: list[BaseException] = []
    # 名称需两两编辑距离 > 3，避免语义去重误杀
    names = [
        "extract_alpha_sign",
        "detect_bravo_plaque",
        "lookup_charlie_crest",
        "compare_delta_emblem",
        "estimate_echo_banner",
    ]

    def worker(name: str) -> None:
        try:
            register_tool(
                _draft_tool(name, f"处理与{name}相关的独特地理线索。"),
                path=tmp_registry,
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    reg = load_registry(tmp_registry)
    assert all(n in reg for n in names)


def test_promote_sun_position_success_with_mocked_rerun(
    tmp_registry: Path,
    tmp_path: Path,
) -> None:
    # 写入一条受影响样本
    out = tmp_path / "output" / "agent1.jsonl"
    out.write_text(
        json.dumps(
            {
                "id": "v1",
                "source_video": "video_a",
                "draft_tool_names": ["sun_position_calc"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def rerun(tool_name: str, video_ids: list[str]) -> dict:
        assert tool_name == "sun_position_calc"
        assert "video_a" in video_ids
        return {"rerun": video_ids, "stages": ["stage4", "stage5", "stage6", "stage7"]}

    report = promote_tool(
        "sun_position_calc",
        "pipeline.tools.sun_position.execute",
        path=tmp_registry,
        stage_rerunner=rerun,
    )
    assert report["status"] == "success"
    reg = load_registry(tmp_registry)
    assert reg["sun_position_calc"].tier is ToolTier.PRODUCTION
    assert reg["sun_position_calc"].executor_ref == "pipeline.tools.sun_position.execute"


def test_promote_failure_rolls_back(tmp_registry: Path) -> None:
    def boom(tool_name: str, video_ids: list[str]) -> dict:
        raise RuntimeError("stage rerun failed")

    # 无受影响样本时默认 rerunner 不会失败；放入假样本
    out_dir = Path(tmp_registry).parent / "output"
    (out_dir / "x.jsonl").write_text(
        json.dumps({"source_video": "v9", "draft_tool_names": ["ocr"]}) + "\n",
        encoding="utf-8",
    )

    # ocr promote 需要可执行 smoke；注入 OCR 引擎
    from pipeline.tools import ocr as ocr_mod

    class _Eng:
        def run(self, image_path: str, bbox: list[float] | None) -> list[str]:
            return ["HELLO"]

    ocr_mod.set_engine(_Eng())
    try:
        with pytest.raises(RuntimeError, match="stage rerun failed"):
            promote_tool(
                "ocr",
                "pipeline.tools.ocr.execute",
                path=tmp_registry,
                stage_rerunner=boom,
            )
    finally:
        ocr_mod.set_engine(None)

    reg = load_registry(tmp_registry)
    assert reg["ocr"].tier is ToolTier.DRAFT
    assert reg["ocr"].executor_ref is None
