"""manage_tools.py CLI 测试：list / stats / register。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manage_tools import main
from pipeline.tools.registry import load_registry


@pytest.fixture()
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = Path(__file__).resolve().parents[1]
    dst = tmp_path / "tool_registry.json"
    dst.write_text((root / "tool_registry.json").read_text(encoding="utf-8"), encoding="utf-8")
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


def test_cli_list_and_stats(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "web_search" in out
    assert "terminal=" in out
    assert main(["stats"]) == 0
    stats = capsys.readouterr().out
    payload = json.loads(stats)
    assert payload["total"] >= 7
    assert "submit_answer" in payload["terminal"]
    assert "by_agent_allow" in payload


def test_cli_register_tool(cli_env: Path, tmp_path: Path) -> None:
    tool_json = tmp_path / "new_tool.json"
    tool_json.write_text(
        json.dumps(
            {
                "name": "detect_facade_ornament",
                "description": "检测建筑立面纹饰图案供粗定位使用。",
                "params": [
                    {
                        "name": "region_hint",
                        "type": "string",
                        "required": True,
                        "description": "立面区域描述，如[upper facade]",
                        "example": "upper facade",
                        "default": None,
                        "enum_values": None,
                    }
                ],
                "observation_fields": [
                    {
                        "name": "status",
                        "type": "string",
                        "nullable": False,
                        "description": "执行状态，取值 success/empty/error",
                        "item_fields": None,
                    },
                    {
                        "name": "error_message",
                        "type": "string",
                        "nullable": True,
                        "description": "status=error 时填写，否则为 null",
                        "item_fields": None,
                    },
                    {
                        "name": "ornament_type",
                        "type": "string",
                        "nullable": True,
                        "description": "纹饰类型标签；无法识别时为 null",
                        "item_fields": None,
                    },
                ],
                "allowed_agents": ["coarse_locator"],
                "is_terminal": False,
                "created_at": "2026-07-14T00:00:00Z",
                "source_video_timestamp": None,
                "source_narration": None,
                "derived_from_existing_tools": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    assert main(["register", "--from-json", str(tool_json)]) == 0
    reg = load_registry(cli_env)
    assert "detect_facade_ornament" in reg
