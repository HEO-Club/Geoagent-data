"""manage_tools.py CLI 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from manage_tools import main


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
    assert main(["list", "--tier", "draft"]) == 0
    assert main(["stats"]) == 0
    stats = capsys.readouterr().out
    assert "draft" in stats


def test_cli_promote_sun_position(cli_env: Path) -> None:
    code = main(
        [
            "promote",
            "sun_position_calc",
            "--executor-ref",
            "pipeline.tools.sun_position.execute",
        ]
    )
    assert code == 0
    from pipeline.tools.registry import load_registry
    from pipeline.schemas import ToolTier

    reg = load_registry(cli_env)
    assert reg["sun_position_calc"].tier is ToolTier.PRODUCTION
