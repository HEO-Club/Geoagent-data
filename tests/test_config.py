"""config.py 配置加载测试。"""

from __future__ import annotations

import pytest

from pipeline.config import Settings, clear_settings_cache, get_settings


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ALLOW_REAL_API", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MAX_REVISION_ROUNDS", raising=False)
    monkeypatch.delenv("DISTANCE_ERROR_THRESHOLD_KM", raising=False)
    # 避免读取仓库 .env 干扰：用空 env 覆盖
    s = Settings(_env_file=None)
    assert s.APP_ENV == "test"
    assert s.ALLOW_REAL_API is False
    assert s.OBS_SYNTH_MAX_RETRY == 3
    assert s.MAX_REVISION_ROUNDS == 2
    assert s.DISTANCE_ERROR_THRESHOLD_KM == 25.0
    assert s.LLM_PROVIDER == "qwen"
    assert s.LLM_MODEL == "qwen3.7-plus"
    assert s.GEMINI_MODEL == "gemini-2.0-flash"
    assert s.TOOL_REGISTRY_PATH == "tool_registry.json"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("LLM_MODEL", "qwen3-vl-plus")
    monkeypatch.setenv("MAX_CONCURRENT_VIDEOS", "4")
    s = Settings(_env_file=None)
    assert s.ALLOW_REAL_API is True
    assert s.LLM_MODEL == "qwen3-vl-plus"
    assert s.MAX_CONCURRENT_VIDEOS == 4


def test_get_settings_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "ci")
    a = get_settings()
    b = get_settings()
    assert a is b
    assert a.APP_ENV == "ci"
