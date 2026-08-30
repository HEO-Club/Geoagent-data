"""Test runs must never inherit a paid-API opt-in from a developer's .env."""

from __future__ import annotations

import pytest

from pipeline.config import clear_settings_cache


@pytest.fixture(autouse=True)
def isolate_real_api_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    monkeypatch.setenv("APP_ENV", "test")
    clear_settings_cache()
    yield
    clear_settings_cache()
