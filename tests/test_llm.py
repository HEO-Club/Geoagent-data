"""llm.py adapter 测试：禁止真实 API，外部调用必须 mock。"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from pipeline.config import clear_settings_cache
from pipeline.llm import RealAPIDisabledError, call_structured, call_text


class _DummyOut(BaseModel):
    label: str


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_call_structured_blocked_when_real_api_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    clear_settings_cache()
    with pytest.raises(RealAPIDisabledError):
        call_structured("hello", _DummyOut)


def test_call_text_blocked_when_real_api_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    clear_settings_cache()
    with pytest.raises(RealAPIDisabledError):
        call_text("judge this")


def test_call_structured_uses_mocked_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-mock")
    clear_settings_cache()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _DummyOut(label="ok")

    def _fake_build(_settings: Any) -> MagicMock:
        return mock_client

    monkeypatch.setattr("pipeline.llm._build_instructor_client", _fake_build)
    result = call_structured("prompt", _DummyOut, model=None)
    assert result.label == "ok"
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gemini-mock"
    assert kwargs["response_model"] is _DummyOut


def test_call_text_uses_mocked_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    clear_settings_cache()

    class _TextResponse(BaseModel):
        text: str

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _TextResponse(text="judged")

    monkeypatch.setattr(
        "pipeline.llm._build_instructor_client",
        lambda _s: mock_client,
    )
    assert call_text("q", model="gemini-mock") == "judged"
