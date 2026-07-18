"""llm.py adapter 测试：禁止真实 API，外部调用必须 mock。"""

from __future__ import annotations

from pathlib import Path
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
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "qwen-vl-plus")
    clear_settings_cache()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _DummyOut(label="ok")

    def _fake_build(_settings: Any) -> MagicMock:
        return mock_client

    monkeypatch.setattr("pipeline.llm._build_instructor_client", _fake_build)
    result = call_structured("prompt", _DummyOut, model=None)
    assert result.label == "ok"
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "qwen-vl-plus"
    assert kwargs["response_model"] is _DummyOut


def test_call_text_uses_mocked_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    clear_settings_cache()

    class _TextResponse(BaseModel):
        text: str

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _TextResponse(text="judged")

    monkeypatch.setattr(
        "pipeline.llm._build_instructor_client",
        lambda _s: mock_client,
    )
    assert call_text("q", model="qwen-vl-plus") == "judged"


def test_qwen_multimodal_embeds_image_data_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "qwen-vl-plus")
    clear_settings_cache()

    img = tmp_path / "a.png"
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (1, 1), color=(200, 100, 50)).save(buf, format="PNG")
    img.write_bytes(buf.getvalue())

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _DummyOut(label="vision")
    monkeypatch.setattr(
        "pipeline.llm._build_instructor_client",
        lambda _s: mock_client,
    )

    call_structured("describe", _DummyOut, images=[str(img)])
    content = mock_client.chat.completions.create.call_args.kwargs["messages"][0][
        "content"
    ]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_unsupported_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("LLM_PROVIDER", "unknown-x")
    monkeypatch.setenv("LLM_MODEL", "x")
    clear_settings_cache()
    with pytest.raises(ValueError, match="不支持的 LLM_PROVIDER"):
        from pipeline.llm import _build_instructor_client
        from pipeline.config import get_settings

        _build_instructor_client(get_settings())
