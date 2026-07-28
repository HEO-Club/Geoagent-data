"""llm.py adapter 测试：禁止真实 API，外部调用必须 mock。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from pipeline.config import clear_settings_cache
from pipeline.llm import (
    RealAPIDisabledError,
    _EndpointConfig,
    _create_structured,
    _resolve_endpoint,
    call_structured,
    call_text,
)
from pipeline.config import Settings


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
    monkeypatch.setenv("LLM_PROVIDER", "kimi")
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "kimi-k3")
    clear_settings_cache()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _DummyOut(label="ok")

    def _fake_build(_endpoint: Any) -> MagicMock:
        return mock_client

    monkeypatch.setattr("pipeline.llm._build_instructor_client", _fake_build)
    result = call_structured("prompt", _DummyOut, model=None)
    assert result.label == "ok"
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "kimi-k3"
    assert kwargs["response_model"] is _DummyOut
    assert kwargs["reasoning_effort"] == "low"
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_call_text_uses_mocked_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("LLM_PROVIDER", "kimi")
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    clear_settings_cache()

    class _TextResponse(BaseModel):
        text: str

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _TextResponse(text="judged")

    monkeypatch.setattr(
        "pipeline.llm._build_instructor_client",
        lambda _e: mock_client,
    )
    assert call_text("q", model="kimi-k3") == "judged"


def test_vlm_lane_embeds_image_data_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("VLM_PROVIDER", "qwen")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("VLM_MODEL", "qwen-vl-plus")
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
        lambda _e: mock_client,
    )

    call_structured("describe", _DummyOut, images=[str(img)], lane="vlm")
    content = mock_client.chat.completions.create.call_args.kwargs["messages"][0][
        "content"
    ]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "qwen-vl-plus"
    assert "reasoning_effort" not in kwargs


def test_kimi_lane_multimodal_embeds_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """stage3+ 主通道 kimi-k3 也支持图片 data URL。"""
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("LLM_PROVIDER", "kimi")
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "kimi-k3")
    monkeypatch.setenv("KIMI_REASONING_EFFORT", "high")
    clear_settings_cache()

    img = tmp_path / "b.jpg"
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(buf, format="JPEG")
    img.write_bytes(buf.getvalue())

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _DummyOut(label="kimi-vision")
    monkeypatch.setattr(
        "pipeline.llm._build_instructor_client",
        lambda _e: mock_client,
    )

    call_structured("see", _DummyOut, images=[str(img)], lane="llm")
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    content = kwargs["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "image_url"
    assert kwargs["reasoning_effort"] == "high"
    assert "temperature" not in kwargs


def test_call_structured_retries_transient_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """瞬时 Connection error 应在 adapter 层重试后成功。"""
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("LLM_PROVIDER", "kimi")
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "kimi-k3")
    clear_settings_cache()

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        ConnectionError("Connection error"),
        _DummyOut(label="recovered"),
    ]
    monkeypatch.setattr(
        "pipeline.llm._build_instructor_client",
        lambda _e: mock_client,
    )
    monkeypatch.setattr("pipeline.llm.time.sleep", lambda _s: None)
    result = call_structured("prompt", _DummyOut)
    assert result.label == "recovered"
    assert mock_client.chat.completions.create.call_count == 2


def test_kimi_create_omits_sampling_params() -> None:
    endpoint = _EndpointConfig(
        provider="kimi",
        base_url="https://api.moonshot.cn/v1",
        model="kimi-k3",
        api_key="x",
        timeout_sec=300.0,
        reasoning_effort="low",
        omit_sampling_params=True,
        multimodal=True,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _DummyOut(label="x")
    _create_structured(
        mock_client,
        endpoint=endpoint,
        response_model=_DummyOut,
        content="hi",
    )
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    for forbidden in (
        "temperature",
        "top_p",
        "n",
        "presence_penalty",
        "frequency_penalty",
    ):
        assert forbidden not in kwargs


def test_resolve_endpoint_lanes() -> None:
    s = Settings(
        _env_file=None,
        VLM_PROVIDER="qwen",
        VLM_MODEL="qwen-vl-plus",
        VLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1",
        DASHSCOPE_API_KEY="dq",
        LLM_PROVIDER="kimi",
        LLM_MODEL="kimi-k3",
        LLM_BASE_URL="https://api.moonshot.cn/v1",
        MOONSHOT_API_KEY="mk",
        KIMI_REASONING_EFFORT="max",
    )
    vlm = _resolve_endpoint("vlm", None, s)
    assert vlm.provider == "qwen"
    assert vlm.model == "qwen-vl-plus"
    assert vlm.reasoning_effort is None
    llm = _resolve_endpoint("llm", None, s)
    assert llm.provider == "kimi"
    assert llm.model == "kimi-k3"
    assert llm.reasoning_effort == "max"
    assert llm.omit_sampling_params is True


def test_unsupported_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("LLM_PROVIDER", "unknown-x")
    monkeypatch.setenv("LLM_MODEL", "x")
    clear_settings_cache()
    with pytest.raises(ValueError, match="不支持的 provider"):
        from pipeline.config import get_settings
        from pipeline.llm import _resolve_endpoint

        _resolve_endpoint("llm", None, get_settings())
