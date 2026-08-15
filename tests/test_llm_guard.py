"""LLM adapter 闸门测试。"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from pipeline import llm
from pipeline.config import Settings, clear_settings_cache
from pipeline.llm import RealAPIDisabledError, call_structured, call_text


class _T(BaseModel):
    x: str


def test_real_api_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    clear_settings_cache()
    with pytest.raises(RealAPIDisabledError):
        call_structured("hi", _T, lane="llm")
    with pytest.raises(RealAPIDisabledError):
        call_text("hi", lane="vlm")


def test_anthropic_endpoint_and_image_format(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        LLM_PROVIDER="anthropic",
        LLM_ANTHROPIC_BASE_URLS="https://relay.example/v1",
        LLM_ANTHROPIC_MODEL="claude-test",
        LLM_ANTHROPIC_API_KEY="test-key",
    )
    endpoint = llm._resolve_endpoint("llm", None, settings)
    assert endpoint.provider == "anthropic"
    assert endpoint.base_url == "https://relay.example"
    assert endpoint.model == "claude-test"

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"not-a-real-image")
    content = llm._build_user_content(
        "prompt",
        [str(image)],
        provider="anthropic",
        multimodal=True,
    )
    assert isinstance(content, list)
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"


def test_anthropic_plain_http_rejected() -> None:
    settings = Settings(
        LLM_PROVIDER="anthropic",
        LLM_ANTHROPIC_BASE_URLS="http://relay.example",
        LLM_ANTHROPIC_MODEL="claude-test",
        LLM_ANTHROPIC_API_KEY="test-key",
        ALLOW_INSECURE_LLM_ENDPOINTS=False,
    )
    with pytest.raises(ValueError, match="HTTP"):
        llm._resolve_endpoint("llm", None, settings)


def test_anthropic_relay_parameter_wrapper_is_unwrapped() -> None:
    class _WrappedResult(BaseModel):
        steps: list[str]

    class _Messages:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            assert kwargs["tools"][0]["input_schema"]["required"] == ["steps"]
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        input={
                            "$FUNCTION_NAME": "_WrappedResult",
                            "$PARAMETER_NAME": {
                                "steps": '["one", "two"]',
                            },
                        },
                    )
                ]
            )

    endpoint = llm._EndpointConfig(
        provider="anthropic",
        base_url="https://relay.example",
        model="claude-test",
        api_key="test-key",
        timeout_sec=30.0,
        max_output_tokens=1024,
        reasoning_effort=None,
        omit_sampling_params=True,
        multimodal=True,
    )
    client = SimpleNamespace(messages=_Messages())
    result = llm._create_anthropic_structured(
        client,
        endpoint=endpoint,
        response_model=_WrappedResult,
        content="prompt",
    )
    assert result.steps == ["one", "two"]


def test_anthropic_relay_same_field_root_wrapper_is_unwrapped() -> None:
    class _WrappedResult(BaseModel):
        steps: list[str]
        notes: str | None = None

    class _Messages:
        def create(self, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        input={
                            "steps": {
                                "steps": ["one", "two"],
                                "notes": None,
                            }
                        },
                    )
                ]
            )

    endpoint = llm._EndpointConfig(
        provider="anthropic",
        base_url="https://relay.example",
        model="claude-test",
        api_key="test-key",
        timeout_sec=30.0,
        max_output_tokens=1024,
        reasoning_effort=None,
        omit_sampling_params=True,
        multimodal=True,
    )
    result = llm._create_anthropic_structured(
        SimpleNamespace(messages=_Messages()),
        endpoint=endpoint,
        response_model=_WrappedResult,
        content="prompt",
    )
    assert result.steps == ["one", "two"]
    assert result.notes is None


@pytest.mark.parametrize(
    "wrapped",
    [
        {"params": {"steps": ["one", "two"], "notes": None}},
        {
            "arguments": '{"steps": ["one", "two"], "notes": null}'
        },
    ],
)
def test_anthropic_relay_generic_protocol_wrapper_is_unwrapped(wrapped) -> None:  # type: ignore[no-untyped-def]
    class _WrappedResult(BaseModel):
        steps: list[str]
        notes: str | None = None

    result = llm._validate_anthropic_tool_payload(wrapped, _WrappedResult)
    assert result.steps == ["one", "two"]
    assert result.notes is None


def test_anthropic_relay_single_field_semantic_envelope_is_unwrapped() -> None:
    class _FrameVerdict(BaseModel):
        kind: str
        reason: str = ""

    result = llm._validate_anthropic_tool_payload(
        {
            "verdict": {
                "kind": "teaching_ui",
                "reason": "地图核验画面",
            }
        },
        _FrameVerdict,
    )
    assert result.kind == "teaching_ui"
    assert result.reason == "地图核验画面"


def test_structured_validation_error_is_retryable() -> None:
    class _RequiredResult(BaseModel):
        value: str

    with pytest.raises(ValidationError) as caught:
        _RequiredResult.model_validate({})
    assert llm._is_transient_llm_error(caught.value) is True


def test_anthropic_stream_uses_final_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StreamResult:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return False

        def get_final_message(self):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", input={"x": "ok"})]
            )

    class _Messages:
        def stream(self, **_kwargs):  # type: ignore[no-untyped-def]
            return _StreamResult()

        def create(self, **_kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("stream=true 时不应走非流式 create")

    monkeypatch.setenv("LLM_ANTHROPIC_STREAM", "true")
    clear_settings_cache()
    endpoint = llm._EndpointConfig(
        provider="anthropic",
        base_url="https://relay.example",
        model="claude-test",
        api_key="test-key",
        timeout_sec=30.0,
        max_output_tokens=1024,
        reasoning_effort=None,
        omit_sampling_params=True,
        multimodal=True,
    )
    result = llm._create_anthropic_structured(
        SimpleNamespace(messages=_Messages()),
        endpoint=endpoint,
        response_model=_T,
        content="prompt",
    )
    assert result.x == "ok"
    clear_settings_cache()


def test_anthropic_stream_can_be_disabled_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Messages:
        def stream(self, **_kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("单帧调用覆盖为非流式后不应走 stream")

        def create(self, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", input={"x": "ok"})]
            )

    monkeypatch.setenv("LLM_ANTHROPIC_STREAM", "true")
    clear_settings_cache()
    endpoint = llm._EndpointConfig(
        provider="anthropic",
        base_url="https://relay.example",
        model="claude-test",
        api_key="test-key",
        timeout_sec=30.0,
        max_output_tokens=1024,
        reasoning_effort=None,
        omit_sampling_params=True,
        multimodal=True,
    )
    result = llm._create_anthropic_structured(
        SimpleNamespace(messages=_Messages()),
        endpoint=endpoint,
        response_model=_T,
        content="prompt",
        stream_override=False,
    )
    assert result.x == "ok"
    clear_settings_cache()


def test_anthropic_stream_has_overall_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StreamResult:
        def __enter__(self):  # type: ignore[no-untyped-def]
            # 总时限必须覆盖 SDK 建立 SSE 会话本身，而非只覆盖最终读取。
            time.sleep(2.0)
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return False

        def get_final_message(self):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", input={"x": "late"})]
            )

    stream = _StreamResult()

    class _Messages:
        def stream(self, **_kwargs):  # type: ignore[no-untyped-def]
            return stream

    monkeypatch.setenv("LLM_ANTHROPIC_STREAM", "true")
    clear_settings_cache()
    endpoint = llm._EndpointConfig(
        provider="anthropic",
        base_url="https://relay.example",
        model="claude-test",
        api_key="test-key",
        timeout_sec=0.1,
        max_output_tokens=1024,
        reasoning_effort=None,
        omit_sampling_params=True,
        multimodal=True,
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="overall timeout"):
        llm._create_anthropic_structured(
            SimpleNamespace(messages=_Messages()),
            endpoint=endpoint,
            response_model=_T,
            content="prompt",
        )
    assert time.monotonic() - started < 0.5
    clear_settings_cache()


def test_omni_audio_helpers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    audio = tmp_path / "sample.m4a"
    audio.write_bytes(b"audio")
    assert llm._audio_format(str(audio)) == "mp4"
    assert llm._omni_delta_text("正文") == "正文"
    assert llm._omni_delta_text([{"type": "text", "text": "江湾大桥"}]) == (
        "江湾大桥"
    )
