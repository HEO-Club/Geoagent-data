"""llm_query 执行器测试；禁止真实付费 API。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.llm_query._query import (
    LlmQueryRequest,
    LlmQueryResponse,
    anthropic_endpoint_kind,
    normalize_anthropic_base_url,
)


class FakeLlmClient:
    """测试替身：记录请求并返回可控原文。"""

    default_model = "fake-default-model"

    def __init__(
        self,
        text: str = "这里像某座铁路桥，仅供核验",
        *,
        response_model: str = "relay-alias-xyz",
        provider: str = "injected",
        endpoint_kind: str = "relay",
    ) -> None:
        self.text = text
        self.response_model = response_model
        self.provider = provider
        self.endpoint_kind = endpoint_kind
        self.calls: list[LlmQueryRequest] = []

    def complete(self, request: LlmQueryRequest) -> LlmQueryResponse:
        self.calls.append(request)
        return LlmQueryResponse(
            text=self.text,
            requested_model=request.model,
            response_model=self.response_model,
            provider=self.provider,
            endpoint_kind=self.endpoint_kind,
        )


def _nested_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(value)
        for item in value.values():
            found.update(_nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_nested_keys(item))
    return found


def _consult(
    *,
    inputs: dict[str, object] | None = None,
    client: FakeLlmClient | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, object] = {"prompt": "这座桥可能是什么？"}
    if inputs:
        payload.update(inputs)
    return _run("consult", payload, client=client, ctx=ctx)


def _enumerate(
    *,
    inputs: dict[str, object] | None = None,
    client: FakeLlmClient | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    payload: dict[str, object] = {
        "prompt": "列出可能的铁路桥名称",
        "item_type": "铁路桥",
    }
    if inputs:
        payload.update(inputs)
    return _run("enumerate", payload, client=client, ctx=ctx)


def _run(
    operation: str,
    inputs: dict[str, object],
    *,
    client: FakeLlmClient | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    runtime = ctx
    if runtime is None:
        runtime = RuntimeContext(
            extras={"llm_query_client": client if client is not None else FakeLlmClient()}
        )
    elif client is not None:
        runtime.extras["llm_query_client"] = client
    return execute(
        "llm_query",
        operation,
        purpose="咨询外部模型",
        inputs=inputs,
        ctx=runtime,
    )


def test_consult_returns_model_suggestion_and_call_metadata() -> None:
    client = FakeLlmClient()
    observation = _consult(
        inputs={"context": "石栏、冬季、铁路桥", "model": "requested-alias"},
        client=client,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert client.calls and client.calls[0].prompt == "这座桥可能是什么？"
    assert "石栏、冬季、铁路桥" in client.calls[0].user
    assert client.calls[0].model == "requested-alias"
    assert observation.result["suggestion"] == "这里像某座铁路桥，仅供核验"
    assert observation.result["evidence_kind"] == "model_suggestion"
    applied = observation.result["applied"]
    assert applied["requested_model"] == "requested-alias"
    assert applied["response_model"] == "relay-alias-xyz"
    assert applied["requested_model"] != applied["response_model"]
    assert applied["endpoint_kind"] == "relay"
    assert applied["provider"] == "injected"
    assumptions = observation.result["assumptions"]
    assert any("不能当作地图数据库" in item for item in assumptions)
    assert any("中转别名" in item for item in assumptions)
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert "raw_content" not in keys
    assert "answer" not in keys


def test_consult_serializes_object_context() -> None:
    client = FakeLlmClient()
    observation = _consult(
        inputs={"context": {"clue": "石栏", "season": "冬"}},
        client=client,
    )
    assert observation.ok is True
    assert '"clue": "石栏"' in client.calls[0].user
    assert client.calls[0].context is not None
    assert "石栏" in client.calls[0].context


def test_enumerate_returns_candidates_with_item_ids() -> None:
    client = FakeLlmClient(
        json.dumps(
            {
                "candidates": [
                    {"text": "南京长江大桥", "rationale": "公路铁路两用"},
                    {"text": "武汉长江大桥", "rationale": "长江第一座"},
                    {"confirmed_location": "MUST NOT LEAK", "raw_content": "secret"},
                ]
            },
            ensure_ascii=False,
        )
    )
    observation = _enumerate(client=client)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["item_type"] == "铁路桥"
    assert observation.result["evidence_kind"] == "model_suggestion"
    candidates = observation.result["candidates"]
    assert candidates[0]["item_id"] == "llm_1"
    assert candidates[0]["text"] == "南京长江大桥"
    assert candidates[1]["item_id"] == "llm_2"
    assert observation.result["applied"]["max_items"] == 10
    assert observation.result["applied"]["item_type"] == "铁路桥"
    dumped = str(observation.result)
    assert "MUST NOT LEAK" not in dumped
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert "raw_content" not in keys


def test_enumerate_parses_fenced_json_and_bare_list() -> None:
    client = FakeLlmClient('```json\n[{"text": "赵州桥"}]\n```')
    observation = _enumerate(client=client)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["candidates"][0]["text"] == "赵州桥"


def test_enumerate_invalid_json_is_invalid_response() -> None:
    observation = _enumerate(client=FakeLlmClient("这不是 JSON"))
    assert observation.ok is False
    assert observation.error_code == "invalid_response"


def test_enumerate_missing_item_type_is_missing_input() -> None:
    observation = execute(
        "llm_query",
        "enumerate",
        purpose="缺类型",
        inputs={"prompt": "列出桥梁"},
        ctx=RuntimeContext(extras={"llm_query_client": FakeLlmClient()}),
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "item_type" in observation.error


def test_enumerate_drops_undeclared_model_and_uses_default() -> None:
    client = FakeLlmClient(
        json.dumps({"candidates": [{"text": "武汉长江大桥"}]}, ensure_ascii=False)
    )
    observation = _enumerate(inputs={"model": "should-be-dropped"}, client=client)
    assert observation.ok is True
    assert client.calls[0].model == "fake-default-model"


def test_enumerate_serializes_object_constraints() -> None:
    client = FakeLlmClient(json.dumps({"candidates": [{"text": "黄河铁路桥"}]}))
    observation = _enumerate(
        inputs={"constraints": {"region": "郑州", "era": "民国"}},
        client=client,
    )
    assert observation.ok is True
    assert "郑州" in (client.calls[0].constraints or "")
    assert "民国" in client.calls[0].user


def test_max_items_is_clamped() -> None:
    client = FakeLlmClient(
        json.dumps(
            {
                "candidates": [
                    {"text": f"桥{index}"} for index in range(1, 8)
                ]
            },
            ensure_ascii=False,
        )
    )
    high = _enumerate(inputs={"max_items": 1000}, client=client)
    assert high.ok is True
    assert client.calls[-1].max_items == 500
    assert high.result is not None
    assert high.result["applied"]["max_items"] == 500

    low_client = FakeLlmClient(
        json.dumps({"candidates": [{"text": "桥1"}, {"text": "桥2"}]})
    )
    low = _enumerate(inputs={"max_items": 0}, client=low_client)
    assert low.ok is True
    assert low_client.calls[0].max_items == 1
    assert low.result is not None
    assert len(low.result["candidates"]) == 1


def test_max_items_truncates_candidates() -> None:
    client = FakeLlmClient(
        json.dumps(
            {"candidates": [{"text": "甲"}, {"text": "乙"}, {"text": "丙"}]},
            ensure_ascii=False,
        )
    )
    observation = _enumerate(inputs={"max_items": 2}, client=client)
    assert observation.ok is True
    assert observation.result is not None
    assert [item["text"] for item in observation.result["candidates"]] == ["甲", "乙"]


def test_invalid_max_items_is_rejected() -> None:
    observation = _enumerate(inputs={"max_items": "many"})
    assert observation.ok is False
    assert observation.error_code == "invalid_max_items"


def test_missing_prompt_is_missing_input() -> None:
    observation = _consult(inputs={"prompt": "  "})
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_allow_real_api_false_without_client_is_unavailable() -> None:
    observation = execute(
        "llm_query",
        "consult",
        purpose="闸门",
        inputs={"prompt": "这座桥可能是什么？"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_API" in observation.error


def test_allow_real_api_true_missing_key_is_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("LLM_QUERY_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("LLM_ANTHROPIC_MODEL", "claude-test")
    monkeypatch.setenv("LLM_ANTHROPIC_BASE_URLS", "https://api.anthropic.com")
    observation = execute(
        "llm_query",
        "consult",
        purpose="缺钥匙",
        inputs={"prompt": "这座桥可能是什么？"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ANTHROPIC" in observation.error


def test_unsupported_provider_is_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("LLM_QUERY_PROVIDER", "gemini")
    observation = execute(
        "llm_query",
        "consult",
        purpose="不支持",
        inputs={"prompt": "这座桥可能是什么？"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "gemini" in observation.error


def test_insecure_anthropic_endpoint_is_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("LLM_QUERY_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("LLM_ANTHROPIC_MODEL", "claude-test")
    monkeypatch.setenv("LLM_ANTHROPIC_BASE_URLS", "http://relay.example")
    monkeypatch.setenv("ALLOW_INSECURE_LLM_ENDPOINTS", "false")
    observation = execute(
        "llm_query",
        "consult",
        purpose="明文",
        inputs={"prompt": "这座桥可能是什么？"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "HTTP" in observation.error


def test_anthropic_endpoint_kind_distinguishes_official_and_relay() -> None:
    assert anthropic_endpoint_kind("https://api.anthropic.com") == "official"
    assert (
        anthropic_endpoint_kind(normalize_anthropic_base_url("https://relay.example/v1"))
        == "relay"
    )
    assert normalize_anthropic_base_url("https://relay.example/v1") == "https://relay.example"


def test_anthropic_client_keeps_request_and_response_model(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                model="claude-sonnet-4-5-20250929",
                content=[SimpleNamespace(type="text", text="像某座桥")],
            )

    class _FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.messages = _Messages()

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    from tool.llm_query._query import AnthropicMessagesClient

    client = AnthropicMessagesClient(
        api_key="test-key",
        base_url="https://relay.example",
        default_model="relay-alias",
        timeout_sec=30.0,
        max_output_tokens=256,
        endpoint_kind="relay",
    )
    response = client.complete(
        LlmQueryRequest(
            operation="consult",
            prompt="这座桥可能是什么？",
            model="relay-alias",
            system="sys",
            user="这座桥可能是什么？",
        )
    )
    assert captured["model"] == "relay-alias"
    assert captured["client"]["api_key"] == "test-key"
    assert captured["client"]["base_url"] == "https://relay.example"
    assert response.requested_model == "relay-alias"
    assert response.response_model == "claude-sonnet-4-5-20250929"
    assert response.endpoint_kind == "relay"
    assert response.text == "像某座桥"


def test_openai_compatible_client_is_relay(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class _Completions:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                model="qwen-relay-alias",
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="可能是铁路桥"))
                ],
            )

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=_Completions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    from tool.llm_query._query import OpenAICompatibleClient

    client = OpenAICompatibleClient(
        api_key="sk-test",
        base_url="https://dashscope.example/compatible-mode/v1",
        default_model="qwen-flash",
        timeout_sec=30.0,
        max_output_tokens=256,
    )
    response = client.complete(
        LlmQueryRequest(
            operation="consult",
            prompt="这座桥可能是什么？",
            model="qwen-flash",
            system="sys",
            user="这座桥可能是什么？",
        )
    )
    assert captured["model"] == "qwen-flash"
    assert response.provider == "openai_compatible"
    assert response.endpoint_kind == "relay"
    assert response.response_model == "qwen-relay-alias"
    assert response.text == "可能是铁路桥"
