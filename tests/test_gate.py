"""真实外网闸门：默认关闭；正式可开；pytest 拦截未 mock 的 HTTP。"""

from __future__ import annotations

import os
import urllib.request

import pytest

from tool import execute
from tool._gate import allow_real_api
from tool.contract import RuntimeContext


def test_allow_real_api_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALLOW_REAL_API", raising=False)
    assert allow_real_api() is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_allow_real_api_accepts_truthy(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", raw)
    assert allow_real_api() is True


@pytest.mark.parametrize("raw", ["", "0", "false", "no", "off"])
def test_allow_real_api_rejects_falsey(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", raw)
    assert allow_real_api() is False


def test_pytest_autouse_forces_allow_real_api_false() -> None:
    assert os.environ.get("ALLOW_REAL_API", "").strip().lower() == "false"
    assert allow_real_api() is False


def test_unmocked_urllib_is_blocked_in_pytest() -> None:
    with pytest.raises(RuntimeError, match="pytest 禁止真实外网"):
        urllib.request.urlopen("https://example.invalid/should-not-hit")


def test_tool_stays_unavailable_when_gate_closed() -> None:
    observation = execute(
        "web_search",
        "keyword_search",
        purpose="闸门",
        inputs={"query": "铁路桥"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_API" in observation.error


def test_opening_gate_without_key_still_does_not_hit_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "")
    monkeypatch.setenv("BRAVE_API_KEY", "")
    observation = execute(
        "web_search",
        "keyword_search",
        purpose="缺钥匙",
        inputs={"query": "铁路桥"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "BRAVE" in observation.error


def test_opened_gate_without_mock_cannot_reach_free_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOW_REAL_API", "true")
    with pytest.raises(RuntimeError, match="pytest 禁止真实外网"):
        execute(
            "weather_archive_query",
            "weather",
            purpose="拦截",
            inputs={
                "area": {"bbox": [113.5, 34.7, 113.8, 34.9]},
                "time_range": "2020-01-01",
            },
            ctx=RuntimeContext(),
        )
