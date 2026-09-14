"""Test runs must never inherit a paid-API opt-in from a developer's .env."""

from __future__ import annotations

import urllib.request
from typing import Any

import pytest

from pipeline.config import clear_settings_cache


class RealNetworkBlockedError(RuntimeError):
    """pytest 拦截未 mock 的真实 HTTP。"""


def _blocked_urlopen(*args: Any, **kwargs: Any) -> Any:
    raise RealNetworkBlockedError(
        "pytest 禁止真实外网；请注入 adapter 或 mock urlopen"
    )


def _blocked_opener_open(self: Any, *args: Any, **kwargs: Any) -> Any:
    del self
    raise RealNetworkBlockedError(
        "pytest 禁止真实外网；请注入 adapter 或 mock urlopen"
    )


def _blocked_httpx_send(self: Any, *args: Any, **kwargs: Any) -> Any:
    del self
    raise RealNetworkBlockedError(
        "pytest 禁止真实外网；请注入 adapter 或 mock HTTP 客户端"
    )


@pytest.fixture(autouse=True)
def isolate_real_api_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    monkeypatch.setenv("ALLOW_REAL_TOOL_API", "false")
    monkeypatch.setenv("ALLOW_CUSTOM_TOOL_ENDPOINTS", "false")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setattr(urllib.request, "urlopen", _blocked_urlopen)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", _blocked_opener_open)
    try:
        import httpx
    except ImportError:
        httpx = None
    if httpx is not None:
        monkeypatch.setattr(httpx.Client, "send", _blocked_httpx_send)
        monkeypatch.setattr(httpx.AsyncClient, "send", _blocked_httpx_send)
    clear_settings_cache()
    yield
    clear_settings_cache()
