"""LLM adapter 闸门测试。"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from pipeline.config import clear_settings_cache
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
