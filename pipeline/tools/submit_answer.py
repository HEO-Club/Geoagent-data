"""submit_answer 为 terminal 本地动作；无 Observation，不提供真实 executor。"""

from __future__ import annotations

from typing import Any


def execute(params: dict[str, Any], image_path: str) -> dict[str, Any]:
    """terminal tool 不应被 execute_action 调用；保留占位以满足导入探测。"""
    _ = params, image_path
    raise RuntimeError("submit_answer 为 terminal Tool，不产生 Observation，禁止调用 execute")
