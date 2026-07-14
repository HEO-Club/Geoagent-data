"""局部放大检视 adapter；描述生成经 LLM adapter，可 mock。"""

from __future__ import annotations

from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from pipeline.config import get_settings


class ZoomDescription(BaseModel):
    """zoom_inspect VLM 结构化输出。"""

    description: str = Field(min_length=1)


_describe_fn: Optional[Callable[[str, list[float]], str]] = None


def set_describe_fn(fn: Optional[Callable[[str, list[float]], str]]) -> None:
    """测试用：注入描述函数。"""
    global _describe_fn
    _describe_fn = fn


def _default_describe(image_path: str, bbox: list[float]) -> str:
    from pipeline.llm import call_structured

    prompt = (
        "Describe the cropped region of this geolocation image in concrete visual detail. "
        f"bbox={bbox}. Do not guess place names if not visible."
    )
    out = call_structured(prompt, ZoomDescription, images=[image_path])
    return out.description


def execute(params: dict[str, Any], image_path: str) -> dict[str, Any]:
    """执行 zoom_inspect。"""
    bbox = params.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return {
            "status": "error",
            "error_message": "bbox 参数无效",
            "description": "",
        }
    try:
        settings = get_settings()
        describe = _describe_fn
        if describe is None:
            if settings.APP_ENV == "test" and not settings.ALLOW_REAL_API:
                raise RuntimeError("test 环境请通过 set_describe_fn 注入描述函数")
            describe = _default_describe
        description = describe(image_path, [float(x) for x in bbox])
        if not description.strip():
            return {
                "status": "empty",
                "error_message": None,
                "description": "",
            }
        return {
            "status": "success",
            "error_message": None,
            "description": description,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": str(exc),
            "description": "",
        }
