"""LLM adapter：经 Instructor 封装结构化输出，禁止业务层直调 SDK。"""

from __future__ import annotations

from typing import Optional, TypeVar

from pydantic import BaseModel

from pipeline.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class RealAPIDisabledError(RuntimeError):
    """ALLOW_REAL_API=false 时拒绝真实网络调用。"""


def _resolve_model(model: Optional[str], settings: Settings) -> str:
    """解析实际使用的模型名；仅允许来自参数或配置。"""
    resolved = model if model is not None else settings.GEMINI_MODEL
    if not resolved:
        raise ValueError("模型名未配置：请设置 GEMINI_MODEL 或传入 model 参数")
    return resolved


def _ensure_real_api_allowed(settings: Settings) -> None:
    if not settings.ALLOW_REAL_API:
        raise RealAPIDisabledError(
            "ALLOW_REAL_API=false，禁止真实付费 API 调用；测试请 mock adapter"
        )


def _build_instructor_client(settings: Settings):  # type: ignore[no-untyped-def]
    """构建 Instructor 客户端（Google GenAI）。依赖注入点，测试可 patch。"""
    import instructor
    from google import genai

    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY 未配置")
    raw = genai.Client(api_key=settings.GOOGLE_API_KEY)
    return instructor.from_genai(raw)


def call_structured(
    prompt: str,
    response_model: type[T],
    images: Optional[list[str]] = None,
    video: Optional[str] = None,
    model: Optional[str] = None,
) -> T:
    """调用 LLM 并强制返回 response_model；不合法由 Instructor 自动重试。

    Args:
        prompt: 文本提示。
        response_model: 期望的 Pydantic 模型类型。
        images: 可选图片路径列表。
        video: 可选视频路径。
        model: 模型名；None 时从配置读取。

    Returns:
        经 Pydantic 校验的结构化结果。
    """
    settings = get_settings()
    _ensure_real_api_allowed(settings)
    model_name = _resolve_model(model, settings)

    # images / video 由上层拼入多模态内容；本阶段先支持文本 + 可选图片路径提示
    content: str = prompt
    if images:
        content = f"{prompt}\n\n[attached_images]={images!r}"
    if video:
        content = f"{content}\n\n[attached_video]={video!r}"

    client = _build_instructor_client(settings)
    result = client.chat.completions.create(
        model=model_name,
        response_model=response_model,
        messages=[{"role": "user", "content": content}],
    )
    return result


def call_text(prompt: str, model: Optional[str] = None) -> str:
    """纯文本调用（如 LLM-as-judge）。禁止传入 groundtruth。

    Args:
        prompt: 文本提示（调用方须确保不含 groundtruth）。
        model: 模型名；None 时从配置读取。

    Returns:
        模型返回的纯文本。
    """
    settings = get_settings()
    _ensure_real_api_allowed(settings)
    model_name = _resolve_model(model, settings)

    client = _build_instructor_client(settings)
    # 用简单字符串模型包装，保证仍走 Instructor 校验路径
    class _TextResponse(BaseModel):
        text: str

    result = client.chat.completions.create(
        model=model_name,
        response_model=_TextResponse,
        messages=[{"role": "user", "content": prompt}],
    )
    return result.text
