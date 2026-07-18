"""LLM adapter：经 Instructor 封装结构化输出，禁止业务层直调 SDK。

默认提供方为通义千问（DashScope OpenAI 兼容接口）；仍保留 Gemini 可选。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

from pydantic import BaseModel

from pipeline.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)

_IMAGE_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


class RealAPIDisabledError(RuntimeError):
    """ALLOW_REAL_API=false 时拒绝真实网络调用。"""


def _resolve_model(model: Optional[str], settings: Settings) -> str:
    """解析实际使用的模型名；仅允许来自参数或配置。"""
    if model is not None and model.strip():
        return model.strip()
    # 主配置 LLM_MODEL；兼容旧 GEMINI_MODEL（provider=gemini 时）
    if settings.LLM_MODEL.strip():
        return settings.LLM_MODEL.strip()
    if settings.LLM_PROVIDER.lower() == "gemini" and settings.GEMINI_MODEL.strip():
        return settings.GEMINI_MODEL.strip()
    raise ValueError(
        "模型名未配置：请设置 LLM_MODEL（或 Gemini 下的 GEMINI_MODEL），或传入 model 参数"
    )


def _ensure_real_api_allowed(settings: Settings) -> None:
    if not settings.ALLOW_REAL_API:
        raise RealAPIDisabledError(
            "ALLOW_REAL_API=false，禁止真实付费 API 调用；测试请 mock adapter"
        )


def _image_to_data_url(image_path: str) -> str:
    """本地图片 → data URL；按配置缩小后以 JPEG 编码，降低请求体。"""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    settings = get_settings()
    max_side = max(64, int(settings.LLM_IMAGE_MAX_SIDE))
    quality = min(95, max(40, int(settings.LLM_IMAGE_JPEG_QUALITY)))

    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            scale = min(1.0, float(max_side) / float(max(w, h)))
            if scale < 1.0:
                img = img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.Resampling.BILINEAR,
                )
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        # 回退：原样上传
        mime = _IMAGE_MIME.get(path.suffix.lower(), "image/jpeg")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"


def _build_user_content(
    prompt: str,
    images: Optional[list[str]],
    video: Optional[str],
    *,
    multimodal: bool,
) -> Union[str, list[dict[str, Any]]]:
    """构造 user message content；multimodal=True 时嵌入图片 data URL。"""
    if not multimodal or not images:
        content = prompt
        if images:
            content = f"{prompt}\n\n[attached_images]={images!r}"
        if video:
            content = f"{content}\n\n[attached_video]={video!r}"
        return content

    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if video:
        parts[0]["text"] = f"{prompt}\n\n[attached_video]={video!r}"
    for image_path in images:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(image_path)},
            }
        )
    return parts


def _build_openai_compatible_client(settings: Settings):  # type: ignore[no-untyped-def]
    """DashScope / 其他 OpenAI 兼容端点。

    Qwen 对 Instructor 默认 TOOLS/function-calling 支持不完整，使用 JSON 模式更稳。
    """
    import instructor
    from openai import OpenAI

    api_key = settings.DASHSCOPE_API_KEY or settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError(
            "未配置 API Key：请设置 DASHSCOPE_API_KEY（推荐）或 OPENAI_API_KEY"
        )
    base_url = settings.LLM_BASE_URL.strip()
    if not base_url:
        raise ValueError("LLM_BASE_URL 未配置")
    raw = OpenAI(api_key=api_key, base_url=base_url)
    return instructor.from_openai(raw, mode=instructor.Mode.JSON)


def _build_gemini_client(settings: Settings):  # type: ignore[no-untyped-def]
    """Google GenAI（Gemini）。"""
    import instructor
    from google import genai

    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY 未配置")
    raw = genai.Client(api_key=settings.GOOGLE_API_KEY)
    return instructor.from_genai(raw)


def _build_instructor_client(settings: Settings):  # type: ignore[no-untyped-def]
    """按 LLM_PROVIDER 构建 Instructor 客户端；测试可 patch 本函数。"""
    provider = settings.LLM_PROVIDER.strip().lower()
    if provider in {"qwen", "dashscope", "openai_compatible", "openai"}:
        return _build_openai_compatible_client(settings)
    if provider == "gemini":
        return _build_gemini_client(settings)
    raise ValueError(
        f"不支持的 LLM_PROVIDER={settings.LLM_PROVIDER!r}；"
        f"可选: qwen / dashscope / gemini"
    )


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
    provider = settings.LLM_PROVIDER.strip().lower()
    multimodal = provider in {"qwen", "dashscope", "openai_compatible", "openai"}

    content = _build_user_content(
        prompt, images, video, multimodal=multimodal
    )
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

    class _TextResponse(BaseModel):
        text: str

    result = client.chat.completions.create(
        model=model_name,
        response_model=_TextResponse,
        messages=[{"role": "user", "content": prompt}],
    )
    return result.text
