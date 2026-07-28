"""LLM adapter：经 Instructor 封装结构化输出，禁止业务层直调 SDK。

双通道：
- lane=vlm：stage3 之前（默认 DashScope / Qwen）
- lane=llm：stage3+ 主通道（默认 Moonshot / kimi-k3，亦支持多模态图片）
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, TypeVar, Union

from pydantic import BaseModel

from pipeline.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)

Lane = Literal["vlm", "llm"]

logger = logging.getLogger(__name__)

# Instructor 的 max_retries 只管 schema 校验；网络/限流需自行重试
_LLM_TRANSIENT_ATTEMPTS = 5
_TRANSIENT_MARKERS: tuple[str, ...] = (
    "connection error",
    "connection reset",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "429",
    "rate limit",
    "too many requests",
    "502",
    "503",
    "504",
)

_IMAGE_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

_OPENAI_COMPAT_PROVIDERS = frozenset(
    {
        "qwen",
        "dashscope",
        "openai_compatible",
        "openai",
        "kimi",
        "moonshot",
    }
)


class RealAPIDisabledError(RuntimeError):
    """ALLOW_REAL_API=false 时拒绝真实网络调用。"""


@dataclass(frozen=True)
class _EndpointConfig:
    """一次调用解析出的端点与模型参数。"""

    provider: str
    base_url: str
    model: str
    api_key: str
    timeout_sec: float
    reasoning_effort: Optional[str]
    omit_sampling_params: bool
    multimodal: bool


def _is_transient_llm_error(exc: BaseException) -> bool:
    """判断是否为可重试的瞬时 LLM/网络错误。"""
    name = type(exc).__name__.lower()
    if any(tok in name for tok in ("timeout", "connection", "ratelimit", "apierror")):
        pass
    blob = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in blob for marker in _TRANSIENT_MARKERS)


def _ensure_real_api_allowed(settings: Settings) -> None:
    if not settings.ALLOW_REAL_API:
        raise RealAPIDisabledError(
            "ALLOW_REAL_API=false，禁止真实付费 API 调用；测试请 mock adapter"
        )


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower()


def _resolve_endpoint(
    lane: Lane,
    model: Optional[str],
    settings: Settings,
) -> _EndpointConfig:
    """按 lane 解析 provider / base_url / model / key 与 Kimi 特殊约束。"""
    if lane == "vlm":
        provider = _normalize_provider(settings.VLM_PROVIDER)
        base_url = settings.VLM_BASE_URL.strip()
        default_model = settings.VLM_MODEL.strip()
    else:
        provider = _normalize_provider(settings.LLM_PROVIDER)
        base_url = settings.LLM_BASE_URL.strip()
        default_model = settings.LLM_MODEL.strip()

    model_name = (model or "").strip() or default_model
    if not model_name:
        if provider == "gemini" and settings.GEMINI_MODEL.strip():
            model_name = settings.GEMINI_MODEL.strip()
        else:
            raise ValueError(
                f"模型名未配置：请设置 {'VLM_MODEL' if lane == 'vlm' else 'LLM_MODEL'}，"
                "或传入 model 参数"
            )

    is_kimi = provider in {"kimi", "moonshot"}
    if is_kimi:
        api_key = (
            settings.MOONSHOT_API_KEY
            or settings.OPENAI_API_KEY
            or settings.DASHSCOPE_API_KEY
        )
        if not base_url:
            base_url = "https://api.moonshot.cn/v1"
        effort = settings.KIMI_REASONING_EFFORT.strip().lower() or "low"
        if effort not in {"low", "high", "max"}:
            raise ValueError(
                f"KIMI_REASONING_EFFORT 无效: {settings.KIMI_REASONING_EFFORT!r}；"
                "可选 low / high / max"
            )
        return _EndpointConfig(
            provider=provider,
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            timeout_sec=float(settings.LLM_TIMEOUT_SEC),
            reasoning_effort=effort,
            omit_sampling_params=True,
            multimodal=True,
        )

    if provider in {"qwen", "dashscope", "openai_compatible", "openai"}:
        api_key = settings.DASHSCOPE_API_KEY or settings.OPENAI_API_KEY
        if not base_url:
            raise ValueError(
                f"{'VLM_BASE_URL' if lane == 'vlm' else 'LLM_BASE_URL'} 未配置"
            )
        return _EndpointConfig(
            provider=provider,
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            timeout_sec=float(settings.LLM_TIMEOUT_SEC),
            reasoning_effort=None,
            omit_sampling_params=False,
            multimodal=True,
        )

    if provider == "gemini":
        return _EndpointConfig(
            provider=provider,
            base_url="",
            model=model_name,
            api_key=settings.GOOGLE_API_KEY,
            timeout_sec=float(settings.LLM_TIMEOUT_SEC),
            reasoning_effort=None,
            omit_sampling_params=False,
            multimodal=False,
        )

    raise ValueError(
        f"不支持的 provider={provider!r}（lane={lane}）；"
        f"可选: qwen / dashscope / kimi / moonshot / gemini"
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


def _build_openai_compatible_client(endpoint: _EndpointConfig):  # type: ignore[no-untyped-def]
    """DashScope / Moonshot / 其他 OpenAI 兼容端点。

    Qwen 与 Kimi 对 Instructor 默认 TOOLS 支持不完整，使用 JSON 模式更稳。
    Kimi K3：禁止传 temperature 等采样参数（由调用侧保证）。
    """
    import instructor
    from openai import OpenAI

    if not endpoint.api_key:
        raise ValueError(
            "未配置 API Key：kimi 请设 MOONSHOT_API_KEY；"
            "qwen 请设 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）"
        )
    if not endpoint.base_url:
        raise ValueError("LLM/VLM BASE_URL 未配置")
    raw = OpenAI(
        api_key=endpoint.api_key,
        base_url=endpoint.base_url,
        max_retries=5,
        timeout=endpoint.timeout_sec,
    )
    return instructor.from_openai(raw, mode=instructor.Mode.JSON)


def _build_gemini_client(endpoint: _EndpointConfig):  # type: ignore[no-untyped-def]
    """Google GenAI（Gemini）。"""
    import instructor
    from google import genai

    if not endpoint.api_key:
        raise ValueError("GOOGLE_API_KEY 未配置")
    raw = genai.Client(api_key=endpoint.api_key)
    return instructor.from_genai(raw)


def _build_instructor_client(endpoint: _EndpointConfig):  # type: ignore[no-untyped-def]
    """按 endpoint.provider 构建 Instructor 客户端；测试可 patch 本函数。"""
    provider = endpoint.provider
    if provider in _OPENAI_COMPAT_PROVIDERS:
        return _build_openai_compatible_client(endpoint)
    if provider == "gemini":
        return _build_gemini_client(endpoint)
    raise ValueError(
        f"不支持的 provider={provider!r}；"
        f"可选: qwen / dashscope / kimi / moonshot / gemini"
    )


def _create_structured(
    client: Any,
    *,
    endpoint: _EndpointConfig,
    response_model: type[T],
    content: Union[str, list[dict[str, Any]]],
) -> T:
    """发起一次结构化调用；Kimi 省略采样参数并附带 reasoning_effort。"""
    kwargs: dict[str, Any] = {
        "model": endpoint.model,
        "response_model": response_model,
        "messages": [{"role": "user", "content": content}],
        "max_retries": 3,
    }
    if endpoint.reasoning_effort is not None:
        kwargs["reasoning_effort"] = endpoint.reasoning_effort
    # omit_sampling_params=True 时绝不传 temperature/top_p/n/penalties
    return client.chat.completions.create(**kwargs)


def call_structured(
    prompt: str,
    response_model: type[T],
    images: Optional[list[str]] = None,
    video: Optional[str] = None,
    model: Optional[str] = None,
    *,
    lane: Lane = "llm",
) -> T:
    """调用 LLM 并强制返回 response_model；不合法由 Instructor 自动重试。

    Args:
        prompt: 文本提示。
        response_model: 期望的 Pydantic 模型类型。
        images: 可选图片路径列表（两通道均可多模态传入）。
        video: 可选视频路径。
        model: 模型名；None 时按 lane 从配置读取。
        lane: ``vlm``=stage3 前视觉通道；``llm``=stage3+ 主通道。

    Returns:
        经 Pydantic 校验的结构化结果。
    """
    settings = get_settings()
    _ensure_real_api_allowed(settings)
    endpoint = _resolve_endpoint(lane, model, settings)

    content = _build_user_content(
        prompt, images, video, multimodal=endpoint.multimodal
    )
    client = _build_instructor_client(endpoint)
    last_error: Optional[BaseException] = None
    for attempt in range(1, _LLM_TRANSIENT_ATTEMPTS + 1):
        try:
            return _create_structured(
                client,
                endpoint=endpoint,
                response_model=response_model,
                content=content,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if not _is_transient_llm_error(exc) or attempt >= _LLM_TRANSIENT_ATTEMPTS:
                raise
            delay = min(2.0**attempt, 20.0)
            logger.warning(
                "call_structured transient failure attempt %s/%s: %s; sleep %.1fs",
                attempt,
                _LLM_TRANSIENT_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def call_text(
    prompt: str,
    model: Optional[str] = None,
    *,
    lane: Lane = "llm",
) -> str:
    """纯文本调用（如 LLM-as-judge）。禁止传入 groundtruth。

    Args:
        prompt: 文本提示（调用方须确保不含 groundtruth）。
        model: 模型名；None 时按 lane 从配置读取。
        lane: ``vlm`` 或 ``llm``。

    Returns:
        模型返回的纯文本。
    """
    settings = get_settings()
    _ensure_real_api_allowed(settings)
    endpoint = _resolve_endpoint(lane, model, settings)

    client = _build_instructor_client(endpoint)

    class _TextResponse(BaseModel):
        text: str

    last_error: Optional[BaseException] = None
    for attempt in range(1, _LLM_TRANSIENT_ATTEMPTS + 1):
        try:
            result = _create_structured(
                client,
                endpoint=endpoint,
                response_model=_TextResponse,
                content=prompt,
            )
            return result.text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if not _is_transient_llm_error(exc) or attempt >= _LLM_TRANSIENT_ATTEMPTS:
                raise
            delay = min(2.0**attempt, 20.0)
            logger.warning(
                "call_text transient failure attempt %s/%s: %s; sleep %.1fs",
                attempt,
                _LLM_TRANSIENT_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error
