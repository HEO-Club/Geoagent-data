"""LLM adapter：Instructor 结构化输出；业务层禁止直调 SDK。"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from pipeline.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)
Lane = Literal["asr", "vlm", "llm"]

logger = logging.getLogger(__name__)

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
    max_output_tokens: int
    reasoning_effort: str | None
    omit_sampling_params: bool
    multimodal: bool


def _is_transient_llm_error(exc: BaseException) -> bool:
    """判断是否为可重试的瞬时 LLM/网络错误。"""
    name = type(exc).__name__.lower()
    if any(tok in name for tok in ("timeout", "connection", "ratelimit", "apierror")):
        return True
    blob = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in blob for marker in _TRANSIENT_MARKERS)


def _ensure_real_api_allowed(settings: Settings) -> None:
    if not settings.ALLOW_REAL_API:
        raise RealAPIDisabledError(
            "ALLOW_REAL_API=false，禁止真实付费 API 调用；测试请 mock adapter"
        )


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    return "anthropic" if normalized == "claude" else normalized


def _split_csv(value: str) -> list[str]:
    """解析逗号分隔配置，去空并按原顺序去重。"""
    result: list[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if item and item not in result:
            result.append(item)
    return result


def _normalize_anthropic_base_url(base_url: str) -> str:
    """Anthropic SDK 会追加 /v1/messages，因此保存站点根地址。"""
    value = base_url.strip().rstrip("/")
    if value.lower().endswith("/v1"):
        value = value[:-3].rstrip("/")
    return value


def _is_insecure_url(base_url: str) -> bool:
    return base_url.strip().lower().startswith("http://")


def _resolve_endpoint(
    lane: Lane,
    model: str | None,
    settings: Settings,
) -> _EndpointConfig:
    """按 lane 解析 provider / base_url / model / key。"""
    if lane == "asr":
        provider = _normalize_provider(settings.ASR_PROVIDER)
        base_url = settings.ASR_BASE_URL.strip()
        default_model = settings.ASR_MODEL.strip()
    elif lane == "vlm":
        provider = _normalize_provider(settings.VLM_PROVIDER)
        base_url = settings.VLM_BASE_URL.strip()
        default_model = settings.VLM_MODEL.strip()
    else:
        provider = _normalize_provider(settings.LLM_PROVIDER)
        if provider == "anthropic":
            bases = _split_csv(settings.LLM_ANTHROPIC_BASE_URLS)
            base_url = bases[0] if bases else settings.LLM_BASE_URL.strip()
            default_model = settings.LLM_ANTHROPIC_MODEL.strip()
        else:
            base_url = settings.LLM_BASE_URL.strip()
            default_model = settings.LLM_MODEL.strip()

    model_name = (model or "").strip() or default_model
    if not model_name:
        if provider == "gemini" and settings.GEMINI_MODEL.strip():
            model_name = settings.GEMINI_MODEL.strip()
        else:
            field = {"asr": "ASR_MODEL", "vlm": "VLM_MODEL", "llm": "LLM_MODEL"}[lane]
            raise ValueError(
                f"模型名未配置：请设置 {field}"
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
                f"KIMI_REASONING_EFFORT 无效: {settings.KIMI_REASONING_EFFORT!r}"
            )
        return _EndpointConfig(
            provider=provider,
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            timeout_sec=float(settings.LLM_TIMEOUT_SEC),
            max_output_tokens=int(settings.LLM_MAX_OUTPUT_TOKENS),
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
            max_output_tokens=int(settings.LLM_MAX_OUTPUT_TOKENS),
            reasoning_effort=None,
            omit_sampling_params=False,
            multimodal=True,
        )

    if provider == "anthropic":
        normalized_base = _normalize_anthropic_base_url(base_url)
        if not normalized_base:
            raise ValueError("LLM_ANTHROPIC_BASE_URLS 未配置")
        if _is_insecure_url(normalized_base) and not settings.ALLOW_INSECURE_LLM_ENDPOINTS:
            raise ValueError("Anthropic 明文 HTTP 端点默认禁用")
        return _EndpointConfig(
            provider=provider,
            base_url=normalized_base,
            model=model_name,
            api_key=(
                settings.LLM_ANTHROPIC_API_KEY or settings.ANTHROPIC_API_KEY
            ),
            timeout_sec=float(settings.LLM_TIMEOUT_SEC),
            max_output_tokens=int(settings.LLM_MAX_OUTPUT_TOKENS),
            reasoning_effort=None,
            omit_sampling_params=True,
            multimodal=True,
        )

    if provider == "gemini":
        return _EndpointConfig(
            provider=provider,
            base_url="",
            model=model_name,
            api_key=settings.GOOGLE_API_KEY,
            timeout_sec=float(settings.LLM_TIMEOUT_SEC),
            max_output_tokens=int(settings.LLM_MAX_OUTPUT_TOKENS),
            reasoning_effort=None,
            omit_sampling_params=False,
            multimodal=False,
        )

    raise ValueError(f"不支持的 provider={provider!r}（lane={lane}）")


def _image_to_base64(image_path: str) -> tuple[str, str]:
    """本地图片 → (MIME, base64)，并按配置缩放压缩。"""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    settings = get_settings()
    max_side = max(64, int(settings.LLM_IMAGE_MAX_SIDE))
    quality = min(95, max(40, int(settings.LLM_IMAGE_JPEG_QUALITY)))

    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(path) as source:
            img = source.convert("RGB")
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
            return "image/jpeg", encoded
    except Exception:  # noqa: BLE001 - Pillow 失败时需回退为原始图片字节
        mime = _IMAGE_MIME.get(path.suffix.lower(), "image/jpeg")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return mime, encoded


def _image_to_data_url(image_path: str) -> str:
    """本地图片 → OpenAI-compatible data URL。"""
    mime, encoded = _image_to_base64(image_path)
    return f"data:{mime};base64,{encoded}"


def _build_user_content(
    prompt: str,
    images: list[str] | None,
    *,
    provider: str,
    multimodal: bool,
) -> str | list[dict[str, Any]]:
    """构造 user message content。"""
    if not multimodal or not images:
        content = prompt
        if images:
            content = f"{prompt}\n\n[attached_images]={images!r}"
        return content

    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in images:
        if provider == "anthropic":
            mime, encoded = _image_to_base64(image_path)
            parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": encoded,
                    },
                }
            )
        else:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_url(image_path)},
                }
            )
    return parts


def _build_openai_compatible_client(endpoint: _EndpointConfig):  # type: ignore[no-untyped-def]
    import instructor
    from openai import OpenAI

    if not endpoint.api_key:
        raise ValueError("未配置 API Key")
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
    import instructor
    from google import genai

    if not endpoint.api_key:
        raise ValueError("GOOGLE_API_KEY 未配置")
    raw = genai.Client(api_key=endpoint.api_key)
    return instructor.from_genai(raw)


def _build_anthropic_client(endpoint: _EndpointConfig):  # type: ignore[no-untyped-def]
    from anthropic import Anthropic

    if not endpoint.api_key:
        raise ValueError("ANTHROPIC_API_KEY 未配置")
    if not endpoint.base_url:
        raise ValueError("Anthropic BASE_URL 未配置")
    return Anthropic(
        api_key=endpoint.api_key,
        base_url=endpoint.base_url,
        max_retries=0,
        timeout=endpoint.timeout_sec,
    )


def _decode_embedded_json(value: Any) -> Any:
    """递归解开 relay 在 Tool 参数内二次序列化的 JSON 字符串。"""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.startswith(("[", "{")):
            return value
        try:
            return _decode_embedded_json(json.loads(stripped))
        except json.JSONDecodeError:
            return value
    if isinstance(value, list):
        return [_decode_embedded_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_embedded_json(item) for key, item in value.items()}
    return value


def _unwrap_anthropic_tool_input(value: Any) -> Any:
    """兼容部分中转站的 ``$PARAMETER_NAME`` Tool 参数包装。"""
    current = value
    while isinstance(current, dict) and "$PARAMETER_NAME" in current:
        nested = current.get("$PARAMETER_NAME")
        if nested is current:
            break
        current = nested
    return _decode_embedded_json(current)


def _create_anthropic_structured(
    client: Any,
    *,
    endpoint: _EndpointConfig,
    response_model: type[T],
    content: str | list[dict[str, Any]],
) -> T:
    """直接调用 Anthropic Messages，并自行解析 relay ToolUseBlock。"""
    tool_name = response_model.__name__
    kwargs = dict(
        model=endpoint.model,
        max_tokens=endpoint.max_output_tokens,
        messages=[{"role": "user", "content": content}],
        tools=[
            {
                "name": tool_name,
                "description": "Return the requested structured result.",
                "input_schema": response_model.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
    )
    if get_settings().LLM_ANTHROPIC_STREAM:
        # SSE 会在长生成期间持续产生事件，避免 Cloudflare 对无响应体的
        # 非流式请求触发 524；最终仍使用完整 Message 做同一套结构校验。
        with client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
    else:
        response = client.messages.create(**kwargs)
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        payload = _unwrap_anthropic_tool_input(getattr(block, "input", None))
        return response_model.model_validate(payload)
    raise ValueError("Anthropic 响应中缺少 tool_use 结构化结果")


def _build_instructor_client(endpoint: _EndpointConfig):  # type: ignore[no-untyped-def]
    provider = endpoint.provider
    if provider in _OPENAI_COMPAT_PROVIDERS:
        return _build_openai_compatible_client(endpoint)
    if provider == "anthropic":
        return _build_anthropic_client(endpoint)
    if provider == "gemini":
        return _build_gemini_client(endpoint)
    raise ValueError(f"不支持的 provider={provider!r}")


def _create_structured(
    client: Any,
    *,
    endpoint: _EndpointConfig,
    response_model: type[T],
    content: str | list[dict[str, Any]],
) -> T:
    if endpoint.provider == "anthropic":
        return _create_anthropic_structured(
            client,
            endpoint=endpoint,
            response_model=response_model,
            content=content,
        )

    kwargs: dict[str, Any] = {
        "model": endpoint.model,
        "response_model": response_model,
        "messages": [{"role": "user", "content": content}],
        "max_retries": 3,
    }
    if endpoint.reasoning_effort is not None:
        kwargs["reasoning_effort"] = endpoint.reasoning_effort
    return client.chat.completions.create(**kwargs)


def _resolve_call_endpoints(
    lane: Lane,
    model: str | None,
    settings: Settings,
) -> list[_EndpointConfig]:
    """解析调用端点；Anthropic 允许按配置顺序切换 HTTPS 中转站。"""
    endpoint = _resolve_endpoint(lane, model, settings)
    if lane != "llm" or endpoint.provider != "anthropic":
        return [endpoint]

    endpoints: list[_EndpointConfig] = []
    for raw_base in _split_csv(settings.LLM_ANTHROPIC_BASE_URLS):
        base = _normalize_anthropic_base_url(raw_base)
        if not base:
            continue
        if _is_insecure_url(base) and not settings.ALLOW_INSECURE_LLM_ENDPOINTS:
            logger.warning("skip insecure Anthropic endpoint: %s", base)
            continue
        endpoints.append(replace(endpoint, base_url=base))
    if not endpoints:
        raise ValueError("没有可用的 Anthropic HTTPS 端点")
    return endpoints


def _safe_error_summary(exc: BaseException) -> str:
    """只保留异常类型与状态码，不记录请求、响应或密钥。"""
    pieces = [type(exc).__name__]
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        pieces.append(f"status={status_code}")
    return " ".join(pieces)


def _retry_delay(exc: BaseException, attempt: int) -> float:
    """尊重中转站 502 的最小退避要求，其余错误使用短指数退避。"""
    blob = f"{type(exc).__name__}: {exc}".lower()
    retry_after = re.search(r"retry_after['\"]?\s*:\s*(\d+)", blob)
    if retry_after:
        return min(120.0, max(1.0, float(retry_after.group(1))))
    if "524" in blob:
        return 120.0
    if "502" in blob:
        return 60.0
    return min(2.0**attempt, 20.0)


def _audio_to_data_url(audio_path: str) -> str:
    """本地音频 → OpenAI-compatible input_audio data URL。"""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"音频不存在: {audio_path}")
    mime = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".webm": "audio/webm",
    }.get(path.suffix.lower(), "audio/mpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _audio_format(audio_path: str) -> str:
    """返回 Qwen-Omni ``input_audio.format`` 所需的格式名。"""
    suffix = Path(audio_path).suffix.lower().lstrip(".")
    return {"m4a": "mp4", "oga": "ogg"}.get(suffix, suffix or "mp3")


def _omni_delta_text(content: Any) -> str:
    """兼容 OpenAI SDK/网关返回的字符串或分块文本内容。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
        else:
            text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _create_audio_text(
    endpoint: _EndpointConfig,
    audio_path: str,
    *,
    language: str,
) -> str:
    """调用 Qwen ASR，并统一返回纯文本。"""
    from openai import OpenAI

    if not endpoint.api_key:
        raise ValueError("未配置 ASR API Key")
    if not endpoint.base_url:
        raise ValueError("ASR_BASE_URL 未配置")
    if endpoint.provider not in {"qwen", "dashscope", "openai_compatible"}:
        raise ValueError(f"ASR 暂不支持 provider={endpoint.provider!r}")

    raw = OpenAI(
        api_key=endpoint.api_key,
        base_url=endpoint.base_url,
        max_retries=5,
        timeout=endpoint.timeout_sec,
    )
    audio_input = {
        "data": _audio_to_data_url(audio_path),
        "format": _audio_format(audio_path),
    }
    messages: Any = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": audio_input,
                }
            ],
        }
    ]

    # Qwen3.5-Omni 是百炼当前推荐的高能力非实时语音识别模型。
    # 它使用流式 Chat Completions，并通过文本指令提供专名保真上下文；
    # 专用 Qwen-ASR 则继续走原有 asr_options 协议。
    if endpoint.model.lower().startswith("qwen3.5-omni"):
        messages[0]["content"].append(
            {
                "type": "text",
                "text": (
                    "请将这段地理定位讲解音频逐字转写为简体中文。"
                    "完整保留地名、人名、道路、桥梁、村镇、编号、距离和方向信息；"
                    "不总结、不翻译、不纠错推测，只输出转写正文。"
                ),
            }
        )
        stream = raw.chat.completions.create(
            model=endpoint.model,
            messages=messages,
            stream=True,
            extra_body={"modalities": ["text"]},
        )
        chunks: list[str] = []
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            chunks.append(_omni_delta_text(getattr(delta, "content", None)))
        return "".join(chunks).strip()

    asr_options: dict[str, Any] = {"enable_itn": True}
    if language:
        asr_options["language"] = language
    completion = raw.chat.completions.create(
        model=endpoint.model,
        messages=messages,
        stream=False,
        extra_body={"asr_options": asr_options},
    )
    return (completion.choices[0].message.content or "").strip()


def call_audio_text(
    audio_path: str,
    model: str | None = None,
    *,
    language: str | None = None,
) -> str:
    """窗口级音频转录；真实调用受 ALLOW_REAL_API 闸门保护。"""
    settings = get_settings()
    _ensure_real_api_allowed(settings)
    endpoint = _resolve_endpoint("asr", model, settings)
    lang = settings.ASR_LANGUAGE.strip() if language is None else language.strip()

    last_error: BaseException | None = None
    for attempt in range(1, _LLM_TRANSIENT_ATTEMPTS + 1):
        try:
            return _create_audio_text(endpoint, audio_path, language=lang)
        except Exception as exc:
            last_error = exc
            if not _is_transient_llm_error(exc) or attempt >= _LLM_TRANSIENT_ATTEMPTS:
                raise
            delay = _retry_delay(exc, attempt)
            logger.warning(
                "call_audio_text transient failure attempt %s/%s: %s; sleep %.1fs",
                attempt,
                _LLM_TRANSIENT_ATTEMPTS,
                _safe_error_summary(exc),
                delay,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def call_structured(
    prompt: str,
    response_model: type[T],
    images: list[str] | None = None,
    model: str | None = None,
    *,
    lane: Lane = "llm",
) -> T:
    """调用 LLM 并强制返回 response_model。"""
    settings = get_settings()
    _ensure_real_api_allowed(settings)
    endpoints = _resolve_call_endpoints(lane, model, settings)
    last_error: BaseException | None = None
    failures: list[str] = []
    for endpoint in endpoints:
        content = _build_user_content(
            prompt,
            images,
            provider=endpoint.provider,
            multimodal=endpoint.multimodal,
        )
        client = _build_instructor_client(endpoint)
        attempts = 1 if len(endpoints) > 1 else _LLM_TRANSIENT_ATTEMPTS
        for attempt in range(1, attempts + 1):
            try:
                result = _create_structured(
                    client,
                    endpoint=endpoint,
                    response_model=response_model,
                    content=content,
                )
                if len(endpoints) > 1:
                    logger.info(
                        "Anthropic route selected base=%s model=%s",
                        endpoint.base_url,
                        endpoint.model,
                    )
                return result
            except Exception as exc:
                last_error = exc
                summary = _safe_error_summary(exc)
                if len(endpoints) > 1:
                    failures.append(f"{endpoint.base_url} ({summary})")
                    logger.warning(
                        "Anthropic route failed base=%s model=%s error=%s",
                        endpoint.base_url,
                        endpoint.model,
                        summary,
                    )
                    break
                if (
                    not _is_transient_llm_error(exc)
                    or attempt >= _LLM_TRANSIENT_ATTEMPTS
                ):
                    raise
                delay = _retry_delay(exc, attempt)
                logger.warning(
                    "call_structured transient failure attempt %s/%s: %s; sleep %.1fs",
                    attempt,
                    _LLM_TRANSIENT_ATTEMPTS,
                    summary,
                    delay,
                )
                time.sleep(delay)

    if failures:
        raise RuntimeError(
            "所有 Anthropic HTTPS 端点均失败：" + "; ".join(failures)
        ) from last_error
    if last_error is not None:
        raise last_error
    raise RuntimeError("没有可执行的 LLM 端点")


def call_text(
    prompt: str,
    model: str | None = None,
    *,
    lane: Lane = "llm",
) -> str:
    """纯文本调用。禁止传入 groundtruth。"""
    class _TextResponse(BaseModel):
        text: str

    result = call_structured(prompt, _TextResponse, model=model, lane=lane)
    return result.text
