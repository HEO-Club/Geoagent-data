"""llm_query 共享执行器：外部模型咨询与候选枚举。

复用 Anthropic Messages / OpenAI 兼容协议与 LLM_* 环境变量，不 import pipeline.llm。
回执一律标为模型建议，不得伪装成地图库或搜索引擎事实。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tool._gate import allow_real_api as _allow_real_api, load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext, declared_inputs

_OP_CONSULT = "consult"
_OP_ENUMERATE = "enumerate"
_DEFAULT_MAX_ITEMS = 10
_MAX_ITEMS_MIN = 1
_MAX_ITEMS_MAX = 500
_DEFAULT_TIMEOUT_SEC = 60.0
_DEFAULT_MAX_OUTPUT_TOKENS = 8192
_PROVIDER_ANTHROPIC = "anthropic"
_PROVIDER_OPENAI_COMPAT = "openai_compatible"
_PROVIDER_INJECTED = "injected"
_ENDPOINT_OFFICIAL = "official"
_ENDPOINT_RELAY = "relay"
_ENDPOINT_INJECTED = "injected"
_EVIDENCE_KIND = "model_suggestion"
_ANTHROPIC_OFFICIAL_HOST = "api.anthropic.com"
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
_CONSULT_FIELDS = ("prompt", "context", "model")
_ENUMERATE_FIELDS = ("prompt", "item_type", "constraints", "max_items")
_JSON_FENCE_RE = re.compile(
    r"^```(?:json)?\s*(.*?)\s*```$",
    re.IGNORECASE | re.DOTALL,
)
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "body",
        "confirmed_location",
        "confirmed_place",
        "content",
        "full_text",
        "location_confirmed",
        "markdown",
        "raw_content",
        "taken_at",
    }
)
_ASSUMPTIONS = [
    "本回执仅为外部模型建议，不能当作地图数据库、POI 或搜索引擎已经返回的事实",
    "中转别名不能由官方 SDK 文档证明实际对应哪个模型",
]
_CONSULT_SYSTEM = (
    "你是外部咨询模型。只根据用户给出的问题与上下文作答。"
    "你的回答是待检验的假设与建议，不是已核实地名，"
    "也不能当作地图数据库、POI、搜索引擎或档案已经返回的事实。"
    "不要编造坐标、URL 或声称已检索外部数据库。"
)
_ENUMERATE_SYSTEM = (
    "你是外部咨询模型，用于生成待检验候选清单。"
    '只输出 JSON 对象，格式为 {"candidates":[{"text":"候选名称或描述","rationale":"简短理由"}]}。'
    "不要输出 Markdown 或其它说明。条目数不超过用户指定的上限。"
    "每条都是待检验假设，不是已核实地名，"
    "也不能当作地图数据库、POI 或搜索引擎已经返回的事实。"
)

class QueryInputError(Exception):
    """prompt / item_type / max_items 等输入无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """真实外部模型未配置、被闸门拒绝或调用失败。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code

class InvalidResponseError(Exception):
    """外部模型回执无法按 enumerate 合同解析。"""

    def __init__(self, message: str, error_code: str = "invalid_response") -> None:
        super().__init__(message)
        self.error_code = error_code

@dataclass(frozen=True)
class LlmQueryRequest:
    """组装后发给外部模型的请求。"""

    operation: str
    prompt: str
    model: str
    system: str
    user: str
    context: str | None = None
    item_type: str | None = None
    constraints: str | None = None
    max_items: int | None = None

@dataclass(frozen=True)
class LlmQueryResponse:
    """外部模型原文及调用元信息。"""

    text: str
    requested_model: str
    response_model: str
    provider: str
    endpoint_kind: str

class EnumeratedItem(BaseModel):
    """enumerate 单条候选。"""

    model_config = ConfigDict(extra="ignore")

    text: str = ""
    rationale: str = ""

    @field_validator("text", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        return str(value)

    @field_validator("rationale", mode="before")
    @classmethod
    def _coerce_rationale(cls, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

class EnumeratePayload(BaseModel):
    """enumerate 结构化回执。"""

    model_config = ConfigDict(extra="ignore")

    candidates: list[EnumeratedItem] = Field(default_factory=list)

    @field_validator("candidates", mode="before")
    @classmethod
    def _coerce_candidates(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("candidates 必须是列表")
        coerced: list[Any] = []
        for item in value:
            if isinstance(item, str):
                coerced.append({"text": item})
            else:
                coerced.append(item)
        return coerced

@runtime_checkable
class ExternalLlmClient(Protocol):
    """可注入的外部模型客户端；测试用 extras['llm_query_client'] 替换。"""

    def complete(self, request: LlmQueryRequest) -> LlmQueryResponse:
        """提交组装后的 prompt，返回原文与调用元信息。"""

class AnthropicMessagesClient:
    """Anthropic Messages API；官方主机标 official，其余 HTTPS 中转标 relay。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        default_model: str,
        timeout_sec: float,
        max_output_tokens: int,
        endpoint_kind: str,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self.default_model = default_model
        self._timeout_sec = timeout_sec
        self._max_output_tokens = max_output_tokens
        self.endpoint_kind = endpoint_kind
        self.provider = _PROVIDER_ANTHROPIC

    def complete(self, request: LlmQueryRequest) -> LlmQueryResponse:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise EngineUnavailableError("未安装 anthropic SDK") from exc
        client = Anthropic(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout_sec,
        )
        try:
            message = client.messages.create(
                model=request.model,
                max_tokens=self._max_output_tokens,
                system=request.system,
                messages=[{"role": "user", "content": request.user}],
            )
        except Exception as exc:
            raise EngineUnavailableError(f"Anthropic 调用失败: {exc}") from exc
        text = _anthropic_text(message)
        response_model = str(getattr(message, "model", "") or request.model)
        return LlmQueryResponse(
            text=text,
            requested_model=request.model,
            response_model=response_model,
            provider=self.provider,
            endpoint_kind=self.endpoint_kind,
        )

class OpenAICompatibleClient:
    """OpenAI Chat Completions 兼容网关；一律标 relay，不声称官方模型。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        default_model: str,
        timeout_sec: float,
        max_output_tokens: int,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self.default_model = default_model
        self._timeout_sec = timeout_sec
        self._max_output_tokens = max_output_tokens
        self.provider = _PROVIDER_OPENAI_COMPAT
        self.endpoint_kind = _ENDPOINT_RELAY

    def complete(self, request: LlmQueryRequest) -> LlmQueryResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EngineUnavailableError("未安装 openai SDK") from exc
        client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout_sec,
        )
        messages: list[dict[str, str]] = []
        if request.system.strip():
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.user})
        try:
            completion = client.chat.completions.create(
                model=request.model,
                max_tokens=self._max_output_tokens,
                messages=messages,
            )
        except Exception as exc:
            raise EngineUnavailableError(f"OpenAI 兼容调用失败: {exc}") from exc
        text = _openai_text(completion)
        response_model = str(getattr(completion, "model", "") or request.model)
        return LlmQueryResponse(
            text=text,
            requested_model=request.model,
            response_model=response_model,
            provider=self.provider,
            endpoint_kind=self.endpoint_kind,
        )

def execute_consult(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """把明确问题及可选上下文提交给外部模型。"""

    del purpose
    return _run(_OP_CONSULT, inputs, ctx)

def execute_enumerate(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """要求外部模型生成候选或结构化清单。"""

    del purpose
    return _run(_OP_ENUMERATE, inputs, ctx)

def anthropic_endpoint_kind(base_url: str) -> str:
    """官方 api.anthropic.com 为 official，其余中转为 relay。"""

    host = (urlparse(base_url).hostname or "").lower()
    if host == _ANTHROPIC_OFFICIAL_HOST:
        return _ENDPOINT_OFFICIAL
    return _ENDPOINT_RELAY

def normalize_anthropic_base_url(base_url: str) -> str:
    """Anthropic SDK 会追加 /v1/messages，因此保存站点根地址。"""

    value = base_url.strip().rstrip("/")
    if value.lower().endswith("/v1"):
        value = value[:-3].rstrip("/")
    return value

def _run(
    operation: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    if operation == _OP_ENUMERATE:
        inputs = declared_inputs(inputs, *_ENUMERATE_FIELDS)
    else:
        inputs = declared_inputs(inputs, *_CONSULT_FIELDS)
    try:
        prompt = _parse_prompt(inputs.get("prompt"))
        context = (
            _stringify_payload(inputs.get("context"), field="context")
            if operation == _OP_CONSULT
            else None
        )
        item_type = (
            _parse_item_type(inputs.get("item_type"))
            if operation == _OP_ENUMERATE
            else None
        )
        constraints = (
            _stringify_payload(inputs.get("constraints"), field="constraints")
            if operation == _OP_ENUMERATE
            else None
        )
        max_items = (
            _parse_max_items(inputs.get("max_items"))
            if operation == _OP_ENUMERATE
            else None
        )
        requested_model = (
            _parse_optional_model(inputs.get("model"))
            if operation == _OP_CONSULT
            else None
        )
        client = _resolve_client(ctx)
        model = _resolve_model(client, requested_model)
        request = LlmQueryRequest(
            operation=operation,
            prompt=prompt,
            model=model,
            system=_system_prompt(operation),
            user=_user_prompt(
                operation,
                prompt=prompt,
                context=context,
                item_type=item_type,
                constraints=constraints,
                max_items=max_items,
            ),
            context=context,
            item_type=item_type,
            constraints=constraints,
            max_items=max_items,
        )
        payload = client.complete(request)
    except QueryInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

    try:
        return _observation(operation, request, payload)
    except InvalidResponseError as exc:
        return _fail(str(exc), exc.error_code)

def _observation(
    operation: str,
    request: LlmQueryRequest,
    payload: LlmQueryResponse,
) -> Observation:
    applied: dict[str, Any] = {
        "provider": payload.provider,
        "requested_model": payload.requested_model,
        "response_model": payload.response_model,
        "endpoint_kind": payload.endpoint_kind,
    }
    if operation == _OP_ENUMERATE:
        applied["item_type"] = request.item_type
        applied["max_items"] = request.max_items
        candidates = _parse_candidates(payload.text, max_items=request.max_items or _DEFAULT_MAX_ITEMS)
        result: dict[str, Any] = {
            "operation": operation,
            "item_type": request.item_type,
            "candidates": candidates,
            "evidence_kind": _EVIDENCE_KIND,
            "applied": applied,
            "assumptions": list(_ASSUMPTIONS),
        }
    else:
        suggestion = payload.text.strip()
        if not suggestion:
            raise InvalidResponseError("外部模型未返回可用文本")
        result = {
            "operation": operation,
            "suggestion": suggestion,
            "evidence_kind": _EVIDENCE_KIND,
            "applied": applied,
            "assumptions": list(_ASSUMPTIONS),
        }
    return Observation(ok=True, result=_strip_forbidden(result))

def _parse_candidates(raw_text: str, *, max_items: int) -> list[dict[str, str]]:
    text = raw_text.strip()
    if not text:
        raise InvalidResponseError("外部模型未返回可用候选 JSON")
    payload = _load_enumerate_json(text)
    try:
        parsed = EnumeratePayload.model_validate(payload)
    except ValidationError as exc:
        raise InvalidResponseError(f"外部模型候选 JSON 不符合合同: {exc}") from exc
    candidates: list[dict[str, str]] = []
    for item in parsed.candidates:
        name = item.text.strip()
        if not name:
            continue
        candidates.append(
            {
                "item_id": f"llm_{len(candidates) + 1}",
                "text": name,
                "rationale": item.rationale.strip(),
            }
        )
        if len(candidates) >= max_items:
            break
    return candidates

def _load_enumerate_json(text: str) -> Any:
    fenced = _JSON_FENCE_RE.match(text)
    blob = fenced.group(1).strip() if fenced else text
    try:
        loaded = json.loads(blob)
    except json.JSONDecodeError:
        loaded = _extract_json_value(blob)
    if isinstance(loaded, list):
        return {"candidates": loaded}
    if isinstance(loaded, dict):
        return loaded
    raise InvalidResponseError("外部模型候选 JSON 必须是对象或列表")

def _extract_json_value(text: str) -> Any:
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise InvalidResponseError("外部模型回执不是合法 JSON")

def _parse_prompt(raw: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise QueryInputError("缺少必填输入 prompt", "missing_input")
    if not isinstance(raw, str):
        raise QueryInputError("prompt 必须是字符串", "invalid_prompt")
    return raw.strip()

def _parse_item_type(raw: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise QueryInputError("缺少必填输入 item_type", "missing_input")
    if not isinstance(raw, str):
        raise QueryInputError("item_type 必须是字符串", "invalid_item_type")
    return raw.strip()

def _parse_optional_model(raw: Any) -> str | None:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise QueryInputError("model 必须是字符串", "invalid_model")
    text = raw.strip()
    return text or None

def _parse_max_items(raw: Any) -> int:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DEFAULT_MAX_ITEMS
    if isinstance(raw, bool):
        raise QueryInputError("max_items 必须是整数", "invalid_max_items")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise QueryInputError("max_items 必须是整数", "invalid_max_items") from exc
    return max(_MAX_ITEMS_MIN, min(_MAX_ITEMS_MAX, value))

def _stringify_payload(raw: Any, *, field: str) -> str | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        text = raw.strip()
        return text or None
    if isinstance(raw, (dict, list)):
        return json.dumps(raw, ensure_ascii=False)
    raise QueryInputError(f"{field} 必须是字符串或对象", f"invalid_{field}")

def _system_prompt(operation: str) -> str:
    if operation == _OP_ENUMERATE:
        return _ENUMERATE_SYSTEM
    return _CONSULT_SYSTEM

def _user_prompt(
    operation: str,
    *,
    prompt: str,
    context: str | None,
    item_type: str | None,
    constraints: str | None,
    max_items: int | None,
) -> str:
    parts = [prompt]
    if operation == _OP_CONSULT:
        if context:
            parts.append(f"上下文：\n{context}")
        return "\n\n".join(parts)
    parts.append(f"对象类型：{item_type}")
    if constraints:
        parts.append(f"约束：\n{constraints}")
    parts.append(f"最多返回 {max_items} 条。")
    return "\n\n".join(parts)

def _resolve_model(client: ExternalLlmClient, requested: str | None) -> str:
    if requested:
        return requested
    default = str(getattr(client, "default_model", "") or "").strip()
    if default:
        return default
    raise EngineUnavailableError("未配置外部模型名：请设置 LLM_QUERY 或 LLM_* MODEL")

def _resolve_client(ctx: RuntimeContext | None) -> ExternalLlmClient:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("llm_query_client")
    if injected is not None:
        if not isinstance(injected, ExternalLlmClient):
            raise EngineUnavailableError("llm_query_client 必须提供 complete(request)")
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError("ALLOW_REAL_API=false，禁止调用真实外部模型 API")
    provider = _normalize_provider(
        _env_value("LLM_QUERY_PROVIDER") or _env_value("LLM_PROVIDER")
    )
    timeout_sec = _env_timeout()
    max_tokens = _env_max_tokens()
    if provider == _PROVIDER_ANTHROPIC:
        return _build_anthropic_client(timeout_sec=timeout_sec, max_tokens=max_tokens)
    if provider in _OPENAI_COMPAT_PROVIDERS:
        return _build_openai_client(timeout_sec=timeout_sec, max_tokens=max_tokens)
    if not provider:
        raise EngineUnavailableError("未配置 LLM_QUERY_PROVIDER 或 LLM_PROVIDER")
    raise EngineUnavailableError(f"llm_query 暂不支持 provider={provider}")

def _build_anthropic_client(
    *,
    timeout_sec: float,
    max_tokens: int,
) -> AnthropicMessagesClient:
    api_key = _env_value("LLM_ANTHROPIC_API_KEY") or _env_value("ANTHROPIC_API_KEY")
    if not api_key:
        raise EngineUnavailableError("未配置 LLM_ANTHROPIC_API_KEY 或 ANTHROPIC_API_KEY")
    bases = _split_csv(_env_value("LLM_ANTHROPIC_BASE_URLS"))
    raw_base = bases[0] if bases else ""
    if not raw_base:
        raise EngineUnavailableError("未配置 LLM_ANTHROPIC_BASE_URLS")
    if _is_insecure_url(raw_base) and not _allow_insecure():
        raise EngineUnavailableError("明文 HTTP 端点默认禁用")
    base_url = normalize_anthropic_base_url(raw_base)
    model = _env_value("LLM_ANTHROPIC_MODEL")
    if not model:
        raise EngineUnavailableError("未配置 LLM_ANTHROPIC_MODEL")
    return AnthropicMessagesClient(
        api_key=api_key,
        base_url=base_url,
        default_model=model,
        timeout_sec=timeout_sec,
        max_output_tokens=max_tokens,
        endpoint_kind=anthropic_endpoint_kind(base_url),
    )

def _build_openai_client(
    *,
    timeout_sec: float,
    max_tokens: int,
) -> OpenAICompatibleClient:
    api_key = (
        _env_value("DASHSCOPE_API_KEY")
        or _env_value("OPENAI_API_KEY")
        or _env_value("MOONSHOT_API_KEY")
    )
    if not api_key:
        raise EngineUnavailableError(
            "未配置 DASHSCOPE_API_KEY、OPENAI_API_KEY 或 MOONSHOT_API_KEY"
        )
    base_url = _env_value("LLM_BASE_URL")
    if not base_url:
        raise EngineUnavailableError("未配置 LLM_BASE_URL")
    if _is_insecure_url(base_url) and not _allow_insecure():
        raise EngineUnavailableError("明文 HTTP 端点默认禁用")
    model = _env_value("LLM_MODEL")
    if not model:
        raise EngineUnavailableError("未配置 LLM_MODEL")
    return OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        default_model=model,
        timeout_sec=timeout_sec,
        max_output_tokens=max_tokens,
    )

def _anthropic_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            if block:
                parts.append(block)
            continue
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
            text = block.get("text")
        else:
            text = getattr(block, "text", None)
        if block_type == "text" and isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)

def _openai_text(completion: Any) -> str:
    choices = getattr(completion, "choices", None)
    if not isinstance(choices, list) or not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if isinstance(content, str):
        return content
    return ""

def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    return "anthropic" if normalized == "claude" else normalized

def _split_csv(value: str) -> list[str]:
    result: list[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if item and item not in result:
            result.append(item)
    return result

def _is_insecure_url(base_url: str) -> bool:
    return base_url.strip().lower().startswith("http://")

def _allow_insecure() -> bool:
    raw = os.environ.get("ALLOW_INSECURE_LLM_ENDPOINTS", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}

def _env_value(name: str, default: str = "") -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default

def _env_timeout() -> float:
    for name in ("LLM_QUERY_TIMEOUT_SEC", "LLM_TIMEOUT_SEC"):
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return _DEFAULT_TIMEOUT_SEC

def _env_max_tokens() -> int:
    raw = os.environ.get("LLM_MAX_OUTPUT_TOKENS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return _DEFAULT_MAX_OUTPUT_TOKENS

def _strip_forbidden(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_forbidden(item)
            for key, item in value.items()
            if key not in _FORBIDDEN_KEYS
        }
    if isinstance(value, list):
        return [_strip_forbidden(item) for item in value]
    return value

def _fail(message: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=message, error_code=error_code)
