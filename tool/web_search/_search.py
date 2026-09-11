"""web_search 共享执行器：Brave Search Web API，只返回标题、链接与摘要。"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tool._gate import allow_real_api as _allow_real_api, load_tool_dotenv as _load_dotenv
from tool.contract import Observation, RuntimeContext, declared_inputs

_DEFAULT_TOP_K = 10
_TOP_K_MIN = 1
_TOP_K_MAX = 100
_BRAVE_MAX_TOP_K = 20
_ENGINE_BRAVE = "brave"
_DEFAULT_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_DEFAULT_TIMEOUT_SEC = 30.0
_RELATIVE_FRESHNESS = frozenset({"pd", "pw", "pm", "py"})
_YEAR_RE = re.compile(r"^(\d{4})$")
_YEAR_RANGE_RE = re.compile(r"^(\d{4})\s*[-–—至到]\s*(\d{4})$")
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
_DATE_RANGE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s*to\s*(\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)
_LANGUAGE_CODE_RE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]{2,8})?$")
_LANGUAGE_ALIASES = {
    "zh": "zh-hans",
    "zh-cn": "zh-hans",
    "zh-hans": "zh-hans",
    "zh-sg": "zh-hans",
    "cmn": "zh-hans",
    "zh-tw": "zh-hant",
    "zh-hk": "zh-hant",
    "zh-mo": "zh-hant",
    "zh-hant": "zh-hant",
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "ja": "ja",
    "jp": "ja",
    "ko": "ko",
    "kr": "ko",
    "中文": "zh-hans",
    "简体": "zh-hans",
    "简体中文": "zh-hans",
    "繁体": "zh-hant",
    "繁體": "zh-hant",
    "繁体中文": "zh-hant",
    "英文": "en",
    "英语": "en",
    "english": "en",
    "chinese": "zh-hans",
    "日文": "ja",
    "日语": "ja",
    "韩文": "ko",
    "韩语": "ko",
}
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
    "结果仅为标题、链接与摘要，不表示已阅读全文",
    "正文证据需继续调用 web_page_read",
]

class SearchInputError(Exception):
    """query / site / top_k 等输入无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code

class EngineUnavailableError(Exception):
    """真实网页搜索引擎未配置、被闸门拒绝或调用失败。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code

@dataclass(frozen=True)
class WebSearchRequest:
    """组装后的网页检索请求；`query` 已含 site: 约束。"""

    query: str
    top_k: int
    language: str | None = None
    freshness: str | None = None

@runtime_checkable
class WebSearchEngine(Protocol):
    """可注入的网页搜索引擎；测试用 extras['web_search_engine'] 替换。"""

    def search(self, request: WebSearchRequest) -> dict[str, Any]:
        """提交检索请求，返回 Brave 风格 `{web: {results: [...]}}` 载荷。"""

class BraveSearchEngine:
    """Brave Search Web API 适配器；密钥与端点只读环境变量。"""

    name = _ENGINE_BRAVE
    max_top_k = _BRAVE_MAX_TOP_K

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        timeout_sec: float,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._timeout_sec = timeout_sec

    def search(self, request: WebSearchRequest) -> dict[str, Any]:
        params: dict[str, str] = {
            "q": request.query,
            "count": str(request.top_k),
        }
        if request.language:
            params["search_lang"] = request.language
        if request.freshness:
            params["freshness"] = request.freshness
        url = _append_query(self._endpoint, urllib.parse.urlencode(params))
        raw = _http_json(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self._api_key,
            },
            timeout_sec=self._timeout_sec,
            error_prefix="Brave Search",
        )
        if not isinstance(raw, dict):
            raise EngineUnavailableError("Brave Search 回执不是 JSON 对象")
        return raw

def execute_keyword_search(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """按关键词和可选域名白名单检索网页，只返回标题、链接与摘要。"""

    del purpose
    return _run_search("keyword_search", inputs, ctx)

def execute_site_search(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """在指定站点内检索网页，只返回标题、链接与摘要。"""

    del purpose
    return _run_search("site_search", inputs, ctx)

def _run_search(
    operation: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    if operation == "site_search":
        inputs = declared_inputs(inputs, "query", "site", "time_range", "top_k")
    else:
        inputs = declared_inputs(
            inputs, "query", "domains", "language", "time_range", "top_k"
        )
    try:
        query = _parse_query(inputs.get("query"))
        domains, site = _parse_scope(operation, inputs)
        language, unsupported_language = _parse_language(
            inputs.get("language") if operation == "keyword_search" else None,
        )
        freshness, unsupported_time = _parse_freshness(inputs.get("time_range"))
        requested_top_k = _parse_top_k(inputs.get("top_k"))
        engine = _resolve_engine(ctx)
        engine_name = str(getattr(engine, "name", "injected"))
        max_top_k = _engine_max_top_k(engine)
        top_k = min(requested_top_k, max_top_k)
        assembled = _with_site_operators(query, domains)
        request = WebSearchRequest(
            query=assembled,
            top_k=top_k,
            language=language,
            freshness=freshness,
        )
        payload = engine.search(request)
    except SearchInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

    results = _normalize_results(payload, top_k=top_k)
    applied: dict[str, Any] = {
        "engine": engine_name,
        "query": assembled,
        "top_k": top_k,
    }
    if site is not None:
        applied["site"] = site
    elif domains:
        applied["domains"] = list(domains)
    if language:
        applied["language"] = language
    if freshness:
        applied["time_range"] = freshness
    unsupported: dict[str, Any] = {}
    if unsupported_language is not None:
        unsupported["language"] = unsupported_language
    if unsupported_time is not None:
        unsupported["time_range"] = unsupported_time
    if unsupported:
        applied["unsupported"] = unsupported

    result: dict[str, Any] = {
        "operation": operation,
        "results": results,
        "applied": applied,
        "assumptions": list(_ASSUMPTIONS),
    }
    return Observation(ok=True, result=_strip_forbidden(result))

def _parse_query(raw: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise SearchInputError("缺少必填输入 query", "missing_input")
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise SearchInputError("query 必须是字符串或字符串列表", "invalid_query")
            stripped = item.strip()
            if stripped:
                parts.append(stripped)
        if not parts:
            raise SearchInputError("缺少必填输入 query", "missing_input")
        return " ".join(parts)
    raise SearchInputError("query 必须是字符串或字符串列表", "invalid_query")

def _parse_scope(
    operation: str,
    inputs: dict[str, Any],
) -> tuple[list[str], str | None]:
    if operation == "site_search":
        site = _parse_site(inputs.get("site"))
        return [site], site
    domains = _parse_domains(inputs.get("domains"))
    return domains, None

def _parse_site(raw: Any) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise SearchInputError("缺少必填输入 site", "missing_input")
    if not isinstance(raw, str):
        raise SearchInputError("site 必须是字符串", "invalid_query")
    site = _normalize_site(raw)
    if not site:
        raise SearchInputError("缺少必填输入 site", "missing_input")
    return site

def _parse_domains(raw: Any) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        stripped = raw.strip()
        return [_normalize_site(stripped)] if stripped else []
    if not isinstance(raw, list):
        raise SearchInputError("domains 必须是字符串列表", "invalid_query")
    domains: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise SearchInputError("domains 必须是字符串列表", "invalid_query")
        site = _normalize_site(item)
        if site and site not in seen:
            domains.append(site)
            seen.add(site)
    return domains

def _normalize_site(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    parsed = urllib.parse.urlparse(text if "://" in text else f"https://{text}")
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host:
        return host
    return text.split("/")[0].strip().lower()

def _with_site_operators(query: str, domains: list[str]) -> str:
    if not domains:
        return query
    if len(domains) == 1:
        constraint = f"site:{domains[0]}"
    else:
        joined = " OR ".join(f"site:{item}" for item in domains)
        constraint = f"({joined})"
    return f"{query} {constraint}".strip()

def _parse_language(raw: Any) -> tuple[str | None, Any | None]:
    if raw is None or raw == "":
        return None, None
    if not isinstance(raw, str):
        return None, raw
    text = raw.strip()
    if not text:
        return None, None
    aliased = _LANGUAGE_ALIASES.get(text.lower()) or _LANGUAGE_ALIASES.get(text)
    if aliased:
        return aliased, None
    code = text.lower()
    if _LANGUAGE_CODE_RE.fullmatch(code):
        return code, None
    return None, raw

def _parse_freshness(raw: Any) -> tuple[str | None, Any | None]:
    if raw is None or raw == "":
        return None, None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None, None
        lower = text.lower()
        if lower in _RELATIVE_FRESHNESS:
            return lower, None
        date_range = _DATE_RANGE_RE.fullmatch(re.sub(r"\s+", "", text))
        if date_range:
            return f"{date_range.group(1)}to{date_range.group(2)}", None
        year = _YEAR_RE.fullmatch(text)
        if year:
            value = year.group(1)
            return f"{value}-01-01to{value}-12-31", None
        year_range = _YEAR_RANGE_RE.fullmatch(text)
        if year_range:
            start, end = year_range.group(1), year_range.group(2)
            return f"{start}-01-01to{end}-12-31", None
        day = _DATE_RE.fullmatch(text)
        if day:
            value = day.group(1)
            return f"{value}to{value}", None
        return None, raw
    if isinstance(raw, dict):
        start = _bound_to_range(raw.get("start"))
        end = _bound_to_range(raw.get("end"))
        if start is not None and end is not None:
            return f"{start[0]}to{end[1]}", None
        return None, raw
    return None, raw

def _bound_to_range(raw: Any) -> tuple[str, str] | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int) and 1000 <= raw <= 9999:
        return f"{raw}-01-01", f"{raw}-12-31"
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if _YEAR_RE.fullmatch(text):
        return f"{text}-01-01", f"{text}-12-31"
    if _DATE_RE.fullmatch(text):
        return text, text
    return None

def _parse_top_k(raw: Any) -> int:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DEFAULT_TOP_K
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SearchInputError("top_k 必须是整数", "invalid_top_k") from exc
    return max(_TOP_K_MIN, min(_TOP_K_MAX, value))

def _engine_max_top_k(engine: WebSearchEngine) -> int:
    raw = getattr(engine, "max_top_k", _BRAVE_MAX_TOP_K)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _BRAVE_MAX_TOP_K
    if value < _TOP_K_MIN:
        return _BRAVE_MAX_TOP_K
    return min(value, _TOP_K_MAX)

def _resolve_engine(ctx: RuntimeContext | None) -> WebSearchEngine:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("web_search_engine")
    if injected is not None:
        if not isinstance(injected, WebSearchEngine):
            raise EngineUnavailableError(
                "web_search_engine 必须提供 search(request)",
            )
        return injected
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError(
            "ALLOW_REAL_API=false，禁止调用真实网页搜索 API",
        )
    api_key = _brave_api_key()
    if not api_key:
        raise EngineUnavailableError("未配置 BRAVE_SEARCH_API_KEY 或 BRAVE_API_KEY")
    return BraveSearchEngine(
        api_key=api_key,
        endpoint=_env_value("BRAVE_SEARCH_ENDPOINT", _DEFAULT_BRAVE_ENDPOINT),
        timeout_sec=_env_timeout(),
    )

def _brave_api_key() -> str:
    return (
        os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
        or os.environ.get("BRAVE_API_KEY", "").strip()
    )

def _env_value(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default

def _env_timeout() -> float:
    raw = os.environ.get("BRAVE_SEARCH_TIMEOUT_SEC", "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    return _DEFAULT_TIMEOUT_SEC

def _http_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout_sec: float,
    error_prefix: str,
) -> Any:
    request = urllib.request.Request(url, method="GET")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise EngineUnavailableError(
            f"{error_prefix} HTTP {exc.code}: {detail[:200]}",
        ) from exc
    except urllib.error.URLError as exc:
        raise EngineUnavailableError(
            f"{error_prefix} 网络失败: {exc.reason}",
        ) from exc
    except (json.JSONDecodeError, TimeoutError, OSError) as exc:
        raise EngineUnavailableError(f"{error_prefix} 调用失败: {exc}") from exc
    if isinstance(raw, dict) and raw.get("error"):
        raise EngineUnavailableError(f"{error_prefix} 失败: {raw['error']}")
    return raw

def _append_query(endpoint: str, query: str) -> str:
    separator = "&" if urllib.parse.urlparse(endpoint).query else "?"
    return f"{endpoint}{separator}{query}"

def _normalize_results(payload: dict[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    web = payload.get("web") if isinstance(payload, dict) else None
    rows = web.get("results") if isinstance(web, dict) else None
    if not isinstance(rows, list):
        return []
    results: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        url = _clean_url(item.get("url"))
        if not url:
            continue
        title = item.get("title")
        snippet = _snippet_from(item)
        result_id = f"ws_{len(results) + 1}"
        results.append(
            {
                "result_id": result_id,
                "title": str(title).strip() if title is not None else "",
                "url": url,
                "snippet": snippet,
            }
        )
        if len(results) >= top_k:
            break
    return results

def _snippet_from(item: dict[str, Any]) -> str:
    parts: list[str] = []
    description = item.get("description")
    if isinstance(description, str) and description.strip():
        parts.append(description.strip())
    extras = item.get("extra_snippets")
    if isinstance(extras, list):
        for extra in extras:
            if isinstance(extra, str) and extra.strip() and extra.strip() not in parts:
                parts.append(extra.strip())
    return " ".join(parts)

def _clean_url(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    if not text:
        return ""
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return text

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
