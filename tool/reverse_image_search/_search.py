"""reverse_image_search 共享执行器：本地裁剪后提交 SerpAPI Lens 或 Vision Web Detection。"""

from __future__ import annotations

import base64
import json
import os
import uuid
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from tool.contract import Observation, RuntimeContext
from tool.image_edit._transform import RegionError, _parse_region, _try_json
from tool.runtime.image_store import ImageResolveError, put_image, resolve_image_ref

_DEFAULT_TOP_K = 10
_TOP_K_MIN = 1
_TOP_K_MAX = 100
_JPEG_QUALITY = 95
_ENGINE_SERPAPI = "serpapi"
_ENGINE_VISION = "google_cloud_vision"
_DEFAULT_ENGINE = _ENGINE_SERPAPI
_ENGINE_ALIASES = {
    "serpapi": _ENGINE_SERPAPI,
    "serpai": _ENGINE_SERPAPI,
    "google_lens": _ENGINE_SERPAPI,
    "lens": _ENGINE_SERPAPI,
    "google_cloud_vision": _ENGINE_VISION,
    "vision": _ENGINE_VISION,
    "web_detection": _ENGINE_VISION,
}
_DEFAULT_VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
_DEFAULT_SERPAPI_SEARCH_ENDPOINT = "https://serpapi.com/search.json"
_DEFAULT_SERPAPI_IMAGE_ENDPOINT = "https://serpapi.com/image"
_DEFAULT_TIMEOUT_SEC = 30.0
_SERPAPI_MAX_IMAGE_BYTES = 500 * 1024
_FORBIDDEN_KEYS = frozenset(
    {
        "confirmed_location",
        "taken_at",
        "location_confirmed",
        "confirmed_place",
    }
)
_MATCH_KEYS = frozenset(
    {"url", "page_url", "image_url", "title", "match_type", "source", "score"}
)
_ASSUMPTIONS = [
    "匹配结果是来源线索，不表示已确认拍摄地点",
    "webEntities 与 bestGuessLabels 只是检索标签，不能当作地点结论",
    "局部搜索先在本地裁剪再提交，未对全网自行建索引",
]


class SearchInputError(Exception):
    """image / engines / top_k 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class EngineUnavailableError(Exception):
    """真实搜图引擎未配置、被闸门拒绝或调用失败。"""

    def __init__(self, message: str, error_code: str = "engine_unavailable") -> None:
        super().__init__(message)
        self.error_code = error_code


@runtime_checkable
class ReverseImageSearchEngine(Protocol):
    """可注入的反向搜图引擎；测试用 extras['reverse_image_search_engine'] 替换。"""

    def search(self, image_bytes: bytes, *, top_k: int) -> dict[str, Any]:
        """提交图片字节，返回 Web Detection 风格载荷。"""


class GoogleVisionWebDetectionEngine:
    """Google Cloud Vision WEB_DETECTION 适配器；密钥与端点只读环境变量。"""

    name = "google_cloud_vision"

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

    def search(self, image_bytes: bytes, *, top_k: int) -> dict[str, Any]:
        payload = {
            "requests": [
                {
                    "image": {
                        "content": base64.b64encode(image_bytes).decode("ascii"),
                    },
                    "features": [
                        {
                            "type": "WEB_DETECTION",
                            "maxResults": top_k,
                        }
                    ],
                }
            ]
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            _endpoint_with_key(self._endpoint, self._api_key),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_sec) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise EngineUnavailableError(
                f"Vision Web Detection HTTP {exc.code}: {detail[:200]}",
            ) from exc
        except urllib.error.URLError as exc:
            raise EngineUnavailableError(
                f"Vision Web Detection 网络失败: {exc.reason}",
            ) from exc
        except (json.JSONDecodeError, TimeoutError, OSError) as exc:
            raise EngineUnavailableError(
                f"Vision Web Detection 调用失败: {exc}",
            ) from exc
        return _extract_web_detection(raw)


class SerpapiGoogleLensEngine:
    """SerpAPI Google Lens 适配器：先上传本地图拿 image_id，再搜视觉匹配。"""

    name = _ENGINE_SERPAPI

    def __init__(
        self,
        *,
        api_key: str,
        search_endpoint: str,
        image_endpoint: str,
        timeout_sec: float,
    ) -> None:
        self._api_key = api_key
        self._search_endpoint = search_endpoint
        self._image_endpoint = image_endpoint
        self._timeout_sec = timeout_sec

    def search(self, image_bytes: bytes, *, top_k: int) -> dict[str, Any]:
        del top_k
        jpeg = _cap_jpeg_bytes(image_bytes, _SERPAPI_MAX_IMAGE_BYTES)
        image_id = self._upload(jpeg)
        raw = self._lens_search(image_id)
        return _lens_to_web_detection(raw)

    def _upload(self, image_bytes: bytes) -> str:
        payload = _multipart_body(
            fields={"api_key": self._api_key},
            files={
                "image": ("query.jpg", image_bytes, "image/jpeg"),
            },
        )
        raw = _http_json(
            self._image_endpoint,
            data=payload.body,
            headers={"Content-Type": payload.content_type},
            timeout_sec=self._timeout_sec,
            error_prefix="SerpAPI Image",
        )
        image_id = raw.get("image_id") if isinstance(raw, dict) else None
        if not isinstance(image_id, str) or not image_id.strip():
            detail = raw.get("error") if isinstance(raw, dict) else raw
            raise EngineUnavailableError(f"SerpAPI 上传图片失败: {detail}")
        return image_id.strip()

    def _lens_search(self, image_id: str) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {
                "engine": "google_lens",
                "image_id": image_id,
                "api_key": self._api_key,
                "type": "all",
            }
        )
        url = _append_query(self._search_endpoint, query)
        raw = _http_json(
            url,
            data=None,
            headers=None,
            timeout_sec=self._timeout_sec,
            error_prefix="SerpAPI Google Lens",
        )
        if not isinstance(raw, dict):
            raise EngineUnavailableError("SerpAPI Google Lens 回执不是 JSON 对象")
        return raw


def execute_search(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """提交整图进行反向搜图，返回来源 URL 与匹配类型。"""

    del purpose
    return _run_search("search", inputs, ctx)


def execute_search_crop(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """先按 region 本地裁剪，再提交裁剪图进行反向搜图。"""

    del purpose
    return _run_search("search_crop", inputs, ctx)


def _run_search(
    operation: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> Observation:
    try:
        source_id, source = _load_image(inputs, ctx)
        region = _parse_optional_region(operation, inputs, source.size, ctx)
        engines, ignored_engines = _parse_engines(inputs.get("engines"))
        top_k = _parse_top_k(inputs.get("top_k"))
        engine, engine_name = _resolve_engine(ctx, requested=engines)
        query, query_id, query_path = _query_image(
            source,
            source_id=source_id,
            region=region,
            ctx=ctx,
        )
        payload = engine.search(_jpeg_bytes(query), top_k=top_k)
    except ImageResolveError as exc:
        return _fail(str(exc), exc.error_code)
    except RegionError as exc:
        return _fail(str(exc), exc.error_code)
    except SearchInputError as exc:
        return _fail(str(exc), exc.error_code)
    except EngineUnavailableError as exc:
        return _fail(str(exc), exc.error_code)

    matches, pages, entities, labels = _normalize_payload(
        payload,
        source=engine_name,
        top_k=top_k,
    )
    applied: dict[str, Any] = {
        "engines": engines,
        "top_k": top_k,
        "engine": engine_name,
    }
    if ignored_engines:
        applied["ignored"] = {"engines": ignored_engines}
    if region is not None:
        applied["region"] = list(region)
    if query_id is not None:
        applied["query_image_id"] = query_id

    result: dict[str, Any] = {
        "operation": operation,
        "image_id": source_id,
        "matches": matches,
        "pages": pages,
        "entities": entities,
        "best_guess_labels": labels,
        "applied": applied,
        "assumptions": list(_ASSUMPTIONS),
    }
    artifacts: dict[str, Any] = {}
    if query_id is not None and query_path is not None:
        result["query_image_id"] = query_id
        artifacts = {
            "query_image_id": query_id,
            "image_path": str(query_path),
        }
    return Observation(ok=True, result=_strip_forbidden(result), artifacts=artifacts)


def _load_image(
    inputs: dict[str, Any],
    ctx: RuntimeContext | None,
) -> tuple[str, Image.Image]:
    image_ref = inputs.get("image")
    if not isinstance(image_ref, str) or not image_ref.strip():
        raise SearchInputError("缺少必填输入 image", "missing_input")
    source_id, source_path = resolve_image_ref(image_ref, ctx)
    try:
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
    except OSError as exc:
        raise ImageResolveError(f"无法读取图片: {exc}", "image_not_found") from exc
    return source_id, source


def _parse_optional_region(
    operation: str,
    inputs: dict[str, Any],
    size: tuple[int, int],
    ctx: RuntimeContext | None,
) -> tuple[int, int, int, int] | None:
    raw_region = inputs.get("region")
    if operation != "search_crop":
        return None
    if raw_region is None:
        raise SearchInputError("缺少必填输入 region", "missing_input")
    return _parse_region(raw_region, size[0], size[1], ctx)


def _query_image(
    source: Image.Image,
    *,
    source_id: str,
    region: tuple[int, int, int, int] | None,
    ctx: RuntimeContext | None,
) -> tuple[Image.Image, str | None, Any]:
    if region is None:
        return source, None, None
    cropped = source.crop(region).convert("RGB")
    image_id, path = put_image(cropped, source_id=source_id, suffix="jpeg", ctx=ctx)
    return cropped, image_id, path


def _parse_engines(raw: Any) -> tuple[list[str], list[str]]:
    values = _parse_string_list(raw, field="engines", error_code="invalid_engines")
    if not values:
        return [_DEFAULT_ENGINE], []
    mapped: list[str] = []
    ignored: list[str] = []
    seen: set[str] = set()
    for item in values:
        key = _ENGINE_ALIASES.get(item.strip().lower())
        if key is None:
            ignored.append(item)
            continue
        if key not in seen:
            mapped.append(key)
            seen.add(key)
    if not mapped:
        raise SearchInputError(
            f"不支持的 engines: {values}",
            "unsupported_engine",
        )
    return mapped, ignored


def _parse_top_k(raw: Any) -> int:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DEFAULT_TOP_K
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SearchInputError("top_k 必须是整数", "invalid_top_k") from exc
    return max(_TOP_K_MIN, min(_TOP_K_MAX, value))


def _parse_string_list(raw: Any, *, field: str, error_code: str) -> list[str]:
    value: Any = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        loaded = _try_json(stripped)
        value = loaded if loaded is not None else ([stripped] if stripped else [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        raise SearchInputError(f"{field} 必须是字符串列表", error_code)
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SearchInputError(f"{field} 必须是字符串列表", error_code)
        if item.strip():
            items.append(item.strip())
    return items


def _resolve_engine(
    ctx: RuntimeContext | None,
    *,
    requested: list[str],
) -> tuple[ReverseImageSearchEngine, str]:
    extras = ctx.extras if ctx is not None else {}
    injected = extras.get("reverse_image_search_engine")
    if injected is not None:
        if not isinstance(injected, ReverseImageSearchEngine):
            raise EngineUnavailableError(
                "reverse_image_search_engine 必须提供 search(image_bytes, top_k=...)",
            )
        name = str(getattr(injected, "name", "injected"))
        return injected, name
    _load_dotenv()
    if not _allow_real_api():
        raise EngineUnavailableError(
            "ALLOW_REAL_API=false，禁止调用真实反向搜图 API",
        )
    last_error: EngineUnavailableError | None = None
    names = requested or [_DEFAULT_ENGINE]
    for name in names:
        try:
            engine = _build_engine(name)
        except EngineUnavailableError as exc:
            last_error = exc
            continue
        return engine, engine.name
    raise last_error or EngineUnavailableError("未配置可用的反向搜图引擎")


def _build_engine(name: str) -> ReverseImageSearchEngine:
    timeout_sec = _env_timeout()
    if name == _ENGINE_SERPAPI:
        api_key = _serpapi_api_key()
        if not api_key:
            raise EngineUnavailableError("未配置 SERPAPI_API_KEY 或 SERPAPI_KEY")
        return SerpapiGoogleLensEngine(
            api_key=api_key,
            search_endpoint=_env_value("SERPAPI_ENDPOINT", _DEFAULT_SERPAPI_SEARCH_ENDPOINT),
            image_endpoint=_env_value(
                "SERPAPI_IMAGE_ENDPOINT",
                _DEFAULT_SERPAPI_IMAGE_ENDPOINT,
            ),
            timeout_sec=timeout_sec,
        )
    if name == _ENGINE_VISION:
        api_key = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()
        if not api_key:
            raise EngineUnavailableError("未配置 GOOGLE_VISION_API_KEY")
        return GoogleVisionWebDetectionEngine(
            api_key=api_key,
            endpoint=_env_value("GOOGLE_VISION_ENDPOINT", _DEFAULT_VISION_ENDPOINT),
            timeout_sec=timeout_sec,
        )
    raise EngineUnavailableError(f"不支持的 engines: {name}")


def _allow_real_api() -> bool:
    raw = os.environ.get("ALLOW_REAL_API", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _serpapi_api_key() -> str:
    return (
        os.environ.get("SERPAPI_API_KEY", "").strip()
        or os.environ.get("SERPAPI_KEY", "").strip()
    )


def _env_value(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default


def _env_timeout() -> float:
    for name in ("SERPAPI_TIMEOUT_SEC", "GOOGLE_VISION_TIMEOUT_SEC"):
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


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


class _MultipartPayload:
    """multipart/form-data 请求体。"""

    def __init__(self, body: bytes, content_type: str) -> None:
        self.body = body
        self.content_type = content_type


def _multipart_body(
    *,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> _MultipartPayload:
    boundary = f"----GeoagentSerpapi{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        chunks.append(value.encode("utf-8"))
        chunks.append(b"\r\n")
    for name, (filename, data, content_type) in files.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return _MultipartPayload(
        body=b"".join(chunks),
        content_type=f"multipart/form-data; boundary={boundary}",
    )


def _http_json(
    url: str,
    *,
    data: bytes | None,
    headers: dict[str, str] | None,
    timeout_sec: float,
    error_prefix: str,
) -> Any:
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    for key, value in (headers or {}).items():
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


def _cap_jpeg_bytes(image_bytes: bytes, max_bytes: int) -> bytes:
    if len(image_bytes) <= max_bytes:
        return image_bytes
    with Image.open(BytesIO(image_bytes)) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        quality = 85
        payload = image_bytes
        while True:
            buffer = BytesIO()
            rgb.save(buffer, format="JPEG", quality=quality)
            payload = buffer.getvalue()
            if len(payload) <= max_bytes or (width <= 32 and quality <= 40):
                return payload
            if quality > 40:
                quality -= 10
                continue
            width = max(32, width // 2)
            height = max(32, height // 2)
            rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)


def _lens_to_web_detection(raw: dict[str, Any]) -> dict[str, Any]:
    """把 Google Lens JSON 收成现有 Web Detection 形状，供统一归一。"""

    full: list[dict[str, Any]] = []
    similar: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    seen_pages: set[str] = set()

    for item in _as_list(raw.get("exact_matches")):
        if not isinstance(item, dict):
            continue
        image_url = _clean_url(item.get("image") or item.get("thumbnail"))
        if image_url:
            full.append({"url": image_url})
        _append_lens_page(pages, seen_pages, item, image_url, match="full")

    for item in _as_list(raw.get("visual_matches")):
        if not isinstance(item, dict):
            continue
        image_url = _clean_url(item.get("image") or item.get("thumbnail"))
        if image_url:
            similar.append({"url": image_url})
        _append_lens_page(pages, seen_pages, item, image_url, match="partial")

    for item in _as_list(raw.get("image_sources")):
        if isinstance(item, dict):
            _append_lens_page(pages, seen_pages, item, "", match="partial")

    labels: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    knowledge = raw.get("knowledge_graph")
    nodes = knowledge if isinstance(knowledge, list) else (
        [knowledge] if isinstance(knowledge, dict) else []
    )
    for node in nodes:
        if not isinstance(node, dict):
            continue
        title = node.get("title")
        if title is None:
            continue
        labels.append({"label": str(title)})
        entity: dict[str, Any] = {"description": str(title)}
        entity_id = node.get("kgmid")
        if entity_id is not None:
            entity["entityId"] = str(entity_id)
        entities.append(entity)

    return {
        "fullMatchingImages": full,
        "partialMatchingImages": [],
        "pagesWithMatchingImages": pages,
        "visuallySimilarImages": similar,
        "webEntities": entities,
        "bestGuessLabels": labels,
    }


def _append_lens_page(
    pages: list[dict[str, Any]],
    seen: set[str],
    item: dict[str, Any],
    image_url: str,
    *,
    match: str,
) -> None:
    page_url = _clean_url(item.get("link"))
    if not page_url or page_url in seen:
        return
    seen.add(page_url)
    title = item.get("title")
    page: dict[str, Any] = {
        "url": page_url,
        "pageTitle": str(title) if title is not None else "",
    }
    if image_url:
        key = "fullMatchingImages" if match == "full" else "partialMatchingImages"
        page[key] = [{"url": image_url}]
    pages.append(page)


def _endpoint_with_key(endpoint: str, api_key: str) -> str:
    parsed = urllib.parse.urlparse(endpoint)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query["key"] = api_key
    return urllib.parse.urlunparse(
        parsed._replace(query=urllib.parse.urlencode(query)),
    )


def _extract_web_detection(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise EngineUnavailableError("Vision 回执不是 JSON 对象")
    responses = raw.get("responses")
    if not isinstance(responses, list) or not responses:
        raise EngineUnavailableError("Vision 回执缺少 responses")
    first = responses[0]
    if not isinstance(first, dict):
        raise EngineUnavailableError("Vision 回执 responses[0] 无效")
    error = first.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or error)
        raise EngineUnavailableError(f"Vision Web Detection 失败: {message}")
    detection = first.get("webDetection")
    if detection is None:
        return {}
    if not isinstance(detection, dict):
        raise EngineUnavailableError("webDetection 不是对象")
    return detection


def _normalize_payload(
    payload: Any,
    *,
    source: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    data = payload if isinstance(payload, dict) else {}
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match_type, key in (
        ("full", "fullMatchingImages"),
        ("partial", "partialMatchingImages"),
        ("page", "pagesWithMatchingImages"),
        ("similar", "visuallySimilarImages"),
    ):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        for item in rows:
            match = (
                _page_match(item, source)
                if match_type == "page"
                else _image_match(item, match_type, source)
            )
            if match is None:
                continue
            url = str(match["url"])
            if url in seen:
                continue
            seen.add(url)
            matches.append(match)
            if len(matches) >= top_k:
                break
        if len(matches) >= top_k:
            break

    pages = [
        page
        for page in (_page_item(item) for item in _as_list(data.get("pagesWithMatchingImages")))
        if page is not None
    ][:top_k]
    entities = [
        entity
        for entity in (_entity_item(item) for item in _as_list(data.get("webEntities")))
        if entity is not None
    ][:top_k]
    labels = [
        label
        for label in (_label_item(item) for item in _as_list(data.get("bestGuessLabels")))
        if label is not None
    ][:top_k]
    return matches, pages, entities, labels


def _as_list(raw: Any) -> list[Any]:
    return raw if isinstance(raw, list) else []


def _image_match(item: Any, match_type: str, source: str) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    url = _clean_url(item.get("url"))
    if not url:
        return None
    row: dict[str, Any] = {
        "url": url,
        "image_url": url,
        "page_url": "",
        "title": "",
        "match_type": match_type,
        "source": source,
    }
    _copy_score(item, row)
    return _keep_match_keys(row)


def _page_match(item: Any, source: str) -> dict[str, Any] | None:
    page = _page_item(item)
    if page is None:
        return None
    image_urls = page.get("image_urls")
    image_url = ""
    if isinstance(image_urls, list) and image_urls and isinstance(image_urls[0], str):
        image_url = image_urls[0]
    row: dict[str, Any] = {
        "url": page["url"],
        "page_url": page["url"],
        "image_url": image_url,
        "title": page.get("title") or "",
        "match_type": "page",
        "source": source,
    }
    return _keep_match_keys(row)


def _page_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    url = _clean_url(item.get("url"))
    if not url:
        return None
    title = item.get("pageTitle")
    if title is None:
        title = item.get("title")
    image_urls: list[str] = []
    for key in ("fullMatchingImages", "partialMatchingImages"):
        nested = item.get(key)
        if not isinstance(nested, list):
            continue
        for nested_item in nested:
            if isinstance(nested_item, dict):
                nested_url = _clean_url(nested_item.get("url"))
                if nested_url:
                    image_urls.append(nested_url)
    page: dict[str, Any] = {"url": url, "title": str(title) if title is not None else ""}
    if image_urls:
        page["image_urls"] = image_urls
    return page


def _entity_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    description = item.get("description")
    entity_id = item.get("entityId")
    if description is None and entity_id is None:
        return None
    row: dict[str, Any] = {}
    if entity_id is not None:
        row["entity_id"] = str(entity_id)
    if description is not None:
        row["description"] = str(description)
    _copy_score(item, row)
    return row


def _label_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    label = item.get("label")
    if label is None:
        return None
    row: dict[str, Any] = {"label": str(label)}
    language = item.get("languageCode")
    if language is not None:
        row["language"] = str(language)
    return row


def _copy_score(src: dict[str, Any], dest: dict[str, Any]) -> None:
    score = src.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        dest["score"] = float(score)


def _clean_url(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.strip()


def _keep_match_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key in _MATCH_KEYS}


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=_JPEG_QUALITY)
    return buffer.getvalue()


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


def _fail(error: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=error, error_code=error_code)
