"""百度识图浏览器引擎：仅在 engines 含 baidu 时由 reverse_image_search 构造。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from tool.reverse_image_search._search import EngineUnavailableError

_INDEX_URL = "https://graph.baidu.com/pcpage/index?tpl_from=pc"
_RESULT_HINT = "graph.baidu.com/s"
_SIMILAR_HINT = "graph.baidu.com/ajax/pcsimi"
_BAIDU_REFUSED = ("功能优化中", "建议您重新上传其他图片")
_CAPTCHA_HINTS = ("安全验证", "人机验证", "captcha")
_DEFAULT_TIMEOUT_SEC = 60.0


@runtime_checkable
class BaiduGraphSession(Protocol):
    """可注入的识图会话；测试用 extras['baidu_graph_session'] 替换，不启动浏览器。"""

    def run(self, image_bytes: bytes) -> dict[str, Any]:
        """上传查询图，返回 similar / guess 原始载荷。"""


class BaiduGraphEngine:
    """把百度识图相似图与猜测映射成 Web Detection 形状。"""

    name = "baidu_graph"

    def __init__(
        self,
        *,
        session: BaiduGraphSession | None = None,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self._session = session
        self._timeout_sec = timeout_sec

    def search(self, image_bytes: bytes, *, top_k: int) -> dict[str, Any]:
        session = self._session or PlaywrightBaiduGraphSession(
            timeout_sec=self._timeout_sec
        )
        raw = session.run(image_bytes)
        return baidu_graph_to_web_detection(raw, top_k=top_k)


class PlaywrightBaiduGraphSession:
    """本机 Playwright 打开百度识图；未安装时在调用期失败。"""

    def __init__(self, *, timeout_sec: float = _DEFAULT_TIMEOUT_SEC) -> None:
        self._timeout_ms = max(1, int(timeout_sec * 1000))

    def run(self, image_bytes: bytes) -> dict[str, Any]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise EngineUnavailableError("未安装 playwright") from exc

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = Path(tmp.name)
        try:
            tmp.write(image_bytes)
            tmp.close()
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    args=["--no-proxy-server"],
                )
                try:
                    page = browser.new_page()
                    page.set_default_timeout(self._timeout_ms)
                    similar_pages: list[dict[str, Any]] = []

                    def _grab(response: Any) -> None:
                        url = str(getattr(response, "url", ""))
                        if _SIMILAR_HINT not in url:
                            return
                        try:
                            data = response.json()
                        except Exception:  # noqa: BLE001
                            return
                        if isinstance(data, dict):
                            similar_pages.append(data)

                    page.on("response", _grab)
                    page.goto(_INDEX_URL, wait_until="domcontentloaded")
                    body = _page_text(page)
                    _raise_if_blocked(body, page_url=page.url)
                    upload = page.query_selector("input[type=file]")
                    if upload is None:
                        raise EngineUnavailableError(
                            "百度识图没有上传入口（页面改版或被拦截）",
                            "provider_error",
                        )
                    upload.set_input_files(str(tmp_path))
                    page.wait_for_timeout(1500)
                    for _ in range(40):
                        if _RESULT_HINT in page.url:
                            break
                        page.wait_for_timeout(250)
                    page.wait_for_timeout(1500)
                    text = _page_text(page)
                    _raise_if_blocked(text, page_url=page.url)
                    similar = _collect_similar(similar_pages)
                    return {
                        "guess": _collect_guesses(text),
                        "similar": similar,
                    }
                finally:
                    browser.close()
        finally:
            tmp_path.unlink(missing_ok=True)


def baidu_graph_to_web_detection(
    raw: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    """把 similar / guess 收成现有 Web Detection 形状。"""

    pages: list[dict[str, Any]] = []
    similar: list[dict[str, Any]] = []
    seen_pages: set[str] = set()
    seen_thumbs: set[str] = set()
    items = raw.get("similar") if isinstance(raw, dict) else None
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            page_url = _http_url(item.get("from"))
            thumb = _http_url(item.get("thumb"))
            if page_url and page_url not in seen_pages:
                seen_pages.add(page_url)
                page: dict[str, Any] = {"url": page_url}
                if thumb:
                    page["partialMatchingImages"] = [{"url": thumb}]
                pages.append(page)
            elif thumb and thumb not in seen_thumbs:
                seen_thumbs.add(thumb)
                similar.append({"url": thumb})
            if len(pages) + len(similar) >= top_k:
                break

    labels: list[dict[str, Any]] = []
    guesses = raw.get("guess") if isinstance(raw, dict) else None
    if isinstance(guesses, list):
        for guess in guesses[:top_k]:
            if isinstance(guess, str) and guess.strip():
                labels.append({"label": guess.strip()})

    return {
        "fullMatchingImages": [],
        "partialMatchingImages": [],
        "pagesWithMatchingImages": pages[:top_k],
        "visuallySimilarImages": similar[:top_k],
        "webEntities": [],
        "bestGuessLabels": labels,
    }


def _page_text(page: Any) -> str:
    try:
        return str(page.evaluate("document.body ? document.body.innerText : ''"))
    except Exception:  # noqa: BLE001
        return ""


def _raise_if_blocked(text: str, *, page_url: str) -> None:
    lowered = text.lower()
    url = page_url.lower()
    if any(hint in text for hint in _CAPTCHA_HINTS) or "captcha" in url:
        raise EngineUnavailableError("百度识图要求人机验证", "provider_error")
    if any(hint in text for hint in _BAIDU_REFUSED):
        raise EngineUnavailableError(
            "百度拒绝处理这张图（功能优化中）",
            "provider_error",
        )
    if "安全验证" in lowered:
        raise EngineUnavailableError("百度识图要求人机验证", "provider_error")


def _collect_guesses(text: str) -> list[str]:
    guesses: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("图中可能是"):
            guesses.append(stripped)
    return guesses


def _collect_similar(pages: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for payload in pages:
        data = payload.get("data") if isinstance(payload, dict) else None
        items = data.get("list") if isinstance(data, dict) else None
        if items is None and isinstance(payload, dict):
            items = payload.get("list")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            thumb = item.get("thumbUrl") or item.get("thumb")
            source = item.get("fromUrl") or item.get("from")
            key = str(item.get("contsign") or thumb or source or "")
            if not key or key in seen:
                continue
            seen.add(key)
            row: dict[str, str] = {}
            if isinstance(thumb, str) and thumb.strip():
                row["thumb"] = thumb.strip()
            if isinstance(source, str) and source.strip():
                row["from"] = source.strip()
            if row:
                rows.append(row)
    return rows


def _http_url(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return text
