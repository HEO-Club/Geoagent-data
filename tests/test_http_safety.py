"""SSRF boundaries for user-controlled page and media URLs."""

from __future__ import annotations

import socket

import pytest

from tool import _http_safety as safety


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://[::1]/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.8/video.mp4",
        "http://localhost:8000/",
        "http://user:secret@example.com/",
    ],
)
def test_private_or_credentialed_urls_are_rejected(url: str) -> None:
    with pytest.raises(safety.UnsafeHttpUrlError):
        safety.require_public_http_url(url)


def test_dns_answer_with_any_private_address_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.10", 443)),
        ]

    monkeypatch.setattr(safety.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(safety.UnsafeHttpUrlError, match="非公网"):
        safety.require_public_http_url("https://example.com/page")


def test_public_dns_answer_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        safety.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    assert safety.require_public_http_url("https://example.com/page") == (
        "https://example.com/page"
    )


def test_redirect_handler_revalidates_target(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_validate(url: str) -> str:
        seen.append(url)
        raise safety.UnsafeHttpUrlError("blocked redirect")

    monkeypatch.setattr(safety, "require_public_http_url", fake_validate)
    handler = safety.PublicHttpRedirectHandler()
    with pytest.raises(safety.UnsafeHttpUrlError, match="blocked redirect"):
        handler.redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1/")
    assert seen == ["http://127.0.0.1/"]
