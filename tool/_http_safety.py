"""User-controlled HTTP URL safety checks shared by Tool downloaders."""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request


class UnsafeHttpUrlError(ValueError):
    """The URL can reach a local, private, reserved, or otherwise unsafe host."""


def require_public_http_url(url: str) -> str:
    """Validate an HTTP(S) URL and every address currently returned by DNS.

    Rejecting the whole hostname when any answer is non-public avoids choosing a
    safe address during validation and a private address during the actual request.
    Redirect handlers must call this function again for every new target.
    """

    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeHttpUrlError("URL 主机或端口无效") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise UnsafeHttpUrlError("仅允许 http 或 https URL")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeHttpUrlError("URL 不允许包含用户名或密码")
    host = parsed.hostname
    if not host:
        raise UnsafeHttpUrlError("URL 缺少主机名")
    lowered = host.rstrip(".").lower()
    if lowered == "localhost" or lowered.endswith((".localhost", ".local")):
        raise UnsafeHttpUrlError("URL 不允许访问本机或局域网主机")

    addresses = _resolved_addresses(host, port or (443 if parsed.scheme.lower() == "https" else 80))
    if not addresses:
        raise UnsafeHttpUrlError("URL 主机没有可用地址")
    unsafe = sorted(address for address in addresses if not _is_public_address(address))
    if unsafe:
        raise UnsafeHttpUrlError(f"URL 解析到非公网地址: {', '.join(unsafe)}")
    return url


def _resolved_addresses(host: str, port: int) -> set[str]:
    try:
        literal = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        try:
            answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise UnsafeHttpUrlError(f"URL 主机无法解析: {host}") from exc
        return {str(answer[4][0]).split("%", 1)[0] for answer in answers}
    return {str(literal)}


def _is_public_address(raw: str) -> bool:
    try:
        return ipaddress.ip_address(raw).is_global
    except ValueError:
        return False


class PublicHttpRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Revalidate every redirect and cap the redirect chain."""

    def __init__(self, max_redirects: int = 5) -> None:
        super().__init__()
        self._max_redirects = max_redirects

    def redirect_request(  # type: ignore[no-untyped-def]
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        redirect_dict = getattr(req, "redirect_dict", {}) if req is not None else {}
        if len(redirect_dict) >= self._max_redirects:
            raise urllib.error.HTTPError(
                str(newurl),
                int(code),
                "redirect limit exceeded",
                headers,
                fp,
            )
        require_public_http_url(str(newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


__all__ = ["PublicHttpRedirectHandler", "UnsafeHttpUrlError", "require_public_http_url"]
