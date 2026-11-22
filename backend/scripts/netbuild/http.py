"""Minimal HTTP client for the build: urllib only, generic User-Agent, retries with backoff.

The build sends no personal data: only the public dataset URLs and SoQL queries.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from scripts.netbuild.constants import HTTP_BACKOFF_S, HTTP_RETRIES, HTTP_TIMEOUT_S, USER_AGENT


class FetchError(RuntimeError):
    """Raised when a URL could not be fetched after all retries (or in offline mode)."""


@dataclass
class RequestLog:
    url: str
    ok: bool
    attempts: int
    seconds: float
    bytes: int = 0
    error: str = ""


@dataclass
class HttpClient:
    """Fetches URLs with retries. ``offline=True`` refuses every request."""

    offline: bool = False
    timeout_s: float = HTTP_TIMEOUT_S
    retries: int = HTTP_RETRIES
    backoff_s: float = HTTP_BACKOFF_S
    sleep: Callable[[float], None] = time.sleep
    log: list[RequestLog] = field(default_factory=list)

    def get(self, url: str, *, timeout_s: float | None = None, accept: str | None = None) -> bytes:
        if self.offline:
            raise FetchError(f"offline mode: refusing to fetch {url}")
        headers = {"User-Agent": USER_AGENT}
        if accept:
            headers["Accept"] = accept
        request = urllib.request.Request(url, headers=headers, method="GET")
        timeout = self.timeout_s if timeout_s is None else timeout_s
        started = time.monotonic()
        last_error = ""
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = response.read()
                self.log.append(RequestLog(url, True, attempt, time.monotonic() - started, len(body)))
                return body
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}: {exc.reason}"
                if 400 <= exc.code < 500 and exc.code not in (408, 429):
                    break  # client errors (bad query, missing dataset) do not improve on retry
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                self.sleep(self.backoff_s * (2 ** (attempt - 1)))
        self.log.append(RequestLog(url, False, attempt, time.monotonic() - started, 0, last_error))
        raise FetchError(f"{url}: {last_error}")

    def get_json(self, url: str, *, timeout_s: float | None = None, accept: str | None = "application/json") -> Any:
        body = self.get(url, timeout_s=timeout_s, accept=accept)
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FetchError(f"{url}: response is not JSON ({exc})") from exc


def soql_url(base: str, params: dict[str, str]) -> str:
    """Build a Socrata query URL with properly encoded ``$``-parameters."""
    return base + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
