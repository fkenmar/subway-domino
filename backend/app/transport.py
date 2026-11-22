"""Per-connection WebSocket writer (ARCHITECTURE §5.2 "Backpressure").

Exactly one task writes to a socket, so JSON messages and binary frames leave in the order
they were produced and no two sends interleave. Binary frames are idempotent keyframes:
:meth:`Outbox.offer_frame` **skips** a frame while the previous one is still pending (queued
or being written). With uvicorn's websockets sans-I/O protocol a send is pending exactly
while the transport's write buffer is above its high-water mark, i.e. while the client is not
keeping up — so a slow client sees a lower frame rate instead of growing latency, and the
server never buffers more than one frame per connection.

JSON messages are never dropped; if a client stops reading entirely, the backlog is bounded
by ``max_backlog`` and the connection is closed (1008) so the session can detach and be
resumed by a healthier reconnect.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from app.protocol import dumps

log = logging.getLogger("app.transport")

CLOSE_NORMAL = 1000
CLOSE_GOING_AWAY = 1001
CLOSE_PROTOCOL_ERROR = 1002
CLOSE_POLICY = 1008
CLOSE_INTERNAL_ERROR = 1011
CLOSE_TRY_AGAIN_LATER = 1013


class SocketLike(Protocol):
    """The subset of ``starlette.websockets.WebSocket`` the writer uses."""

    async def send_text(self, data: str) -> None: ...

    async def send_bytes(self, data: bytes) -> None: ...

    async def close(self, code: int = 1000, reason: str | None = None) -> None: ...


@dataclass(slots=True)
class OutboxStats:
    messages_sent: int = 0
    frames_sent: int = 0
    frames_skipped: int = 0
    bytes_sent: int = 0  # payload bytes before permessage-deflate


_TEXT, _BYTES, _CLOSE = 0, 1, 2


class Outbox:
    """Ordered single-writer queue for one socket."""

    def __init__(self, ws: SocketLike, *, max_backlog: int = 512, close_timeout_s: float = 3.0) -> None:
        self._ws = ws
        self._queue: deque[tuple[int, Any]] = deque()
        self._wake = asyncio.Event()
        self._frame_pending = False
        self._closing = False
        self._closed = False
        self._max_backlog = int(max_backlog)
        self._close_timeout_s = float(close_timeout_s)
        self._task: asyncio.Task[None] | None = None
        self.close_code: int | None = None
        self.stats = OutboxStats()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run(), name="ws-writer")

    @property
    def open(self) -> bool:
        """True while messages can still be queued (not closing, not closed)."""
        return not (self._closing or self._closed)

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------ producers
    def send_json(self, message: Mapping[str, Any]) -> bool:
        """Queue one JSON message; ``False`` if the socket is closing/closed."""
        if not self.open:
            return False
        if len(self._queue) >= self._max_backlog:
            log.warning("client not reading (backlog %d); closing", len(self._queue))
            self.close_soon(CLOSE_POLICY, "client too slow")
            return False
        self._queue.append((_TEXT, dumps(message)))
        self._wake.set()
        return True

    def offer_frame(self, data: bytes) -> bool:
        """Queue a binary keyframe unless the previous one is still pending (then skip it)."""
        if not self.open:
            return False
        if self._frame_pending:
            self.stats.frames_skipped += 1
            return False
        self._frame_pending = True
        self._queue.append((_BYTES, data))
        self._wake.set()
        return True

    @property
    def frame_pending(self) -> bool:
        return self._frame_pending

    def close_soon(self, code: int = CLOSE_NORMAL, reason: str = "") -> None:
        """Send everything already queued, then a close frame (non-blocking)."""
        if not self.open:
            return
        self._closing = True
        self.close_code = code
        self._queue.append((_CLOSE, (code, reason)))
        self._wake.set()

    async def close(self, code: int = CLOSE_NORMAL, reason: str = "") -> None:
        """:meth:`close_soon` and wait (bounded) until the writer has finished."""
        self.close_soon(code, reason)
        await self.wait_closed(self._close_timeout_s)

    async def wait_closed(self, timeout: float | None = None) -> None:
        task = self._task
        if task is None:
            self._closed = True
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self.abort()

    def abort(self) -> None:
        """Stop writing immediately (the peer is gone); queued messages are dropped."""
        self._closed = True
        self._queue.clear()
        if self._task is not None and not self._task.done():
            self._task.cancel()

    # ------------------------------------------------------------------ writer
    async def _run(self) -> None:
        ws = self._ws
        try:
            while True:
                while not self._queue:
                    self._wake.clear()
                    await self._wake.wait()
                kind, payload = self._queue.popleft()
                if kind == _TEXT:
                    await ws.send_text(payload)
                    self.stats.messages_sent += 1
                    self.stats.bytes_sent += len(payload)
                elif kind == _BYTES:
                    try:
                        await ws.send_bytes(payload)
                    finally:
                        self._frame_pending = False
                    self.stats.frames_sent += 1
                    self.stats.bytes_sent += len(payload)
                else:
                    code, reason = payload
                    with contextlib.suppress(Exception):
                        await ws.close(code, reason)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the peer went away mid-send (WebSocketDisconnect, OSError, RuntimeError)
            log.debug("writer stopped: %r", exc)
        finally:
            self._closed = True
            self._queue.clear()
