"""Token-bucket rate limiting for client messages (protocol error code ``rate_limited``)."""

from __future__ import annotations

import time
from collections.abc import Callable


class TokenBucket:
    """``rate`` tokens per second, at most ``burst`` banked; :meth:`take` spends one."""

    __slots__ = ("rate", "burst", "_tokens", "_t", "_clock")

    def __init__(self, rate: float, burst: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        if rate <= 0 or burst < 1:
            raise ValueError("rate must be > 0 and burst >= 1")
        self.rate = float(rate)
        self.burst = float(burst)
        self._clock = clock
        self._tokens = float(burst)
        self._t = clock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.burst, self._tokens + (now - self._t) * self.rate)
        self._t = now

    def take(self, n: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= n:
            self._tokens -= n
            return True
        return False

    @property
    def tokens(self) -> float:
        self._refill()
        return self._tokens
