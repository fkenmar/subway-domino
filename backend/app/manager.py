"""Session registry, capacity cap, resume tokens and the CPU governor (ARCHITECTURE §5.1-§5.2).

* **Capacity.** At most ``MAX_SESSIONS`` *attached* sessions. A connection reserves a slot
  when it passes the check (before its ``hello``), so concurrent connects cannot overshoot;
  the slot becomes the session's on ``hello`` or is released.
* **Resume.** When a socket drops, its session is paused and kept for ``DETACH_GRACE_S``
  under its resume token; detached sessions do not count toward ``MAX_SESSIONS``. A
  ``hello`` carrying the token of a *detached* session reattaches to the exact state (the
  token is rotated); any other token falls back to the client's ``restore`` payload. The
  detached pool is bounded (oldest evicted first) so dropped connections cannot pile up
  simulations in memory.
* **CPU governor.** One process-wide EWMA of CPU seconds per wall second. Above
  ``CPU_BUDGET`` every session's tick budget is scaled by the same factor, so all sessions
  slow equally; the factor is reported as ``metrics.effective_speed``. The load is the
  steady per-session work (ticks, inference, encoding, networking): one-off resets on the
  worker thread are excluded, because scaling tick budgets cannot reduce them — counting
  them made a burst of arrivals throttle every session to the floor for several seconds.
* **Worker thread.** Simulation construction and resets (≈0.1 s of warm-up on the real
  network) run in one worker thread, so a preset click never stalls every other session.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import secrets
import time
import uuid
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from sim.graph import Graph

from app import protocol as P
from app.callouts import CalloutWriter
from app.config import Settings
from app.session import PolicyLike, Session, SessionContext
from app.transport import Outbox

log = logging.getLogger("app.manager")

T = TypeVar("T")

GOVERNOR_PERIOD_S = 0.5


class CpuGovernor:
    """EWMA of CPU-seconds per wall-second → one speed factor shared by every session.

    The load is ``cpu_clock`` per wall second. :class:`SessionManager` passes the process's
    CPU time (``time.process_time``: simulation, inference, frame encoding, compression and
    networking) minus the metered one-off reset work, which tick scaling cannot reduce and
    which the headroom under ``budget`` absorbs. The control law is
    multiplicative — the scale that would have put the observed load on budget,
    ``scale · budget / load`` — which converges even though part of the load (frames,
    networking) does not scale with speed. Decreases apply at once; increases are limited
    to ``max_raise`` per sample so recovering from a spike does not oscillate. A ±5 %
    deadband around the budget keeps ``effective_speed`` from flickering.
    """

    def __init__(
        self,
        budget: float,
        *,
        alpha: float = 0.5,
        min_scale: float = 0.05,
        max_raise: float = 1.25,
        deadband: float = 0.05,
        cpu_clock: Callable[[], float] = time.process_time,
        wall_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < budget:
            raise ValueError("budget must be > 0")
        self.budget = float(budget)
        self.alpha = float(alpha)
        self.min_scale = float(min_scale)
        self.max_raise = float(max_raise)
        self.deadband = float(deadband)
        self._cpu = cpu_clock
        self._wall = wall_clock
        self._cpu0 = cpu_clock()
        self._wall0 = wall_clock()
        self._primed = False
        self.load = 0.0
        self.scale = 1.0

    def sample(self) -> None:
        """Measure CPU use since the previous sample and update ``load`` and ``scale``."""
        cpu, wall = self._cpu(), self._wall()
        dw = wall - self._wall0
        if dw <= 1e-6:
            return
        dc = cpu - self._cpu0
        self._cpu0, self._wall0 = cpu, wall
        self.observe(max(0.0, dc) / dw)

    def observe(self, inst_load: float) -> None:
        """Feed one instantaneous load measurement (CPU-s per wall-s)."""
        x = max(0.0, float(inst_load))
        self.load = x if not self._primed else self.alpha * x + (1.0 - self.alpha) * self.load
        self._primed = True
        lo, hi = self.budget * (1.0 - self.deadband), self.budget * (1.0 + self.deadband)
        if lo <= self.load <= hi:
            return
        target = 1.0 if self.load <= 1e-9 else self.scale * self.budget / self.load
        target = min(1.0, max(self.min_scale, target))
        if target > self.scale:
            target = min(target, self.scale * self.max_raise)
        self.scale = target


class SessionManager:
    """Owns every session; the WebSocket endpoint calls :meth:`try_reserve`, :meth:`open`
    and :meth:`connection_closed`."""

    def __init__(
        self,
        *,
        graph: Graph,
        policy: PolicyLike | None,
        settings: Settings,
        clock: Callable[[], float] = time.monotonic,
        governor: CpuGovernor | None = None,
        offload: bool = True,
    ) -> None:
        self.graph = graph
        self.policy = policy
        self.settings = settings
        self.clock = clock
        self._blocking_cpu_s = 0.0  # CPU spent in run_blocking work (written by the worker thread)
        self.governor = (
            governor if governor is not None else CpuGovernor(settings.cpu_budget, cpu_clock=self._governed_cpu_s)
        )
        self.started_at = clock()
        self.counters: Counter[str] = Counter()
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim-worker") if offload else None
        )
        self._sessions: dict[str, Session] = {}
        self._token_to_sid: dict[str, str] = {}
        self._sid_to_token: dict[str, str] = {}
        self._detached: dict[str, asyncio.TimerHandle] = {}  # insertion order = detach order
        self._outboxes: set[Outbox] = set()
        self._reserved = 0
        self._governor_task: asyncio.Task[None] | None = None
        self.ctx = SessionContext(
            graph=graph,
            policy=policy,
            settings=settings,
            speed_scale=lambda: self.governor.scale,
            run_blocking=self.run_blocking,
            clock=clock,
            writer=CalloutWriter(graph),
        )

    # ------------------------------------------------------------------ properties
    @property
    def model_loaded(self) -> bool:
        return self.policy is not None

    @property
    def active_sessions(self) -> int:
        """Attached sessions (detached ones do not count toward ``MAX_SESSIONS``)."""
        return len(self._sessions) - len(self._detached)

    @property
    def detached_sessions(self) -> int:
        return len(self._detached)

    @property
    def reserved(self) -> int:
        return self._reserved

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self._governor_task is None:
            self._governor_task = asyncio.get_running_loop().create_task(self._govern(), name="cpu-governor")

    async def shutdown(self) -> None:
        if self._governor_task is not None:
            self._governor_task.cancel()
            try:
                await self._governor_task
            except asyncio.CancelledError:
                pass
            self._governor_task = None
        for handle in self._detached.values():
            handle.cancel()
        self._detached.clear()
        for out in list(self._outboxes):
            out.close_soon(1001, "server shutting down")
        sessions = list(self._sessions.values())
        self._sessions.clear()
        self._token_to_sid.clear()
        self._sid_to_token.clear()
        await asyncio.gather(*(s.close() for s in sessions), return_exceptions=True)
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _governed_cpu_s(self) -> float:
        """Process CPU time minus the one-off worker-thread work (resets, construction)."""
        return time.process_time() - self._blocking_cpu_s

    def _metered(self, fn: Callable[..., T], *args: Any) -> T:
        cpu0 = time.thread_time()
        try:
            return fn(*args)
        finally:
            self._blocking_cpu_s += time.thread_time() - cpu0  # single worker thread: no race

    async def run_blocking(self, fn: Callable[..., T], *args: Any) -> T:
        """Run ``fn(*args)`` on the simulation worker thread (inline when offload is off).

        Its CPU time is metered and left out of the governor's load (see module docstring).
        """
        if self._executor is None:
            return self._metered(fn, *args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(self._metered, fn, *args))

    async def prewarm(self, profiles: tuple[str, ...] = P.PROFILES) -> float:
        """Build each profile's cached service tables once, before the first visitor."""
        from sim import NYCSubwaySim, Scenario

        t0 = time.perf_counter()
        for profile in profiles:
            await self.run_blocking(NYCSubwaySim, self.graph, Scenario(seed=0, profile=profile))
        return time.perf_counter() - t0

    async def _govern(self) -> None:
        while True:
            await asyncio.sleep(GOVERNOR_PERIOD_S)
            prev = self.governor.scale
            self.governor.sample()
            if abs(self.governor.scale - prev) >= 0.05 and (self.governor.scale < 1.0 or prev < 1.0):
                log.info("governor load=%.2f budget=%.2f scale %.2f -> %.2f sessions=%d",
                         self.governor.load, self.governor.budget, prev, self.governor.scale, self.active_sessions)

    # ------------------------------------------------------------------ capacity
    def try_reserve(self) -> bool:
        """Claim a session slot for a new connection (``False`` → at capacity)."""
        if self.active_sessions + self._reserved >= self.settings.max_sessions:
            self.counters["rejected_capacity"] += 1
            return False
        self._reserved += 1
        return True

    def release(self) -> None:
        """Give back a reservation that never became a session."""
        self._reserved = max(0, self._reserved - 1)

    # ------------------------------------------------------------------ sessions
    def register_outbox(self, out: Outbox) -> None:
        self._outboxes.add(out)

    def _issue_token(self, session: Session) -> str:
        old = self._sid_to_token.pop(session.session_id, None)
        if old is not None:
            self._token_to_sid.pop(old, None)
        token = secrets.token_urlsafe(32)
        self._token_to_sid[token] = session.session_id
        self._sid_to_token[session.session_id] = token
        return token

    async def open(self, out: Outbox, hello: P.Hello) -> Session:
        """Turn a reserved connection into a session: resume a detached one, or create one."""
        try:
            sid = self._token_to_sid.get(hello.resume_token) if hello.resume_token else None
            if sid is not None and sid in self._detached:
                session = self._sessions[sid]
                self._detached.pop(sid).cancel()
                token = self._issue_token(session)
                session.attach(out, resume_token=token, resumed=True, model_loaded=self.model_loaded)
                self.counters["sessions_resumed"] += 1
                log.info("session=%s resumed (active=%d)", sid[:8], self.active_sessions)
                return session
            session = await Session.create(self.ctx, session_id=uuid.uuid4().hex, restore=hello.restore)
            self._sessions[session.session_id] = session
            token = self._issue_token(session)
            session.attach(out, resume_token=token, resumed=False, model_loaded=self.model_loaded)
            session.start()
            self.counters["sessions_created"] += 1
            log.info("session=%s created profile=%s mode=%s (active=%d)", session.session_id[:8], session.profile,
                     session.mode, self.active_sessions)
            return session
        finally:
            self._reserved = max(0, self._reserved - 1)

    def connection_closed(self, session: Session, out: Outbox) -> None:
        """The socket bound to ``session`` is gone: detach it (resumable) or drop it."""
        self.retire_outbox(out)
        sid = session.session_id
        if sid not in self._sessions:
            return
        if session.closed:
            self._forget(sid)
            return
        if session.attached and not session.bound_to(out):
            return  # a newer socket already owns the session
        session.detach()
        if sid in self._detached:
            return
        if self.settings.detach_grace_s <= 0 or self.settings.detached_cap == 0:
            self._forget(sid)
            return
        loop = asyncio.get_running_loop()
        self._detached[sid] = loop.call_later(self.settings.detach_grace_s, self._expire, sid)
        while len(self._detached) > self.settings.detached_cap:
            oldest = next(iter(self._detached))
            self.counters["sessions_evicted"] += 1
            self._forget(oldest)
        log.info("session=%s detached (active=%d detached=%d)", sid[:8], self.active_sessions, len(self._detached))

    def _expire(self, sid: str) -> None:
        if sid in self._detached:
            self.counters["sessions_expired"] += 1
            log.info("session=%s expired after %.0f s detached", sid[:8], self.settings.detach_grace_s)
            self._forget(sid)

    def _forget(self, sid: str) -> None:
        handle = self._detached.pop(sid, None)
        if handle is not None:
            handle.cancel()
        session = self._sessions.pop(sid, None)
        token = self._sid_to_token.pop(sid, None)
        if token is not None:
            self._token_to_sid.pop(token, None)
        if session is not None:
            asyncio.get_running_loop().create_task(session.close())

    def retire_outbox(self, out: Outbox) -> None:
        """Fold a finished connection's writer stats into the totals (idempotent)."""
        if out in self._outboxes:
            self._outboxes.discard(out)
            for key in ("messages_sent", "frames_sent", "frames_skipped", "bytes_sent"):
                self.counters[key] += getattr(out.stats, key)

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, Any]:
        totals = Counter(self.counters)
        for out in self._outboxes:
            for key in ("messages_sent", "frames_sent", "frames_skipped", "bytes_sent"):
                totals[key] += getattr(out.stats, key)
        live = [s for s in self._sessions.values() if s.attached]
        return {
            "sessions": self.active_sessions,
            "detached": self.detached_sessions,
            "connecting": self._reserved,
            "max_sessions": self.settings.max_sessions,
            "governor_load": round(self.governor.load, 3),
            "governor_scale": round(self.governor.scale, 3),
            "cpu_budget": self.governor.budget,
            "process_cpu_s": round(time.process_time(), 3),  # all threads; Δ over Δuptime_s = cores used
            "reset_cpu_s": round(self._blocking_cpu_s, 3),  # worker-thread share, excluded from the governor
            "uptime_s": round(self.clock() - self.started_at, 1),
            "graph_version": self.graph.graph_version,
            "model_loaded": self.model_loaded,
            "modes": dict(Counter(s.mode for s in live)),
            "speeds": dict(Counter(s.speed for s in live)),
            "sim_cpu_s": round(sum(s.stats.cpu_s for s in self._sessions.values()), 3),
            "totals": {k: int(v) for k, v in sorted(totals.items())},
        }
