"""One visitor's simulation and its asyncio loop (ARCHITECTURE §5.2 — CONTRACT).

Every session owns an isolated :class:`~sim.NYCSubwaySim` (plus, in AI mode, a Normal
*twin*); sessions share only the read-only :class:`~sim.Graph` and the single ONNX policy.

**Loop.** One asyncio task per session. Every ``1 / FRAME_HZ`` s it adds
``speed_sim_per_s / FRAME_HZ / DT_S × governor_scale`` ticks to a fractional accumulator
and runs the whole ticks owed (at most ``MAX_TICKS_PER_FRAME``). Each tick:

* **AI mode** — take the tick's ``pending_decisions()`` batch, run ``OnnxPolicy.act`` on the
  whole batch (masked argmax), ``apply_actions``, then ``step(1)``;
* **Normal mode** — ``step(1)`` only (every decision resolves as CONTINUE_SCHEDULED);
* **twin** (AI mode only) — ``twin.step(1)``; its decisions resolve as CONTINUE.

After the ticks: callouts (rate-limited), disruption expiry events, one binary frame
(skipped while the previous one is still being written) and, at ``METRICS_HZ``, a metrics
message comparing the live sim with its twin.

**Twin.** Created with ``sim.copy()`` when the session switches to AI (or is reset/restored
while in AI mode); user disruption changes are mirrored to it; dropped on the switch back to
Normal, on reset and on a profile change.

**Messages** are handled on the session task between frame cycles, so the simulation is
only ever touched by one coroutine. Resets (profile change, preset, reset) run the sim's
warm-up in a worker thread so other sessions keep their frame rate meanwhile; while one is
in flight nothing else reads the sim (a socket that reattaches meanwhile gets its metrics
from the reset handler). A reset ends every active disruption, so each one is reported as
``disruption_cleared`` before the new state's ``disruptions`` list. A preset is served as
blockages that last until the user clears them (:func:`as_blockages`).

There is never a heuristic fallback: without a model, or after an inference error, AI mode
is unavailable and the client is told so (``ai_unavailable``).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, TypeVar

import numpy as np

from sim import NYCSubwaySim, Scenario, constants as C, preset_scenario
from sim.engine import ActionEvent
from sim.graph import Graph

from app import protocol as P
from app.callouts import CalloutGate, CalloutWriter
from app.config import FRAME_HZ, MAX_TICKS_PER_FRAME, METRICS_HZ, Settings
from app.ratelimit import TokenBucket
from app.transport import CLOSE_INTERNAL_ERROR, CLOSE_NORMAL, Outbox

log = logging.getLogger("app.session")

T = TypeVar("T")

RESET_RATE_PER_S = 0.5  # sustained resets (preset / reset / set_profile) per session ...
RESET_BURST = 3  # ... after a burst of this many


class PolicyLike(Protocol):
    def act(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray: ...


RunBlocking = Callable[..., Awaitable[Any]]


async def run_inline(fn: Callable[..., T], *args: Any) -> T:
    """Default :data:`RunBlocking`: call ``fn`` on the event loop (tests, tiny graphs)."""
    return fn(*args)


def as_blockages(scenario: Scenario) -> Scenario:
    """A preset as the UI serves it: every disruption is a blockage that lasts until cleared.

    ``duration_s`` becomes ``None`` (no end) and a ``signal_delay`` becomes a ``local_block`` on
    the same target (both kinds accept segments and interlockings, §3.5). Only the served copy
    changes: ``network.json`` and :func:`sim.preset_scenario` keep their timed presets, so
    ``graph_version`` and the WP2 calibration presets are untouched.
    """
    return replace(scenario, disruptions=tuple(
        replace(d, kind="local_block" if d.kind == "signal_delay" else d.kind, duration_s=None)
        for d in scenario.disruptions))


@dataclass(frozen=True, slots=True)
class SessionContext:
    """What every session shares: read-only graph, the one policy, settings and services."""

    graph: Graph
    policy: PolicyLike | None
    settings: Settings
    speed_scale: Callable[[], float] = lambda: 1.0  # CPU governor factor (same for every session)
    run_blocking: RunBlocking = run_inline  # offload sim construction/resets (manager: worker thread)
    clock: Callable[[], float] = time.monotonic  # real time (idle timer, callout rate limit)
    writer: CalloutWriter | None = None

    def callout_writer(self) -> CalloutWriter:
        return self.writer if self.writer is not None else CalloutWriter(self.graph)


@dataclass(slots=True)
class SessionStats:
    ticks: int = 0
    cycles: int = 0
    cpu_s: float = 0.0  # event-loop CPU spent in this session's frame cycles
    decisions: int = 0
    actions_applied: int = 0
    callouts: int = 0
    resets: int = 0


def ticks_due(acc: float, speed_sim_per_s: float, scale: float) -> tuple[int, float]:
    """§5.2 accumulator: ``(whole ticks to run now, new fractional accumulator)``.

    Owed ticks per frame are ``speed / FRAME_HZ / DT_S`` scaled by the governor; at most
    ``MAX_TICKS_PER_FRAME`` run per frame and any backlog beyond the cap is dropped (only its
    fraction is kept), so a stalled loop never bursts afterwards.
    """
    total = acc + max(0.0, speed_sim_per_s) / FRAME_HZ / C.DT_S * max(0.0, scale)
    whole = int(total)
    if whole > MAX_TICKS_PER_FRAME:
        return MAX_TICKS_PER_FRAME, total - math.floor(total)
    return whole, total - whole


class Session:
    """A visitor's isolated simulation; create with :meth:`create` (async)."""

    def __init__(
        self,
        ctx: SessionContext,
        *,
        session_id: str,
        sim: NYCSubwaySim,
        speed: P.Speed = "1x",
        profile: P.Profile = "am_peak",
    ) -> None:
        self.ctx = ctx
        self.session_id = session_id
        self.sim = sim
        self.twin: NYCSubwaySim | None = None
        self.mode: P.Mode = "normal"
        self.speed: P.Speed = speed
        self.profile: P.Profile = profile
        self.ai_since_t: float | None = None
        self.stats = SessionStats()
        self.created_at = ctx.clock()
        self._out: Outbox | None = None
        self._attached = asyncio.Event()
        self._inbox: deque[Any] = deque()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._seq = 0
        self._acc = 0.0
        self._last_activity = ctx.clock()
        self._gate = CalloutGate(ctx.callout_writer(), clock=ctx.clock)
        self._resets = TokenBucket(RESET_RATE_PER_S, RESET_BURST, clock=ctx.clock)
        self._policy_failed = False
        self._resetting = False  # a reset is running in the worker thread: the sim is off-limits
        self._known: dict[int, P.DisruptionInfo] = {}  # active disruptions as last reported
        self._twin_ids: dict[int, int] = {}  # live disruption id -> twin disruption id
        self._next_expiry = math.inf
        self._sync_known()

    # ================================================================== construction
    @classmethod
    async def create(cls, ctx: SessionContext, *, session_id: str, restore: P.Restore | None) -> "Session":
        """Build a session from the client's ``restore`` payload (settings + disruptions)."""
        r = restore if restore is not None else P.Restore()
        sim = await ctx.run_blocking(NYCSubwaySim, ctx.graph, Scenario(seed=0, profile=r.profile))
        s = cls(ctx, session_id=session_id, sim=sim, speed=r.speed, profile=r.profile)
        skipped = 0
        for d in r.disruptions[: ctx.settings.max_disruptions]:
            try:
                sim.add_disruption(d.kind, d.target.type, d.target.idx, d.duration_s)
            except ValueError:
                skipped += 1  # e.g. indices from another graph_version: restore what still applies
        if skipped:
            log.info("session=%s restore skipped %d invalid disruption(s)", session_id, skipped)
        s._sync_known()
        if r.mode == "ai" and s.ai_available:
            s._enter_ai()
        return s

    # ================================================================== properties
    @property
    def attached(self) -> bool:
        return self._out is not None

    @property
    def closed(self) -> bool:
        return self._closed

    def bound_to(self, out: Outbox) -> bool:
        """True when ``out`` is the socket this session currently writes to."""
        return self._out is out

    @property
    def ai_available(self) -> bool:
        return self.ctx.policy is not None and not self._policy_failed

    @property
    def last_activity(self) -> float:
        return self._last_activity

    def settings_dict(self) -> P.SettingsDict:
        return {"mode": self.mode, "speed": self.speed, "profile": self.profile}

    # ================================================================== attach / detach
    def attach(self, out: Outbox, *, resume_token: str, resumed: bool, model_loaded: bool) -> None:
        """Bind a connected socket: welcome, current disruptions and metrics go out first."""
        self._out = out
        self._last_activity = self.ctx.clock()
        self._inbox.clear()
        out.send_json(
            P.welcome_message(
                session_id=self.session_id,
                resume_token=resume_token,
                resumed=resumed,
                graph_version=self.ctx.graph.graph_version,
                model_loaded=model_loaded,
                ai_available=self.ai_available,
                settings=self.settings_dict(),
                idle_timeout_s=self.ctx.settings.idle_timeout_s,
                frame_hz=FRAME_HZ,
            )
        )
        out.send_json(P.disruptions_message(list(self._known.values())))
        if not self._resetting:  # else the worker thread owns the sim; the reset handler sends metrics
            self._send_metrics()
        self._attached.set()
        self._wake.set()

    def detach(self) -> None:
        """Unbind the socket; the session pauses (no ticks) until reattached or closed."""
        self._out = None
        self._attached.clear()
        self._inbox.clear()
        self._wake.set()

    def start(self) -> None:
        if self._task is None and not self._closed:
            self._task = asyncio.get_running_loop().create_task(self._run(), name=f"session-{self.session_id}")

    async def close(self) -> None:
        """Stop the loop and release the simulations (idempotent)."""
        self._closed = True
        self._out = None
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self.twin = None

    # ================================================================== client input
    def touch(self) -> None:
        """Any client message except ``ping`` resets the idle timer (§5.2)."""
        self._last_activity = self.ctx.clock()

    def submit(self, msg: Any) -> None:
        """Queue a validated client message for the session task (handled in order)."""
        if self._out is None or self._closed:
            return
        self._inbox.append(msg)
        self._wake.set()

    # ================================================================== loop
    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        period = 1.0 / FRAME_HZ
        metrics_period = 1.0 / METRICS_HZ
        next_frame = loop.time()
        next_metrics = next_frame + metrics_period
        idle_s = self.ctx.settings.idle_timeout_s
        try:
            while not self._closed:
                out = self._out
                if out is None or not out.open:
                    self._attached.clear()
                    await self._attached.wait()
                    next_frame = loop.time()
                    next_metrics = next_frame + metrics_period
                    continue
                if self._inbox:
                    await self._handle(self._inbox.popleft())
                    continue
                if self.ctx.clock() - self._last_activity >= idle_s:
                    self._idle_out(out)
                    continue
                now = loop.time()
                if now >= next_frame:
                    send_metrics = now >= next_metrics
                    self._cycle(out, send_metrics=send_metrics)
                    if send_metrics:
                        next_metrics = max(next_metrics + metrics_period, now)
                    next_frame += period
                    if now - next_frame > period:  # more than a cycle behind: resync, don't burst
                        next_frame = now + period
                    await asyncio.sleep(0)  # one cycle at a time: the other sessions and writers run
                    continue
                self._wake.clear()
                if self._inbox or self._out is not out:
                    continue
                try:
                    async with asyncio.timeout(next_frame - now):
                        await self._wake.wait()
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("session=%s crashed; closing it", self.session_id)
            self._closed = True
            out = self._out
            if out is not None:
                out.close_soon(CLOSE_INTERNAL_ERROR, "internal error")

    def _idle_out(self, out: Outbox) -> None:
        log.info("session=%s idle for %.0f s: closing socket", self.session_id, self.ctx.settings.idle_timeout_s)
        out.send_json(P.idle_timeout_message())
        out.close_soon(CLOSE_NORMAL, "idle timeout")
        self._attached.clear()

    # ================================================================== frame cycle
    def _cycle(self, out: Outbox, *, send_metrics: bool) -> None:
        cpu0 = time.thread_time()
        self.stats.cycles += 1
        n, self._acc = ticks_due(self._acc, C.SPEEDS[self.speed], self.ctx.speed_scale())
        if n:
            for ev in self._advance(n):
                self._gate.offer(ev)
            if self.sim.clock_s >= self._next_expiry:
                self._report_expired(out)
        callout = self._gate.poll()
        if callout is not None:
            self.stats.callouts += 1
            out.send_json(P.events_message([P.callout_item(
                id=callout.id, t=callout.t, train_id=callout.train_id, route=callout.route, action=callout.action,
                station_idx=callout.station_idx, text=callout.text, more=callout.more,
            )]))
        if not out.frame_pending:  # skip-on-backpressure: don't even encode a frame we'd drop
            frame = P.encode_frame(
                self.sim.frame(), seq=self._seq, clock_s=self.sim.clock_s, mode=self.mode, speed=self.speed,
                twin_active=self.twin is not None,
            )
            if out.offer_frame(frame):
                self._seq = (self._seq + 1) & 0xFFFFFFFF
        else:
            out.stats.frames_skipped += 1
        if send_metrics:
            self._send_metrics()
        self.stats.cpu_s += time.thread_time() - cpu0

    def _advance(self, n: int) -> list[ActionEvent]:
        """Run ``n`` ticks (§5.2 "Each tick"); returns the applied AI actions' events."""
        events: list[ActionEvent] = []
        sim, twin = self.sim, self.twin
        for _ in range(n):
            if self.mode == "ai":
                events.extend(self._decide())
            sim.step(1)
            if twin is not None:
                twin.step(1)
        self.stats.ticks += n
        return events

    def _decide(self) -> list[ActionEvent]:
        batch = self.sim.pending_decisions()
        if len(batch) == 0:
            return []
        policy = self.ctx.policy
        assert policy is not None  # AI mode is only entered with a policy
        try:
            actions = np.asarray(policy.act(batch.obs, batch.masks), dtype=np.int64).reshape(-1)
            if actions.shape[0] != len(batch):
                raise ValueError(f"policy returned {actions.shape[0]} actions for {len(batch)} decisions")
        except Exception:
            log.exception("session=%s policy inference failed; AI mode disabled for this session", self.session_id)
            self._policy_failed = True
            self._leave_ai()
            self._send(P.error_message(
                "ai_unavailable", "The AI Dispatcher stopped after an internal error. Normal Schedule is running."
            ))
            return []
        self.stats.decisions += len(batch)
        applied = [ev for ev in self.sim.apply_actions(batch.train_ids, actions) if ev.applied]
        self.stats.actions_applied += len(applied)
        return applied

    # ================================================================== outbound helpers
    def _send(self, message: dict[str, Any]) -> None:
        out = self._out
        if out is not None:
            out.send_json(message)

    def _send_metrics(self) -> None:
        live = P.metric_values(self.sim.metrics())
        twin = P.metric_values(self.twin.metrics()) if self.twin is not None else None
        self._send(P.metrics_message(
            t=self.sim.clock_s, mode=self.mode, effective_speed=self.ctx.speed_scale(), live=live, twin=twin,
            ai_since_t=self.ai_since_t, actions=self.sim.action_counts(),
        ))

    def _send_disruptions(self) -> None:
        self._send(P.disruptions_message(list(self._known.values())))

    def _send_events(self, items: list[P.EventItem]) -> None:
        if items:
            self._send(P.events_message(items))

    def _error(self, code: P.ErrorCode, message: str) -> None:
        self._send(P.error_message(code, message))

    # ================================================================== disruptions bookkeeping
    def _sync_known(self) -> None:
        self._known = {int(d["id"]): P.disruption_info(d) for d in self.sim.disruptions()}
        self._update_next_expiry()

    def _update_next_expiry(self) -> None:
        ends = [d["end_t"] for d in self._known.values() if d["end_t"] is not None]
        self._next_expiry = min(ends) if ends else math.inf

    def _report_expired(self, out: Outbox) -> None:
        current = {int(d["id"]) for d in self.sim.disruptions()}
        gone = [i for i in self._known if i not in current]
        if gone:
            items: list[P.EventItem] = []
            for i in gone:
                items.append(P.disruption_event("disruption_expired", self._known.pop(i)))
                self._twin_ids.pop(i, None)  # the twin's copy expires on the same tick
            out.send_json(P.events_message(items))
            out.send_json(P.disruptions_message(list(self._known.values())))
        self._update_next_expiry()

    # ================================================================== AI mode / twin
    def _enter_ai(self) -> None:
        self.mode = "ai"
        self.twin = self.sim.copy()
        self._twin_ids = {i: i for i in self._known}  # the copy carries the same disruption ids
        self.ai_since_t = float(self.sim.clock_s)

    def _leave_ai(self) -> None:
        self.mode = "normal"
        self.twin = None
        self._twin_ids = {}
        self.ai_since_t = None

    async def _reset_sim(self, scenario: Scenario) -> list[P.EventItem]:
        """Reset to ``scenario`` (warm-up off the event loop); the twin restarts in AI mode.

        Returns a ``disruption_cleared`` event for every disruption the reset ended.
        """
        was_ai = self.mode == "ai"
        cleared: list[P.EventItem] = [P.disruption_event("disruption_cleared", d) for d in self._known.values()]
        self.twin = None
        self._twin_ids = {}
        self._resetting = True
        try:
            await self.ctx.run_blocking(self.sim.reset, scenario)
        finally:
            self._resetting = False
        self.stats.resets += 1
        self.profile = scenario.profile  # type: ignore[assignment]
        self._acc = 0.0
        self._gate.reset()
        self._sync_known()
        if was_ai and self.mode == "ai":
            self._enter_ai()
        return cleared

    # ================================================================== message handlers
    async def _handle(self, msg: Any) -> None:
        handler = _HANDLERS.get(type(msg))
        if handler is None:
            self._error("bad_request", f"unexpected message {getattr(msg, 'type', '?')!r}")
            return
        await handler(self, msg)

    async def _on_set_mode(self, msg: P.SetMode) -> None:
        if msg.mode == "ai":
            if not self.ai_available:
                self._error("ai_unavailable", "The AI Dispatcher is unavailable right now (no model loaded).")
                return
            if self.mode != "ai":
                self._enter_ai()
                log.info("session=%s AI on at t=%.0f", self.session_id, self.sim.clock_s)
        elif self.mode != "normal":
            self._leave_ai()
            log.info("session=%s AI off", self.session_id)
        self._send_metrics()

    async def _on_set_speed(self, msg: P.SetSpeed) -> None:
        self.speed = msg.speed  # the next frame header carries it

    def _reset_allowed(self) -> bool:
        if self._resets.take():
            return True
        self._error("rate_limited", "Too many resets in a row — please wait a moment.")
        return False

    async def _on_set_profile(self, msg: P.SetProfile) -> None:
        if not self._reset_allowed():
            return
        self._send_events(await self._reset_sim(Scenario(seed=0, profile=msg.profile)))
        self._send_disruptions()
        self._send_metrics()

    async def _on_reset(self, msg: P.Reset) -> None:
        if not self._reset_allowed():
            return
        self._send_events(await self._reset_sim(Scenario(seed=0, profile=self.profile)))
        self._send_disruptions()
        self._send_metrics()

    async def _on_preset(self, msg: P.Preset) -> None:
        graph = self.ctx.graph
        if msg.id not in graph.preset_index:
            self._error("bad_request", f"Unknown preset {msg.id!r}.")
            return
        if not self._reset_allowed():
            return
        cleared = await self._reset_sim(as_blockages(preset_scenario(msg.id, graph)))
        self._send_events(cleared + [P.disruption_event("disruption_started", d) for d in self._known.values()])
        self._send_disruptions()
        self._send_metrics()

    async def _on_add_disruption(self, msg: P.AddDisruption) -> None:
        limit = self.ctx.settings.max_disruptions
        if len(self._known) >= limit:
            self._error("bad_request", f"At most {limit} disruptions can be active at once.")
            return
        try:
            did = self.sim.add_disruption(msg.kind, msg.target.type, msg.target.idx, msg.duration_s)
        except ValueError as exc:
            self._error("invalid_target", f"{exc}.")
            return
        if self.twin is not None:
            self._twin_ids[did] = self.twin.add_disruption(msg.kind, msg.target.type, msg.target.idx, msg.duration_s)
        info = next(P.disruption_info(d) for d in self.sim.disruptions() if int(d["id"]) == did)
        self._known[did] = info
        self._update_next_expiry()
        self._send_events([P.disruption_event("disruption_started", info)])
        self._send_disruptions()

    async def _on_clear_disruption(self, msg: P.ClearDisruption) -> None:
        info = self._known.pop(msg.id, None)
        if info is not None:  # unknown ids (e.g. it just expired) are a no-op: clear is idempotent
            self.sim.clear_disruption(msg.id)
            twin_id = self._twin_ids.pop(msg.id, None)
            if self.twin is not None and twin_id is not None:
                self.twin.clear_disruption(twin_id)
            self._update_next_expiry()
            self._send_events([P.disruption_event("disruption_cleared", info)])
        self._send_disruptions()

    async def _on_clear_all(self, msg: P.ClearAll) -> None:
        items: list[P.EventItem] = [P.disruption_event("disruption_cleared", d) for d in self._known.values()]
        self.sim.clear_all_disruptions()
        if self.twin is not None:
            self.twin.clear_all_disruptions()
        self._known = {}
        self._twin_ids = {}
        self._next_expiry = math.inf
        self._send_events(items)
        self._send_disruptions()

    async def _on_activity(self, msg: P.Activity) -> None:
        return None  # the endpoint already reset the idle timer


_Handler = Callable[[Session, Any], Awaitable[None]]
_HANDLERS: dict[type, _Handler] = {
    P.SetMode: Session._on_set_mode,
    P.SetSpeed: Session._on_set_speed,
    P.SetProfile: Session._on_set_profile,
    P.Reset: Session._on_reset,
    P.Preset: Session._on_preset,
    P.AddDisruption: Session._on_add_disruption,
    P.ClearDisruption: Session._on_clear_disruption,
    P.ClearAll: Session._on_clear_all,
    P.Activity: Session._on_activity,
}


__all__ = [
    "Session",
    "SessionContext",
    "SessionStats",
    "as_blockages",
    "PolicyLike",
    "RunBlocking",
    "run_inline",
    "ticks_due",
]
