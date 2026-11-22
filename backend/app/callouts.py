"""Plain-English dispatch callouts (ARCHITECTURE §5.4 — CONTRACT behavior).

Every applied non-CONTINUE AI action yields at most one callout, built only from the
:class:`~sim.engine.ActionEvent` (``detail`` facts) and graph names. The text states the
situation and the action ("Holding a 4 train at 59 St with doors open — the local track
ahead is blocked at Grand Central-42 St, …"); it never claims an inner reasoning.

:class:`CalloutWriter` holds one template family per action, with variants for the main
``detail`` cases. :class:`CalloutGate` enforces the rate limit (≤ 1 callout per 1.5 s real
time per session): actions arriving inside the window are aggregated, and when the window
reopens the most significant pending action is announced with "(+N more dispatch
actions)". A hold that merely extends a hold already announced for the same train at the
same station (the decision recurs every ``HOLD_STEP_S``) is a continuation, not a new
dispatch decision: it yields no callout and is not counted in "+N more" (it still counts in
``metrics.actions``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from sim import constants as C
from sim.engine import ActionEvent
from sim.graph import Graph

from app.config import CALLOUT_MIN_INTERVAL_S

# Which pending action a window announces: the most disruptive maneuver first.
PRIORITY: dict[int, int] = {
    C.A_SHORT_TURN: 6,
    C.A_REROUTE: 5,
    C.A_YARD: 4,
    C.A_EXPRESS: 3,
    C.A_LOCAL: 3,
    C.A_SKIP: 2,
    C.A_HOLD: 1,
}
_HOLD_CONTINUATION_S = C.HOLD_STEP_S + 3 * C.DT_S  # next hold step decided within this sim time
_VOWEL_SOUND_ROUTES = frozenset("AEFHILMNORSX8")


@dataclass(frozen=True, slots=True)
class Callout:
    id: int
    t: float
    train_id: int
    route: str
    action: str
    station_idx: int
    text: str
    more: int


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------
def _rider_count(x: float) -> int:
    n = max(0, int(round(float(x))))
    return int(round(n, -1)) if n >= 100 else n


def format_riders(x: float) -> str:
    """``"1 rider"``, ``"37 riders"``, ``"840 riders"`` (nearest 10 from 100 up), ``"1,250 riders"``."""
    n = _rider_count(x)
    return "1 rider" if n == 1 else f"{n:,} riders"


def riders_verb(x: float, singular: str, plural: str) -> str:
    """``"1 rider rides"`` / ``"37 riders ride"``: the rider count with an agreeing verb."""
    return f"{format_riders(x)} {singular if _rider_count(x) == 1 else plural}"


def format_duration(seconds: float) -> str:
    """``"45 s"``, ``"3.5 min"``, ``"4 min"``, ``"12 min"``."""
    s = max(0.0, float(seconds))
    if s < 60.0:
        return f"{int(round(s))} s"
    m = s / 60.0
    if m < 10.0:
        txt = f"{m:.1f}".rstrip("0").rstrip(".")
        return f"{txt} min"
    return f"{int(round(m))} min"


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# --------------------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------------------
class CalloutWriter:
    """Builds callout text from an :class:`ActionEvent` and the graph's names."""

    def __init__(self, graph: Graph) -> None:
        self.g = graph

    # ---------------------------------------------------------------- names
    def station(self, idx: Any) -> str:
        i = _number(idx)
        if i is None or not (0 <= int(i) < self.g.n_stations):
            return "the next station"
        return self.g.station_name[int(i)]

    def route_id(self, route_idx: int) -> str:
        if 0 <= int(route_idx) < self.g.n_routes:
            return self.g.route_id[int(route_idx)]
        return "?"

    def route_phrase(self, route_idx: int, *, article: bool = True) -> str:
        """``"a 4 train"``, ``"an A train"``, ``"a 6 express train"``, ``"a 42 St Shuttle train"``."""
        r = int(route_idx)
        if not (0 <= r < self.g.n_routes):
            return "a train" if article else "train"
        rid = self.g.route_id[r]
        name = self.g.route_name[r]
        if bool(self.g.route_shuttle[r]) and "shuttle" in name.lower():
            label = name
            vowel = label[:1].upper() in "AEIOU"
        elif len(rid) == 2 and rid.endswith("X"):
            label = f"{rid[0]} express"
            vowel = rid[0] in _VOWEL_SOUND_ROUTES
        else:
            label = rid
            vowel = rid[:1] in _VOWEL_SOUND_ROUTES
        phrase = f"{label} train"
        if not article:
            return phrase
        return f"{'an' if vowel else 'a'} {phrase}"

    def blocked_clause(self, blocked: Mapping[str, Any] | None, here: int) -> str | None:
        """The §4.2 ``blocked_ahead`` fact in words (``None`` when nothing is blocked ahead)."""
        if not blocked:
            return None
        kind = str(blocked.get("kind", ""))
        st = _number(blocked.get("station_idx"))
        st_i = int(st) if st is not None else -1
        name = self.station(st_i)
        same = st_i == int(here)
        if kind in ("local_block", "express_block"):
            track = "local" if kind == "local_block" else "express"
            return f"the {track} track just ahead is blocked" if same else f"the {track} track ahead is blocked at {name}"
        if kind == "full_station_closure":
            return "this station is closed" if same else f"{name} ahead is closed"
        if kind == "signal_delay":
            return "signal problems are slowing trains just ahead" if same else f"signal problems are slowing trains at {name}"
        if kind == "congestion":
            return f"trains are stopped in the tunnel ahead near {name}"
        return None

    # ---------------------------------------------------------------- per action
    def text(self, ev: ActionEvent) -> str:
        d = ev.detail or {}
        route = self.route_phrase(ev.route_idx)
        here = self.station(ev.station_idx)
        clause = self.blocked_clause(d.get("blocked_ahead"), ev.station_idx)
        builder = {
            C.A_HOLD: self._hold,
            C.A_EXPRESS: self._express,
            C.A_LOCAL: self._local,
            C.A_SKIP: self._skip,
            C.A_SHORT_TURN: self._short_turn,
            C.A_REROUTE: self._reroute,
            C.A_YARD: self._yard,
        }.get(int(ev.action))
        if builder is None:
            return f"Dispatching {route} at {here}."
        return builder(ev, d, route, here, clause)

    def _hold(self, ev: ActionEvent, d: Mapping[str, Any], route: str, here: str, clause: str | None) -> str:
        riders = _number(d.get("riders_onboard")) or 0.0
        blocked = d.get("blocked_ahead") or {}
        if clause and blocked.get("kind") == "congestion":
            return f"Holding {route} at {here} with doors open — {clause}, so it waits at the platform instead of joining the queue."
        if clause:
            who = f"its {riders_verb(riders, 'waits', 'wait')}" if riders >= 0.5 else "it waits"
            return f"Holding {route} at {here} with doors open — {clause}, so {who} on a platform instead of in a stalled tunnel."
        gap = _number(d.get("leader_gap_s"))
        headway = _number(d.get("sched_headway_s"))
        step = format_duration(C.HOLD_STEP_S)
        if gap is not None and headway is not None:
            return (f"Holding {route} at {here} with doors open for {step} — the train ahead is only "
                    f"{format_duration(gap)} ahead, with service scheduled every {format_duration(headway)}.")
        return f"Holding {route} at {here} with doors open for {step}."

    def _bypassed(self, d: Mapping[str, Any]) -> int:
        stops = d.get("bypassed_station_idx")
        return len(stops) if isinstance(stops, (list, tuple)) else 0

    def _displaced_sentence(self, d: Mapping[str, Any]) -> str:
        n = _number(d.get("riders_displaced")) or 0.0
        if n < 0.5:
            return ""
        return f" {format_riders(n)} bound for those stops {'rides' if _rider_count(n) == 1 else 'ride'} on to the next stop."

    def _express(self, ev: ActionEvent, d: Mapping[str, Any], route: str, here: str, clause: str | None) -> str:
        head = f"Switching {route} to the express track at {here}"
        if clause:
            head += f" — {clause}"
        rejoin = self.station(d.get("rejoin_station_idx"))
        k = self._bypassed(d)
        tail = f". It rejoins its route at {rejoin}"
        if k:
            tail += f", passing {_plural(k, 'local stop')}"
        return head + tail + "." + (self._displaced_sentence(d) if k else "")

    def _local(self, ev: ActionEvent, d: Mapping[str, Any], route: str, here: str, clause: str | None) -> str:
        head = f"Moving {route} to the local track at {here}"
        if clause:
            head += f" — {clause}"
        rejoin = self.station(d.get("rejoin_station_idx"))
        return head + f". It makes all local stops until {rejoin}."

    def _skip(self, ev: ActionEvent, d: Mapping[str, Any], route: str, here: str, clause: str | None) -> str:
        skipped_idx = _number(d.get("skipped_station_idx"))
        skipped = self.station(skipped_idx)
        blocked = d.get("blocked_ahead") or {}
        closed = (
            blocked.get("kind") == "full_station_closure"
            and skipped_idx is not None
            and _number(blocked.get("station_idx")) == skipped_idx
        )
        if closed:
            return f"Running {route} through {skipped} without stopping — the station is closed."
        n = _number(d.get("riders_overshot")) or 0.0
        text = f"{route[0].upper()}{route[1:]} at {here} will pass {skipped} without stopping"
        if n >= 0.5:
            text += f"; {format_riders(n)} bound there {'rides' if _rider_count(n) == 1 else 'ride'} on to the next stop"
        return text + "."

    def _short_turn(self, ev: ActionEvent, d: Mapping[str, Any], route: str, here: str, clause: str | None) -> str:
        head = f"Short-turning {route} at {here}"
        if clause:
            head += f" — {clause}"
        riders = _number(d.get("riders_forced_off")) or 0.0
        latency = _number(d.get("turnback_latency_s"))
        wait = f" after about {format_duration(latency)}" if latency is not None else ""
        if riders >= 0.5:
            body = (f". {riders_verb(riders, 'steps', 'step')} off to wait for the next train, and it reverses{wait} "
                    "for the crew to change ends and test the brakes.")
        else:
            body = f". The empty train reverses{wait} for the crew to change ends and test the brakes."
        return head + body

    def _reroute(self, ev: ActionEvent, d: Mapping[str, Any], route: str, here: str, clause: str | None) -> str:
        via = self.station(d.get("via_station_idx"))
        head = f"Rerouting {route} at {here} via {via}"
        if clause:
            head += f" — {clause}"
        rejoin_idx = _number(d.get("rejoin_station_idx"))
        rejoin = self.station(rejoin_idx)
        terminal = rejoin_idx is not None and 0 <= int(rejoin_idx) < self.g.n_stations and bool(
            self.g.station_terminal[int(rejoin_idx)]
        )
        tail = f". It runs to {rejoin}." if terminal else f". It rejoins its route at {rejoin}."
        return head + tail + (self._displaced_sentence(d) if self._bypassed(d) else "")

    def _yard(self, ev: ActionEvent, d: Mapping[str, Any], route: str, here: str, clause: str | None) -> str:
        y = _number(d.get("yard_idx"))
        yard = self.g.yard_name[int(y)] if y is not None and 0 <= int(y) < self.g.n_yards else "the yard"
        lead = self.station(d.get("lead_station_idx"))
        dead = _number(d.get("deadhead_s"))
        when = f" in about {format_duration(dead)}" if dead is not None else ""
        return f"Sending a gap train out of {yard} — it enters service as {route} at {lead}{when}."


# --------------------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------------------
def more_suffix(n: int) -> str:
    return f" (+{n} more dispatch action{'' if n == 1 else 's'})" if n > 0 else ""


class CalloutGate:
    """≤ 1 callout per ``min_interval_s`` real seconds; excess actions aggregate as "+N more".

    ``offer`` every applied action as it happens; ``poll`` once per frame cycle. Memory is
    O(1) in the burst size: only the most significant pending action and a count are kept.
    """

    def __init__(
        self,
        writer: CalloutWriter,
        *,
        min_interval_s: float = CALLOUT_MIN_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.writer = writer
        self.min_interval_s = float(min_interval_s)
        self._clock = clock
        self._next_id = 1
        self._last_emit = -math.inf
        self._best: ActionEvent | None = None
        self._best_priority = -1
        self._pending = 0
        self._holds: dict[int, tuple[int, float]] = {}  # train -> (station, sim time of its last hold)

    def reset(self) -> None:
        """Forget pending actions and hold history (the sim was reset; sim time jumped)."""
        self._best = None
        self._best_priority = -1
        self._pending = 0
        self._holds.clear()

    def _is_hold_continuation(self, ev: ActionEvent) -> bool:
        prev = self._holds.get(ev.train_id)
        self._holds[ev.train_id] = (ev.station_idx, ev.t)
        if len(self._holds) > 4 * C.MAX_TRAINS:  # bounded: drop holds that cannot continue anymore
            horizon = ev.t - _HOLD_CONTINUATION_S
            self._holds = {k: v for k, v in self._holds.items() if v[1] >= horizon}
        return prev is not None and prev[0] == ev.station_idx and 0.0 <= ev.t - prev[1] <= _HOLD_CONTINUATION_S

    def offer(self, ev: ActionEvent) -> None:
        if not ev.applied or ev.action == C.A_CONTINUE:
            return
        if ev.action == C.A_HOLD:
            if self._is_hold_continuation(ev):
                return
        else:
            self._holds.pop(ev.train_id, None)
        self._pending += 1
        prio = PRIORITY.get(int(ev.action), 0)
        if self._best is None or prio > self._best_priority:
            self._best, self._best_priority = ev, prio

    @property
    def pending(self) -> int:
        return self._pending

    def poll(self) -> Callout | None:
        """The window's callout if one is due now, else ``None``."""
        ev = self._best
        if ev is None:
            return None
        now = self._clock()
        if now - self._last_emit < self.min_interval_s:
            return None
        more = self._pending - 1
        self._best, self._best_priority, self._pending = None, -1, 0
        self._last_emit = now
        cid = self._next_id
        self._next_id += 1
        return Callout(
            id=cid,
            t=float(ev.t),
            train_id=int(ev.train_id),
            route=self.writer.route_id(ev.route_idx),
            action=C.ACTION_NAMES[int(ev.action)],
            station_idx=int(ev.station_idx),
            text=self.writer.text(ev) + more_suffix(more),
            more=more,
        )
