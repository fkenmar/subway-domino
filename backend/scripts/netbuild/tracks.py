"""Express-track inference (ARCHITECTURE §2.3 step 4).

Every directed elementary hop has a local (L) track. It also has an express (X) track when
the timetable proves one: two trains cannot change order on a single track, so a scheduled
**overtake** (a trip departs ``u`` after another and reaches ``v`` first, the two stopping
differently in between) proves a second track over the bypass chain ``u..v``. Evidence of
both directions is pooled (``>= MIN_OVERTAKES``): a 4-track trunk has express tracks both
ways and a 3-track middle track serves both directions. Curated corridors from
``infrastructure.json`` add middle tracks that carry no express service in this pick.

Patterns are then assigned a track class per elementary hop. Extension rules only trust an
**alongside local**: a pattern of a *different* route that runs on the L track of the same hop
in a *shared demand-profile window*. That is what makes a hop part of a 4-track trunk, and it
keeps late-night locals of the express route itself (e.g. the overnight Q, or the overnight 4
through the 2-track Lexington Av tunnel) from dragging X onto 2-track connectors.

* **R1 evidence.** A pattern runs X over a bypass range whose hops all have an X track when
  at least one bypassed station is a stop of a concurrent pattern of another route (it
  really passes a local). A local that merely skips a closed platform stays on L.
* **R2 continuation.** Inside a bypass range, X spreads hop by hop from an adjacent X hop
  while an alongside local of that hop also uses the next hop (the D past 135 St, the N/Q
  past 8 St and Prince St; the Manhattan Bridge stays single-track).
* **R3 gap fill.** Up to ``MAX_GAP_FILL_HOPS`` stopping hops between two X runs of one
  pattern stay X when one alongside local runs the whole stretch (2/3 Times Sq -> 34 St).
* **R4 tails.** An X run extends over up to ``MAX_TAIL_HOPS`` stopping hops shared with an
  alongside local when the pattern diverges from that local right after (B/D 47-50 Sts and
  Broadway-Lafayette St, 4/5 Atlantic Av -> Nevins St).

Skip-stop service (J/Z) has no overtakes, so it stays on one track.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from scripts.netbuild.constants import (
    MAX_EXTENSION_HOP_M,
    MAX_GAP_FILL_HOPS,
    MAX_TAIL_HOPS,
    MIN_CONCURRENT_FRACTION,
    MIN_OVERTAKES,
)
from scripts.netbuild.gtfs import GtfsTrip
from scripts.netbuild.patterns import Decomposition, Hop, Pattern

L, X = "L", "X"


@dataclass(frozen=True)
class ResolvedCorridor:
    """A curated station range resolved onto elementary hops (§2.4)."""

    kind: str  # "middle" | "three_track" | "cbtc"
    name: str
    routes: tuple[str, ...]
    stations: tuple[str, ...]  # ordered elementary station chain from_gtfs .. to_gtfs
    peak: str = ""
    group: int = -1  # bidir group id for middle / three-track corridors

    def hops(self) -> list[Hop]:
        return list(zip(self.stations, self.stations[1:]))


@dataclass
class TrackModel:
    classes: dict[int, list[str]]  # pid -> class per path hop
    xtrack: set[Hop]  # directed elementary hops that have an X track
    evidence: dict[Hop, int]  # raw directed overtake counts per elementary hop
    bidir_group: dict[Hop, int]  # X hops of a 3-track / middle-track group
    reasons: dict[Hop, set[str]] = field(default_factory=dict)


def overtake_evidence(trips: list[GtfsTrip], dec: Decomposition, elementary: set[Hop]) -> dict[Hop, int]:
    """Count timetable overtakes on every bypass chain, attributed to its elementary hops."""
    at: dict[str, dict[int, int]] = defaultdict(dict)
    for ti, t in enumerate(trips):
        for i, s in enumerate(t.stations):
            at[s][ti] = i
    evidence: dict[Hop, int] = defaultdict(int)
    for (u, v), chain in sorted(dec.chains.items()):
        interior = set(chain[1:-1])
        a_dep, a_arr, b_dep, b_arr = [], [], [], []
        for ti, i in at.get(u, {}).items():
            j = at.get(v, {}).get(ti)
            if j is None or j <= i:
                continue
            t = trips[ti]
            if j == i + 1:
                a_dep.append(t.dep[i])
                a_arr.append(t.arr[j])
            elif set(t.stations[i + 1 : j]) <= interior:
                b_dep.append(t.dep[i])
                b_arr.append(t.arr[j])
        if not a_dep or not b_dep:
            continue
        ad, aa = np.asarray(a_dep)[:, None], np.asarray(a_arr)[:, None]
        bd, ba = np.asarray(b_dep)[None, :], np.asarray(b_arr)[None, :]
        n = int(np.count_nonzero((ad > bd) & (aa < ba)))
        if n:
            for h in zip(chain, chain[1:]):
                if h in elementary:
                    evidence[h] += n
    return dict(evidence)


def _bypass_ranges(p: Pattern) -> list[tuple[int, int]]:
    """Path-hop index ranges ``[i0, i1]`` of stop-hops that bypass at least one station."""
    out = []
    for k in range(len(p.stops) - 1):
        i0, i1 = p.stop_pos[k], p.stop_pos[k + 1] - 1
        if i1 > i0:
            out.append((i0, i1))
    return out


def _runs(cls: list[str]) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, c in enumerate(cls + [L]):
        if c == X and start is None:
            start = i
        elif c != X and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


class _Inference:
    def __init__(
        self, kept: list[Pattern], xtrack: set[Hop], bidir: dict[Hop, int], hop_len: dict[Hop, float]
    ) -> None:
        self.kept = kept
        self.bidir = bidir
        self.hop_len = hop_len
        self.by_pid = {p.pid: p for p in kept}
        self.hourly = {p.pid: np.bincount([min(23, t.start_s // 3600) for t in p.trips], minlength=24) for p in kept}
        self.hops = {p.pid: p.path_hops() for p in kept}
        self.index = {p.pid: {h: i for i, h in enumerate(self.hops[p.pid])} for p in kept}
        self.cls = {p.pid: [L] * len(self.hops[p.pid]) for p in kept}
        self.ranges = {p.pid: _bypass_ranges(p) for p in kept}
        # Hop indices where the pattern runs non-stop (inside a bypass range).
        self.bypassing = {pid: {i for i0, i1 in r for i in range(i0, i1 + 1)} for pid, r in self.ranges.items()}
        # Routes that run non-stop over each hop in some pattern: their stopping patterns on
        # that hop are local *variants* of an express (e.g. the overnight Q, which crosses to
        # the bridge track south of Prince St), not evidence of a separate local track.
        self.express_routes: dict[Hop, set[str]] = defaultdict(set)
        for p in kept:
            for i in self.bypassing[p.pid]:
                self.express_routes[self.hops[p.pid][i]].add(p.route_id)
        self.xtrack = xtrack
        self.reasons: dict[Hop, set[str]] = defaultdict(set)
        self.users: dict[Hop, list[int]] = defaultdict(list)
        for p in kept:
            for h in self.hops[p.pid]:
                self.users[h].append(p.pid)
        # (station, in-hop) -> patterns that stop there arriving over that hop
        self.stoppers: dict[tuple[str, Hop], list[int]] = defaultdict(list)
        for p in kept:
            for j in range(1, len(p.path)):
                if p.is_stop[j]:
                    self.stoppers[(p.path[j], self.hops[p.pid][j - 1])].append(p.pid)

    # ------------------------------------------------------------------ relations
    def concurrent(self, a: int, b: int) -> bool:
        """Both run at the same time of day: overlapping origin departures per hour cover at
        least ``MIN_CONCURRENT_FRACTION`` of the smaller pattern (and at least one trip)."""
        ha, hb = self.hourly[a], self.hourly[b]
        overlap = int(np.minimum(ha, hb).sum())
        return overlap >= max(1.0, MIN_CONCURRENT_FRACTION * min(int(ha.sum()), int(hb.sum())))

    def passes_local(self, pid: int, i0: int, i1: int) -> bool:
        """Does bypass range ``[i0, i1]`` pass a station where a concurrent local of another route stops?"""
        p, hops = self.by_pid[pid], self.hops[pid]
        return any(
            self.other_route(pid, q) and self.concurrent(pid, q)
            for j in range(i0 + 1, i1 + 1)
            for q in self.stoppers.get((p.path[j], hops[j - 1]), ())
        )

    def extensible(self, pid: int, i: int) -> bool:
        """Gap fills and tails start only from 4-track express runs (never from a middle track)
        and cover only short hops (a river crossing is never an unproven second track)."""
        return self.hops[pid][i] not in self.bidir

    def other_route(self, a: int, b: int) -> bool:
        return self.by_pid[a].route_id != self.by_pid[b].route_id

    def uses_l(self, q: int, h: Hop) -> bool:
        k = self.index[q].get(h)
        return k is not None and self.cls[q][k] == L

    def runs_local(self, q: int, h: Hop) -> bool:
        """``q`` is a true local on ``h``: it runs on L, stops at both ends, and its route never
        runs express over ``h`` (an express not yet classified X, or a local variant of an
        express route, is never treated as one)."""
        k = self.index[q].get(h)
        return (
            k is not None
            and self.cls[q][k] == L
            and k not in self.bypassing[q]
            and self.by_pid[q].route_id not in self.express_routes[h]
        )

    def alongside(self, pid: int, i: int) -> set[int]:
        """Concurrent locals of other routes on the L track of the pattern's hop ``i``."""
        h = self.hops[pid][i]
        return {
            q for q in self.users[h]
            if q != pid and self.other_route(pid, q) and self.concurrent(pid, q) and self.runs_local(q, h)
        }

    def set_x(self, pid: int, i: int, why: str) -> bool:
        if self.cls[pid][i] == X:
            return False
        self.cls[pid][i] = X
        self.reasons[self.hops[pid][i]].add(why)
        return True

    # ------------------------------------------------------------------ rules
    def r1_evidence(self) -> bool:
        changed = False
        for p in self.kept:
            hops, cls = self.hops[p.pid], self.cls[p.pid]
            for i0, i1 in self.ranges[p.pid]:
                if all(c == X for c in cls[i0 : i1 + 1]):
                    continue
                if not all(hops[i] in self.xtrack for i in range(i0, i1 + 1)):
                    continue
                if self.passes_local(p.pid, i0, i1):
                    for i in range(i0, i1 + 1):
                        changed |= self.set_x(p.pid, i, "bypass")
        return changed

    def r2_continuation(self) -> bool:
        changed = False
        for p in self.kept:
            hops, cls, n = self.hops[p.pid], self.cls[p.pid], len(self.hops[p.pid])
            for i0, i1 in self.ranges[p.pid]:
                if not self.passes_local(p.pid, i0, i1):
                    continue
                progress = True
                while progress:
                    progress = False
                    for i in range(i0, i1 + 1):
                        if cls[i] == X:
                            continue
                        for nb in (i - 1, i + 1):
                            if not (0 <= nb < n and i0 - 1 <= nb <= i1 + 1 and cls[nb] == X):
                                continue
                            if any(self.uses_l(q, hops[i]) for q in self.alongside(p.pid, nb)):
                                progress = self.set_x(p.pid, i, "continuation")
                                changed |= progress
                                break
        return changed

    def r3_gap_fill(self) -> bool:
        changed = False
        for p in self.kept:
            hops = self.hops[p.pid]
            runs = _runs(self.cls[p.pid])
            for (_s0, e0), (s1, _e1) in zip(runs, runs[1:]):
                gap = range(e0 + 1, s1)
                if not 0 < len(gap) <= MAX_GAP_FILL_HOPS:
                    continue
                if not (self.extensible(p.pid, e0) and self.extensible(p.pid, s1)):
                    continue
                if any(self.hop_len[hops[g]] > MAX_EXTENSION_HOP_M for g in gap):
                    continue
                common = self.alongside(p.pid, e0) & self.alongside(p.pid, s1)
                if any(all(self.uses_l(q, hops[g]) for g in gap) for q in common):
                    for g in gap:
                        changed |= self.set_x(p.pid, g, "gap")
        return changed

    def _diverges(self, pid: int, j: int, step: int, sharing: set[int]) -> bool:
        """After shared hop ``j`` (walking ``step``), does the pattern part from every sharing local?"""
        hops = self.hops[pid]
        nxt = j + step
        p_next = hops[nxt] if 0 <= nxt < len(hops) else None
        for q in sharing:
            q_hops = self.hops[q]
            qn = self.index[q][hops[j]] + step
            q_next = q_hops[qn] if 0 <= qn < len(q_hops) else None
            # The local ending here is no junction (the B at Bedford Park Blvd); the express
            # ending here while the local runs on is one (the B at Brighton Beach).
            if q_next is None or p_next == q_next:
                return False
        return True

    def r4_tails(self) -> bool:
        changed = False
        for p in self.kept:
            hops, cls, n = self.hops[p.pid], self.cls[p.pid], len(self.hops[p.pid])
            for s, e in _runs(cls):
                for step, end in ((1, e), (-1, s)):
                    if not self.extensible(p.pid, end):
                        continue
                    sharing = self.alongside(p.pid, end)
                    cand: list[int] = []
                    j = end + step
                    while sharing and 0 <= j < n and len(cand) < MAX_TAIL_HOPS and cls[j] == L:
                        if self.hop_len[hops[j]] > MAX_EXTENSION_HOP_M:
                            break
                        sharing = {q for q in sharing if self.uses_l(q, hops[j])}
                        if not sharing:
                            break
                        cand.append(j)
                        if self._diverges(p.pid, j, step, sharing):
                            for k in cand:
                                changed |= self.set_x(p.pid, k, "tail")
                            break
                        j += step
        return changed

    def close_xtrack(self, elementary: set[Hop]) -> bool:
        """Hops run on X have an X track, and an X track exists in both directions."""
        changed = False
        for p in self.kept:
            for i, h in enumerate(self.hops[p.pid]):
                if self.cls[p.pid][i] == X and h not in self.xtrack:
                    self.xtrack.add(h)
                    changed = True
        for h in sorted(self.xtrack):
            rev = (h[1], h[0])
            if rev in elementary and rev not in self.xtrack:
                self.xtrack.add(rev)
                self.reasons[rev].add("reverse")
                changed = True
        return changed


def infer_tracks(
    kept: list[Pattern],
    trips: list[GtfsTrip],
    dec: Decomposition,
    corridors: Iterable[ResolvedCorridor],
    hop_len: dict[Hop, float],
) -> TrackModel:
    """Assign L/X per pattern hop. ``hop_len`` is the straight-line length of each elementary hop."""
    elementary = set(dec.elementary)
    evidence = overtake_evidence(trips, dec, elementary)
    xtrack: set[Hop] = set()
    reasons: dict[Hop, set[str]] = defaultdict(set)
    for h in elementary:
        if evidence.get(h, 0) + evidence.get((h[1], h[0]), 0) >= MIN_OVERTAKES:
            xtrack.add(h)
            reasons[h].add("overtakes")
    bidir: dict[Hop, int] = {}
    for c in corridors:
        if c.kind not in ("middle", "three_track"):
            continue
        for a, b in c.hops():
            for h in ((a, b), (b, a)):
                if h not in elementary:
                    raise ValueError(f"corridor {c.name!r}: hop {h} is not an elementary hop")
                xtrack.add(h)
                reasons[h].add(f"curated:{c.kind}")
                bidir[h] = c.group
    inf = _Inference(kept, xtrack, bidir, hop_len)
    for h, why in reasons.items():
        inf.reasons[h] |= why
    for _ in range(64):
        changed = inf.r1_evidence()
        changed |= inf.r2_continuation()
        changed |= inf.r3_gap_fill()
        changed |= inf.r4_tails()
        changed |= inf.close_xtrack(elementary)
        if not changed:
            break
    else:  # pragma: no cover - rules are monotone (L -> X only), so this cannot happen
        raise RuntimeError("track inference did not converge")
    return TrackModel(
        classes=inf.cls,
        xtrack=inf.xtrack,
        evidence=evidence,
        bidir_group=bidir,
        reasons={h: set(r) for h, r in inf.reasons.items()},
    )
