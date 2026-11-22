"""Stop patterns and elementary hops (ARCHITECTURE §2.3 steps 1-2).

A *pattern* is a ``(route, ordered parent-station sequence)`` of weekday trips. A stop-to-stop
hop that bypasses stations is decomposed into the chain of *elementary hops* of a pattern
that stops at the intermediate stations (the chain with the most trips), recursively.

Orientation (which patterns of a route run "the same way") is derived from station order
alone, never from GTFS ``direction_id`` (inconsistent on the shared J/M track, §0).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping

from scripts.netbuild.constants import MAX_CHAIN_DETOUR, MIN_PATTERN_TRIPS, PROFILE_WINDOWS_H
from scripts.netbuild.gtfs import GtfsTrip

Hop = tuple[str, str]


@dataclass
class Pattern:
    route_id: str
    stops: tuple[str, ...]
    trips: list[GtfsTrip]
    orientation: int = 1
    pid: int = -1
    # Demand-profile windows with at least one origin departure (see ``service_profiles``).
    profiles: frozenset[str] = frozenset()
    # Filled by ``expand_paths``: elementary station path (stops and pass-throughs).
    path: tuple[str, ...] = ()
    is_stop: tuple[bool, ...] = ()
    stop_pos: tuple[int, ...] = ()
    # For every path hop i (path[i] -> path[i+1]): index of the stop-hop it belongs to.
    hop_stop_hop: tuple[int, ...] = ()

    @property
    def n_trips(self) -> int:
        return len(self.trips)

    @property
    def origin(self) -> str:
        return self.stops[0]

    @property
    def dest(self) -> str:
        return self.stops[-1]

    def path_hops(self) -> list[Hop]:
        return list(zip(self.path, self.path[1:]))

    def stop_hop_is_bypass(self, k: int) -> bool:
        return self.stop_pos[k + 1] - self.stop_pos[k] > 1


@dataclass
class Decomposition:
    """Elementary decomposition of every stop-to-stop hop of the kept patterns."""

    chains: dict[Hop, tuple[str, ...]]  # bypass hop -> full elementary station chain
    elementary: set[Hop] = field(default_factory=set)
    rejected: dict[Hop, str] = field(default_factory=dict)


def service_profiles(p: Pattern) -> frozenset[str]:
    """Demand profiles in whose window the pattern has at least one origin departure."""
    out = set()
    for prof, (h0, h1) in PROFILE_WINDOWS_H.items():
        lo, hi = h0 * 3600, h1 * 3600
        if any(lo <= t.start_s < hi for t in p.trips):
            out.add(prof)
    return frozenset(out)


def group_patterns(trips: Iterable[GtfsTrip]) -> list[Pattern]:
    groups: dict[tuple[str, tuple[str, ...]], list[GtfsTrip]] = defaultdict(list)
    for t in trips:
        groups[(t.route_id, t.stations)].append(t)
    out = [Pattern(r, s, sorted(ts, key=lambda x: (x.start_s, x.trip_id))) for (r, s), ts in groups.items()]
    out.sort(key=lambda p: (p.route_id, -p.n_trips, p.stops))
    return out


def _relative_orientation(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """+1 if the stations ``a`` and ``b`` share appear in the same order, -1 if reversed, 0 if unknown."""
    pos_b = {s: i for i, s in enumerate(b)}
    seq = [pos_b[s] for s in a if s in pos_b]
    if len(seq) < 2:
        return 0
    concordant = sum(1 for x, y in zip(seq, seq[1:]) if y > x)
    discordant = len(seq) - 1 - concordant
    if concordant == discordant:
        return 0
    return 1 if concordant > discordant else -1


def orient_patterns(patterns: list[Pattern]) -> None:
    """Assign ``orientation`` (+1/-1) per route relative to the route's busiest pattern."""
    by_route: dict[str, list[Pattern]] = defaultdict(list)
    for p in patterns:
        by_route[p.route_id].append(p)
    for plist in by_route.values():
        plist.sort(key=lambda p: (-p.n_trips, p.stops))
        known: list[Pattern] = [plist[0]]
        plist[0].orientation = 1
        pending = plist[1:]
        progress = True
        while pending and progress:
            progress = False
            still: list[Pattern] = []
            for p in pending:
                for ref in known:
                    rel = _relative_orientation(p.stops, ref.stops)
                    if rel:
                        p.orientation = rel * ref.orientation
                        known.append(p)
                        progress = True
                        break
                else:
                    still.append(p)
            pending = still
        for p in pending:  # shares < 2 stations with every oriented pattern: arbitrary but stable
            p.orientation = 1


def select_patterns(patterns: list[Pattern], min_trips: int = MIN_PATTERN_TRIPS) -> list[Pattern]:
    """Keep patterns with >= ``min_trips`` weekday trips, or the only one of their route+orientation."""
    count: dict[tuple[str, int], int] = defaultdict(int)
    for p in patterns:
        count[(p.route_id, p.orientation)] += 1
    return [p for p in patterns if p.n_trips >= min_trips or count[(p.route_id, p.orientation)] == 1]


def _dist(xy: Mapping[str, tuple[float, float]], a: str, b: str) -> float:
    (ax, ay), (bx, by) = xy[a], xy[b]
    return math.hypot(ax - bx, ay - by)


def decompose_hops(kept: list[Pattern], station_xy: Mapping[str, tuple[float, float]]) -> Decomposition:
    """Find, for every bypassing stop-hop, the elementary chain through the skipped stations.

    The chain comes from the pattern(s) with the most trips that stop at the intermediate
    stations between ``u`` and ``v`` in the same direction. When no pattern does (e.g. a
    platform closed in one direction only, as at 52 St and 69 St on the Flushing line in
    this pick), the chain of the reverse direction is used, reversed: the trains still run
    through those stations, they just do not stop.
    """
    positions = [{s: i for i, s in enumerate(p.stops)} for p in kept]
    hops: set[Hop] = {h for p in kept for h in zip(p.stops, p.stops[1:])}
    best: dict[Hop, tuple[str, ...]] = {}
    rejected: dict[Hop, str] = {}
    for u, v in sorted(hops):
        weights: dict[tuple[str, ...], int] = defaultdict(int)
        for p, pos in zip(kept, positions):
            iu, iv = pos.get(u), pos.get(v)
            if iu is not None and iv is not None and iv - iu >= 2:
                weights[p.stops[iu : iv + 1]] += p.n_trips
        if not weights:
            for p, pos in zip(kept, positions):
                iu, iv = pos.get(u), pos.get(v)
                if iu is not None and iv is not None and iu - iv >= 2:
                    weights[tuple(reversed(p.stops[iv : iu + 1]))] += p.n_trips
        if not weights:
            continue
        direct = max(_dist(station_xy, u, v), 1.0)
        ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
        for chain, _w in ranked:
            length = sum(_dist(station_xy, a, b) for a, b in zip(chain, chain[1:]))
            if length <= MAX_CHAIN_DETOUR * direct:
                best[(u, v)] = chain
                break
        else:
            rejected[(u, v)] = f"every candidate chain detours > {MAX_CHAIN_DETOUR}x the direct hop"

    memo: dict[Hop, tuple[str, ...]] = {}

    def expand(hop: Hop, stack: frozenset[Hop]) -> tuple[str, ...]:
        if hop in memo:
            return memo[hop]
        chain = best.get(hop)
        if chain is None or hop in stack:
            return hop
        out: list[str] = [hop[0]]
        for a, b in zip(chain, chain[1:]):
            out.extend(expand((a, b), stack | {hop})[1:])
        result = tuple(out)
        if len(set(result)) != len(result):  # a decomposition must not revisit a station
            rejected[hop] = "recursive decomposition revisits a station"
            result = hop
        memo[hop] = result
        return result

    chains: dict[Hop, tuple[str, ...]] = {}
    for hop in sorted(hops):
        full = expand(hop, frozenset())
        if len(full) > 2:
            chains[hop] = full
    elementary: set[Hop] = set()
    for hop in hops:
        seq = chains.get(hop, hop)
        elementary.update(zip(seq, seq[1:]))
    return Decomposition(chains=chains, elementary=elementary, rejected=rejected)


def expand_paths(kept: list[Pattern], dec: Decomposition) -> None:
    """Fill each pattern's elementary ``path`` with stop flags and stop-hop membership."""
    for p in kept:
        path: list[str] = [p.stops[0]]
        stop_pos = [0]
        hop_owner: list[int] = []
        for k, hop in enumerate(zip(p.stops, p.stops[1:])):
            seq = dec.chains.get(hop, hop)
            path.extend(seq[1:])
            hop_owner.extend([k] * (len(seq) - 1))
            stop_pos.append(len(path) - 1)
        if len(set(path)) != len(path):
            raise ValueError(f"pattern {p.route_id} {p.stops[0]}->{p.stops[-1]} revisits a station after expansion")
        is_stop = [False] * len(path)
        for i in stop_pos:
            is_stop[i] = True
        p.path = tuple(path)
        p.is_stop = tuple(is_stop)
        p.stop_pos = tuple(stop_pos)
        p.hop_stop_hop = tuple(hop_owner)
