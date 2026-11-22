"""Resolve curated station ranges of ``infrastructure.json`` onto elementary hops (§2.3.4, §2.3.8).

A corridor is ``(routes, from_gtfs, to_gtfs)``. It is resolved to the ordered elementary station
chain between the two stations along the busiest pattern of one of its routes (pass-through
stations included), so CBTC territory and middle tracks follow the physical line even where
the curated routes run express. A corridor that does not resolve fails the build.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from scripts.netbuild.infra import Corridor, Infrastructure, InfrastructureError
from scripts.netbuild.patterns import Hop, Pattern
from scripts.netbuild.tracks import ResolvedCorridor


def resolve_chain(patterns: Sequence[Pattern], routes: Iterable[str], a: str, b: str) -> tuple[str, ...]:
    """Elementary station chain from ``a`` to ``b`` along the busiest pattern of ``routes`` that visits both."""
    wanted = set(routes)
    best: tuple[int, int, tuple[str, ...]] | None = None
    for p in patterns:
        if p.route_id not in wanted:
            continue
        pos = {s: i for i, s in enumerate(p.path)}
        if a not in pos or b not in pos:
            continue
        i, j = pos[a], pos[b]
        chain = p.path[i : j + 1] if i < j else tuple(reversed(p.path[j : i + 1]))
        key = (-p.n_trips, len(chain), chain)
        if best is None or key < best:
            best = key
    if best is None:
        raise InfrastructureError(f"no pattern of {sorted(wanted)} visits both {a} and {b}")
    return best[2]


def chain_hops(chain: Sequence[str]) -> list[Hop]:
    return list(zip(chain, chain[1:]))


def _resolve(c: Corridor, kind: str, group: int, patterns: Sequence[Pattern], elementary: set[Hop]) -> ResolvedCorridor:
    try:
        chain = resolve_chain(patterns, c.routes, c.from_gtfs, c.to_gtfs)
    except InfrastructureError as exc:
        raise InfrastructureError(f"{kind} corridor {c.name!r}: {exc}") from exc
    for h in chain_hops(chain):
        if h not in elementary:
            raise InfrastructureError(f"{kind} corridor {c.name!r}: hop {h} is not an elementary hop")
        if kind != "cbtc" and (h[1], h[0]) not in elementary:
            raise InfrastructureError(f"{kind} corridor {c.name!r}: hop {h} is one-way, a middle track needs both")
    return ResolvedCorridor(kind=kind, name=c.name, routes=c.routes, stations=chain, peak=c.peak, group=group)


def resolve_corridors(
    infra: Infrastructure, patterns: Sequence[Pattern], elementary: set[Hop]
) -> tuple[list[ResolvedCorridor], list[ResolvedCorridor]]:
    """Return ``(track_corridors, cbtc_corridors)``.

    Track corridors (3-track peak groups first, then middle tracks) get consecutive bidir
    group ids in file order. A hop may belong to at most one group.
    """
    track: list[ResolvedCorridor] = []
    group = 0
    for kind, items in (("three_track", infra.three_track_peak), ("middle", infra.middle_tracks)):
        for c in items:
            track.append(_resolve(c, kind, group, patterns, elementary))
            group += 1
    owner: dict[Hop, str] = {}
    for rc in track:
        for a, b in rc.hops():
            for h in ((a, b), (b, a)):
                if h in owner and owner[h] != rc.name:
                    raise InfrastructureError(f"hop {h} is in both {owner[h]!r} and {rc.name!r}")
                owner[h] = rc.name
    cbtc = [_resolve(c, "cbtc", -1, patterns, elementary) for c in infra.cbtc]
    return track, cbtc


def cbtc_hop_set(cbtc: Iterable[ResolvedCorridor], elementary: set[Hop]) -> set[Hop]:
    """Both directions (where they exist) of every hop of the CBTC corridors."""
    out: set[Hop] = set()
    for rc in cbtc:
        for a, b in rc.hops():
            for h in ((a, b), (b, a)):
                if h in elementary:
                    out.add(h)
    return out
