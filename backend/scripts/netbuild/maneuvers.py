"""Dispatch maneuvers physically available upstream of a disruption (preset design check).

The presets exist to show the AI dispatcher beating the timetable (ARCHITECTURE §2.4), so every
preset disruption must leave the dispatcher something to do. This module reads a ``network.json``
document and, for each disruption, finds the in-service patterns it affects and the maneuvers the
track graph physically allows upstream of it (the masks' need clauses are the simulator's job):

* ``hold`` - a scheduled stop (platform) upstream: hold there instead of queueing in a tunnel;
* ``short_turn`` - a pocket turnback leaving one of the pattern's nodes (not at its terminals);
* ``express`` / ``local`` - an L2X / X2L crossover whose parallel track runs past the whole
  affected stretch, unaffected, along the pattern's stations (RUN_EXPRESS / RUN_LOCAL);
* ``reroute`` - a diverge junction with an unaffected alternate that reaches the pattern beyond
  the disruption or a terminal of the route (same division by construction: no edge crosses);
* ``gap_train`` - a yard of the route whose lead node lies on the pattern.

The footprint follows the disruption semantics of §3.5: blocks take the L (or X) running edges of
a segment, or those entering/leaving a station's L (X) nodes; a closure closes every node of the
station(s); a signal delay affects every edge of the segment or touching the interlocking.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

UPSTREAM_STATIONS = 12  # how far upstream (distinct stations on the pattern's path) maneuvers are sought
REROUTE_MAX_HOPS = 30  # matches the simulator's reroute horizon (§4.2: rejoin within 30 stations)
STRUCTURAL = ("short_turn", "express", "local", "reroute", "gap_train")
MANEUVERS = ("hold",) + STRUCTURAL


@dataclass(frozen=True)
class Footprint:
    edges: frozenset[int]  # running / timed edges blocked or slowed
    closed_nodes: frozenset[int]

    def hits(self, edge: Mapping[str, Any]) -> bool:
        return edge["idx"] in self.edges or edge["to"] in self.closed_nodes


@dataclass(frozen=True)
class PatternManeuvers:
    pattern: int
    route: str
    maneuvers: Mapping[str, tuple[str, ...]]  # kind -> "Station name@stations upstream" (yard names for gap trains)

    @property
    def structural(self) -> bool:
        return any(self.maneuvers.get(k) for k in STRUCTURAL)


class _Graph:
    def __init__(self, doc: Mapping[str, Any], profile: str) -> None:
        self.doc = doc
        self.E = doc["edges"]
        self.N = doc["nodes"]
        self.S = doc["stations"]
        allowed: dict[int, set[int]] = defaultdict(set)
        for pd in doc["peak_dir"]:
            if pd["profile"] == profile:
                allowed[pd["bidir_group"]].update(pd["allowed_dir_edges"])
        # Middle-track (bidir group) X edges are usable only in the profile's peak direction (§3.3).
        self.usable = {e["idx"] for e in self.E if e["bidir_group"] < 0 or e["idx"] in allowed[e["bidir_group"]]}
        self.out_run: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        self.out_all: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for e in self.E:
            self.out_all[e["from"]].append(e)
            if e["kind"] == "run":
                self.out_run[e["from"]].append(e)

    def station_of(self, node: int) -> int:
        return int(self.N[node]["station"])

    def open(self, edge: Mapping[str, Any], fp: Footprint) -> bool:
        return edge["idx"] in self.usable and not fp.hits(edge)


def footprint(doc: Mapping[str, Any], disruption: Mapping[str, Any]) -> Footprint:
    """Edges and nodes a disruption blocks, closes or slows (§3.5)."""
    E, N = doc["edges"], doc["nodes"]
    kind, target = disruption["kind"], disruption["target"]
    typ, idx = target["type"], int(target["idx"])
    track = {"local_block": "L", "express_block": "X"}.get(kind)
    edges: set[int] = set()
    closed: set[int] = set()
    if typ == "segment":
        for ei in doc["segments"][idx]["edges"]:
            if kind == "signal_delay" or E[ei]["track"] == track:
                edges.add(ei)
        return Footprint(frozenset(edges), frozenset())
    if typ == "station":
        stations = {idx}
    elif typ == "interlocking":
        stations = {int(doc["interlockings"][idx]["station"])}
    else:
        stations = set(doc["complexes"][idx]["stations"])
    nodes = {n["idx"] for n in N if n["station"] in stations}
    if kind == "full_station_closure":
        return Footprint(frozenset(), frozenset(nodes))
    for e in E:
        if e["from"] in nodes or e["to"] in nodes:
            if kind == "signal_delay" or (e["kind"] == "run" and e["track"] == track):
                edges.add(e["idx"])
    return Footprint(frozenset(edges), frozenset())


def _pattern_stations(g: _Graph, p: Mapping[str, Any]) -> list[int]:
    return [g.station_of(g.E[p["edges"][0]]["from"])] + [g.station_of(g.E[e]["to"]) for e in p["edges"]]


def _pattern_nodes_by_station(g: _Graph, p: Mapping[str, Any]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    for e in p["edges"]:
        for n in (g.E[e]["from"], g.E[e]["to"]):
            out[g.station_of(n)].add(n)
    return out


def _parallel_bypass(g: _Graph, p: Mapping[str, Any], pos: int, xover: Mapping[str, Any], fp: Footprint) -> bool:
    """RUN_EXPRESS / RUN_LOCAL: the track entered by ``xover`` (taken where pattern edge ``pos`` starts)
    runs open along the pattern's stations past its last affected edge and crosses back onto the
    pattern there (or merges into it)."""
    stations = _pattern_stations(g, p)
    hits = [i for i, e in enumerate(p["edges"]) if fp.hits(g.E[e])]
    if not hits:
        return False
    end = hits[-1] + 1  # index into ``stations`` of the last affected edge's to-station
    own = _pattern_nodes_by_station(g, p)
    j, node = pos, xover["to"]
    while j + 1 < len(stations):
        if stations[j + 1] == stations[j]:  # the pattern's own crossover position
            j += 1
            continue
        nxt = [r for r in g.out_run[node] if g.station_of(r["to"]) == stations[j + 1] and g.open(r, fp)]
        if not nxt:
            return False
        node, j = nxt[0]["to"], j + 1
        if j >= end:
            here = own[stations[j]]
            if node in here or any(x["kind"] == "crossover" and x["to"] in here and g.open(x, fp)
                                   for x in g.out_all[node]):
                return True
    return False


def _reroute(g: _Graph, p: Mapping[str, Any], node: int, fp: Footprint, route_terminals: set[int]) -> bool:
    """REROUTE: from a diverge node, an open path over a non-pattern out-edge reaches a station of the
    pattern beyond the disruption, or a terminal of the route, within ``REROUTE_MAX_HOPS``."""
    hits = [i for i, e in enumerate(p["edges"]) if fp.hits(g.E[e])]
    beyond = {g.station_of(g.E[e]["to"]) for e in p["edges"][hits[-1] + 1:]}
    on_pattern = set(_pattern_stations(g, p))
    own_edges = set(p["edges"])
    for first in g.out_run[node]:
        if first["idx"] in own_edges or not g.open(first, fp):
            continue
        queue = deque([(first["to"], 1)])
        seen = {first["to"]}
        while queue:
            n, hops = queue.popleft()
            st = g.station_of(n)
            if st in beyond or (st in route_terminals and st not in on_pattern):
                return True
            if hops >= REROUTE_MAX_HOPS:
                continue
            for r in g.out_run[n]:
                if r["to"] not in seen and g.open(r, fp):
                    seen.add(r["to"])
                    queue.append((r["to"], hops + 1))
    return False


def upstream_maneuvers(doc: Mapping[str, Any], disruption: Mapping[str, Any], profile: str,
                       max_stations: int = UPSTREAM_STATIONS) -> list[PatternManeuvers]:
    """Maneuvers upstream of ``disruption`` for every pattern in service in ``profile`` that it affects."""
    g = _Graph(doc, profile)
    fp = footprint(doc, disruption)
    served = {row["pattern"] for row in doc["service"][profile]}
    routes = doc["routes"]
    terminals: dict[int, set[int]] = defaultdict(set)
    for q in doc["patterns"]:
        terminals[q["route"]].update((q["origin"], q["dest"]))
    out: list[PatternManeuvers] = []
    for p in doc["patterns"]:
        if p["idx"] not in served:
            continue
        if g.E[p["edges"][0]]["from"] in fp.closed_nodes:
            hit = 0
        else:
            hit = next((i for i, e in enumerate(p["edges"]) if fp.hits(g.E[e])), None)
            if hit is None:
                continue
        found: dict[str, list[str]] = defaultdict(list)
        order: list[int] = []
        ends = {p["origin"], p["dest"]}
        for i in range(hit, -1, -1):
            n = g.E[p["edges"][i]]["from"]
            st = g.station_of(n)
            if n in fp.closed_nodes:
                continue
            if st not in order:
                if len(order) >= max_stations:
                    break
                order.append(st)
            label = f"{g.S[st]['name']}@{order.index(st)}"
            stops_here = i == 0 or bool(p["stops"][i - 1])
            if stops_here and g.N[n]["platform"] >= 0:
                found["hold"].append(label)
            for e in g.out_all[n]:
                if not g.open(e, fp):
                    continue
                if e["kind"] == "turnback" and e["pocket"] and st not in ends:
                    found["short_turn"].append(label)
                elif e["kind"] == "crossover" and _parallel_bypass(g, p, i, e, fp):
                    found["express" if g.N[n]["track"] == "L" else "local"].append(label)
            if len({g.station_of(r["to"]) for r in g.out_run[n]}) >= 2 and _reroute(g, p, n, fp, terminals[p["route"]]):
                found["reroute"].append(label)
        on_path = {g.E[e]["to"] for e in p["edges"]} | {g.E[p["edges"][0]]["from"]}
        for y in doc["yards"]:
            if p["route"] in y["routes"]:
                leads = {g.E[e]["to"] for e in y["lead_edges"] if g.station_of(g.E[e]["from"]) == -1}
                if leads & on_path:
                    found["gap_train"].append(y["name"])
        out.append(PatternManeuvers(
            pattern=int(p["idx"]),
            route=str(routes[p["route"]]["id"]),
            maneuvers={k: tuple(dict.fromkeys(found[k])) for k in MANEUVERS if found.get(k)},
        ))
    return out


def preset_maneuver_summary(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per preset and disruption: affected patterns and the maneuver kinds available (for the manifest)."""
    summary = []
    for preset in doc["presets"]:
        items = []
        for d in preset["disruptions"]:
            pms = upstream_maneuvers(doc, d, preset["profile"])
            kinds = {k: sorted({pm.route for pm in pms if pm.maneuvers.get(k)}) for k in MANEUVERS}
            items.append({
                "kind": d["kind"],
                "target": dict(d["target"]),
                "affected_patterns": len(pms),
                "routes_by_maneuver": {k: v for k, v in kinds.items() if v},
                "nearest": {k: min((loc for pm in pms for loc in pm.maneuvers.get(k, ()) if "@" in loc),
                                   key=lambda s: int(s.rsplit("@", 1)[1]), default=None)
                            for k in ("short_turn", "express", "local", "reroute")},
            })
        summary.append({"id": preset["id"], "profile": preset["profile"], "disruptions": items})
    return summary


def check_preset_maneuvers(doc: Mapping[str, Any]) -> list[str]:
    """Every preset disruption affects in-service patterns and leaves a structural maneuver upstream."""
    errors = []
    for preset in doc["presets"]:
        for j, d in enumerate(preset["disruptions"]):
            pms = upstream_maneuvers(doc, d, preset["profile"])
            where = f"preset {preset['id']} disruption {j} ({d['kind']})"
            if not pms:
                errors.append(f"{where}: affects no pattern in service in {preset['profile']}")
            elif not any(pm.structural for pm in pms):
                errors.append(f"{where}: no short-turn, express/local switch, reroute or gap train upstream")
    return errors


def describe(pms: Sequence[PatternManeuvers]) -> list[str]:
    """Human-readable lines (debugging and reports)."""
    return [f"{pm.route} p{pm.pattern}: " + ", ".join(f"{k} {list(v)[:3]}" for k, v in pm.maneuvers.items())
            for pm in pms]
