"""Track nodes, edges, platforms, segments and interlockings (ARCHITECTURE §2.3 steps 3, 5-9, 12).

**Track nodes are through-route components.** Every pattern pass through a station produces an
incoming occurrence ``(station, in-hop, "i", class)`` and an outgoing occurrence ``(station,
out-hop, "o", class)``. Occurrences are union-found: a pass that keeps its track class joins
its two occurrences, and passes that share an incoming or outgoing hop on the same class share
a node. A pass that changes class keeps two nodes linked by a crossover edge. Physical
direction never comes from ``direction_id`` or the N/S suffix (CLAUDE.md): at each station
the neighbor stations are 2-colored into two *sides* so that every pass crosses from one side
to the other, and disconnected groups (two lines sharing one GTFS station) are aligned by
geometry.

At a class change on a scheduled stop the stop is placed on the X node (expresses use the
express platform of an express station): after the crossover for L->X, before it for X->L.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from scripts.netbuild.constants import LAYOVER_S, TURNBACK_S
from scripts.netbuild.patterns import Hop, Pattern
from scripts.netbuild.tracks import L, X

Occ = tuple[str, Hop, str, str]  # (station, hop, side "i"/"o", track class)
Y = "Y"  # yard node / yard lead track label


class TopologyError(ValueError):
    """The inferred topology violates a structural rule (bad curated data or inference bug)."""


# --------------------------------------------------------------------------- data model


@dataclass
class Node:
    key: tuple
    station: str  # GTFS parent station id ("" for a yard node)
    track: str  # "L" | "X" | "Y"
    direction: int  # 0/1, local to the station (-1 for a yard node)
    division: str
    idx: int = -1
    platform: int = -1
    yard: int = -1


@dataclass
class Edge:
    kind: str  # "run" | "crossover" | "turnback" | "yard_lead"
    src: int  # node key index into Topology.nodes (final idx after finalize)
    dst: int
    track: str
    hop: Hop | None = None
    station: str = ""  # crossover / turnback station
    pocket: bool = False
    cbtc: bool = False
    bidir_group: int = -1
    yard: int = -1
    run_s: float = 0.0
    idx: int = -1


@dataclass
class PatternPath:
    pid: int
    edges: list[int]
    stops: list[int]
    origin: str
    dest: str


@dataclass
class Platform:
    station: str
    direction: int
    nodes: list[int]
    idx: int = -1


@dataclass
class Segment:
    a: str
    b: str
    edges: list[int]
    idx: int = -1


@dataclass
class Interlocking:
    station: str
    edges: list[int]
    kinds: list[str]
    idx: int = -1


@dataclass
class YardSite:
    name: str
    division: str
    xy: tuple[float, float]
    node: int
    lead_edges: list[int]
    routes: tuple[str, ...]
    fleet_cap: int
    deadhead_s: int
    idx: int = -1


@dataclass
class Topology:
    nodes: list[Node]
    edges: list[Edge]
    platforms: list[Platform]
    paths: dict[int, PatternPath]
    next_patterns: dict[int, list[int]]
    segments: list[Segment]
    interlockings: list[Interlocking]
    yards: list[YardSite]
    station_routes: dict[str, set[str]]  # routes that stop at each station
    terminal_stations: set[str]
    turnback_evidence: dict[str, list[str]]  # station -> routes (non-terminal turnback evidence)
    stats: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StationRef:
    gtfs: str
    division: str
    xy: tuple[float, float]


@dataclass(frozen=True)
class TurnbackRequest:
    station: str
    routes: tuple[str, ...]
    source: str  # "evidence" | "curated"


@dataclass(frozen=True)
class YardRequest:
    name: str
    division: str
    xy: tuple[float, float]
    lead_stations: tuple[str, ...]
    routes: tuple[str, ...]
    fleet_cap: int
    deadhead_s: int


# --------------------------------------------------------------------------- union-find helpers


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Occ, Occ] = {}

    def add(self, x: Occ) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: Occ) -> Occ:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: Occ, b: Occ) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            self.parent[hi] = lo


class _ParityUF:
    """Union-find with parity: ``side(a) xor side(b) == diff`` constraints."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.parity: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.parity[x] = 0

    def find(self, x: str) -> tuple[str, int]:
        par = 0
        node = x
        while self.parent[node] != node:
            par ^= self.parity[node]
            node = self.parent[node]
        return node, par

    def union(self, a: str, b: str, diff: int) -> bool:
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            return (pa ^ pb) == diff
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        self.parent[hi] = lo
        self.parity[hi] = pa ^ pb ^ diff
        return True


def _unit(dx: float, dy: float) -> tuple[float, float]:
    n = math.hypot(dx, dy)
    return (0.0, 0.0) if n == 0 else (dx / n, dy / n)


# --------------------------------------------------------------------------- builder


class _Builder:
    def __init__(
        self,
        kept: Sequence[Pattern],
        classes: Mapping[int, list[str]],
        xtrack: set[Hop],
        stations: Mapping[str, StationRef],
        station_order: Sequence[str],
        crossover_s: float,
    ) -> None:
        self.kept = list(kept)
        self.classes = classes
        self.xtrack = xtrack
        self.st = stations
        self.st_rank = {g: i for i, g in enumerate(station_order)}
        self.crossover_s = crossover_s
        self.warnings: list[str] = []
        self.uf = _UnionFind()
        self.passes: set[tuple[str, str, str]] = set()
        self.node_of: dict[Occ, int] = {}
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.edge_index: dict[tuple, int] = {}

    # ------------------------------------------------------------------ occurrences
    def occurrences(self) -> None:
        elementary: set[Hop] = set()
        used_l: set[Hop] = set()
        for p in self.kept:
            hops, cls = p.path_hops(), self.classes[p.pid]
            elementary.update(hops)
            used_l.update(h for h, c in zip(hops, cls) if c == L)
            for j, s in enumerate(p.path):
                occ_in = (s, hops[j - 1], "i", cls[j - 1]) if j > 0 else None
                occ_out = (s, hops[j], "o", cls[j]) if j < len(hops) else None
                for o in (occ_in, occ_out):
                    if o is not None:
                        self.uf.add(o)
                if occ_in and occ_out:
                    self.passes.add((hops[j - 1][0], s, hops[j][1]))
                    if occ_in[3] == occ_out[3]:
                        self.uf.union(occ_in, occ_out)
        # Every elementary hop has an L track; hops in ``xtrack`` also have an X track,
        # which runs straight through a station wherever a pass continues on X-capable hops.
        for h in sorted(elementary):
            self.uf.add((h[0], h, "o", L))
            self.uf.add((h[1], h, "i", L))
        for h in sorted(self.xtrack):
            self.uf.add((h[0], h, "o", X))
            self.uf.add((h[1], h, "i", X))
        for u, s, w in sorted(self.passes):
            if (u, s) in self.xtrack and (s, w) in self.xtrack:
                self.uf.union((s, (u, s), "i", X), (s, (s, w), "o", X))
        missing_l = sorted(elementary - used_l)
        if missing_l:
            self.warnings.append(f"{len(missing_l)} elementary hops have no scheduled L traversal (virtual L track)")
        self.elementary = elementary

    # ------------------------------------------------------------------ directions
    def directions(self) -> dict[Occ, int]:
        comps: dict[Occ, list[Occ]] = defaultdict(list)
        for occ in sorted(self.uf.parent):
            comps[self.uf.find(occ)].append(occ)
        by_station: dict[str, list[list[Occ]]] = defaultdict(list)
        for root in sorted(comps):
            by_station[root[0]].append(comps[root])
        passes_at: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for u, s, w in sorted(self.passes):
            passes_at[s].append((u, w))

        occ_dir: dict[Occ, int] = {}
        for s in sorted(by_station):
            puf = _ParityUF()
            node_groups = by_station[s]
            for group in node_groups:
                for st, hop, side, _ in group:
                    puf.add(hop[0] if side == "i" else hop[1])
            for u, w in passes_at[s]:
                if not puf.union(u, w, 1):
                    self.warnings.append(f"station {s}: pass {u}->{s}->{w} conflicts with side assignment")
            for group in node_groups:
                froms = sorted({hop[0] for _, hop, side, _ in group if side == "i"})
                tos = sorted({hop[1] for _, hop, side, _ in group if side == "o"})
                for f in froms[1:]:
                    puf.union(froms[0], f, 0)
                for t in tos[1:]:
                    puf.union(tos[0], t, 0)
                if froms and tos and not puf.union(froms[0], tos[0], 1):
                    self.warnings.append(f"station {s}: node {froms}->{tos} conflicts with side assignment")
            # Align disconnected side groups geometrically (two lines sharing one station).
            sx, sy = self.st[s].xy
            vec: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
            weight: dict[str, int] = defaultdict(int)
            for group in node_groups:
                for _, hop, side, _ in group:
                    nb = hop[0] if side == "i" else hop[1]
                    root, par = puf.find(nb)
                    nx, ny = self.st[nb].xy
                    # Direction-0 trains come from side-0 neighbors and leave toward side-1 ones.
                    if side == "i":
                        ux, uy = _unit(sx - nx, sy - ny)
                        sign = 1.0 if par == 0 else -1.0
                    else:
                        ux, uy = _unit(nx - sx, ny - sy)
                        sign = 1.0 if par == 1 else -1.0
                    vec[root][0] += sign * ux
                    vec[root][1] += sign * uy
                    weight[root] += 1
            roots = sorted(vec, key=lambda r: (-weight[r], r))
            flip: dict[str, int] = {r: 0 for r in roots}
            if roots:
                ref = vec[roots[0]]
                for r in roots[1:]:
                    if ref[0] * vec[r][0] + ref[1] * vec[r][1] < 0:
                        flip[r] = 1
            for group in node_groups:
                for occ in group:
                    _, hop, side, _ = occ
                    nb = hop[0] if side == "i" else hop[1]
                    root, par = puf.find(nb)
                    side_nb = par ^ flip[root]
                    occ_dir[occ] = side_nb if side == "i" else 1 - side_nb
        return occ_dir

    # ------------------------------------------------------------------ nodes
    def make_nodes(self) -> None:
        occ_dir = self.directions()
        comps: dict[Occ, list[Occ]] = defaultdict(list)
        for occ in sorted(self.uf.parent):
            comps[self.uf.find(occ)].append(occ)
        staged = []
        for root, occs in comps.items():
            s, track = root[0], root[3]
            dirs = sorted({occ_dir[o] for o in occs})
            if len(dirs) != 1:
                self.warnings.append(f"station {s}: node {root} mixes directions {dirs}; using {dirs[0]}")
            staged.append((self.st_rank[s], dirs[0], track, root, occs))
        staged.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        for rank, direction, track, root, occs in staged:
            s = root[0]
            node = Node(key=root, station=s, track=track, direction=direction, division=self.st[s].division)
            node.idx = len(self.nodes)
            self.nodes.append(node)
            for o in occs:
                self.node_of[o] = node.idx

    # ------------------------------------------------------------------ edges
    def add_edge(self, kind: str, src: int, dst: int, *, hop: Hop | None = None, track: str | None = None,
                 station: str = "", pocket: bool = False, run_s: float = 0.0, yard: int = -1) -> int:
        key = (kind, src, dst, hop)
        if key in self.edge_index:
            e = self.edges[self.edge_index[key]]
            e.pocket = e.pocket or pocket
            if kind == "turnback":
                e.run_s = TURNBACK_S if e.pocket else LAYOVER_S
            return self.edge_index[key]
        if self.nodes[src].division != self.nodes[dst].division:
            raise TopologyError(f"{kind} edge {self.nodes[src].key} -> {self.nodes[dst].key} crosses divisions")
        e = Edge(kind=kind, src=src, dst=dst, track=track or self.nodes[src].track, hop=hop, station=station,
                 pocket=pocket, run_s=run_s, yard=yard)
        self.edge_index[key] = len(self.edges)
        self.edges.append(e)
        return self.edge_index[key]

    def run_edges(self) -> None:
        for h in sorted(self.elementary):
            self.add_edge("run", self.node_of[(h[0], h, "o", L)], self.node_of[(h[1], h, "i", L)], hop=h, track=L)
        for h in sorted(self.xtrack):
            self.add_edge("run", self.node_of[(h[0], h, "o", X)], self.node_of[(h[1], h, "i", X)], hop=h, track=X)

    def pattern_paths(self) -> dict[int, PatternPath]:
        paths: dict[int, PatternPath] = {}
        for p in self.kept:
            hops, cls, n = p.path_hops(), self.classes[p.pid], len(p.path_hops())
            edges: list[int] = []
            stops: list[int] = []
            for j, h in enumerate(hops):
                src = self.node_of[(h[0], h, "o", cls[j])]
                dst = self.node_of[(h[1], h, "i", cls[j])]
                arrive_stop = j == n - 1 or p.is_stop[j + 1]
                switch = j + 1 < n and cls[j] != cls[j + 1]
                l_to_x = switch and cls[j] == L
                edges.append(self.edge_index[("run", src, dst, h)])
                stops.append(1 if arrive_stop and not l_to_x else 0)
                if switch:
                    nxt = hops[j + 1]
                    out = self.node_of[(nxt[0], nxt, "o", cls[j + 1])]
                    edges.append(self.add_edge("crossover", dst, out, station=h[1], run_s=self.crossover_s))
                    stops.append(1 if arrive_stop and l_to_x else 0)
            paths[p.pid] = PatternPath(p.pid, edges, stops, p.path[0], p.path[-1])
        return paths


# --------------------------------------------------------------------------- public entry point


def build_topology(
    kept: Sequence[Pattern],
    classes: Mapping[int, list[str]],
    xtrack: set[Hop],
    bidir: Mapping[Hop, int],
    stations: Mapping[str, StationRef],
    station_order: Sequence[str],
    turnback_requests: Iterable[TurnbackRequest],
    yard_requests: Sequence[YardRequest],
    cbtc_hops: set[Hop],
    crossover_s: float,
) -> Topology:
    """Build the track graph. ``cbtc_hops`` holds both directions of every CBTC hop."""
    b = _Builder(kept, classes, xtrack, stations, station_order, crossover_s)
    b.occurrences()
    b.make_nodes()
    b.run_edges()
    paths = b.pattern_paths()
    by_pid = {p.pid: p for p in kept}

    # ---------------------------------------------------------------- stop nodes, platforms
    edges, nodes = b.edges, b.nodes
    stop_nodes: set[int] = set()
    route_stops: dict[tuple[str, str], set[int]] = defaultdict(set)  # (route, station) -> stop nodes
    through_nodes: set[int] = set()  # nodes some pattern passes through (not its first/last)
    for pid, path in paths.items():
        p = by_pid[pid]
        first = edges[path.edges[0]].src
        stop_nodes.add(first)
        route_stops[(p.route_id, nodes[first].station)].add(first)
        for k, (e, stop) in enumerate(zip(path.edges, path.stops)):
            dst = edges[e].dst
            if stop:
                stop_nodes.add(dst)
                route_stops[(p.route_id, nodes[dst].station)].add(dst)
            if k + 1 < len(path.edges):
                through_nodes.add(dst)
    plat_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    for n in sorted(stop_nodes):
        plat_key[(nodes[n].station, nodes[n].direction)].append(n)
    platforms: list[Platform] = []
    for (s, d) in sorted(plat_key, key=lambda k: (b.st_rank[k[0]], k[1])):
        pl = Platform(station=s, direction=d, nodes=sorted(plat_key[(s, d)]), idx=len(platforms))
        platforms.append(pl)
        for n in pl.nodes:
            nodes[n].platform = pl.idx

    # ---------------------------------------------------------------- adjacency helpers
    out_run: dict[int, set[str]] = defaultdict(set)
    in_run: dict[int, set[str]] = defaultdict(set)
    for e in edges:
        if e.kind == "run":
            out_run[e.src].add(nodes[e.dst].station)
            in_run[e.dst].add(nodes[e.src].station)
    station_nodes: dict[str, list[int]] = defaultdict(list)
    for n in nodes:
        station_nodes[n.station].append(n.idx)

    # ---------------------------------------------------------------- crossovers
    xover_before = sum(1 for e in edges if e.kind == "crossover")
    # (1) X-run ends: an express track that begins/ends at a station is reached from/left to
    #     the parallel local track of the same direction.
    for n in list(nodes):
        if n.track != X:
            continue
        parallel = [m for m in station_nodes[n.station] if nodes[m].track == L and nodes[m].direction == n.direction]
        has_x_in = any(nodes[e.src].track == X for e in edges if e.kind == "run" and e.dst == n.idx)
        has_x_out = any(nodes[e.dst].track == X for e in edges if e.kind == "run" and e.src == n.idx)
        if not has_x_in:
            for m in parallel:
                if out_run[m] & out_run[n.idx]:
                    b.add_edge("crossover", m, n.idx, station=n.station, run_s=crossover_s)
        if not has_x_out:
            for m in parallel:
                if in_run[m] & in_run[n.idx]:
                    b.add_edge("crossover", n.idx, m, station=n.station, run_s=crossover_s)
    # (2) Standard placement: express stations (an L and an X node with platforms) get
    #     crossovers in both senses in both directions between parallel L and X nodes.
    standard = 0
    for s in station_order:
        ns = station_nodes.get(s, [])
        if not (any(nodes[n].track == L and nodes[n].platform >= 0 for n in ns)
                and any(nodes[n].track == X and nodes[n].platform >= 0 for n in ns)):
            continue
        standard += 1
        for xn in (n for n in ns if nodes[n].track == X):
            for ln in (n for n in ns if nodes[n].track == L and nodes[n].direction == nodes[xn].direction):
                if (out_run[ln] & out_run[xn]) or (in_run[ln] & in_run[xn]):
                    b.add_edge("crossover", ln, xn, station=s, run_s=crossover_s)
                    b.add_edge("crossover", xn, ln, station=s, run_s=crossover_s)

    # ---------------------------------------------------------------- next patterns
    by_origin: dict[tuple[str, str], list[Pattern]] = defaultdict(list)
    for p in kept:
        by_origin[(p.route_id, p.path[0])].append(p)
    next_patterns: dict[int, list[int]] = {}
    for p in kept:
        last = edges[paths[p.pid].edges[-1]].dst
        cands = []
        for q in by_origin.get((p.route_id, p.path[-1]), []):
            first = edges[paths[q.pid].edges[0]].src
            if nodes[first].direction != nodes[last].direction:
                cands.append(q)
        cands.sort(key=lambda q: (-q.n_trips, q.pid))
        next_patterns[p.pid] = [q.pid for q in cands]

    # ---------------------------------------------------------------- turnbacks
    def relay(a: int, d: int, s: str) -> None:
        pocket = a in through_nodes or d in through_nodes
        b.add_edge("turnback", a, d, station=s, pocket=pocket, run_s=TURNBACK_S if pocket else LAYOVER_S)

    # (a) terminal relays toward the next patterns
    for p in kept:
        last = edges[paths[p.pid].edges[-1]].dst
        for qid in next_patterns[p.pid]:
            relay(last, edges[paths[qid].edges[0]].src, p.path[-1])
    # (b) evidence stations and (c) curated pockets
    req_list = sorted(set(turnback_requests), key=lambda r: (b.st_rank.get(r.station, 1 << 30), r.routes, r.source))
    pocket_stations: dict[str, set[str]] = defaultdict(set)
    for req in req_list:
        if req.station not in station_nodes:
            raise TopologyError(f"turnback at {req.station}: station is not in the network")
        made = 0
        for r in req.routes:
            ns = sorted(route_stops.get((r, req.station), ()))
            for a in ns:
                opp = [d for d in ns if nodes[d].direction != nodes[a].direction]
                same = [d for d in opp if nodes[d].track == nodes[a].track]
                for d in same or opp:
                    b.add_edge("turnback", a, d, station=req.station, pocket=True, run_s=TURNBACK_S)
                    made += 1
        if made == 0:
            msg = f"turnback at {req.station} for {list(req.routes)}: no platform nodes of those routes"
            if req.source == "curated":
                raise TopologyError(msg)
            b.warnings.append(msg)
        else:
            pocket_stations[req.station].update(req.routes)

    # ---------------------------------------------------------------- yards
    yards: list[YardSite] = []
    for yi, y in enumerate(yard_requests):
        node = Node(key=("yard", y.name), station="", track=Y, direction=-1, division=y.division, yard=yi)
        node.idx = len(nodes)
        nodes.append(node)
        lead_nodes = []
        for s in y.lead_stations:
            if s not in station_nodes:
                raise TopologyError(f"yard {y.name!r}: lead station {s} is not in the network")
            for r in y.routes:
                lead_nodes.extend(route_stops.get((r, s), ()))
        lead_nodes = sorted(set(lead_nodes))
        if not lead_nodes:
            raise TopologyError(f"yard {y.name!r}: no platform node of {list(y.routes)} at {list(y.lead_stations)}")
        lead_edges = []
        for n in lead_nodes:
            if nodes[n].division != y.division:
                raise TopologyError(f"yard {y.name!r} ({y.division}) leads to a division-{nodes[n].division} node")
            lead_edges.append(b.add_edge("yard_lead", node.idx, n, track=Y, station=nodes[n].station,
                                         run_s=float(y.deadhead_s), yard=yi))
            lead_edges.append(b.add_edge("yard_lead", n, node.idx, track=Y, station=nodes[n].station,
                                         run_s=float(y.deadhead_s), yard=yi))
        yards.append(YardSite(y.name, y.division, y.xy, node.idx, lead_edges, y.routes, y.fleet_cap, y.deadhead_s))

    # ---------------------------------------------------------------- attributes
    for e in edges:
        if e.kind == "run":
            e.cbtc = e.hop in cbtc_hops
            if e.track == X and e.hop in bidir:
                e.bidir_group = bidir[e.hop]

    # ---------------------------------------------------------------- final edge order
    kind_rank = {"run": 0, "crossover": 1, "turnback": 2, "yard_lead": 3}

    def seg_key(e: Edge) -> tuple:
        if e.kind == "run":
            a, c = nodes[e.src].station, nodes[e.dst].station
            lo, hi = sorted((a, c), key=lambda g: b.st_rank[g])
            return (b.st_rank[lo], b.st_rank[hi], 0 if a == lo else 1, e.track)
        return (e.src, e.dst)

    order = sorted(range(len(edges)), key=lambda i: (kind_rank[edges[i].kind], seg_key(edges[i])))
    remap = {old: new for new, old in enumerate(order)}
    edges[:] = [edges[i] for i in order]
    for i, e in enumerate(edges):
        e.idx = i
    for path in paths.values():
        path.edges = [remap[e] for e in path.edges]
    for y in yards:
        y.lead_edges = sorted(remap[e] for e in y.lead_edges)

    # ---------------------------------------------------------------- segments
    seg_edges: dict[tuple[str, str], list[int]] = defaultdict(list)
    for e in edges:
        if e.kind == "run":
            a, c = nodes[e.src].station, nodes[e.dst].station
            lo, hi = sorted((a, c), key=lambda g: b.st_rank[g])
            seg_edges[(lo, hi)].append(e.idx)
    segments = [Segment(a, c, sorted(es), idx=i) for i, ((a, c), es) in
                enumerate(sorted(seg_edges.items(), key=lambda kv: (b.st_rank[kv[0][0]], b.st_rank[kv[0][1]])))]

    # ---------------------------------------------------------------- interlockings
    il_edges: dict[str, list[int]] = defaultdict(list)
    il_kinds: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        if e.kind == "crossover":
            il_edges[e.station].append(e.idx)
            il_kinds[e.station].add("L2X" if nodes[e.src].track == L else "X2L")
        elif e.kind == "turnback":
            il_edges[e.station].append(e.idx)
            il_kinds[e.station].add("turnback")
    out_edges: dict[int, list[int]] = defaultdict(list)
    for e in edges:
        if e.kind == "run":
            out_edges[e.src].append(e.idx)
    for n in nodes:
        if n.track in (L, X) and len(out_run[n.idx]) >= 2:
            il_kinds[n.station].add("junction")
            il_edges[n.station].extend(out_edges[n.idx])  # the diverging running edges
    interlockings = [
        Interlocking(s, sorted(il_edges.get(s, [])), sorted(il_kinds[s]), idx=i)
        for i, s in enumerate(sorted(il_kinds, key=lambda g: b.st_rank[g]))
    ]

    # ---------------------------------------------------------------- station facts
    station_routes: dict[str, set[str]] = defaultdict(set)
    for r, s in route_stops:
        station_routes[s].add(r)
    passing: dict[str, set[str]] = defaultdict(set)
    ending: dict[str, set[str]] = defaultdict(set)
    for p in kept:
        ending[p.path[0]].add(p.route_id)
        ending[p.path[-1]].add(p.route_id)
        for s in p.path[1:-1]:
            passing[s].add(p.route_id)
    terminal_stations = {s for s, rs in ending.items() if rs - passing.get(s, set())}

    stats = {
        "crossovers_from_patterns": xover_before,
        "standard_placement_stations": standard,
        "patterns_without_next": sum(1 for v in next_patterns.values() if not v),
        "pocket_turnback_stations": len({e.station for e in edges if e.kind == "turnback" and e.pocket}),
    }
    return Topology(
        nodes=nodes,
        edges=edges,
        platforms=platforms,
        paths=paths,
        next_patterns=next_patterns,
        segments=segments,
        interlockings=interlockings,
        yards=yards,
        station_routes={s: set(r) for s, r in station_routes.items()},
        terminal_stations=terminal_stations,
        turnback_evidence={s: sorted(r) for s, r in pocket_stations.items()},
        stats=stats,
        warnings=b.warnings,
    )
