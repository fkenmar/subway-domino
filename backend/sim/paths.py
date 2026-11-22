"""Compiled train paths (ARCHITECTURE §3.3).

Every train runs along an explicit *path*: a sequence of edges with stop flags. Paths are
pattern templates, template suffixes (gap trains, short-turn continuations) or custom
rewrites (express/local runs, reroutes). :func:`compile_path` turns an edge sequence into
the flat arrays the engine keeps per train slot:

* per path position ``i`` (edge ``edge[i]``, arrival at its to-node): stop flag, *visit*
  index (station count from the path start), nominal arrival offset and template position;
* per visit ``v`` (station along the path; visit 0 is the origin): station, node, last
  path position at the station, the edge leaving it, stop flag, nominal arrival offset and
  stop index;
* per stop ``j``: station (destination buckets are indexed by stops of the actual path).

Compiling is pure Python over at most a few hundred edges and only happens on dispatch of
a new template (cached) or on a rare path rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sim import constants as C
from sim.graph import Graph, nominal_schedule


@dataclass(frozen=True, eq=False)
class CompiledPath:
    """Arrays describing one path; see the module docstring."""

    edge: np.ndarray  # int16[L]
    stop: np.ndarray  # bool[L]
    visit: np.ndarray  # int16[L]
    sched: np.ndarray  # float32[L] nominal arrival offset at the to-node
    tpos: np.ndarray  # int16[L] template position on the train's pattern, -1 off-template
    v_station: np.ndarray  # int16[V]
    v_node: np.ndarray  # int16[V]
    v_pos: np.ndarray  # int16[V] last path position at the visit (-1: origin, no edge yet)
    v_out: np.ndarray  # int16[V] first edge leaving the visit (sentinel n_edges if none)
    v_stop: np.ndarray  # bool[V]
    v_time: np.ndarray  # float32[V]
    v_sidx: np.ndarray  # int16[V] stop index of the visit, -1 if not a stop
    s_station: np.ndarray  # int16[S]
    s_pos: np.ndarray  # int16[S]

    @property
    def n_edges(self) -> int:
        return int(self.edge.shape[0])

    @property
    def n_visits(self) -> int:
        return int(self.v_station.shape[0])

    @property
    def n_stops(self) -> int:
        return int(self.s_station.shape[0])


def compile_path(
    g: Graph,
    edges: Sequence[int],
    stops: Sequence[bool],
    tpos: Sequence[int] | None = None,
    sched: Sequence[float] | None = None,
) -> CompiledPath:
    """Compile an edge sequence (consecutive edges) into a :class:`CompiledPath`.

    ``sched`` gives the nominal arrival offsets; when omitted it is the nominal schedule
    from departure at the path start (:func:`sim.graph.nominal_schedule`).
    """
    e = np.asarray(edges, dtype=np.int64)
    L = int(e.shape[0])
    if L == 0:
        raise ValueError("empty path")
    st = np.asarray(stops, dtype=bool).copy()
    if st.shape[0] != L:
        raise ValueError("stops length mismatch")
    if tpos is None:
        tp = np.full(L, -1, dtype=np.int16)
    else:
        tp = np.asarray(tpos, dtype=np.int16)
    if sched is None:
        sc = nominal_schedule(g, e.tolist(), st.tolist()).astype(np.float32)
    else:
        sc = np.asarray(sched, dtype=np.float32)

    ef, et = g.edge_from, g.edge_to
    fst, tst = g.edge_from_station, g.edge_to_station
    sentinel = g.n_edges

    visit = np.zeros(L, dtype=np.int16)
    v_station = [int(fst[e[0]])]
    v_node = [int(ef[e[0]])]
    v_pos = [-1]
    v_stop = [True]  # the origin is where the train boards
    v_time = [0.0]
    v_sidx = [-1]
    s_station: list[int] = []
    s_pos: list[int] = []
    v = 0
    for i in range(L):
        ei = int(e[i])
        if int(tst[ei]) != int(fst[ei]):
            v += 1
            v_station.append(int(tst[ei]))
            v_node.append(int(et[ei]))
            v_pos.append(i)
            v_stop.append(False)
            v_time.append(float(sc[i]))
            v_sidx.append(-1)
        else:
            v_pos[v] = i
        visit[i] = v
        if st[i]:
            v_node[v] = int(et[ei])
            v_stop[v] = True
            v_sidx[v] = len(s_station)
            s_station.append(int(tst[ei]))
            s_pos.append(i)
    n_v = v + 1
    v_out = np.full(n_v, sentinel, dtype=np.int16)
    for vv in range(n_v):
        nxt = v_pos[vv] + 1
        if nxt < L:
            v_out[vv] = int(e[nxt])
    return CompiledPath(
        edge=e.astype(np.int16),
        stop=st,
        visit=visit,
        sched=sc,
        tpos=tp,
        v_station=np.asarray(v_station, dtype=np.int16),
        v_node=np.asarray(v_node, dtype=np.int16),
        v_pos=np.asarray(v_pos, dtype=np.int16),
        v_out=v_out,
        v_stop=np.asarray(v_stop, dtype=bool),
        v_time=np.asarray(v_time, dtype=np.float32),
        v_sidx=np.asarray(v_sidx, dtype=np.int16),
        s_station=np.asarray(s_station, dtype=np.int16),
        s_pos=np.asarray(s_pos, dtype=np.int16),
    )


def template_path(g: Graph, pattern: int, start: int = 0) -> CompiledPath:
    """The pattern template from edge position ``start`` (cached on the graph).

    The nominal schedule of a suffix is re-based so that offset 0 is the departure from
    the suffix's first node (the template's dwell at that stop is not included).
    """
    key = ("tmpl_path", int(pattern), int(start))
    cached = g._cache.get(key)
    if cached is not None:
        return cached
    L = int(g.pat_len[pattern])
    if not (0 <= start < L):
        raise ValueError(f"template start {start} outside pattern {pattern} (length {L})")
    edges = g.tmpl_edge[pattern, start:L]
    stops = g.tmpl_stop[pattern, start:L]
    sched = g.tmpl_sched[pattern, start:L].astype(np.float64)
    if start > 0:
        base = float(g.tmpl_sched[pattern, start - 1])
        if bool(g.tmpl_stop[pattern, start - 1]):
            base += C.DWELL_SCHED_S
        sched = np.maximum(sched - base, 0.0)
    tpos = np.arange(start, L, dtype=np.int16)
    cp = compile_path(g, edges.tolist(), stops.tolist(), tpos=tpos, sched=sched)
    g._cache[key] = cp
    return cp


def path_capacity_dims(g: Graph) -> tuple[int, int, int]:
    """(P, V, S) row capacities for per-train path arrays."""
    P = int(g.path_capacity)
    max_vis = int(g.tmpl_visit.max(initial=0)) + 1 if g.n_patterns else 1
    V = int(np.ceil((max_vis + 32) / 8.0) * 8)
    S = int(g.stop_capacity)
    return P, V, max(S, V)
