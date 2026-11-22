"""``NYCSubwaySim`` — the shared NYC subway simulation engine (ARCHITECTURE §3, §4.1-§4.4).

One instance simulates the whole network (or a Stage-1 route subset) in fixed ``DT_S``
ticks. State is struct-of-arrays over ``MAX_TRAINS`` train slots; per-tick kinematics,
boarding and costs are vectorized NumPy. Rare structural changes (dispatch, path rewrites
by actions, disruptions) are Python-side.

Tick phases (``_tick``):

1. scheduled disruptions start/expire;
2. timers: dwell/hold/crossover/key-by/short-turn timers; dwells that end make trains
   *ready to depart* (decision candidates);
3. trains standing at a timed edge (crossover, turnback) try to enter it;
4. kinematics: movement authority from one braking-curve kernel (leader under fixed-block
   or CBTC rules, red signals at blocked edges, closed stations, stops, key-by points),
   closed-form motion over the tick, merge resolution, arrivals;
5. passengers: arrivals at platforms, wait-age shift, spillover;
6. costs (PRD §4.3) and tunnel-stuck bookkeeping;
7. clock advances; terminal arrivals retire to their pool; dispatches, put-ins and yard
   gap trains spawn;
8. occupancy tables (who leads whom on each edge) for the next tick and the decisions;
9. decisions: masks for ready trains; trains with >= 1 valid non-CONTINUE action form the
   tick's :class:`DecisionBatch`.

Normal Schedule is literally "every decision resolves as CONTINUE": unanswered decisions
are resolved as CONTINUE at the start of the next step, and CONTINUE has no side effect.
"""

from __future__ import annotations

import copy as _copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from sim import constants as C
from sim import disruptions as D
from sim.cost import I_TOTAL, tick_cost_vectors, tick_costs
from sim.graph import KIND_RUN, KIND_TURNBACK, TRACK_L, TRACK_X, Graph, ServiceView, shift_buckets
from sim.kinematics import fixed_block_authority, motion_profile
from sim.paths import CompiledPath, compile_path, path_capacity_dims, template_path
from sim.scenarios import Scenario, ScheduledDisruption

M_RUN, M_DWELL, M_READY, M_TIMED = 0, 1, 2, 3
TK_NONE, TK_EDGE, TK_KEYBY = 0, 1, 2
INF = math.inf
DT = C.DT_S
_Q_TAIL = math.exp(-1.0 / C.DEST_DECAY_STOPS)
_K_LOOK = 1 + C.LOOKAHEAD_EDGES
_AGE_SHIFT_TICKS = int(round(C.AGE_BIN_S / C.DT_S))
_DEC11 = np.exp(-np.arange(C.N_DEST - 1) / C.DEST_DECAY_STOPS)
_NON_CONTINUE = C.ACTION_NAMES[1:]


# --------------------------------------------------------------------------------------
# Public value types
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DecisionBatch:
    train_ids: np.ndarray  # int32[n]
    obs: np.ndarray  # float32[n, OBS_DIM], values in [-1, 1]
    masks: np.ndarray  # bool[n, N_ACTIONS]
    context: list[dict]  # per decision: {"station_idx", "route_idx", "node_idx"}

    def __len__(self) -> int:
        return int(self.train_ids.shape[0])


@dataclass(frozen=True)
class ActionEvent:
    t: float
    train_id: int
    route_idx: int
    action: int
    station_idx: int
    applied: bool
    detail: dict = field(default_factory=dict)


def empty_batch() -> DecisionBatch:
    return DecisionBatch(
        train_ids=np.zeros(0, dtype=np.int32),
        obs=np.zeros((0, C.OBS_DIM), dtype=np.float32),
        masks=np.zeros((0, C.N_ACTIONS), dtype=bool),
        context=[],
    )


# --------------------------------------------------------------------------------------
# Static tables (per graph / per profile), shared read-only by every sim instance
# --------------------------------------------------------------------------------------
class _Static:
    """Profile-independent lookups with a sentinel slot (edge ``n_edges``, node ``n_nodes``)."""

    def __init__(self, g: Graph) -> None:
        ne, nn = g.n_edges, g.n_nodes
        self.SE, self.SN = ne, nn
        self.elen = np.append(g.edge_length, 0.0)
        self.etimed = np.append(g.edge_timed, False)
        self.eto = np.append(g.edge_to, nn).astype(np.intp)
        self.efrom = np.append(g.edge_from, nn).astype(np.intp)
        self.ekind = np.append(g.edge_kind, -1).astype(np.int8)
        self.etrack = np.append(g.edge_track, 2).astype(np.int8)
        self.ecbtc = np.append(g.edge_cbtc, False)
        self.epocket = np.append(g.edge_pocket, False)
        self.erun = np.append(g.edge_run_s, 0.0)
        self.evmax = np.append(np.where(g.edge_timed, 0.0, g.edge_vmax), 0.0)
        self.ediv = np.append(g.node_division[g.edge_from], -1).astype(np.int8)
        self.npl = np.append(g.node_platform, -1).astype(np.intp)
        self.nst = np.append(g.node_station, -1).astype(np.intp)
        self.ndirpl = np.append(g.node_dir_platform, -1).astype(np.intp)
        self.ntrack = np.append(g.node_track, 2).astype(np.int8)
        self.ndiv = np.append(g.node_division, -1).astype(np.int8)
        self.nl2x = np.append(g.node_l2x, -1)
        self.nx2l = np.append(g.node_x2l, -1)
        self.npocket = np.append(g.node_has_pocket, False)
        self.ndiverge = np.append(g.node_diverge, False)
        self.st_hub = np.append(g.station_hub, False)
        self.st_term = np.append(g.station_terminal, False)
        self.st_xfer = np.append(g.station_transfer, False)
        self.pl_station = g.platform_station.astype(np.intp)
        self.pl_cap = g.platform_capacity.astype(np.float64)
        self.age_c = np.asarray(C.age_bin_centers_min(), dtype=np.float64)
        self.in_edges = [np.asarray(g.node_in(n), dtype=np.intp) for n in range(nn)]
        self.out_edges = [np.asarray(g.node_out(n), dtype=np.intp) for n in range(nn)]
        # running (non-timed) in/out edges per node, as plain lists for the interlocking checks
        timed = g.edge_timed
        self.run_in = [[int(e) for e in g.node_in(n) if not timed[e]] for n in range(nn)]
        self.run_out = [[int(e) for e in g.node_out(n) if not timed[e]] for n in range(nn)]
        self.nopp = _opposite_nodes(g)
        # station -> platforms (CSR, as in the graph) for per-station frame values
        self.st_pl_ptr = g.station_platform_ptr.astype(np.intp)
        self.st_pl_idx = g.station_platform_idx.astype(np.intp)
        self.st_has_pl = np.diff(self.st_pl_ptr) > 0
        # route -> platforms served (incidence) for route-level wait features
        inc = np.zeros((g.n_routes, g.n_platforms), dtype=np.float32)
        for p in range(g.n_patterns):
            r = int(g.pat_route[p])
            pe = g.pattern_edges(p)
            pls = g.node_platform[g.edge_to[pe]]
            inc[r, pls[pls >= 0]] = 1.0
            o = int(g.node_platform[g.pat_origin_node[p]])
            if o >= 0:
                inc[r, o] = 1.0
        self.route_pl = inc
        # per-detour arrays (edges, to-nodes, original edges, original to-nodes)
        self._det: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self.div_cap = np.array([C.TRAIN_CAPACITY["A"], C.TRAIN_CAPACITY["B"]], dtype=np.float64)
        self.div_len = np.array([C.TRAIN_LENGTH_M["A"], C.TRAIN_LENGTH_M["B"]], dtype=np.float64)
        self._rows: _TemplateRows | None = None

    def template_rows(self, sim: "NYCSubwaySim") -> "_TemplateRows":
        """Padded per-pattern path rows (origin templates), built once per graph."""
        if self._rows is None:
            self._rows = _TemplateRows(sim.g, self, sim._PC, sim._VC, sim._SC)
        return self._rows

    def detour_arrays(self, det: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = id(det)
        hit = self._det.get(key)
        if hit is None:
            de = np.asarray(det.edges, dtype=np.intp)
            oe = np.asarray(det.orig_edges, dtype=np.intp)
            hit = (de, self.eto[de], oe, self.eto[oe])
            self._det[key] = hit
        return hit


class _TemplateRows:
    """Every pattern's origin template as padded rows, laid out exactly like a train slot.

    Timetable dispatch copies whole rows with one fancy-indexed assignment per array.
    """

    def __init__(self, g: Graph, S: "_Static", P: int, V: int, Sc: int) -> None:
        n = g.n_patterns
        self.edge = np.full((n, P), S.SE, dtype=np.int16)
        self.stop = np.zeros((n, P), dtype=bool)
        self.visit = np.zeros((n, P), dtype=np.int16)
        self.sched = np.zeros((n, P), dtype=np.float32)
        self.tpos = np.full((n, P), -1, dtype=np.int16)
        self.v_station = np.full((n, V), -1, dtype=np.int16)
        self.v_node = np.full((n, V), S.SN, dtype=np.int16)
        self.v_pos = np.full((n, V), -1, dtype=np.int16)
        self.v_out = np.full((n, V), S.SE, dtype=np.int16)
        self.v_stop = np.zeros((n, V), dtype=bool)
        self.v_time = np.zeros((n, V), dtype=np.float32)
        self.v_sidx = np.full((n, V), -1, dtype=np.int16)
        self.s_station = np.full((n, Sc), -1, dtype=np.int16)
        self.s_pos = np.full((n, Sc), -1, dtype=np.int16)
        self.plen = np.zeros(n, dtype=np.int64)
        self.nvis = np.zeros(n, dtype=np.int64)
        self.nstops = np.zeros(n, dtype=np.int64)
        self.paths: list[CompiledPath] = []
        for p in range(n):
            cp = template_path(g, p, 0)
            self.paths.append(cp)
            L, nv, ns = cp.n_edges, cp.n_visits, cp.n_stops
            self.edge[p, :L] = cp.edge
            self.stop[p, :L] = cp.stop
            self.visit[p, :L] = cp.visit
            self.visit[p, L:] = cp.visit[-1] + 1
            self.sched[p, :L] = cp.sched
            self.sched[p, L:] = cp.sched[-1]
            self.tpos[p, :L] = cp.tpos
            self.v_station[p, :nv] = cp.v_station
            self.v_node[p, :nv] = cp.v_node
            self.v_pos[p, :nv] = cp.v_pos
            self.v_out[p, :nv] = cp.v_out
            self.v_stop[p, :nv] = cp.v_stop
            self.v_time[p, :nv] = cp.v_time
            self.v_time[p, nv:] = cp.v_time[-1]
            self.v_sidx[p, :nv] = cp.v_sidx
            self.s_station[p, :ns] = cp.s_station
            self.s_pos[p, :ns] = cp.s_pos
            self.plen[p], self.nvis[p], self.nstops[p] = L, nv, ns
        for a in vars(self).values():
            if isinstance(a, np.ndarray):
                a.flags.writeable = False


def _opposite_nodes(g: Graph) -> np.ndarray:
    """Node of the same station and track class in the opposite physical direction."""
    nn = g.n_nodes
    opp = np.full(nn + 1, nn, dtype=np.intp)
    pred: list[set[int]] = [set() for _ in range(nn)]
    succ: list[set[int]] = [set() for _ in range(nn)]
    for e in range(g.n_edges):
        if int(g.edge_kind[e]) != KIND_RUN:
            continue
        a, b = int(g.edge_from[e]), int(g.edge_to[e])
        succ[a].add(int(g.node_station[b]))
        pred[b].add(int(g.node_station[a]))
    for e in range(g.n_edges):
        if int(g.edge_kind[e]) == KIND_TURNBACK:
            a, b = int(g.edge_from[e]), int(g.edge_to[e])
            if opp[a] == nn:
                opp[a] = b
    for n in range(nn):
        if opp[n] != nn or g.node_station[n] < 0:
            continue
        s = int(g.node_station[n])
        for m in g.station_nodes(s):
            m = int(m)
            if m == n or g.node_track[m] != g.node_track[n]:
                continue
            if (succ[n] & pred[m]) or (pred[n] & succ[m]):
                opp[n] = m
                break
    return opp


def _static(g: Graph) -> _Static:
    s = g._cache.get("engine_static")
    if s is None:
        s = _Static(g)
        g._cache["engine_static"] = s
    return s


def route_platform_incidence(g: Graph) -> np.ndarray:
    """``float32[n_routes, n_platforms]``: 1 where a pattern of the route serves the platform.

    The same matrix the route-level wait features use (read-only, cached per graph); training
    credit assignment (``rl.credit``) scopes rewards with it.
    """
    inc = _static(g).route_pl
    out = inc.view()
    out.flags.writeable = False
    return out


class _Prof:
    """Profile (and route subset) tables derived from a :class:`ServiceView`."""

    def __init__(self, g: Graph, S: _Static, view: ServiceView) -> None:
        self.view = view
        self.rate = view.platform_rate.astype(np.float64)
        self.attr = view.attraction.astype(np.float64)
        self.edge_h = np.append(view.edge_sched_h, INF)
        # per node (+ sentinel): the out-edge with the most frequent service (sentinel if none)
        best_h = np.full(S.SN + 1, INF)
        self.node_out_edge = np.full(S.SN + 1, S.SE, dtype=np.intp)
        for e in np.argsort(view.edge_sched_h, kind="stable")[::-1].tolist():
            h = float(view.edge_sched_h[e])
            n = int(g.edge_from[e])
            if np.isfinite(h) and h <= best_h[n]:
                best_h[n] = h
                self.node_out_edge[n] = e
        self.pl_h = view.platform_sched_h.astype(np.float64)
        self.pat_h = view.pattern_headway.astype(np.float64)
        sp = view.svc_pattern.astype(np.intp)
        self.svc_pattern = sp
        self.svc_h = view.svc_headway.astype(np.float64)
        self.svc_phase = view.svc_phase.astype(np.float64)
        self.n_svc = int(sp.shape[0])
        self.pool_keys = view.pool_keys
        self.pool_index = {k: i for i, k in enumerate(view.pool_keys)}
        self.n_pools = len(view.pool_keys)
        fam = g.route_family[g.pat_route[sp]] if self.n_svc else np.zeros(0, dtype=np.int16)
        self.svc_pool = np.array(
            [self.pool_index[(int(f), int(g.pat_origin_station[p]))] for f, p in zip(fam, sp)], dtype=np.intp
        )
        self.svc_origin_node = g.pat_origin_node[sp].astype(np.intp) if self.n_svc else np.zeros(0, dtype=np.intp)
        self.pool_target = view.pool_target.astype(np.int64)
        self.pool_putin = view.pool_putin_period.astype(np.float64)
        self.pool_surplus = view.pool_surplus.astype(bool)
        # transfer / spill weights by platform rate (uniform when the rates are all zero)
        self.xfer_ptr = g.station_xfer_ptr.astype(np.intp)
        self.xfer_idx = g.station_xfer_idx.astype(np.intp)
        self.xfer_w = _group_weights(self.xfer_ptr, self.xfer_idx, self.rate)
        self.spill_ptr = g.platform_spill_ptr.astype(np.intp)
        self.spill_idx = g.platform_spill_idx.astype(np.intp)
        self.spill_w = _group_weights(self.spill_ptr, self.spill_idx, self.rate)
        self.has_spill = np.diff(self.spill_ptr) > 0
        # yard leads per route among patterns in service
        self.route_leads: dict[int, list[Any]] = {}
        for lead in g.yard_leads:
            if np.isfinite(self.pat_h[lead.pattern]):
                self.route_leads.setdefault(int(g.pat_route[lead.pattern]), []).append(lead)
        # Yard starvation tables (masks.lead_etas / yard_choice). Per in-service lead: every
        # in-service (pattern, position) using the edge leaving the lead node on the lead's
        # track (the "next train" to reach it may be any of them), as a CSR over the leads.
        edge_users: dict[int, list[tuple[int, int]]] = {}
        for q in np.flatnonzero(np.isfinite(self.pat_h)).tolist():
            for j, e in enumerate(g.tmpl_edge[q, : int(g.pat_len[q])].tolist()):
                edge_users.setdefault(int(e), []).append((int(q), int(j)))
        n_leads = len(g.yard_leads)
        svc_leads = sorted({int(ld.lead_id) for leads in self.route_leads.values() for ld in leads})
        pq: list[int] = []
        pj: list[int] = []
        ptr = [0]
        self.lead_thr = np.full(n_leads, INF)
        for lid in svc_leads:
            lead = g.yard_leads[lid]
            assert lead.lead_id == lid
            e0 = int(g.tmpl_edge[lead.pattern, lead.position])
            for q, j in edge_users.get(e0, [(int(lead.pattern), int(lead.position))]):
                pq.append(q)
                pj.append(j)
            ptr.append(len(pq))
            h0 = float(self.edge_h[e0])
            h0 = h0 if np.isfinite(h0) else C.SCHED_H_FALLBACK_S
            h0 = min(max(h0, C.SCHED_H_MIN_S), C.SCHED_H_MAX_S)
            self.lead_thr[lid] = max(C.YARD_STARVED_MULT * h0, C.YARD_STARVED_MIN_S)
        self.lead_svc = np.asarray(svc_leads, dtype=np.intp)
        self.lead_pair_q = np.asarray(pq, dtype=np.intp)
        self.lead_pair_j = np.asarray(pj, dtype=np.intp)
        self.lead_pair_ptr = np.asarray(ptr, dtype=np.intp)
        self.lead_yard = np.array([int(ld.yard) for ld in g.yard_leads], dtype=np.intp)
        self.lead_node = np.array([int(ld.node) for ld in g.yard_leads], dtype=np.intp)
        width = max((len(v) for v in self.route_leads.values()), default=0)
        self.route_lead_mat = np.full((g.n_routes, max(width, 1)), -1, dtype=np.intp)
        for r, leads in self.route_leads.items():
            self.route_lead_mat[r, : len(leads)] = [int(ld.lead_id) for ld in leads]
        # service entries by pattern (next departure of a pattern, for yard starvation)
        self.pat_svc: dict[int, list[int]] = {}
        for k, p in enumerate(sp.tolist()):
            self.pat_svc.setdefault(int(p), []).append(k)
        self._suffix_cache: dict[int, tuple[CompiledPath, np.ndarray]] = {}
        self._tmpl_suffix: np.ndarray | None = None

    def template_suffix(self, sim: "NYCSubwaySim", T: _TemplateRows) -> np.ndarray:
        """Destination suffix rows ``[n_patterns, S + 1]`` of the origin templates."""
        if self._tmpl_suffix is None:
            out = np.zeros((len(T.paths), sim._SC + 1), dtype=np.float64)
            for p, cp in enumerate(T.paths):
                out[p, : cp.n_stops] = self.suffix(cp)
            out.flags.writeable = False
            self._tmpl_suffix = out
        return self._tmpl_suffix

    def suffix(self, cp: CompiledPath) -> np.ndarray:
        """Attraction-weighted destination suffix sums for a compiled path's stops."""
        key = id(cp)
        entry = self._suffix_cache.get(key)
        hit = entry[1] if entry is not None and entry[0] is cp else None
        if hit is None:
            ns = cp.n_stops
            if ns:
                w = self.attr[cp.s_station.astype(np.intp)] * np.exp(-np.arange(ns) / C.DEST_DECAY_STOPS)
                hit = np.cumsum(w[::-1])[::-1].astype(np.float64)
            else:
                hit = np.zeros(0)
            if len(self._suffix_cache) < 4096:
                self._suffix_cache[key] = (cp, hit)
        return hit


def _group_weights(ptr: np.ndarray, idx: np.ndarray, rate: np.ndarray) -> np.ndarray:
    w = np.zeros(idx.shape[0], dtype=np.float64)
    for i in range(ptr.shape[0] - 1):
        a, b = int(ptr[i]), int(ptr[i + 1])
        if b <= a:
            continue
        r = rate[idx[a:b]]
        s = float(r.sum())
        w[a:b] = r / s if s > 0 else 1.0 / (b - a)
    return w


def _prof(g: Graph, S: _Static, profile: str, routes: tuple[int, ...] | None) -> _Prof:
    key = ("engine_prof", profile, routes)
    p = g._cache.get(key)
    if p is None:
        p = _Prof(g, S, g.service_view(profile, routes))
        g._cache[key] = p
    return p


# --------------------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------------------
_SHARED_ATTRS = frozenset({"g", "_S", "_P", "_ark", "_N", "_PC", "_VC", "_SC"})
_TRANSIENT_ATTRS = frozenset(
    {"_batch", "_pl_stats", "_pl_stats_tick", "_lead_eta_cache", "_tmpl_dis_cache", "_route_stats_cache"}
)
# per-session identity counters (not physics); excluded from state_hash()
_SESSION_ATTRS = frozenset({"_next_dis_id"})
# opt-in training accounting (per-platform / per-route cost vectors); excluded from state_hash()
_ACCOUNTING_ATTRS = frozenset({"track_cost_vectors", "cost_plat", "cost_route_train"})
# derived every tick from the primary state; excluded from state_hash()
_DERIVED_ATTRS = frozenset(
    {"_rear_min", "_rear_id", "_front_max", "_front_id", "_lead_rear", "_lead_id", "_foll_front", "_foll_id",
     "_e_stopped", "_e_timed_occ", "_node_res", "_node_tail", "_node_route",
     "_st_pocket_busy", "_dwell_log", "_t0"}
)


class NYCSubwaySim:
    """The citywide simulation (public API: ARCHITECTURE §3.1)."""

    # ------------------------------------------------------------------ construction
    def __init__(self, graph: Graph, scenario: Scenario | None = None, *, warmup: bool = True) -> None:
        self.g = graph
        self._S = _static(graph)
        self._N = C.MAX_TRAINS
        self._PC, self._VC, self._SC = path_capacity_dims(graph)
        self._ark = np.arange(_K_LOOK, dtype=np.intp)[None, :]
        self._next_dis_id = 1
        #: accumulate cost_vectors() in every tick (training credit only; the server leaves it off)
        self.track_cost_vectors = False
        self.reset(scenario, warmup=warmup)

    def _alloc(self) -> None:
        g, N, P, V, S = self.g, self._N, self._PC, self._VC, self._SC
        SE = self._S.SE
        # trains
        self.tr_active = np.zeros(N, dtype=bool)
        self.tr_route = np.full(N, -1, dtype=np.int16)
        self.tr_pattern = np.full(N, -1, dtype=np.int32)
        self.tr_div = np.zeros(N, dtype=np.int8)
        self.tr_cap = np.full(N, 1.0, dtype=np.float64)
        self.tr_len = np.zeros(N, dtype=np.float64)
        self.tr_ptr = np.zeros(N, dtype=np.int64)
        self.tr_off = np.zeros(N, dtype=np.float64)
        self.tr_v = np.zeros(N, dtype=np.float64)
        self.tr_mode = np.zeros(N, dtype=np.int8)
        self.tr_timer = np.zeros(N, dtype=np.float64)
        self.tr_tkind = np.zeros(N, dtype=np.int8)
        self.tr_at_start = np.zeros(N, dtype=bool)
        self.tr_stop_done = np.full(N, -1, dtype=np.int64)
        self.tr_keyby_done = np.full(N, -1, dtype=np.int64)
        self.tr_ready_tick = np.zeros(N, dtype=np.int64)
        self.tr_next_dec = np.full(N, INF)
        self.tr_hold_total = np.zeros(N, dtype=np.float64)
        self.tr_held = np.zeros(N, dtype=bool)
        self.tr_skipped_prev = np.zeros(N, dtype=bool)
        self.tr_skip_next = np.zeros(N, dtype=bool)
        self.tr_deviated = np.zeros(N, dtype=bool)
        self.tr_switch = np.zeros(N, dtype=bool)
        self.tr_rerouted = np.zeros(N, dtype=bool)
        self.tr_gap = np.zeros(N, dtype=bool)
        self.tr_finishing = np.zeros(N, dtype=bool)
        self.tr_in_closed = np.zeros(N, dtype=bool)
        self.tr_late = np.zeros(N, dtype=np.float64)
        self.tr_sched_dep = np.zeros(N, dtype=np.float64)
        self.tr_arr_time = np.zeros(N, dtype=np.float64)
        self.tr_stopped_since = np.full(N, INF)
        self.tr_at_pl = np.zeros(N, dtype=bool)
        self.tr_tunnel = np.zeros(N, dtype=bool)
        self.tr_sptr = np.zeros(N, dtype=np.int64)
        self.tr_plen = np.zeros(N, dtype=np.int64)
        self.tr_nvis = np.zeros(N, dtype=np.int64)
        self.tr_nstops = np.zeros(N, dtype=np.int64)
        self.tr_onboard = np.zeros((N, C.N_DEST), dtype=np.float64)
        self.p_edge = np.full((N, P), SE, dtype=np.int16)
        self.p_stop = np.zeros((N, P), dtype=bool)
        self.p_visit = np.zeros((N, P), dtype=np.int16)
        self.p_sched = np.zeros((N, P), dtype=np.float32)
        self.p_tpos = np.full((N, P), -1, dtype=np.int16)
        self.v_station = np.full((N, V), -1, dtype=np.int16)
        self.v_node = np.full((N, V), self._S.SN, dtype=np.int16)
        self.v_pos = np.full((N, V), -1, dtype=np.int16)
        self.v_out = np.full((N, V), SE, dtype=np.int16)
        self.v_stop = np.zeros((N, V), dtype=bool)
        self.v_time = np.zeros((N, V), dtype=np.float32)
        self.v_sidx = np.full((N, V), -1, dtype=np.int16)
        self.s_station = np.full((N, S), -1, dtype=np.int16)
        self.s_pos = np.full((N, S), -1, dtype=np.int16)
        self.s_suffix = np.zeros((N, S + 1), dtype=np.float64)
        # platforms / nodes
        self.waiting = np.zeros((g.n_platforms, C.N_AGE_BINS), dtype=np.float64)
        self.pl_last_dep = np.zeros(g.n_platforms, dtype=np.float64)
        self.nd_last_dep = np.zeros(g.n_nodes + 1, dtype=np.float64)
        self.rate_eff = np.zeros(g.n_platforms, dtype=np.float64)
        # disruption effect arrays
        self.e_blocked = np.zeros(SE + 1, dtype=bool)
        self.e_bkind = np.zeros(SE + 1, dtype=np.int8)
        self.e_sig = np.zeros(SE + 1, dtype=bool)
        self.e_xoff = np.zeros(SE + 1, dtype=bool)
        self.e_vmax = self._S.evmax.copy()
        self.e_peak_ok = np.ones(SE + 1, dtype=bool)
        self.n_closed = np.zeros(g.n_nodes + 1, dtype=bool)
        self.n_ckind = np.zeros(g.n_nodes + 1, dtype=np.int8)
        self.st_closed = np.zeros(g.n_stations, dtype=bool)
        self.pl_closed = np.zeros(g.n_platforms, dtype=bool)
        # occupancy tables (rebuilt every tick)
        self._rear_min = np.full(SE + 1, INF)  # rear offset of the rearmost train on each edge
        self._rear_id = np.full(SE + 1, -1, dtype=np.int64)  # ... and its slot
        self._front_max = np.full(SE + 1, -INF)
        self._lead_rear = np.full(N, INF)  # rear offset of the train ahead on the same edge
        self._lead_id = np.full(N, -1, dtype=np.int64)
        self._front_id = np.full(SE + 1, -1, dtype=np.int64)  # slot of the frontmost train on each edge
        self._foll_front = np.full(N, -INF)  # front offset of the train behind on the same edge
        self._foll_id = np.full(N, -1, dtype=np.int64)
        self._e_stopped = np.zeros(SE + 1, dtype=np.int64)
        self._e_timed_occ = np.zeros(SE + 1, dtype=bool)
        # Interlocking reservations held by crossover occupants (see _try_enter_timed):
        # _node_res = the slot holding a route through the node (-1 = free); the occupant's
        # from-node is a *tail* node (its body still extends back over it), its to-node a
        # *route* node (the home signal in front of it is red for everybody else).
        self._node_res = np.full(g.n_nodes + 1, -1, dtype=np.int64)
        self._node_tail = np.zeros(g.n_nodes + 1, dtype=bool)
        self._node_route = np.zeros(g.n_nodes + 1, dtype=bool)
        self._st_pocket_busy = np.zeros(g.n_stations, dtype=bool)
        # yards
        self.yard_stock = g.yard_fleet_cap.astype(np.int64).copy()
        self.yard_cool = np.full(g.n_yards, -INF)
        # accounting
        self.cost = np.zeros(len(C.COST_KEYS), dtype=np.float64)
        self.cost_plat = np.zeros(g.n_platforms, dtype=np.float64)  # wait + crowd per platform
        self.cost_route_train = np.zeros(g.n_routes, dtype=np.float64)  # tunnel + hold per train route
        self.counts = np.zeros(C.N_ACTIONS, dtype=np.int64)

    # ------------------------------------------------------------------ reset
    def reset(self, scenario: Scenario | None = None, *, warmup: bool = True) -> None:
        """Rebuild the initial state from ``scenario`` (default ``Scenario()``)."""
        sc = scenario if scenario is not None else Scenario(seed=0, profile="am_peak")
        if not isinstance(sc, Scenario):
            raise TypeError("scenario must be a sim.Scenario")
        g = self.g
        self.scenario = sc
        self.profile = sc.profile
        self.demand_scale = float(sc.demand_scale)
        self.dwell_scale = float(sc.dwell_scale)
        self.speed_scale = float(sc.speed_scale)
        self.episode_s = float(sc.episode_s)
        routes = g.route_ids_to_idx(sc.routes)
        self._P = _prof(g, self._S, sc.profile, routes)
        self._alloc()
        ss = np.random.SeedSequence([int(sc.seed) & 0xFFFFFFFF, 0x5D])
        self._rng_dispatch = np.random.Generator(np.random.PCG64(ss.spawn(1)[0]))
        self._rng_action = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(sc.seed) & 0xFFFFFFFF, 0xA7])))

        self.clock_s = float(C.PROFILE_START_S[sc.profile] - (C.WARMUP_S if warmup else 0.0))
        self.tick = 0
        self._tick_abs = 0  # monotonic tick counter (never reset after warm-up)
        self._age_phase = 0
        self.riders_served = 0.0
        self.riders_displaced = 0.0
        self.missed_departures = 0
        self.laid_up = 0
        self.put_ins = 0
        self._dis_active: list[dict[str, Any]] = []
        self._dis_scheduled: list[ScheduledDisruption] = list(sc.disruptions)
        self._ep_start = self.clock_s + (C.WARMUP_S if warmup else 0.0)
        self._gap_queue: list[tuple[float, int, int, int, int]] = []
        self._retire: list[int] = []
        self._dwell_log: list[float] = []
        self._decide = False
        self._mask_epoch = 0
        self._clear_pending()
        self._batch: DecisionBatch | None = None
        self._pl_stats: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._pl_stats_tick = -1
        self._lead_eta_cache: dict[Any, float] = {}
        self._tmpl_dis_cache: dict[Any, np.ndarray] = {}
        self._route_stats_cache = None
        self._t0 = self.clock_s

        self._apply_disruption_effects(initial=True)
        self._init_service(float(sc.headway_jitter))
        self._occupancy()
        if warmup:
            for _ in range(int(round(C.WARMUP_S / DT))):
                self._tick()
            self.cost[:] = 0.0
            self.cost_plat[:] = 0.0
            self.cost_route_train[:] = 0.0
            self.counts[:] = 0
            self.riders_served = 0.0
            self.riders_displaced = 0.0
            self.missed_departures = 0
            self.tick = 0
            self._dwell_log = []
            self._pl_stats = None
            self._lead_eta_cache = {}
        self._ep_start = self.clock_s
        self._decide = True
        self._activate_scheduled()
        self._occupancy()
        self._collect_decisions(np.flatnonzero(self.tr_active & (self.tr_mode == M_READY)))

    # ------------------------------------------------------------------ service init
    def _init_service(self, jitter: float) -> None:
        g, P = self.g, self._P
        t0 = self.clock_s
        rng = self._rng_dispatch
        w0 = C.PROFILE_WINDOWS_H[self.profile][0] * 3600.0
        n = P.n_svc
        self.svc_next = np.zeros(n, dtype=np.float64)
        self.svc_n = np.zeros(n, dtype=np.int64)
        horizon = C.WARMUP_S + max(self.episode_s, 3600.0) + 7200.0
        n_j = 64
        if n:
            n_j = int(min(max(np.ceil(horizon / max(float(P.svc_h.min()), 60.0)) + 2, 8), 2048))
        self.svc_jit = (rng.random((n, n_j)) * jitter).astype(np.float64)
        init_j = rng.random((n, 64)) * 2.0 - 1.0
        # pool_rel[k, j]: release time of the train laid over in slot j of pool k (it may
        # depart once released); +inf marks an empty slot. Seeded trains are released.
        self.pool_rel = np.full((P.n_pools, C.POOL_MAX), INF)
        for k in range(P.n_pools):
            m = int(min(P.pool_target[k], C.POOL_MAX))
            self.pool_rel[k, :m] = t0 - C.LAYOVER_S
        self.pool_putin_next = np.where(np.isfinite(P.pool_putin), t0 + 0.5 * P.pool_putin, INF)

        placements: list[tuple[int, int, float, float]] = []  # (svc k, pattern, tau, dep time)
        for k in range(n):
            p = int(P.svc_pattern[k])
            h = float(P.svc_h[k])
            ph = float(P.svc_phase[k])
            L = int(g.pat_len[p])
            T = float(g.tmpl_sched[p, L - 1])
            m_hi = math.floor((t0 - w0 - ph) / h)
            t_hi = w0 + ph + m_hi * h
            self.svc_next[k] = t_hi + h
            self.svc_n[k] = 0
            j = 0
            t_d = t_hi
            while t0 - t_d < T and j < 64:
                tau = t0 - t_d + float(init_j[k, j]) * jitter * h
                if 0.0 <= tau < T:
                    placements.append((k, p, tau, t_d))
                elif tau < 0.0 and j == 0:
                    pass  # departs later; the timetable will dispatch it
                j += 1
                t_d -= h
        for k, p, tau, t_d in placements:
            slot = self._free_slot()
            if slot < 0:
                break
            self._place_initial(slot, p, tau, t_d)

        # initial queues (steady state), last-departure clocks
        view = P.view
        self.waiting[:, :] = view.init_waiting.astype(np.float64) * self.demand_scale
        self.waiting[self.pl_closed] = 0.0
        self.pl_last_dep[:] = t0 - np.minimum(0.5 * np.where(np.isfinite(P.pl_h), P.pl_h, 3600.0), 1800.0)
        out_h = np.full(g.n_nodes + 1, 3600.0)
        eh = P.edge_h[:-1]
        np.minimum.at(out_h, g.edge_from.astype(np.intp), np.where(np.isfinite(eh), eh, 3600.0))
        self.nd_last_dep[:] = t0 - np.minimum(0.5 * out_h, 1800.0)

    def _place_initial(self, slot: int, p: int, tau: float, t_dep: float) -> None:
        g = self.g
        cp = template_path(g, p, 0)
        self._spawn_train(slot, cp, p, sched_dep=t_dep)
        L = cp.n_edges
        sched = cp.sched.astype(np.float64)
        stops = cp.stop
        dep = sched + np.where(stops, C.DWELL_SCHED_S, 0.0)
        # find the position at elapsed time tau
        prev_dep = 0.0
        S = self._S
        n_served = 0
        for i in range(L):
            arr = float(sched[i])
            e = int(cp.edge[i])
            if tau < arr:
                # running on edge i
                if S.etimed[e]:
                    # stand at the edge's from-node (before the crossover)
                    self._set_position(slot, i, at_start=True, off=0.0, v=0.0, mode=M_RUN)
                else:
                    span = max(arr - prev_dep, 1e-6)
                    f = min(max((tau - prev_dep) / span, 0.0), 0.98)
                    off = f * float(S.elen[e])
                    rem = float(S.elen[e]) - off
                    v = min(0.85 * float(self.e_vmax[e]), math.sqrt(2.0 * C.BRAKE * max(rem, 0.0)))
                    self._set_position(slot, i, at_start=False, off=off, v=v, mode=M_RUN)
                break
            if stops[i] and tau < float(dep[i]) and i < L - 1:
                # dwelling at the stop at the end of edge i
                self._set_position(slot, i, at_start=False, off=float(S.elen[e]), v=0.0, mode=M_DWELL)
                self.tr_timer[slot] = float(dep[i]) - tau
                self.tr_stop_done[slot] = i
                n_served += 1
                break
            if stops[i]:
                n_served += 1
            prev_dep = float(dep[i]) if stops[i] else arr
        else:  # pragma: no cover - tau < T guarantees a break
            self._set_position(slot, L - 1, at_start=False, off=float(S.elen[int(cp.edge[L - 1])]), v=0.0, mode=M_RUN)
        n_served = min(n_served, cp.n_stops)
        loads = self._P.view.init_onboard.get(p)
        if loads is not None:
            self.tr_onboard[slot, :] = loads[min(n_served, loads.shape[0] - 1)] * self.demand_scale
            self.tr_onboard[slot] *= min(1.0, self.tr_cap[slot] / max(self.tr_onboard[slot].sum(), 1e-9))
        self.tr_sptr[slot] = n_served

    def _set_position(self, slot: int, ptr: int, *, at_start: bool, off: float, v: float, mode: int) -> None:
        self.tr_ptr[slot] = ptr
        self.tr_at_start[slot] = at_start
        self.tr_off[slot] = off
        self.tr_v[slot] = v
        self.tr_mode[slot] = mode
        self.tr_timer[slot] = 0.0
        self.tr_tkind[slot] = TK_NONE
        self.tr_stop_done[slot] = -1

    # ------------------------------------------------------------------ slots & paths
    def _free_slot(self) -> int:
        free = np.flatnonzero(~self.tr_active)
        return int(free[0]) if free.size else -1

    def _write_path(self, slot: int, cp: CompiledPath) -> None:
        P, V, S = self._PC, self._VC, self._SC
        L = cp.n_edges
        nv = cp.n_visits
        ns = cp.n_stops
        if L > P or nv > V or ns > S:
            raise RuntimeError(f"path does not fit the row capacity (L={L}/{P}, V={nv}/{V}, S={ns}/{S})")
        SE, SN = self._S.SE, self._S.SN
        self.p_edge[slot, :L] = cp.edge
        self.p_edge[slot, L:] = SE
        self.p_stop[slot, :L] = cp.stop
        self.p_stop[slot, L:] = False
        self.p_visit[slot, :L] = cp.visit
        self.p_visit[slot, L:] = cp.visit[-1] + 1
        self.p_sched[slot, :L] = cp.sched
        self.p_sched[slot, L:] = cp.sched[-1]
        self.p_tpos[slot, :L] = cp.tpos
        self.p_tpos[slot, L:] = -1
        self.v_station[slot, :nv] = cp.v_station
        self.v_station[slot, nv:] = -1
        self.v_node[slot, :nv] = cp.v_node
        self.v_node[slot, nv:] = SN
        self.v_pos[slot, :nv] = cp.v_pos
        self.v_pos[slot, nv:] = -1
        self.v_out[slot, :nv] = cp.v_out
        self.v_out[slot, nv:] = SE
        self.v_stop[slot, :nv] = cp.v_stop
        self.v_stop[slot, nv:] = False
        self.v_time[slot, :nv] = cp.v_time
        self.v_time[slot, nv:] = cp.v_time[-1]
        self.v_sidx[slot, :nv] = cp.v_sidx
        self.v_sidx[slot, nv:] = -1
        self.s_station[slot, :ns] = cp.s_station
        self.s_station[slot, ns:] = -1
        self.s_pos[slot, :ns] = cp.s_pos
        self.s_pos[slot, ns:] = -1
        suf = self._P.suffix(cp)
        self.s_suffix[slot, :ns] = suf
        self.s_suffix[slot, ns:] = 0.0
        self.tr_plen[slot] = L
        self.tr_nvis[slot] = nv
        self.tr_nstops[slot] = ns

    def _reset_trains(self, slots: np.ndarray, patterns: np.ndarray, sched_dep: np.ndarray, *, gap: bool) -> None:
        """Activate ``slots`` as fresh trains of ``patterns`` standing at their path start."""
        g = self.g
        routes = g.pat_route[patterns]
        div = g.route_division[routes]
        self.tr_active[slots] = True
        self.tr_route[slots] = routes
        self.tr_pattern[slots] = patterns
        self.tr_div[slots] = div
        self.tr_cap[slots] = self._S.div_cap[div]
        self.tr_len[slots] = self._S.div_len[div]
        self.tr_ptr[slots] = 0
        self.tr_at_start[slots] = True
        self.tr_off[slots] = 0.0
        self.tr_v[slots] = 0.0
        self.tr_mode[slots] = M_DWELL
        self.tr_timer[slots] = 0.0
        self.tr_tkind[slots] = TK_NONE
        self.tr_stop_done[slots] = -1
        self.tr_keyby_done[slots] = -1
        self.tr_ready_tick[slots] = -1
        self.tr_next_dec[slots] = INF
        self.tr_hold_total[slots] = 0.0
        for flags in (self.tr_held, self.tr_skipped_prev, self.tr_skip_next, self.tr_deviated, self.tr_switch,
                      self.tr_rerouted, self.tr_finishing, self.tr_in_closed, self.tr_tunnel):
            flags[slots] = False
        self.tr_gap[slots] = gap
        self.tr_late[slots] = 0.0
        self.tr_sched_dep[slots] = sched_dep
        self.tr_arr_time[slots] = self.clock_s
        self.tr_stopped_since[slots] = INF
        self.tr_at_pl[slots] = True
        self.tr_sptr[slots] = 0
        self.tr_onboard[slots] = 0.0

    def _spawn_train(self, slot: int, cp: CompiledPath, pattern: int, *, sched_dep: float, gap: bool = False) -> None:
        """Activate one train on an arbitrary compiled path (initial placement, gap trains)."""
        self._write_path(slot, cp)
        self._reset_trains(np.array([slot]), np.array([pattern]), np.array([sched_dep]), gap=gap)

    def _spawn_templates(self, slots: np.ndarray, patterns: np.ndarray, sched_dep: np.ndarray) -> None:
        """Activate trains at the origin of their pattern templates (batched timetable dispatch)."""
        T = self._S.template_rows(self)
        self.p_edge[slots] = T.edge[patterns]
        self.p_stop[slots] = T.stop[patterns]
        self.p_visit[slots] = T.visit[patterns]
        self.p_sched[slots] = T.sched[patterns]
        self.p_tpos[slots] = T.tpos[patterns]
        self.v_station[slots] = T.v_station[patterns]
        self.v_node[slots] = T.v_node[patterns]
        self.v_pos[slots] = T.v_pos[patterns]
        self.v_out[slots] = T.v_out[patterns]
        self.v_stop[slots] = T.v_stop[patterns]
        self.v_time[slots] = T.v_time[patterns]
        self.v_sidx[slots] = T.v_sidx[patterns]
        self.s_station[slots] = T.s_station[patterns]
        self.s_pos[slots] = T.s_pos[patterns]
        self.s_suffix[slots] = self._P.template_suffix(self, T)[patterns]
        self.tr_plen[slots] = T.plen[patterns]
        self.tr_nvis[slots] = T.nvis[patterns]
        self.tr_nstops[slots] = T.nstops[patterns]
        self._reset_trains(slots, patterns, sched_dep, gap=False)

    # ------------------------------------------------------------------ public API
    def copy(self) -> "NYCSubwaySim":
        """Deep copy of the mutable state; shares the read-only graph (the Normal twin)."""
        new = object.__new__(NYCSubwaySim)
        d = new.__dict__
        for k, v in self.__dict__.items():
            if k in _SHARED_ATTRS:
                d[k] = v
            elif isinstance(v, np.ndarray):
                d[k] = v.copy()
            elif k in _TRANSIENT_ATTRS:
                d[k] = None if k in ("_batch", "_pl_stats") else ({} if isinstance(v, dict) else v)
            elif k == "_pend_info":
                d[k] = dict(v)
            elif isinstance(v, (list, dict, np.random.Generator)):
                d[k] = _copy.deepcopy(v)
            else:
                d[k] = v
        d["_batch"] = self._batch
        d["_pl_stats_tick"] = -1
        return new

    def step(self, n_ticks: int = 1) -> None:
        """Advance ``n_ticks`` x ``DT_S``; unanswered decisions resolve as CONTINUE."""
        for _ in range(int(n_ticks)):
            self._clear_pending()
            self._tick()

    def pending_decisions(self) -> DecisionBatch:
        """The decisions requested at the current tick (one snapshot); empty batch if none."""
        if self._batch is not None:
            return self._batch
        ids = self._pend_ids
        if ids.size == 0:
            self._batch = empty_batch()
            return self._batch
        from sim.obs import build_obs

        obs = build_obs(self, ids, self._pend_masks, self._pend_info)
        S = self._S
        nodes = self._pend_info["node"]
        ctx = [
            {"station_idx": int(S.nst[n]), "route_idx": int(self.tr_route[t]), "node_idx": int(n)}
            for t, n in zip(ids.tolist(), nodes.tolist())
        ]
        self._batch = DecisionBatch(
            train_ids=ids.astype(np.int32), obs=obs, masks=self._pend_masks.copy(), context=ctx
        )
        return self._batch

    def apply_actions(self, train_ids: np.ndarray, actions: np.ndarray) -> list[ActionEvent]:
        """Apply actions in order, re-validated against the current mask."""
        from sim import actions as A

        ids = np.asarray(train_ids, dtype=np.int64).ravel()
        acts = np.asarray(actions, dtype=np.int64).ravel()
        if ids.shape != acts.shape:
            raise ValueError("train_ids and actions must have the same length")
        events: list[ActionEvent] = []
        for tid, a in zip(ids.tolist(), acts.tolist()):
            if not (0 <= a < C.N_ACTIONS):
                raise ValueError(f"action {a} out of range")
            row = self._pend_pos.pop(tid, None)
            if a == C.A_CONTINUE:
                continue
            ev = A.apply(self, tid, a, row)
            events.append(ev)
        return events

    def add_disruption(self, kind: str, target_type: str, target_idx: int, duration_s: float | None = None) -> int:
        eff = D.resolve(self.g, kind, target_type, target_idx)
        if duration_s is not None and not (float(duration_s) > 0):
            raise ValueError("duration_s must be > 0 or None")
        did = self._next_dis_id
        self._next_dis_id += 1
        start = self.clock_s
        self._dis_active.append(
            {
                "id": did,
                "kind": eff.kind,
                "target": {"type": eff.target_type, "idx": int(eff.target_idx)},
                "label": eff.label,
                "start_s": float(start),
                "end_s": None if duration_s is None else float(start + float(duration_s)),
            }
        )
        self._apply_disruption_effects()
        return did

    def clear_disruption(self, disruption_id: int) -> bool:
        for i, d in enumerate(self._dis_active):
            if d["id"] == int(disruption_id):
                del self._dis_active[i]
                self._apply_disruption_effects()
                return True
        return False

    def clear_all_disruptions(self) -> None:
        self._dis_active = []
        self._dis_scheduled = []
        self._apply_disruption_effects()

    def disruptions(self) -> list[dict]:
        return [_copy.deepcopy(d) for d in self._dis_active]

    def metrics(self) -> dict:
        tot, mean_w, _ = self._platform_stats()
        riders_waiting = float(tot.sum())
        wsum = float((self.waiting @ self._S.age_c).sum())
        avg = wsum / riders_waiting if riders_waiting > 1e-9 else 0.0
        over = int(np.count_nonzero((~self.pl_closed) & (tot > C.OVERCROWD_FRAC * self._S.pl_cap)))
        act = self.tr_active
        stuck = int(np.count_nonzero(self._tunnel_stuck_mask()))
        return {
            "clock_s": float(self.clock_s),
            "avg_wait_min": float(avg),
            "overcrowded_platforms": over,
            "tunnel_stuck_trains": stuck,
            "riders_served": float(self.riders_served),
            "riders_waiting": riders_waiting,
            "riders_onboard": float(self.tr_onboard[act].sum()),
            "trains_active": int(np.count_nonzero(act)),
        }

    def cost_totals(self) -> dict[str, float]:
        return {k: float(v) for k, v in zip(C.COST_KEYS, self.cost.tolist())}

    def cost_vectors(self) -> dict[str, np.ndarray]:
        """Where the cumulative flow costs since reset arose (copies; training credit assignment).

        ``platform``: float64[n_platforms] wait + crowd; ``route_train``: float64[n_routes]
        tunnel + hold of each route's trains. They accumulate only while
        ``track_cost_vectors`` is set, in the same tick code as ``cost_totals()``; then
        ``platform.sum() == wait + crowd`` and ``route_train.sum() == tunnel + hold`` up to
        float rounding. Immediate action costs are not included (they belong to the actor).
        """
        return {"platform": self.cost_plat.copy(), "route_train": self.cost_route_train.copy()}

    def action_counts(self) -> dict[str, int]:
        return {name: int(self.counts[i + 1]) for i, name in enumerate(_NON_CONTINUE)}

    def frame(self) -> dict[str, np.ndarray]:
        S = self._S
        idx = np.flatnonzero(self.tr_active)
        e = self.p_edge[idx, self.tr_ptr[idx]].astype(np.intp)
        L = S.elen[e]
        off = np.minimum(self.tr_off[idx], L)
        frac = np.where(L > 0, np.rint(off / np.maximum(L, 1e-9) * 65535.0), 0.0)
        frac = np.clip(frac, 0, 65535).astype(np.uint16)
        status = self._status_bits(idx, e)
        load = np.minimum(255, np.rint(100.0 * self.tr_onboard[idx].sum(1) / self.tr_cap[idx])).astype(np.uint8)
        crowd, wait = self._station_frame()
        return {
            "train_id": idx.astype(np.uint16),
            "edge": e.astype(np.uint16),
            "frac": frac,
            "route": self.tr_route[idx].astype(np.uint8),
            "status": status,
            "load": load,
            "station_crowd": crowd,
            "station_wait": wait,
        }

    def state_hash(self) -> str:
        """SHA-1 of every mutable simulation quantity (tests, determinism checks).

        Session bookkeeping is excluded: disruption ids keep increasing across resets of one
        session, so active disruptions are hashed by kind, target and timing only.
        """
        h = hashlib.sha1()
        for k in sorted(self.__dict__):
            if (k in _SHARED_ATTRS or k in _TRANSIENT_ATTRS or k in _DERIVED_ATTRS or k in _SESSION_ATTRS
                    or k in _ACCOUNTING_ATTRS or k.startswith("_pend")):
                continue
            v = self.__dict__[k]
            h.update(k.encode())
            if k == "_dis_active":
                v = [(d["kind"], d["target"]["type"], d["target"]["idx"], d["start_s"], d["end_s"]) for d in v]
            if isinstance(v, np.ndarray):
                h.update(str(v.dtype).encode())
                h.update(np.ascontiguousarray(v).tobytes())
            elif isinstance(v, np.random.Generator):
                h.update(repr(v.bit_generator.state).encode())
            elif isinstance(v, Scenario):
                h.update(repr(v).encode())
            else:
                h.update(repr(v).encode())
        return h.hexdigest()

    # ---------------------------------------------------------------- non-contract helpers
    @property
    def graph(self) -> Graph:
        return self.g

    @property
    def episode_time_s(self) -> float:
        """Sim seconds since the episode start (after warm-up)."""
        return float(self.clock_s - self._ep_start)

    def active_train_ids(self) -> np.ndarray:
        return np.flatnonzero(self.tr_active).astype(np.int32)

    def masks_for(self, train_ids: Sequence[int]) -> np.ndarray:
        """Current masks for trains *at a stop* (recomputed; for tests and debugging)."""
        from sim.masks import compute_masks

        ids = np.asarray(train_ids, dtype=np.int64)
        return compute_masks(self, ids)[0]

    def train_info(self, tid: int) -> dict[str, Any]:
        """Debug view of one train slot."""
        S = self._S
        t = int(tid)
        e = int(self.p_edge[t, self.tr_ptr[t]])
        node = int(S.efrom[e]) if self.tr_at_start[t] else int(S.eto[e])
        return {
            "active": bool(self.tr_active[t]),
            "route": self.g.route_id[int(self.tr_route[t])] if self.tr_route[t] >= 0 else None,
            "pattern": int(self.tr_pattern[t]),
            "edge": e,
            "ptr": int(self.tr_ptr[t]),
            "offset": float(self.tr_off[t]),
            "v": float(self.tr_v[t]),
            "mode": ["run", "dwell", "ready", "timed"][int(self.tr_mode[t])],
            "timer": float(self.tr_timer[t]),
            "at_start": bool(self.tr_at_start[t]),
            "node": node,
            "station": int(S.nst[node]),
            "onboard": float(self.tr_onboard[t].sum()),
            "lateness_s": float(self.tr_late[t]),
            "hold_total": float(self.tr_hold_total[t]),
            "in_closed": bool(self.tr_in_closed[t]),
            "gap_train": bool(self.tr_gap[t]),
            "deviated": bool(self.tr_deviated[t]),
        }

    def diagnostics(self) -> dict[str, Any]:
        """Calibration diagnostics (not part of the contract)."""
        dl = np.asarray(self._dwell_log, dtype=np.float64)
        return {
            "dwell_median_s": float(np.median(dl)) if dl.size else float("nan"),
            "dwell_p90_s": float(np.percentile(dl, 90)) if dl.size else float("nan"),
            "dwell_samples": int(dl.size),
            "missed_departures": int(self.missed_departures),
            "laid_up": int(self.laid_up),
            "put_ins": int(self.put_ins),
            "riders_displaced": float(self.riders_displaced),
            "pool_trains": int(np.count_nonzero(np.isfinite(self.pool_rel))),
        }

    # ------------------------------------------------------------------ tick
    def _tick(self) -> None:
        self._t0 = self.clock_s
        if self._dis_scheduled or any(d["end_s"] is not None for d in self._dis_active):
            self._update_disruptions()
        cands = self._timers()
        self._timed_entries()
        self._move()
        self._passengers()
        self._accumulate_costs()
        self.clock_s = self._t0 + DT
        self.tick += 1
        self._tick_abs += 1
        self._batch = None
        if self._retire:
            self._retire_trains()
        self._occupancy()
        self._spawn_phase()
        if self._decide:
            waiting = np.flatnonzero(
                self.tr_active & (self.tr_mode == M_READY) & (self.tr_next_dec <= self.clock_s + 1e-6) & ~self.tr_in_closed
            )
            if waiting.size:
                self._topup_board(waiting)
                self.tr_next_dec[waiting] = self.clock_s + C.HOLD_STEP_S
            if cands.size or waiting.size:
                self._collect_decisions(np.union1d(cands, waiting))

    # ------------------------------------------------------------------ timers
    def _timers(self) -> np.ndarray:
        act = self.tr_active
        mode = self.tr_mode
        cands = np.zeros(0, dtype=np.int64)
        d = np.flatnonzero(act & (mode == M_DWELL))
        if d.size:
            self.tr_timer[d] -= DT
            done = d[self.tr_timer[d] <= 1e-9]
            if done.size:
                cands = self._dwell_done(done)
        t = np.flatnonzero(act & (mode == M_TIMED))
        if t.size:
            self.tr_timer[t] -= DT
            done = t[self.tr_timer[t] <= 1e-9]
            if done.size:
                self._timed_done(done)
        return cands

    def _node_of(self, idx: np.ndarray) -> np.ndarray:
        S = self._S
        e = self.p_edge[idx, self.tr_ptr[idx]].astype(np.intp)
        return np.where(self.tr_at_start[idx], S.efrom[e], S.eto[e])

    def _dwell_done(self, idx: np.ndarray) -> np.ndarray:
        fin = self.tr_finishing[idx]
        if fin.any():
            for t in idx[fin].tolist():
                if t not in self._retire:
                    self._retire.append(t)
            self.tr_timer[idx[fin]] = 0.0
        r = idx[~fin]
        if r.size == 0:
            return r
        self._topup_board(r)
        self.tr_mode[r] = M_READY
        self.tr_ready_tick[r] = self._tick_abs
        self.tr_held[r] = False
        self.tr_next_dec[r] = self._t0 + DT + C.HOLD_STEP_S
        return r

    def _timed_done(self, idx: np.ndarray) -> None:
        S = self._S
        for t in idx.tolist():
            if self.tr_tkind[t] == TK_KEYBY:
                self.tr_mode[t] = M_RUN
                self.tr_tkind[t] = TK_NONE
                self.tr_keyby_done[t] = int(self.tr_ptr[t]) + (0 if self.tr_at_start[t] else 1)
                self.tr_timer[t] = 0.0
                continue
            pos = int(self.tr_ptr[t])
            e = int(self.p_edge[t, pos])
            n = int(S.eto[e])
            # A crossover occupant holds the route into n: it leaves as soon as n is
            # physically free. A pocket train merges back onto the main line, which needs
            # a fresh route into n (approach locking: it yields to approaching trains).
            pocket = bool(S.epocket[e])
            if self.n_closed[n] or not self._node_clear(n, ignore=t, approach=pocket):
                self.tr_timer[t] = 0.0
                continue
            if not pocket:
                self._release_route(t, e)
            self.tr_tkind[t] = TK_NONE
            self.tr_timer[t] = 0.0
            self.tr_v[t] = 0.0
            if pos + 1 >= int(self.tr_plen[t]):
                self.tr_at_start[t] = False
                self.tr_off[t] = 0.0
                self._arrive(np.array([t]), np.array([n]), np.array([DT]), np.array([pos]))
            else:
                self.tr_ptr[t] = pos + 1
                self.tr_at_start[t] = True
                self.tr_off[t] = 0.0
                self.tr_stop_done[t] = -1
                if self.p_stop[t, pos]:
                    self._arrive(np.array([t]), np.array([n]), np.array([DT]), np.array([pos]))
                else:
                    self.tr_mode[t] = M_RUN
                self._register_at_start(np.array([t]))

    # ------------------------------------------------------------------ timed edges
    def _next_edge_index(self, t: int) -> int:
        return int(self.tr_ptr[t]) + (0 if self.tr_at_start[t] else 1)

    def _timed_entries(self) -> None:
        """Trains standing at the entry of a timed edge (crossover/turnback) try to enter.

        Runs at the start of the tick, before the kinematics, so every route is set against
        the true positions of the approaching trains (which then see the reservation and
        stop at its home signal). A train that reaches a timed edge entry during the
        kinematics stops there and requests its route at the next tick. Requests are served
        in order of scheduled departure (then slot): a deterministic first-come rule.
        """
        S = self._S
        mode = self.tr_mode
        act = self.tr_active & ~self.tr_in_closed
        act &= (mode == M_RUN) | ((mode == M_READY) & (self.tr_ready_tick < self._tick_abs))
        cand = np.flatnonzero(act)
        if cand.size == 0:
            return
        ptr = self.tr_ptr[cand]
        at_start = self.tr_at_start[cand]
        k = ptr + (~at_start)
        e_cur = self.p_edge[cand, ptr].astype(np.intp)
        e_next = self.p_edge[cand, np.minimum(k, self._PC - 1)].astype(np.intp)
        # standing exactly at the node (the kinematics stop a train at a timed-edge entry
        # exactly), and done with the stop there if it is one (a train still braking into
        # its platform must berth and dwell first)
        at_end = (self.tr_off[cand] >= S.elen[e_cur] - 1e-3) & (self.tr_v[cand] < C.STOPPED_V)
        at_end &= ~self.p_stop[cand, ptr] | (self.tr_stop_done[cand] == ptr)
        at_node = at_start | at_end
        sel = (k < self.tr_plen[cand]) & S.etimed[e_next] & at_node
        if not sel.any():
            return
        idx = cand[sel]
        if idx.size > 1:
            idx = idx[np.lexsort((idx, self.tr_sched_dep[idx]))]
        for t in idx.tolist():
            self._try_enter_timed(t)

    def _try_enter_timed(self, t: int) -> bool:
        """Set the route over the timed edge ahead of train ``t`` if the interlocking allows.

        A crossover (non-pocket) needs both of its nodes free of other routes and approach
        locking on its exit node; the occupant then reserves both nodes until it leaves. A
        pocket turnback leads onto a center track that fouls no running line: it only needs
        the pocket itself (exclusive).
        """
        S = self._S
        k = self._next_edge_index(t)
        if k >= int(self.tr_plen[t]):
            return False
        e = int(self.p_edge[t, k])
        if not S.etimed[e]:
            return False
        n_to = int(S.eto[e])
        node_here = int(S.efrom[e])
        pocket = bool(S.epocket[e])
        ok = not (self.e_xoff[e] or self.n_closed[n_to] or self._e_timed_occ[e])
        if ok and not pocket:
            r_here, r_to = int(self._node_res[node_here]), int(self._node_res[n_to])
            ok = r_here in (-1, t) and r_to in (-1, t) and self._node_clear(n_to, ignore=t)
        if not ok:
            self.tr_v[t] = 0.0
            return False
        was_ready = self.tr_mode[t] == M_READY
        self.tr_ptr[t] = k
        self.tr_at_start[t] = False
        self.tr_off[t] = 0.0
        self.tr_v[t] = 0.0
        self.tr_mode[t] = M_TIMED
        self.tr_tkind[t] = TK_EDGE
        self.tr_timer[t] = float(S.erun[e])
        self._e_timed_occ[e] = True
        if not pocket:
            self._reserve_route(t, node_here, n_to)
        if was_ready:
            self._record_departure(np.array([t]), np.array([node_here]))
        # The train keeps its record on the run edge it is leaving until the end-of-tick
        # rebuild: its rear is still clearing that block (followers stay conservative).
        return True

    def _reserve_route(self, t: int | np.ndarray, a: int | np.ndarray, b: int | np.ndarray) -> None:
        """Lock nodes ``a`` (tail) and ``b`` (route) for the crossover occupant(s) ``t``."""
        self._node_res[a] = t
        self._node_res[b] = t
        self._node_tail[a] = True
        self._node_route[b] = True

    def _release_route(self, t: int, e: int) -> None:
        """Free the nodes of timed edge ``e`` held by train ``t`` (it has left the edge)."""
        S = self._S
        for n in (int(S.efrom[e]), int(S.eto[e])):
            if self._node_res[n] == t:
                self._node_res[n] = -1
                self._node_tail[n] = False
                self._node_route[n] = False

    def _node_clear(self, n: int, *, ignore: int = -1, approach: bool = True) -> bool:
        """Whether a train may be put at node ``n`` (route setting / spawning).

        ``approach=True`` (approach locking: entering a crossover or pocket exit towards
        ``n``, spawning a train at ``n``): ``n`` is not reserved by another train's route,
        no train front is within ``NODE_CLEAR_M`` of ``n`` on a running in-edge, and no
        train body straddles ``n``. ``approach=False`` (a crossover occupant leaving onto
        the node it has reserved): only physical occupation counts, i.e. a front within
        ``NODE_OCCUPIED_M`` or a body over ``n``.

        A train standing at ``n`` to leave it over a timed edge (a crossover or turnback
        of its own) never blocks: it is beside the route, not on it, and is itself held by
        the reservations (this is what lets two trains swap tracks through crossovers in
        opposite senses). Uses the occupancy tables (rebuilt once per tick, with spawns and
        timed-edge exits registered incrementally); ``ignore`` is a train that may be the
        one found.
        """
        S = self._S
        if n >= S.SN:
            return False
        if approach:
            r = int(self._node_res[n])
            if r >= 0 and r != ignore:
                return False
        reach = C.NODE_CLEAR_M if approach else C.NODE_OCCUPIED_M
        for e in S.run_in[n]:
            f = int(self._front_id[e])
            if f < 0 or f == ignore or self._front_max[e] < S.elen[e] - reach:
                continue
            if self._front_max[e] >= S.elen[e] - 0.5 and self._leaves_by_timed_edge(f):
                continue
            return False
        for e in S.run_out[n]:
            r = int(self._rear_id[e])
            if r >= 0 and r != ignore and self._rear_min[e] < 1.0:
                return False
        return True

    def _leaves_by_timed_edge(self, t: int) -> bool:
        """Train ``t`` stands at the end of its edge and its path continues over a timed edge."""
        k = int(self.tr_ptr[t]) + (0 if self.tr_at_start[t] else 1)
        return k < int(self.tr_plen[t]) and bool(self._S.etimed[int(self.p_edge[t, k])])

    # ------------------------------------------------------------------ kinematics
    def _move(self) -> None:
        S = self._S
        act = self.tr_active
        mode = self.tr_mode
        sel = act & ((mode == M_RUN) | ((mode == M_READY) & (self.tr_ready_tick < self._tick_abs) & ~self.tr_in_closed))
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            return
        # standing at a timed edge entry from the node: handled by _timed_entries
        at_start = self.tr_at_start[idx]
        ptr = self.tr_ptr[idx]
        e0 = self.p_edge[idx, ptr].astype(np.intp)
        keep = ~(at_start & S.etimed[e0])
        if not keep.all():
            idx, at_start, ptr, e0 = idx[keep], at_start[keep], ptr[keep], e0[keep]
            if idx.size == 0:
                return
        n = idx.size
        off = self.tr_off[idx]
        v0 = self.tr_v[idx]
        cols = np.minimum(ptr[:, None] + self._ark, self._PC - 1)
        rows = idx[:, None]
        E = self.p_edge[rows, cols].astype(np.intp)
        elen = S.elen[E]
        ends = np.cumsum(elen, axis=1) - off[:, None]
        starts = ends - elen
        to_nodes = S.eto[E]

        # --- hard obstacles ----------------------------------------------------------
        tmd = S.etimed[E]
        tmd[:, 0] = False
        d_timed = np.where(tmd, starts, INF).min(axis=1)
        # A timed edge (crossover/turnback) is a barrier: the train stops at its entry and
        # traverses it by time (_try_enter_timed / _timed_done, which handle a stop at its
        # end). Stops at or beyond the first timed edge are not braking targets here.
        stp = self.p_stop[rows, cols] & (np.cumsum(tmd, axis=1) == 0)
        stp[:, 0] &= self.tr_stop_done[idx] != ptr
        d_stop = np.where(stp, ends, INF).min(axis=1)
        blk = self.e_blocked[E].copy()
        blk[:, 0] &= at_start
        sig = self.e_sig[E]
        sig_occ = sig & (self._rear_min[E] < INF)
        sig_occ[:, 0] = sig[:, 0] & at_start & (self._lead_rear[idx] < INF)
        red = blk | sig_occ
        # A red signal at the departure end of a platform where the train stops: it still
        # berths at the platform (the stop point is the signal); elsewhere it stops short.
        berth = np.zeros_like(red)
        berth[:, 1:] = stp[:, :-1]
        d_red = np.where(red, starts - np.where(berth, 0.0, C.FIXED_BLOCK_STOP_MARGIN_M), INF).min(axis=1)
        kb = sig.copy()
        kb[:, 0] &= at_start
        kb &= cols != self.tr_keyby_done[idx][:, None]
        d_kb = np.where(kb, starts, INF).min(axis=1)
        clo = self.n_closed[to_nodes]
        d_clo = np.where(clo, ends - C.CLOSED_NODE_MARGIN_M, INF).min(axis=1)
        horizon = ends[:, -1]
        rem0 = elen[:, 0] - off

        # --- interlocking reservations (crossover occupants, see _try_enter_timed) ------
        # A route node is protected by a red home signal for everybody else; a tail node
        # carries the occupant's body (a leader whose rear is one train length short of
        # it). A train already standing at the node is not approaching it: it keeps its
        # own authority, and the occupant waits for it to clear (_node_clear).
        rsv_route = self._node_route[to_nodes]
        rsv_tail = self._node_tail[to_nodes]
        if rsv_route.any() or rsv_tail.any():
            away = rem0 > C.NODE_OCCUPIED_M
            rsv_route[:, 0] &= away
            rsv_tail[:, 0] &= away
            d_red = np.minimum(d_red, np.where(rsv_route, ends - C.FIXED_BLOCK_STOP_MARGIN_M, INF).min(axis=1))
        else:
            rsv_tail = None

        # --- leaders -----------------------------------------------------------------
        d_lead = self._lead_rear[idx] - off
        rm = self._rear_min[E[:, 1:]]
        d_lead = np.minimum(d_lead, (starts[:, 1:] + rm).min(axis=1))
        if rsv_tail is not None and rsv_tail.any():
            d_lead = np.minimum(d_lead, np.where(rsv_tail, ends - self.tr_len[idx][:, None], INF).min(axis=1))
        cb = np.where(rem0 > 1.0, S.ecbtc[E[:, 0]], S.ecbtc[E[:, 1]])
        fin = np.isfinite(d_lead)
        auth_lead = np.full(n, INF)
        if fin.any():
            if cb.any():
                auth_lead = np.where(cb & fin, d_lead - C.CBTC_MARGIN_M, auth_lead)
            fb = fin & ~cb
            if fb.any():
                fba = fixed_block_authority(d_lead[fb], starts[fb], elen[fb])
                auth_lead[fb] = fba
            # Station close-in (station-time signals): a train may always berth at its next
            # stop once the leader's rear has cleared the platform end.
            close_in = fin & (d_stop < d_lead - C.FIXED_BLOCK_STOP_MARGIN_M)
            auth_lead = np.where(close_in, np.maximum(auth_lead, d_stop), auth_lead)

        auth = np.minimum.reduce([d_stop, d_timed, d_red, d_kb, d_clo, horizon, auth_lead])

        # --- speed envelope (braking curves to lower downstream limits) -----------------
        vm_e = self.e_vmax[E]
        env = np.sqrt(vm_e * vm_e + 2.0 * C.BRAKE * np.maximum(starts, 0.0))
        env[:, 0] = np.where(rem0 > 1.0, vm_e[:, 0], INF)
        vm = env.min(axis=1)
        vm = np.where(np.isfinite(vm), vm, vm_e[:, 1])

        x, v1, t_arr = motion_profile(v0, vm, auth, np.full(n, DT))

        # --- events ------------------------------------------------------------------
        tol = 0.01
        is_stop = (d_stop <= auth + 1e-6) & (x >= d_stop - tol)
        is_timed = ~is_stop & (d_timed <= auth + 1e-6) & (x >= d_timed - tol)
        is_kb = ~is_stop & ~is_timed & (d_kb <= auth + 1e-6) & (x >= d_kb - tol)

        # --- advance along the path --------------------------------------------------
        k_new = (ends[:, : _K_LOOK - 1] < x[:, None] - 1e-6).sum(axis=1)
        moved = x > 1e-3
        # merge rule: one train per newly entered edge per tick (earliest boundary first)
        entering = k_new > 0
        if entering.sum() > 1:
            ent_edge = E[np.arange(n), k_new]
            ei = np.flatnonzero(entering)
            ee = ent_edge[ei]
            uniq, cnt = np.unique(ee, return_counts=True)
            if (cnt > 1).any():
                for edge in uniq[cnt > 1].tolist():
                    grp = ei[ee == edge]
                    dist = starts[grp, k_new[grp]]
                    order = np.lexsort((idx[grp], self.tr_sched_dep[idx[grp]], dist))
                    losers = grp[order[1:]]
                    k_new[losers] -= 1
                    x[losers] = ends[losers, k_new[losers]]
                    v1[losers] = 0.0
                    is_stop[losers] = False
                    is_timed[losers] = False
                    is_kb[losers] = False
        moved = x > 1e-3
        new_ptr = ptr + k_new
        new_off = x - starts[np.arange(n), k_new]
        new_len = elen[np.arange(n), k_new]
        new_off = np.clip(new_off, 0.0, new_len)

        departed = (mode[idx] == M_READY) & moved
        if departed.any():
            di = idx[departed]
            node_dep = np.where(at_start[departed], S.efrom[E[departed, 0]], to_nodes[departed, 0])
            self.tr_mode[di] = M_RUN
            if self._decide and len(self._dwell_log) < 200000:
                st_dep = di[~at_start[departed]]
                if st_dep.size:
                    self._dwell_log.extend((self._t0 - self.tr_arr_time[st_dep]).tolist())
            self._record_departure(di, node_dep)
        # nodes passed through
        t_end = self._t0 + DT
        for j in range(_K_LOOK - 1):
            pj = k_new > j
            if j == 0:
                pj &= ~at_start | (k_new > 0)
            if pj.any():
                nodes = to_nodes[pj, j]
                self.nd_last_dep[nodes] = t_end
                pl = S.npl[nodes]
                ok = pl >= 0
                if ok.any():
                    self.pl_last_dep[pl[ok]] = t_end
        st_moved = at_start & moved
        if st_moved.any():
            self.tr_at_start[idx[st_moved]] = False
        self.tr_ptr[idx] = new_ptr
        self.tr_off[idx] = new_off
        self.tr_v[idx] = v1

        # --- arrivals, timed entries, key-by ---------------------------------------------
        if is_stop.any():
            ai = idx[is_stop]
            ta = np.where(np.isfinite(t_arr[is_stop]), t_arr[is_stop], DT)
            nodes = S.eto[self.p_edge[ai, self.tr_ptr[ai]].astype(np.intp)]
            self.tr_off[ai] = S.elen[self.p_edge[ai, self.tr_ptr[ai]].astype(np.intp)]
            self._arrive(ai, nodes, ta, self.tr_ptr[ai].copy())
        if is_timed.any():
            # standing at the entry of a timed edge: the route is requested at the start of
            # the next tick (_timed_entries), against the then-current positions
            self.tr_v[idx[is_timed]] = 0.0
        if is_kb.any():
            ki = idx[is_kb]
            was_ready = ki[self.tr_mode[ki] == M_READY]
            if was_ready.size:
                self._record_departure(was_ready, self._node_of(was_ready))
            self.tr_v[ki] = 0.0
            self.tr_mode[ki] = M_TIMED
            self.tr_tkind[ki] = TK_KEYBY
            self.tr_timer[ki] = C.SIGNAL_DELAY_KEYBY_S - (DT - np.minimum(np.where(np.isfinite(t_arr[is_kb]), t_arr[is_kb], DT), DT))

    def _record_departure(self, idx: np.ndarray, nodes: np.ndarray) -> None:
        S = self._S
        t = self._t0
        self.nd_last_dep[nodes] = t
        pl = S.npl[nodes]
        ok = pl >= 0
        if ok.any():
            self.pl_last_dep[pl[ok]] = t
        self.tr_held[idx] = False
        self.tr_next_dec[idx] = INF

    # ------------------------------------------------------------------ stops
    def _arrive(self, idx: np.ndarray, nodes: np.ndarray, t_in: np.ndarray, pos: np.ndarray) -> None:
        S = self._S
        n = idx.size
        pl = S.npl[nodes]
        st = S.nst[nodes]
        ob = self.tr_onboard[idx]
        sptr = self.tr_sptr[idx]
        term = sptr >= self.tr_nstops[idx] - 1
        tot = ob.sum(axis=1)
        alight = np.where(term, tot, ob[:, 0])
        xf = np.where(S.st_xfer[st], C.TRANSFER_FRAC, 0.0)
        x_amt = alight * xf
        self.riders_served += float((alight - x_amt).sum())
        if (x_amt > 0).any():
            self._transfer(st[x_amt > 0], x_amt[x_amt > 0])
        ob2 = shift_buckets(ob, _Q_TAIL)
        ob2[term] = 0.0
        self.tr_onboard[idx] = ob2
        self.tr_sptr[idx] = sptr + 1
        board = np.zeros(n)
        nt = ~term & (pl >= 0)
        if nt.any():
            board[nt] = self._board(idx[nt], pl[nt])
        load = self.tr_onboard[idx].sum(axis=1) / self.tr_cap[idx]
        base = C.DWELL_MIN_S + C.BOARD_S_PER_RIDER * (alight + board)
        crowd = 1.0 + C.CROWD_DWELL_K * np.maximum(0.0, load - C.CROWD_DWELL_LOAD0) ** 2
        dwell = np.minimum(base * crowd * self.dwell_scale, C.DWELL_MAX_S)
        dwell = np.where(term, np.minimum(C.DWELL_MIN_S + C.BOARD_S_PER_RIDER * alight, C.DWELL_MAX_S), dwell)
        self.tr_mode[idx] = M_DWELL
        self.tr_tkind[idx] = TK_NONE
        self.tr_timer[idx] = dwell - (DT - t_in)
        self.tr_v[idx] = 0.0
        self.tr_finishing[idx] = term
        arr_t = self._t0 + t_in
        self.tr_late[idx] = arr_t - (self.tr_sched_dep[idx] + self.p_sched[idx, pos].astype(np.float64))
        self.tr_arr_time[idx] = arr_t
        run_arr = ~self.tr_at_start[idx]
        self.tr_stop_done[idx] = np.where(run_arr, self.tr_ptr[idx], -1)
        self.tr_skipped_prev[idx] = self.tr_skip_next[idx]
        self.tr_skip_next[idx] = False
        self.tr_hold_total[idx] = 0.0
        self.tr_held[idx] = False

    def _dest_weights(self, idx: np.ndarray) -> np.ndarray:
        """Destination bucket weights for boarders (next 11 stops + 12+ tail), normalized."""
        sptr = self.tr_sptr[idx]
        ns = self.tr_nstops[idx]
        j = sptr[:, None] + np.arange(C.N_DEST - 1)[None, :]
        valid = j < ns[:, None]
        jj = np.minimum(j, self._SC - 1)
        stations = self.s_station[idx[:, None], jj].astype(np.intp)
        w = np.where(valid, self._P.attr[np.maximum(stations, 0)] * _DEC11[None, :], 0.0)
        tail_j = np.minimum(sptr + C.N_DEST - 1, self._SC)
        tail = self.s_suffix[idx, tail_j] * np.exp(np.minimum(sptr, 600) / C.DEST_DECAY_STOPS)
        out = np.concatenate([w, tail[:, None]], axis=1)
        s = out.sum(axis=1)
        bad = s <= 1e-12
        if bad.any():
            out[bad] = 0.0
            out[bad, 0] = 1.0
            s[bad] = 1.0
        return out / s[:, None]

    def _board(self, idx: np.ndarray, pl: np.ndarray) -> np.ndarray:
        """Board waiting riders (oldest first) at platforms ``pl``; returns riders boarded.

        Trains sharing a platform board one after another in batch order (rare: a single
        vectorized pass handles the usual all-distinct case).
        """
        n = idx.size
        out = np.zeros(n)
        if n == 0:
            return out
        if n > 1:
            ps = np.sort(pl)
            if (ps[1:] == ps[:-1]).any():
                remaining = np.arange(n)
                while remaining.size:
                    _, first = np.unique(pl[remaining], return_index=True)
                    cur = remaining[np.sort(first)]
                    out[cur] = self._board_distinct(idx[cur], pl[cur])
                    keep = np.ones(remaining.size, dtype=bool)
                    keep[np.searchsorted(remaining, cur)] = False
                    remaining = remaining[keep]
                return out
        return self._board_distinct(idx, pl)

    def _board_distinct(self, idx: np.ndarray, pl: np.ndarray) -> np.ndarray:
        W = self.waiting[pl]
        tot = W.sum(axis=1)
        room = np.maximum(self.tr_cap[idx] - self.tr_onboard[idx].sum(axis=1), 0.0)
        b = np.minimum(tot, room)
        if (b > 0).any():
            Wr = W[:, ::-1]
            cum = np.cumsum(Wr, axis=1)
            take = np.clip(b[:, None] - (cum - Wr), 0.0, Wr)
            self.waiting[pl] = (Wr - take)[:, ::-1]
            self.tr_onboard[idx] += b[:, None] * self._dest_weights(idx)
        return b

    def _topup_board(self, idx: np.ndarray) -> None:
        if idx.size == 0:
            return
        nodes = self._node_of(idx)
        pl = self._S.npl[nodes]
        ok = (pl >= 0) & ~self.tr_in_closed[idx] & ~self.tr_finishing[idx]
        if ok.any():
            self._board(idx[ok], pl[ok])

    def _transfer(self, stations: np.ndarray, amounts: np.ndarray) -> None:
        P = self._P
        a = P.xfer_ptr[stations]
        b = P.xfer_ptr[stations + 1]
        cnt = b - a
        tot = int(cnt.sum())
        if tot == 0:
            self.riders_served += float(amounts.sum())
            return
        rep = np.repeat(np.arange(stations.size), cnt)
        offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        pos = np.repeat(a, cnt) + offs
        np.add.at(self.waiting[:, 0], P.xfer_idx[pos], amounts[rep] * P.xfer_w[pos])
        lost = amounts[cnt == 0]
        if lost.size:
            self.riders_served += float(lost.sum())
        if self.pl_closed.any():
            self.riders_displaced += float(self.waiting[self.pl_closed].sum())
            self.waiting[self.pl_closed] = 0.0

    # ------------------------------------------------------------------ passengers
    def _passengers(self) -> None:
        self.waiting[:, 0] += self.rate_eff * DT
        self._age_phase += 1
        if self._age_phase >= _AGE_SHIFT_TICKS:
            self._age_phase = 0
            w = self.waiting
            w[:, -1] += w[:, -2]
            w[:, 1:-1] = w[:, :-2].copy()
            w[:, 0] = 0.0
            self._spillover()

    def _spillover(self) -> None:
        P = self._P
        b0 = C.SPILLOVER_MIN_AGE_MIN
        stale = P.has_spill & ~self.pl_closed & (self._t0 - self.pl_last_dep > C.SPILLOVER_HEADWAY_MULT * P.pl_h)
        if not stale.any():
            return
        src = np.flatnonzero(stale)
        moved = self.waiting[src, b0:] * C.SPILLOVER_RATE_PER_MIN
        if moved.sum() <= 0:
            return
        self.waiting[src, b0:] -= moved
        a = P.spill_ptr[src]
        cnt = P.spill_ptr[src + 1] - a
        tot = int(cnt.sum())
        rep = np.repeat(np.arange(src.size), cnt)
        offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        pos = np.repeat(a, cnt) + offs
        dst = P.spill_idx[pos]
        contrib = moved[rep] * P.spill_w[pos][:, None]
        np.add.at(self.waiting[:, b0:], dst, contrib)
        if self.pl_closed.any():
            self.riders_displaced += float(self.waiting[self.pl_closed].sum())
            self.waiting[self.pl_closed] = 0.0

    def _platform_stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(total waiting, mean wait min, oldest non-empty bin center min) per platform."""
        if self._pl_stats is not None and self._pl_stats_tick == self.tick:
            return self._pl_stats
        w = self.waiting
        S = self._S
        tot = w.sum(axis=1)
        mean = (w @ S.age_c) / np.maximum(tot, 1e-9)
        nz = w > 0.5
        last = C.N_AGE_BINS - 1 - np.argmax(nz[:, ::-1], axis=1)
        oldest = np.where(nz.any(axis=1), S.age_c[last], 0.0)
        self._pl_stats = (tot, mean, oldest)
        self._pl_stats_tick = self.tick
        return self._pl_stats

    # ------------------------------------------------------------------ costs
    def _accumulate_costs(self) -> None:
        act = self.tr_active
        # tunnel-stopped trains (any duration) and tunnel-stuck timers
        self._compute_at_pl()
        stopped = act & (self.tr_v < C.STOPPED_V) & ~self.tr_at_pl
        stopped &= (self.tr_mode == M_RUN) | ((self.tr_mode == M_TIMED) & (self.tr_tkind == TK_KEYBY))
        newly = stopped & ~np.isfinite(self.tr_stopped_since)
        self.tr_stopped_since[newly] = self._t0
        self.tr_stopped_since[~stopped] = INF
        self.tr_tunnel = stopped
        held = act & ((self.tr_held & (self.tr_mode == M_DWELL)) | self.tr_in_closed)
        trapped_min = np.where(stopped, (self._t0 + DT - self.tr_stopped_since) / 60.0, 0.0)
        onboard = self.tr_onboard.sum(axis=1)
        terms = tick_costs(self.waiting, self._S.pl_cap, onboard, stopped, held, DT, trapped_min)
        self.cost[: terms.size] += terms
        self.cost[I_TOTAL] += float(terms.sum())
        if self.track_cost_vectors:
            plat, per_train = tick_cost_vectors(self.waiting, self._S.pl_cap, onboard, stopped, held, DT, trapped_min)
            self.cost_plat += plat
            t = np.flatnonzero(per_train)
            r = self.tr_route[t]
            if t.size and bool((r >= 0).all()):
                self.cost_route_train += np.bincount(r, weights=per_train[t], minlength=self.cost_route_train.size)
            elif t.size:  # pragma: no cover - an active train always has a route
                ok = r >= 0
                self.cost_route_train += np.bincount(r[ok], weights=per_train[t][ok],
                                                     minlength=self.cost_route_train.size)

    def _compute_at_pl(self) -> np.ndarray:
        """Refresh ``tr_at_pl`` (front at a platform node, or crossing at a station); returns edges."""
        S = self._S
        act = self.tr_active
        e_all = self.p_edge[np.arange(self._N), self.tr_ptr].astype(np.intp)
        at_st = self.tr_at_start
        node = np.where(at_st, S.efrom[e_all], S.eto[e_all])
        has_pl = S.npl[node] >= 0
        near = at_st | (self.tr_off >= S.elen[e_all] - C.AT_PLATFORM_M)
        timed_edge = (self.tr_mode == M_TIMED) & (self.tr_tkind == TK_EDGE)
        self.tr_at_pl = act & ((has_pl & near & ~timed_edge) | timed_edge)
        return e_all

    def _tunnel_stuck_mask(self) -> np.ndarray:
        return self.tr_tunnel & (self.clock_s - self.tr_stopped_since >= C.TUNNEL_STUCK_S - 1e-6)

    def charge(self, key: str, amount: float) -> None:
        """Add an immediate (action-time) cost term."""
        i = C.COST_KEYS.index(key)
        self.cost[i] += amount
        self.cost[I_TOTAL] += amount

    # ------------------------------------------------------------------ occupancy
    def _occupancy(self) -> None:
        S = self._S
        act = self.tr_active
        timed_edge = (self.tr_mode == M_TIMED) & (self.tr_tkind == TK_EDGE)
        on_run = act & ~timed_edge
        idx = np.flatnonzero(on_run)
        e = self.p_edge[idx, self.tr_ptr[idx]].astype(np.intp)
        off = self.tr_off[idx]
        tl = self.tr_len[idx]
        rear = off - tl
        # straddlers: register the body part on the previous path edge (diverging followers)
        ptr = self.tr_ptr[idx]
        strad = (rear < 0.0) & ~self.tr_at_start[idx] & (ptr >= 1)
        rec_e = e
        rec_off = off
        rec_real = np.ones(idx.size, dtype=bool)
        rec_id = idx
        if strad.any():
            si = idx[strad]
            pe = self.p_edge[si, ptr[strad] - 1].astype(np.intp)
            okp = ~S.etimed[pe]
            if okp.any():
                si, pe = si[okp], pe[okp]
                v_off = S.elen[pe] + off[strad][okp]
                rec_e = np.concatenate([e, pe])
                rec_off = np.concatenate([off, v_off])
                rec_real = np.concatenate([rec_real, np.zeros(si.size, dtype=bool)])
                rec_id = np.concatenate([idx, si])
        rec_len = self.tr_len[rec_id]
        key = rec_e.astype(np.float64) * 1.0e5 + rec_off
        o = np.argsort(key, kind="stable")
        eo = rec_e[o]
        offo = rec_off[o]
        rearo = offo - rec_len[o]
        realo = rec_real[o]
        ido = rec_id[o]
        m = eo.size
        same_next = np.zeros(m, dtype=bool)
        if m > 1:
            same_next[:-1] = eo[1:] == eo[:-1]
        lead = np.full(m, INF)
        if m > 1:
            lead[:-1] = np.where(same_next[:-1], rearo[1:], INF)
        self._lead_rear.fill(INF)
        self._lead_rear[ido[realo]] = lead[realo]
        lead_id = np.full(m, -1, dtype=np.int64)
        if m > 1:
            lead_id[:-1] = np.where(same_next[:-1], ido[1:], -1)
        self._lead_id.fill(-1)
        self._lead_id[ido[realo]] = lead_id[realo]
        # follower front: previous *real* record on the same edge (virtual straddler
        # records sit beyond the edge end, so they never precede a real record)
        foll = np.full(m, -INF)
        foll_id = np.full(m, -1, dtype=np.int64)
        if m > 1:
            prev_same = np.zeros(m, dtype=bool)
            prev_same[1:] = same_next[:-1] & realo[:-1]
            foll[1:] = np.where(prev_same[1:], offo[:-1], -INF)
            foll_id[1:] = np.where(prev_same[1:], ido[:-1], -1)
        self._foll_front.fill(-INF)
        self._foll_front[ido[realo]] = foll[realo]
        self._foll_id.fill(-1)
        self._foll_id[ido[realo]] = foll_id[realo]
        first = np.ones(m, dtype=bool)
        if m > 1:
            first[1:] = ~same_next[:-1]
        self._rear_min.fill(INF)
        self._rear_min[eo[first]] = rearo[first]
        self._rear_id.fill(-1)
        self._rear_id[eo[first]] = ido[first]
        # frontmost real train on each edge
        self._front_max.fill(-INF)
        self._front_id.fill(-1)
        ri = np.flatnonzero(realo)
        if ri.size:
            er = eo[ri]
            last = np.ones(ri.size, dtype=bool)
            last[:-1] = er[1:] != er[:-1]
            self._front_max[er[last]] = offo[ri[last]]
            self._front_id[er[last]] = ido[ri[last]]
        # platform status and stopped-outside-platform counts
        e_all = self._compute_at_pl()
        stopped = on_run & (self.tr_v < C.STOPPED_V) & ~self.tr_at_pl & (self.tr_mode == M_RUN)
        self._e_stopped.fill(0)
        if stopped.any():
            np.add.at(self._e_stopped, e_all[stopped], 1)
        # timed-edge occupancy and node obstacles
        self._e_timed_occ.fill(False)
        self._node_res.fill(-1)
        self._node_tail.fill(False)
        self._node_route.fill(False)
        self._st_pocket_busy.fill(False)
        tm = np.flatnonzero(act & timed_edge)
        if tm.size:
            te = e_all[tm]
            self._e_timed_occ[te] = True
            pk = S.epocket[te]
            if (~pk).any():
                self._reserve_route(tm[~pk], S.efrom[te[~pk]], S.eto[te[~pk]])
            if pk.any():
                self._st_pocket_busy[S.nst[S.efrom[te[pk]]]] = True

    # ------------------------------------------------------------------ retire & spawn
    def _retire_trains(self) -> None:
        g = self.g
        P = self._P
        S = self._S
        for t in self._retire:
            if not self.tr_active[t]:
                continue
            e = int(self.p_edge[t, self.tr_ptr[t]])
            node = int(S.efrom[e]) if self.tr_at_start[t] else int(S.eto[e])
            st = int(S.nst[node])
            fam = int(g.route_family[int(self.tr_route[t])])
            k = P.pool_index.get((fam, st))
            joined = False
            if k is not None:
                row = self.pool_rel[k]
                n_in = int(np.count_nonzero(np.isfinite(row)))
                if not (P.pool_surplus[k] and n_in >= P.pool_target[k]):
                    free = np.flatnonzero(~np.isfinite(row))
                    if free.size:
                        row[free[0]] = float(self.tr_arr_time[t]) + C.LAYOVER_S
                        joined = True
            if not joined:
                self.laid_up += 1
            self._free_train(t)
        self._retire = []

    def _free_train(self, t: int) -> None:
        self.tr_active[t] = False
        self.tr_route[t] = -1
        self.tr_pattern[t] = -1
        self.tr_mode[t] = M_RUN
        self.tr_tkind[t] = TK_NONE
        self.tr_v[t] = 0.0
        self.tr_onboard[t] = 0.0
        self.tr_finishing[t] = False
        self.tr_in_closed[t] = False
        self.tr_tunnel[t] = False
        self.tr_at_pl[t] = False
        self.tr_held[t] = False
        self.tr_stopped_since[t] = INF
        self.tr_ptr[t] = 0
        self.p_edge[t, :] = self._S.SE
        self.tr_plen[t] = 0

    def _spawn_phase(self) -> None:
        P = self._P
        now = self.clock_s
        # put-ins (yard -> terminal pool) for structurally short pools
        if P.n_pools:
            due = np.flatnonzero(self.pool_putin_next <= now)
            for k in due.tolist():
                row = self.pool_rel[k]
                free = np.flatnonzero(~np.isfinite(row))
                if free.size and np.count_nonzero(np.isfinite(row)) < max(int(P.pool_target[k]), 1):
                    row[free[0]] = now
                    self.put_ins += 1
                self.pool_putin_next[k] += float(P.pool_putin[k])
        # scheduled departures (admission is sequential and cheap; activation is batched)
        if P.n_svc:
            jit = self.svc_jit[np.arange(P.n_svc), self.svc_n % self.svc_jit.shape[1]] * P.svc_h
            due = np.flatnonzero(self.svc_next - C.ORIGIN_DWELL_S + jit <= now + 1e-6)
            if due.size:
                self._dispatch(due)
        # yard gap trains
        if self._gap_queue:
            keep = []
            for item in self._gap_queue:
                t_enter, y, p, j, node = item
                if t_enter <= now + 1e-6:
                    t = self._spawn_gap(y, p, j, node)
                    if t >= 0:
                        self._register_at_start(np.array([t], dtype=np.int64))
                        continue
                keep.append(item)
            self._gap_queue = keep

    def _dispatch(self, due: np.ndarray) -> None:
        """Dispatch the due timetable departures ``due`` (service entries, ascending)."""
        P = self._P
        S = self._S
        now = self.clock_s
        free = np.flatnonzero(~self.tr_active).tolist()
        used_nodes: set[int] = set()
        slots: list[int] = []
        ks: list[int] = []
        for k in due.tolist():
            h = float(P.svc_h[k])
            dep = float(self.svc_next[k])
            origin = int(P.svc_origin_node[k])
            row = self.pool_rel[int(P.svc_pool[k])]
            j = -1
            if free and origin not in used_nodes and not self.n_closed[origin]:
                avail = np.flatnonzero(row <= now)
                if avail.size and self._node_clear(origin):
                    j = int(avail[np.argmin(row[avail])])
            if j < 0:
                if now > dep + C.LATE_DEPARTURE_MAX_FRAC * h:  # give up: a missed departure (service gap)
                    self.missed_departures += 1
                    self.svc_next[k] += h
                    self.svc_n[k] += 1
                continue
            row[j] = INF
            used_nodes.add(origin)
            slots.append(free.pop(0))
            ks.append(k)
        if not slots:
            return
        idx = np.asarray(slots, dtype=np.int64)
        kk = np.asarray(ks, dtype=np.intp)
        dep = self.svc_next[kk].copy()
        self._spawn_templates(idx, P.svc_pattern[kk], dep)
        self.tr_timer[idx] = np.maximum(dep - now, C.DWELL_MIN_S)
        self.tr_late[idx] = np.maximum(now + self.tr_timer[idx] - dep, 0.0)
        self.svc_next[kk] += P.svc_h[kk]
        self.svc_n[kk] += 1
        pl = S.npl[P.svc_origin_node[kk]]
        ok = pl >= 0
        ok[ok] = ~self.pl_closed[pl[ok]]
        if ok.any():
            self._board(idx[ok], pl[ok])
        self._register_at_start(idx)

    def _spawn_gap(self, y: int, p: int, j: int, node: int) -> int:
        g = self.g
        if self.n_closed[node] or not self._node_clear(node):
            return -1
        slot = self._free_slot()
        if slot < 0:
            return -1
        cp = template_path(g, p, j)
        self._spawn_train(slot, cp, p, sched_dep=self.clock_s + C.DWELL_MIN_S, gap=True)
        self.tr_timer[slot] = C.DWELL_MIN_S * self.dwell_scale
        pl = int(self._S.npl[node])
        if pl >= 0 and not self.pl_closed[pl]:
            self._board(np.array([slot]), np.array([pl]))
        return slot

    def _register_at_start(self, idx: np.ndarray) -> None:
        """Add trains now standing at offset 0 of their current edge to the occupancy tables.

        O(1) per train: the edge's previous rearmost train becomes the leader, and the new
        train becomes that leader's follower.
        """
        S = self._S
        for t in np.asarray(idx).tolist():
            e = int(self.p_edge[t, self.tr_ptr[t]])
            rear = -float(self.tr_len[t])
            prev_id = int(self._rear_id[e])
            self._lead_rear[t] = self._rear_min[e]
            self._lead_id[t] = prev_id
            self._foll_front[t] = -INF
            self._foll_id[t] = -1
            if rear < self._rear_min[e]:
                self._rear_min[e] = rear
                self._rear_id[e] = t
            if self._front_max[e] < 0.0:
                self._front_max[e] = 0.0
                self._front_id[e] = t
            if prev_id >= 0 and prev_id != t and self._foll_front[prev_id] < 0.0:
                self._foll_front[prev_id] = 0.0
                self._foll_id[prev_id] = t
            self.tr_at_pl[t] = bool(S.npl[S.efrom[e]] >= 0)

    # ------------------------------------------------------------------ decisions
    def _clear_pending(self) -> None:
        self._pend_ids = np.zeros(0, dtype=np.int64)
        self._pend_masks = np.zeros((0, C.N_ACTIONS), dtype=bool)
        self._pend_info: dict[str, Any] = {}
        self._pend_pos: dict[int, int] = {}
        self._pend_epoch = self._mask_epoch if hasattr(self, "_mask_epoch") else 0

    def _collect_decisions(self, cands: np.ndarray) -> None:
        from sim.masks import compute_masks, subset_info

        self._clear_pending()
        self._batch = None
        if cands.size == 0:
            return
        cands = np.unique(cands)
        ok = self.tr_active[cands] & (self.tr_mode[cands] == M_READY) & ~self.tr_in_closed[cands] & ~self.tr_finishing[cands]
        cands = cands[ok]
        if cands.size == 0:
            return
        masks, info = compute_masks(self, cands)
        fire = masks[:, 1:].any(axis=1)
        if not fire.any():
            return
        sel = np.flatnonzero(fire)
        self._pend_ids = cands[sel]
        self._pend_masks = masks[sel]
        self._pend_info = subset_info(info, sel)
        self._pend_pos = {int(t): i for i, t in enumerate(self._pend_ids.tolist())}
        self._pend_epoch = self._mask_epoch

    # ------------------------------------------------------------------ disruptions
    def _activate_scheduled(self) -> None:
        if not self._dis_scheduled:
            return
        now = self.clock_s
        keep: list[ScheduledDisruption] = []
        changed = False
        for sd in self._dis_scheduled:
            start = self._ep_start + float(sd.start_s)
            if start <= now + 1e-6:
                eff = D.resolve(self.g, sd.kind, sd.target_type, sd.target_idx)
                did = self._next_dis_id
                self._next_dis_id += 1
                end = None if sd.duration_s is None else start + float(sd.duration_s)
                if end is not None and end <= now:
                    continue
                self._dis_active.append(
                    {"id": did, "kind": eff.kind, "target": {"type": eff.target_type, "idx": int(eff.target_idx)},
                     "label": eff.label, "start_s": float(start), "end_s": end}
                )
                changed = True
            else:
                keep.append(sd)
        self._dis_scheduled = keep
        if changed:
            self._apply_disruption_effects()

    def _update_disruptions(self) -> None:
        if not self._decide:
            return  # scheduled disruptions are relative to the episode start (after warm-up)
        now = self._t0
        before = len(self._dis_active)
        self._dis_active = [d for d in self._dis_active if d["end_s"] is None or d["end_s"] > now + 1e-6]
        if len(self._dis_active) != before:
            self._apply_disruption_effects()
        if self._dis_scheduled:
            save = self.clock_s
            self.clock_s = now
            self._activate_scheduled()
            self.clock_s = save

    def _apply_disruption_effects(self, initial: bool = False) -> None:
        g = self.g
        S = self._S
        self.e_blocked[:] = False
        self.e_bkind[:] = 0
        self.e_sig[:] = False
        self.e_xoff[:] = False
        self.n_closed[:] = False
        self.n_ckind[:] = 0
        prev_closed = self.st_closed.copy()
        self.st_closed[:] = False
        for d in self._dis_active:
            eff = D.resolve(g, d["kind"], d["target"]["type"], d["target"]["idx"])
            code = D.KIND_CODE[eff.kind]
            if eff.blocked_edges.size:
                self.e_blocked[eff.blocked_edges] = True
                self.e_bkind[eff.blocked_edges] = code
            if eff.delayed_edges.size:
                self.e_sig[eff.delayed_edges] = True
                self.e_bkind[eff.delayed_edges] = np.where(
                    self.e_bkind[eff.delayed_edges] == 0, code, self.e_bkind[eff.delayed_edges]
                )
            if eff.disabled_edges.size:
                self.e_xoff[eff.disabled_edges] = True
            if eff.closed_nodes.size:
                self.n_closed[eff.closed_nodes] = True
                self.n_ckind[eff.closed_nodes] = code
                self.st_closed[eff.closed_stations] = True
        self.pl_closed[:] = self.st_closed[S.pl_station]
        vmax = S.evmax * self.speed_scale
        vmax[self.e_sig] = np.minimum(vmax[self.e_sig], C.SIGNAL_DELAY_VMAX)
        self.e_vmax[:] = vmax
        pk = self.g.peak_ok[self.profile]
        self.e_peak_ok[:-1] = pk
        self.rate_eff[:] = self._P.rate * self.demand_scale * (~self.pl_closed)
        # newly closed stations: displace waiting riders, freeze trains inside
        newly = self.st_closed & ~prev_closed
        if newly.any() and not initial:
            pls = self.pl_closed & newly[S.pl_station]
            self.riders_displaced += float(self.waiting[pls].sum())
            self.waiting[pls] = 0.0
        self._update_closed_trains()
        self._mask_epoch += 1

    def _update_closed_trains(self) -> None:
        S = self._S
        act = np.flatnonzero(self.tr_active)
        if act.size == 0:
            return
        e = self.p_edge[act, self.tr_ptr[act]].astype(np.intp)
        at_st = self.tr_at_start[act]
        node = np.where(at_st, S.efrom[e], S.eto[e])
        mode = self.tr_mode[act]
        timed = (mode == M_TIMED) & (self.tr_tkind[act] == TK_EDGE)
        inside_pos = at_st | (self.tr_off[act] >= S.elen[e] - C.CLOSED_NODE_MARGIN_M)
        inside = self.n_closed[node] & inside_pos & ~timed
        was = self.tr_in_closed[act]
        enter = inside & ~was
        if enter.any():
            ti = act[enter]
            snap = ~self.tr_at_start[ti]
            self.tr_off[ti[snap]] = S.elen[e[enter][snap]]
            self.tr_v[ti] = 0.0
            fin = self.tr_finishing[ti]
            nonfin = ti[~fin]
            run_in = self.tr_mode[nonfin] == M_RUN
            if run_in.any():
                self.tr_stop_done[nonfin[run_in]] = self.tr_ptr[nonfin[run_in]]
            self.tr_mode[nonfin] = M_READY
            self.tr_ready_tick[nonfin] = self._tick_abs
            self.tr_held[nonfin] = False
        self.tr_in_closed[act] = inside
        leave = was & ~inside
        if leave.any():
            self.tr_next_dec[act[leave]] = self.clock_s

    # ------------------------------------------------------------------ frame helpers
    def _status_bits(self, idx: np.ndarray, e: np.ndarray) -> np.ndarray:
        S = self._S
        st = np.zeros(idx.size, dtype=np.uint8)
        st |= np.where(self.tr_at_pl[idx], C.ST_AT_PLATFORM, 0).astype(np.uint8)
        stuck = self._tunnel_stuck_mask()[idx]
        st |= np.where(stuck, C.ST_TUNNEL_STUCK, 0).astype(np.uint8)
        held = (self.tr_held[idx] & (self.tr_mode[idx] == M_DWELL)) | self.tr_in_closed[idx]
        st |= np.where(held, C.ST_HELD, 0).astype(np.uint8)
        stt = (self.tr_mode[idx] == M_TIMED) & (self.tr_tkind[idx] == TK_EDGE) & S.epocket[e]
        st |= np.where(stt, C.ST_SHORT_TURNING, 0).astype(np.uint8)
        st |= np.where(self.tr_switch[idx], C.ST_TRACK_SWITCH, 0).astype(np.uint8)
        st |= np.where(self.tr_rerouted[idx], C.ST_REROUTED, 0).astype(np.uint8)
        st |= np.where(self.tr_gap[idx], C.ST_GAP_TRAIN, 0).astype(np.uint8)
        st |= np.where(self.tr_skip_next[idx], C.ST_SKIPPING_NEXT, 0).astype(np.uint8)
        return st

    def _station_frame(self) -> tuple[np.ndarray, np.ndarray]:
        S = self._S
        g = self.g
        tot, _, _ = self._platform_stats()
        ratio = np.where(self.pl_closed, 0.0, tot / S.pl_cap)
        wsum = self.waiting @ S.age_c
        ptr, pidx = S.st_pl_ptr, S.st_pl_idx
        n_st = g.n_stations
        crowd = np.zeros(n_st)
        wt = np.zeros(n_st)
        tt = np.zeros(n_st)
        if pidx.size:
            starts = ptr[:-1]
            has = S.st_has_pl
            r = np.maximum.reduceat(ratio[pidx], np.minimum(starts, pidx.size - 1))
            crowd = np.where(has, r, 0.0)
            ws = np.add.reduceat(wsum[pidx], np.minimum(starts, pidx.size - 1))
            ts = np.add.reduceat(tot[pidx], np.minimum(starts, pidx.size - 1))
            wt = np.where(has, ws, 0.0)
            tt = np.where(has, ts, 0.0)
        c8 = np.minimum(254, np.rint(100.0 * crowd)).astype(np.uint8)
        c8[self.st_closed] = 255
        mean = np.where(tt > 1e-9, wt / np.maximum(tt, 1e-9), 0.0)
        w8 = np.minimum(255, np.rint(4.0 * mean)).astype(np.uint8)
        return c8, w8
