"""Action masks (ARCHITECTURE §4.2): one code path for training and serving.

:func:`compute_masks` evaluates every predicate for a batch of trains standing at a stop,
ready to depart. The common quantities (scheduled headways, leader/follower gaps, the
blocked-ahead horizon) are vectorized over the batch; the structural actions (express,
local, reroute, short-turn, yard) are pre-filtered with vectorized flags and only the few
candidates are checked against their precomputed alternate paths.

Definitions (contract text in quotes; see docs/progress/WP2.md for the clarifications):

* **Scheduled headway is a property of the track.** ``sched_h(e)`` is the combined
  scheduled headway of every in-service pattern using edge ``e`` (equal to the route's
  headway on an unshared track; on a shared trunk the leader is often another route's
  train, so the track headway is the meaningful reference), clipped to
  ``[SCHED_H_MIN_S, SCHED_H_MAX_S]``. The decision's ``sched_h`` is the headway on the
  train's next edge; each gap is compared with the headway *where it is measured*: the
  leader's edge for the leader gap, the follower's edge for the follower gap and the
  reverse service for the opposite-direction gap.
* **Gaps are headways in time.** ``leader_gap_s`` is the nominal time-to-reach (timetable
  speed profile and nominal dwells along this train's path) from the train's departure to
  the leader's front, plus the time the leader has been standing at its platform (capped
  at ``GAP_DWELL_CAP_S``). An on-time leader one headway ahead gives one headway; a stuck
  leader just ahead gives a small gap ("A stuck leader just ahead therefore gives a small
  gap"); a red signal at a disruption ahead, with no train in between, counts as a stuck
  leader. The follower gap is the same quantity seen from the follower.
* **Bunching is measured against the timetable.** On a shared trunk the timetable itself
  spaces trains irregularly (patterns with different headways and phases merge without
  coordination; on the real network 39 % of calm departures on multi-pattern track have a
  leader closer than 0.75 x the combined headway *by schedule*). A gap is therefore "short"
  only if it is short against both the track headway and the planned gap to that very
  train: ``ref_h = min(sched_h, planned)`` with ``planned = gap - (late_behind -
  late_ahead)`` (the lateness difference of the two trains is exactly how far their
  spacing has drifted from plan). The bunched-leader clause of HOLD and the short-follower
  clauses of SKIP_STOP and SHORT_TURN_RELAY use ``ref_h``; the "headway not already wide"
  clause keeps ``sched_h``. On an unshared, regular track ``planned == sched_h`` and the
  definition reduces to the contract text.
* ``blocked_ahead`` — "within the next K_AHEAD stations of the path, any edge or node is
  blocked or closed by a disruption, OR >= 2 trains are stopped outside platforms".
* A *critical wait* at a platform: mean wait >= 10 min or oldest non-empty bin >= 20 min.
* SHORT_TURN_RELAY's "a disruption blocks the path between here and the terminal" requires
  the train to reach the first blocked edge / closed station within
  ``SHORT_TURN_BLOCK_HORIZON_S`` of nominal running: a relay is for the turnbacks in front
  of a blockage. On the real network the literal reading made every pocket up the line valid
  (lex-union-sq: southbound 4/5 trains relayed in the Bronx, ~25 min before 14 St), stranding
  their riders where service was still running; northbound relays at Bowling Green (~8 min
  before 14 St) stay valid.
* DISPATCH_YARD_GAP_TRAIN's "expected time until the next train reaches [the lead node]"
  considers every in-service train on the lead's track (all patterns using the edge leaving
  the lead node), matching the combined ``sched_h`` it is compared with. Reading it per lead
  pattern made rare variants (e.g. a 2 pattern every 3 h) look "starved" on a trunk served
  every 6 min: DISPATCH was valid ~5,000 times per calm am_peak hour. Evaluated for all
  leads at once per tick (:func:`lead_etas`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from sim import constants as C

if TYPE_CHECKING:  # pragma: no cover
    from sim.engine import NYCSubwaySim

W_AHEAD = 16  # path edges in the forward window (leader search, blocked-ahead horizon)
W_BACK = 12  # path edges searched behind for the follower
_KIND_NAMES = ("none",) + C.DISRUPTION_KINDS
INF = np.inf


# --------------------------------------------------------------------------------------
# Headways and path windows
# --------------------------------------------------------------------------------------
def edge_headway(sim: "NYCSubwaySim", e: np.ndarray, fallback: np.ndarray | float) -> np.ndarray:
    """Scheduled headway of the service on edges ``e`` (``fallback`` where none runs)."""
    h = sim._P.edge_h[np.asarray(e, dtype=np.intp)]
    h = np.where(np.isfinite(h), h, fallback)
    return np.clip(h, C.SCHED_H_MIN_S, C.SCHED_H_MAX_S)


def pattern_headway(sim: "NYCSubwaySim", idx: np.ndarray) -> np.ndarray:
    """Scheduled headway of each train's own pattern (``SCHED_H_FALLBACK_S`` if none)."""
    pat = sim.tr_pattern[idx]
    ph = np.where(pat >= 0, sim._P.pat_h[np.maximum(pat, 0)], INF)
    return np.where(np.isfinite(ph), ph, C.SCHED_H_FALLBACK_S)


def train_geometry(sim: "NYCSubwaySim", idx: np.ndarray) -> dict[str, np.ndarray]:
    """Position facts for trains standing at a node (decision trains)."""
    S = sim._S
    at_start = sim.tr_at_start[idx]
    ptr = sim.tr_ptr[idx]
    nxt = ptr + (~at_start).astype(np.int64)
    PC = sim._PC
    e_cur = sim.p_edge[idx, ptr].astype(np.intp)
    e_next = sim.p_edge[idx, np.minimum(nxt, PC - 1)].astype(np.intp)
    e_next = np.where(nxt < sim.tr_plen[idx], e_next, S.SE)
    node = np.where(at_start, S.efrom[e_cur], S.eto[e_cur])
    cv = np.where(nxt >= 1, sim.p_visit[idx, np.maximum(nxt - 1, 0)], 0).astype(np.int64)
    return {"at_start": at_start, "ptr": ptr, "nxt": nxt, "e_cur": e_cur, "e_next": e_next, "node": node,
            "station": S.nst[node], "cv": cv}


def _nominal_times(sim: "NYCSubwaySim", idx: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nominal (timetable) departure and arrival offsets of path positions ``cols``.

    ``t_dep`` is when the train nominally leaves the from-node of the edge at that position
    (the previous arrival plus the nominal dwell if it was a stop), ``t_arr`` when it reaches
    the to-node. Both include acceleration/braking losses and dwells, so differences are
    time-to-reach in the same units as the scheduled headway.
    """
    PC = sim._PC
    rows = idx[:, None] if cols.ndim == 2 else idx
    cc = np.clip(cols, 0, PC - 1)
    t_arr = sim.p_sched[rows, cc].astype(np.float64)
    pc = np.clip(cols - 1, 0, PC - 1)
    t_prev = sim.p_sched[rows, pc].astype(np.float64) + C.DWELL_SCHED_S * sim.p_stop[rows, pc]
    t_dep = np.where(cols >= 1, t_prev, 0.0)
    return t_dep, np.maximum(t_arr, t_dep)


def _time_at(t_dep: np.ndarray, t_arr: np.ndarray, off: np.ndarray, elen: np.ndarray, vme: np.ndarray) -> np.ndarray:
    """Nominal time of the point at offset ``off`` along an edge (linear in distance).

    Offsets outside the edge (a body straddling into the previous edge, a front already on
    the next one) are converted at the edge speed.
    """
    inside = np.clip(off, 0.0, elen)
    frac = np.where(elen > 0, inside / np.maximum(elen, 1e-9), 0.0)
    return t_dep + frac * (t_arr - t_dep) + (off - inside) / vme


def _standing_s(sim: "NYCSubwaySim", ids: np.ndarray) -> np.ndarray:
    """Time trains ``ids`` have stood at their platform (capped); 0 if running/in a tunnel."""
    from sim.engine import M_DWELL, M_READY

    ok = ids >= 0
    i = np.maximum(ids, 0)
    mode = sim.tr_mode[i]
    standing = ok & ((mode == M_DWELL) | (mode == M_READY)) & sim.tr_at_pl[i]
    return np.where(standing, np.clip(sim.clock_s - sim.tr_arr_time[i], 0.0, C.GAP_DWELL_CAP_S), 0.0)


def path_window(sim: "NYCSubwaySim", idx: np.ndarray, geo: dict[str, np.ndarray], width: int = W_AHEAD) -> dict[str, Any]:
    """The next ``width`` path edges of each train, with lengths and nominal times."""
    S = sim._S
    cols = geo["nxt"][:, None] + np.arange(width)[None, :]
    valid = cols < sim.tr_plen[idx][:, None]
    cc = np.minimum(cols, sim._PC - 1)
    rows = idx[:, None]
    E = np.where(valid, sim.p_edge[rows, cc], S.SE).astype(np.intp)
    elen = S.elen[E]
    t_dep, t_arr = _nominal_times(sim, idx, cols)
    return {
        "cols": cols,
        "valid": valid,
        "E": E,
        "elen": elen,
        "starts": np.cumsum(elen, axis=1) - elen,
        "vme": np.maximum(sim.e_vmax[E], 1.0),
        "t_dep": t_dep,
        "t_arr": t_arr,
        "vis": sim.p_visit[rows, cc].astype(np.int64),
    }


# --------------------------------------------------------------------------------------
# Gaps
# --------------------------------------------------------------------------------------
def leader_gaps(sim: "NYCSubwaySim", idx: np.ndarray, geo: dict[str, np.ndarray],
                win: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Headway to the train ahead along the path (see the module docstring).

    Returns ``gap`` (s, inf if no train within the window), ``dist`` (m, front to the
    leader's rear), ``edge`` (the leader's edge, sentinel if none) and ``id`` (slot, -1).
    """
    S = sim._S
    win = win if win is not None else path_window(sim, idx, geo)
    n = idx.size
    E, elen, vme = win["E"], win["elen"], win["vme"]
    rm = sim._rear_min[E]
    rid = sim._rear_id[E]
    at_start = geo["at_start"]
    rm[:, 0] = np.where(at_start, sim._lead_rear[idx], rm[:, 0])
    rid[:, 0] = np.where(at_start, sim._lead_id[idx], rid[:, 0])
    found = np.isfinite(rm) & win["valid"]
    has = found.any(axis=1)
    k = np.argmax(found, axis=1)
    ar = np.arange(n)
    lid = np.where(has, rid[ar, k], -1)
    r = np.where(has, rm[ar, k], 0.0)
    # the leader's front may already be past the end of the edge holding its rear: carry it
    # forward through the window so the nominal dwell at the stop in between is counted
    kf = k.copy()
    front = r + sim.tr_len[np.maximum(lid, 0)]
    width = E.shape[1]
    for _ in range(4):
        over = has & (front > elen[ar, kf]) & (kf + 1 < width) & win["valid"][ar, np.minimum(kf + 1, width - 1)]
        if not over.any():
            break
        front = np.where(over, front - elen[ar, kf], front)
        kf = np.where(over, kf + 1, kf)
    t_front = _time_at(win["t_dep"][ar, kf], win["t_arr"][ar, kf], front, elen[ar, kf], vme[ar, kf])
    gap = t_front - win["t_dep"][:, 0] + _standing_s(sim, lid)
    return {
        "gap": np.where(has, np.maximum(gap, 0.0), INF),
        "dist": np.where(has, np.maximum(win["starts"][ar, k] + r, 0.0), INF),
        "edge": np.where(has, E[ar, k], S.SE),
        "id": lid,
    }


def follower_gaps(sim: "NYCSubwaySim", idx: np.ndarray, geo: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Headway to the train behind, measured along this train's own path behind it.

    Returns ``gap`` (s, inf if none within ``W_BACK`` edges) and ``edge`` (the follower's).
    """
    S = sim._S
    ptr = geo["ptr"]
    jj = ptr[:, None] - np.arange(W_BACK)[None, :]
    vb = jj >= 0
    B = np.where(vb, sim.p_edge[idx[:, None], np.maximum(jj, 0)], S.SE).astype(np.intp)
    blen = S.elen[B]
    vmb = np.maximum(sim.e_vmax[B], 1.0)
    fm = sim._front_max[B]
    fid = sim._front_id[B]
    fm[:, 0] = sim._foll_front[idx]
    fid[:, 0] = sim._foll_id[idx]
    fm = np.where(vb, fm, -INF)
    t_dep, t_arr = _nominal_times(sim, idx, jj)
    t_self, _ = _nominal_times(sim, idx, geo["nxt"])  # this train's nominal departure
    found = np.isfinite(fm)
    has = found.any(axis=1)
    k = np.argmax(found, axis=1)
    ar = np.arange(idx.size)
    f_off = np.where(has, fm[ar, k], 0.0)
    f_id = np.where(has, fid[ar, k], -1)
    t_front = _time_at(t_dep[ar, k], t_arr[ar, k], f_off, blen[ar, k], vmb[ar, k]) + _standing_s(sim, f_id)
    return {
        "gap": np.where(has, np.maximum(t_self - t_front, 0.0), INF),
        "edge": np.where(has, B[ar, k], S.SE),
        "id": f_id,
    }


def planned_ref(sim: "NYCSubwaySim", gap: np.ndarray, h: np.ndarray, ahead: np.ndarray, behind: np.ndarray) -> np.ndarray:
    """Reference headway ``min(h, planned gap)`` for a measured gap between two trains.

    ``planned = gap - (late[behind] - late[ahead])``: the timetable's spacing of the pair at
    this location (their lateness difference is exactly the drift from plan). Where either
    train is unknown (no train, or a red signal standing in for the leader) the track
    headway ``h`` is returned unchanged.
    """
    ok = (ahead >= 0) & (behind >= 0) & np.isfinite(gap)
    la = sim.tr_late[np.maximum(ahead, 0)]
    lb = sim.tr_late[np.maximum(behind, 0)]
    planned = np.maximum(np.where(ok, gap - (lb - la), INF), C.GAP_PLAN_MIN_S)
    return np.where(ok, np.minimum(h, planned), h)


# --------------------------------------------------------------------------------------
# Blocked-ahead horizon
# --------------------------------------------------------------------------------------
def horizon(sim: "NYCSubwaySim", idx: np.ndarray, geo: dict[str, np.ndarray],
            win: dict[str, Any] | None = None) -> dict[str, Any]:
    """Blocked-ahead evaluation over the next ``K_AHEAD`` stations.

    Also returns the nominal time-to-reach of the first disruption obstacle (a blocked
    edge's entry signal or a closed station), ``obst_gap``, and its edge, ``obst_edge``.
    """
    S = sim._S
    win = win if win is not None else path_window(sim, idx, geo)
    E = win["E"]
    inh = win["valid"] & (win["vis"] <= geo["cv"][:, None] + C.K_AHEAD)
    to_n = S.eto[E]
    dis_e = sim.e_blocked[E] & inh
    dis_n = sim.n_closed[to_n] & inh
    dis = dis_e | dis_n
    stopped = sim._e_stopped[E] * inh
    n_stopped = stopped.sum(axis=1)
    any_dis = dis.any(axis=1)
    blocked = any_dis | (n_stopped >= C.BLOCKED_STOPPED_TRAINS)
    n = idx.size
    ar = np.arange(n)
    k = np.argmax(dis, axis=1)
    t_obst = np.where(dis_e[ar, k], win["t_dep"][ar, k], win["t_arr"][ar, k]) - win["t_dep"][:, 0]
    obst_gap = np.where(any_dis, np.maximum(t_obst, 0.0), INF)
    obst_edge = np.where(any_dis, E[ar, k], S.SE)
    detail: list[dict[str, Any] | None] = [None] * n
    for i in np.flatnonzero(blocked).tolist():
        if any_dis[i]:
            kk = int(k[i])
            e = int(E[i, kk])
            if dis_e[i, kk]:
                kind = _KIND_NAMES[int(sim.e_bkind[e])]
                st = int(S.nst[S.efrom[e]])
            else:
                kind = _KIND_NAMES[int(sim.n_ckind[to_n[i, kk]])]
                st = int(S.nst[to_n[i, kk]])
            detail[i] = {"kind": kind, "station_idx": st}
        else:
            kk = int(np.argmax(stopped[i] > 0))
            detail[i] = {"kind": "congestion", "station_idx": int(S.nst[to_n[i, kk]])}
    return {"blocked": blocked, "detail": detail, "n_stopped": n_stopped, "obst_gap": obst_gap, "obst_edge": obst_edge}


# --------------------------------------------------------------------------------------
# Structural alternates
# --------------------------------------------------------------------------------------
def _detour_open(sim: "NYCSubwaySim", det: Any) -> bool:
    de, dn, _, _ = sim._S.detour_arrays(det)
    if sim.e_blocked[de].any() or sim.n_closed[dn].any() or sim.e_xoff[de].any():
        return False
    return bool(sim.e_peak_ok[de].all())


def _orig_affected(sim: "NYCSubwaySim", det: Any) -> bool:
    _, _, oe, on = sim._S.detour_arrays(det)
    if sim.e_blocked[oe].any() or sim.n_closed[on].any():
        return True
    return int(sim._e_stopped[oe].sum()) >= C.BLOCKED_STOPPED_TRAINS


def _dis_cum_matrix(sim: "NYCSubwaySim") -> np.ndarray:
    """Prefix counts ``[n_patterns, P + 1]`` of disrupted edges/nodes along every template.

    ``cum[q, b] - cum[q, a] > 0`` iff a blocked edge or closed node lies on template edges
    ``a..b-1`` of pattern ``q``. Cached per disruption epoch.
    """
    cache = sim.__dict__.setdefault("_tmpl_dis_cache", {})
    hit = cache.get(sim._mask_epoch)
    if hit is None:
        cache.clear()
        S = sim._S
        g = sim.g
        te = g.tmpl_edge.astype(np.intp)
        valid = np.arange(te.shape[1])[None, :] < g.pat_len[:, None]
        te = np.where(valid, te, S.SE)
        flags = (sim.e_blocked[te] | sim.n_closed[S.eto[te]]) & valid
        hit = np.zeros((te.shape[0], te.shape[1] + 1), dtype=np.int32)
        np.cumsum(flags, axis=1, out=hit[:, 1:])
        cache[sim._mask_epoch] = hit
    return hit


def lead_etas(sim: "NYCSubwaySim") -> np.ndarray:
    """Expected time (s) until the next train reaches each yard lead node (``g.yard_leads`` order).

    "The next train" is any in-service train whose template uses the edge leaving the lead
    node along the lead's pattern (every pattern sharing that track, any route), matching the
    combined headway the starvation threshold is measured against; a lead on a low-frequency
    variant pattern is not "starved" while the trunk it joins is served. Per (pattern ``q``,
    position ``j``) candidates are the nearest on-template train of ``q`` upstream of the node
    (nominal remaining time; infinite when a disruption lies in between, which then also holds
    for every train further upstream) and the pattern's next scheduled departure from its
    origin. Leads out of service are ``inf``. Vectorized; cached per tick and disruption epoch.
    """
    key = (sim._tick_abs, sim._mask_epoch)
    cache = sim.__dict__.setdefault("_lead_eta_cache", {})
    hit = cache.get(key)
    if hit is not None:
        return hit
    cache.clear()
    g, P = sim.g, sim._P
    out = np.full(len(g.yard_leads), INF)
    q, j = P.lead_pair_q, P.lead_pair_j
    if q.size:
        cum = _dis_cum_matrix(sim)
        tmpl = g.tmpl_sched
        t_lead = np.where(j >= 1, tmpl[q, np.maximum(j - 1, 0)], 0.0).astype(np.float64)
        # nearest upstream train per pair: sort on-template trains by (pattern, eff, at_start)
        # where eff = tp (+1 if past the from-node); "upstream of j" <=> 2*eff + at_start <= 2j + 1
        on = np.flatnonzero(sim.tr_active & (sim.tr_pattern >= 0) & ~sim.tr_deviated)
        tp = sim.p_tpos[on, sim.tr_ptr[on]].astype(np.int64)
        keep = tp >= 0
        on, tp = on[keep], tp[keep]
        at = sim.tr_at_start[on].astype(np.int64)
        code = 2 * (tp + 1 - at) + at
        m2 = 2 * (tmpl.shape[1] + 2)
        keys = np.sort(sim.tr_pattern[on].astype(np.int64) * m2 + code)
        pos = np.searchsorted(keys, q * m2 + 2 * j + 1, side="right") - 1
        kk = keys[np.maximum(pos, 0)] if keys.size else np.zeros(q.size, dtype=np.int64)
        found = (pos >= 0) & (kk // m2 == q)
        c = kk % m2
        tpn = np.where(c % 2 == 1, c // 2, c // 2 - 1)
        t_here = np.where(tpn >= 1, tmpl[q, np.clip(tpn - 1, 0, tmpl.shape[1] - 1)], 0.0)
        clear = cum[q, j] - cum[q, np.clip(tpn, 0, j)] == 0
        eta = np.where(found & clear, np.maximum(t_lead - t_here, 0.0), INF)
        # next scheduled departure of each pattern from its origin
        nd = np.full(g.n_patterns, INF)
        if P.n_svc:
            np.minimum.at(nd, P.svc_pattern, sim.svc_next)
        sched = np.where(cum[q, j] - cum[q, 0] == 0, np.maximum(nd[q] - sim.clock_s, 0.0) + t_lead, INF)
        out[P.lead_svc] = np.minimum.reduceat(np.minimum(eta, sched), P.lead_pair_ptr[:-1])
    cache[key] = out
    return out


def lead_eta(sim: "NYCSubwaySim", lead: Any) -> float:
    """:func:`lead_etas` for one :class:`~sim.graph.YardLead`."""
    return float(lead_etas(sim)[int(lead.lead_id)])


def yard_choice(sim: "NYCSubwaySim", routes: np.ndarray) -> np.ndarray:
    """Per route: the starved, dispatchable yard lead with the longest wait (-1 if none).

    A lead qualifies when its yard has stock, its cooldown is over, its node is open and the
    expected time until the next train reaches it exceeds ``max(2 sched_h, 480 s)``.
    """
    P = sim._P
    eta = lead_etas(sim)
    ok = ((sim.yard_stock[P.lead_yard] > 0) & (sim.clock_s >= sim.yard_cool[P.lead_yard])
          & ~sim.n_closed[P.lead_node] & (eta > P.lead_thr))
    R = P.route_lead_mat[np.asarray(routes, dtype=np.intp)]
    valid = (R >= 0) & ok[np.maximum(R, 0)]
    score = np.where(valid, eta[np.maximum(R, 0)], -1.0)
    best = np.argmax(score, axis=1)
    pick = R[np.arange(R.shape[0]), best]
    return np.where(valid.any(axis=1), pick, -1)


def platform_wait_stats(sim: "NYCSubwaySim", pl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mean wait min, oldest non-empty bin center min) of platforms ``pl`` only.

    Same definitions as ``NYCSubwaySim._platform_stats`` (which covers every platform), for
    the few platforms a decision batch looks at.
    """
    w = sim.waiting[pl]
    age_c = sim._S.age_c
    tot = w.sum(axis=1)
    mean = (w @ age_c) / np.maximum(tot, 1e-9)
    nz = w > 0.5
    last = C.N_AGE_BINS - 1 - np.argmax(nz[:, ::-1], axis=1)
    oldest = np.where(nz.any(axis=1), age_c[last], 0.0)
    return mean, oldest


# --------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------
def compute_masks(sim: "NYCSubwaySim", idx: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Masks ``bool[n, N_ACTIONS]`` and per-train facts for trains ready at a stop."""
    S = sim._S
    g = sim.g
    P = sim._P
    idx = np.asarray(idx, dtype=np.int64)
    n = idx.size
    masks = np.zeros((n, C.N_ACTIONS), dtype=bool)
    masks[:, 0] = True
    geo = train_geometry(sim, idx)
    node = geo["node"]
    station = geo["station"]
    e_next = geo["e_next"]
    nxt = geo["nxt"]
    pat_h = pattern_headway(sim, idx)
    sched_h = edge_headway(sim, e_next, pat_h)
    win = path_window(sim, idx, geo)
    lead = leader_gaps(sim, idx, geo, win)
    hz = horizon(sim, idx, geo, win)
    # the effective leader is the nearest of the train ahead and a red signal ahead
    obst_first = hz["obst_gap"] < lead["gap"]
    lead_gap = np.where(obst_first, hz["obst_gap"], lead["gap"])
    lead_h = edge_headway(sim, np.where(obst_first, hz["obst_edge"], lead["edge"]), sched_h)
    foll = follower_gaps(sim, idx, geo)
    foll_gap = foll["gap"]
    foll_h = edge_headway(sim, foll["edge"], sched_h)
    # timetable-relative references for the "short gap" clauses (module docstring)
    lead_ref = planned_ref(sim, lead_gap, lead_h, np.where(obst_first, -1, lead["id"]), idx)
    foll_ref = planned_ref(sim, foll_gap, foll_h, idx, foll["id"])
    blocked = hz["blocked"]
    late = sim.tr_late[idx]
    has_pl = S.npl[node] >= 0
    has_next_edge = nxt < sim.tr_plen[idx]

    # 1 HOLD_IN_STATION
    masks[:, C.A_HOLD] = (
        has_pl
        & has_next_edge
        & (sim.tr_hold_total[idx] < C.MAX_HOLD_S - 1e-6)
        & (lead_gap < C.HOLD_GAP_WIDE_MULT * lead_h)
        & ((lead_gap < C.HOLD_GAP_BUNCHED_MULT * lead_ref) | blocked)
    )

    # 4 SKIP_STOP
    sptr = sim.tr_sptr[idx]
    ns = sim.tr_nstops[idx]
    js = np.minimum(sptr, sim._SC - 1)
    nst = sim.s_station[idx, js].astype(np.intp)
    nspos = sim.s_pos[idx, js].astype(np.intp)
    nsnode = S.eto[sim.p_edge[idx, np.maximum(nspos, 0)].astype(np.intp)]
    nspl = S.npl[nsnode]
    mean_w, oldest = platform_wait_stats(sim, np.maximum(nspl, 0))
    crit = (nspl >= 0) & ((mean_w >= C.CRITICAL_MEAN_WAIT_MIN) | (oldest >= C.CRITICAL_OLDEST_WAIT_MIN))
    closed_next = sim.n_closed[nsnode]
    masks[:, C.A_SKIP] = (
        (sptr < ns - 1)
        & (nst >= 0)
        & ~S.st_term[np.maximum(nst, 0)]
        & ~S.st_hub[np.maximum(nst, 0)]
        & ~sim.tr_skipped_prev[idx]
        & (~crit | closed_next)
        & ((late > C.SKIP_LATENESS_S) | (foll_gap < C.SKIP_FOLLOWER_MULT * foll_ref))
        & has_next_edge
    )

    # template position of the next edge (structural actions need an on-template train)
    pat = sim.tr_pattern[idx].astype(np.int64)
    tpos = sim.p_tpos[idx, np.minimum(nxt, sim._PC - 1)].astype(np.int64)
    flat = np.where((tpos >= 0) & (pat >= 0) & has_next_edge, g.pat_edge_ptr[np.maximum(pat, 0)] + tpos, -1)
    fl0 = np.maximum(flat, 0)
    on_tmpl = flat >= 0

    det_x: list[Any] = [None] * n
    det_l: list[Any] = [None] * n
    det_r: list[Any] = [None] * n
    st_opt: list[Any] = [None] * n
    yard_pick: list[Any] = [None] * n
    opp_gap = np.full(n, INF)
    opp_h = sched_h.copy()

    # 2 RUN_EXPRESS / 3 RUN_LOCAL
    cx = on_tmpl & g.pp_has_express[fl0] & (S.ntrack[node] == 0)
    for i in np.flatnonzero(cx).tolist():
        for det in g.express_detours.get(int(flat[i]), ()):
            if _orig_affected(sim, det) and _detour_open(sim, det):
                det_x[i] = det
                masks[i, C.A_EXPRESS] = True
                break
    cl = on_tmpl & g.pp_has_local[fl0] & (S.ntrack[node] == 1)
    for i in np.flatnonzero(cl).tolist():
        for det in g.local_detours.get(int(flat[i]), ()):
            if _orig_affected(sim, det) and _detour_open(sim, det):
                det_l[i] = det
                masks[i, C.A_LOCAL] = True
                break

    # 6 REROUTE_VIA_COMPATIBLE_TRUNK
    cr = on_tmpl & g.pp_has_reroute[fl0] & S.ndiverge[node]
    for i in np.flatnonzero(cr).tolist():
        for det in g.reroute_detours.get(int(flat[i]), ()):
            if _orig_affected(sim, det) and _detour_open(sim, det):
                det_r[i] = det
                masks[i, C.A_REROUTE] = True
                break

    # 5 SHORT_TURN_RELAY
    cs = on_tmpl & g.pp_has_short_turn[fl0] & has_pl
    cs &= (station != g.pat_dest_station[np.maximum(pat, 0)]) & (station != g.pat_origin_station[np.maximum(pat, 0)])
    cs &= ~sim._st_pocket_busy[np.maximum(station, 0)]
    for i in np.flatnonzero(cs).tolist():
        opt = g.short_turn_options[int(flat[i])]
        tb = int(opt.turnback_edge)
        n_to = int(S.eto[tb])
        if sim.e_xoff[tb] or sim.n_closed[n_to] or sim._e_timed_occ[tb]:
            continue
        # "a disruption blocks the path": the train would reach it within
        # SHORT_TURN_BLOCK_HORIZON_S of nominal running (module docstring)
        t = int(idx[i])
        a, b = int(nxt[i]), int(sim.tr_plen[t])
        rem = sim.p_edge[t, a:b].astype(np.intp)
        hit = sim.e_blocked[rem] | sim.n_closed[S.eto[rem]]
        path_blocked = False
        if hit.any():
            k = a + int(np.argmax(hit))
            col = np.array([k])
            t_dep, t_arr = _nominal_times(sim, np.array([t]), col)
            t_here, _ = _nominal_times(sim, np.array([t]), np.array([a]))
            reach = (t_dep[0] if sim.e_blocked[int(sim.p_edge[t, k])] else t_arr[0]) - t_here[0]
            path_blocked = reach <= C.SHORT_TURN_BLOCK_HORIZON_S
        og = sim.clock_s - float(sim.nd_last_dep[n_to])
        oh = float(edge_headway(sim, np.array([int(g.tmpl_edge[opt.pattern, opt.position])]), sched_h[i])[0])
        opp_gap[i] = og
        opp_h[i] = oh
        imbalance = (og > C.SHORT_TURN_OPPOSITE_GAP_MULT * oh) and (foll_gap[i] < C.SHORT_TURN_FOLLOWER_MULT * foll_ref[i])
        if path_blocked or imbalance:
            st_opt[i] = opt
            masks[i, C.A_SHORT_TURN] = True

    # 7 DISPATCH_YARD_GAP_TRAIN
    routes = sim.tr_route[idx].astype(np.int64)
    pick = yard_choice(sim, np.maximum(routes, 0))
    pick = np.where(routes >= 0, pick, -1)
    masks[:, C.A_YARD] = pick >= 0
    for i in np.flatnonzero(pick >= 0).tolist():
        yard_pick[i] = g.yard_leads[int(pick[i])]

    # trains whose path ends here (terminal) never decide
    masks[~has_next_edge, 1:] = False
    info: dict[str, Any] = {
        "node": node,
        "station": station,
        "e_next": e_next,
        "cv": geo["cv"],
        "sched_h": sched_h,
        "lead_gap": lead_gap,
        "lead_h": lead_h,
        "lead_dist": lead["dist"],
        "foll_gap": foll_gap,
        "foll_h": foll_h,
        "lead_ref": lead_ref,
        "foll_ref": foll_ref,
        "blocked": blocked,
        "block_detail": hz["detail"],
        "late": late,
        "opp_gap": opp_gap,
        "opp_h": opp_h,
        "det_express": det_x,
        "det_local": det_l,
        "det_reroute": det_r,
        "short_turn": st_opt,
        "yard": yard_pick,
    }
    return masks, info


def subset_info(info: dict[str, Any], sel: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sl = sel.tolist()
    for k, v in info.items():
        if isinstance(v, np.ndarray):
            out[k] = v[sel]
        else:
            out[k] = [v[i] for i in sl]
    return out


def disruption_kind_name(code: int) -> str:
    return _KIND_NAMES[int(code)]


__all__ = [
    "compute_masks",
    "subset_info",
    "lead_eta",
    "lead_etas",
    "yard_choice",
    "train_geometry",
    "path_window",
    "leader_gaps",
    "follower_gaps",
    "horizon",
    "edge_headway",
    "pattern_headway",
]
