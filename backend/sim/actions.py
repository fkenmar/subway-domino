"""Action effects (ARCHITECTURE §4.2 "effect" column) and their immediate PRD §4.3 costs.

:func:`apply` re-validates the requested action against the train's *current* mask (the
batch mask is reused unless a structural action changed the state since the snapshot),
applies it, charges the immediate cost and returns the :class:`ActionEvent`.

Onboard riders are bucketed by the k-th next stop of the train's actual path, so every
path rewrite remaps buckets by destination station: riders whose stop is still served keep
it; riders whose stop is bypassed overshoot to the next served stop and are counted as
displaced (skip / reroute displacement cost).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from sim import constants as C
from sim.graph import nominal_schedule
from sim.paths import compile_path

if TYPE_CHECKING:  # pragma: no cover
    from sim.engine import ActionEvent, NYCSubwaySim


def _finite(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def _event(sim: "NYCSubwaySim", t: int, a: int, station: int, applied: bool, detail: dict[str, Any]) -> "ActionEvent":
    from sim.engine import ActionEvent

    return ActionEvent(
        t=float(sim.clock_s),
        train_id=int(t),
        route_idx=int(sim.tr_route[t]),
        action=int(a),
        station_idx=int(station),
        applied=bool(applied),
        detail=detail,
    )


def _current(sim: "NYCSubwaySim", t: int, row: int | None) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Mask row and facts for train ``t``: the batch snapshot if still current, else recomputed."""
    from sim.masks import compute_masks, subset_info

    if row is None or not sim.tr_active[t]:
        return None, None
    if sim._pend_epoch == sim._mask_epoch:
        info = subset_info(sim._pend_info, np.array([row]))
        return sim._pend_masks[row], info
    m, info = compute_masks(sim, np.array([t], dtype=np.int64))
    return m[0], info


def apply(sim: "NYCSubwaySim", t: int, a: int, row: int | None) -> "ActionEvent":
    mask, info = _current(sim, t, row)
    if mask is None or info is None:
        st = int(sim._S.nst[sim._node_of(np.array([t]))[0]]) if sim.tr_active[t] else -1
        return _event(sim, t, a, st, False, {"reason": "no pending decision"})
    station = int(info["station"][0])
    detail: dict[str, Any] = {
        "blocked_ahead": info["block_detail"][0],
        "riders_onboard": float(sim.tr_onboard[t].sum()),
        "leader_gap_s": _finite(info["lead_gap"][0]),
        "sched_headway_s": float(info["sched_h"][0]),
    }
    if not mask[a]:
        detail["reason"] = "action no longer valid"
        return _event(sim, t, a, station, False, detail)
    if a == C.A_HOLD:
        _hold(sim, t)
    elif a == C.A_SKIP:
        detail.update(_skip(sim, t))
    elif a == C.A_EXPRESS:
        detail.update(_detour(sim, t, info["det_express"][0], "express"))
    elif a == C.A_LOCAL:
        detail.update(_detour(sim, t, info["det_local"][0], "local"))
    elif a == C.A_REROUTE:
        detail.update(_detour(sim, t, info["det_reroute"][0], "reroute"))
    elif a == C.A_SHORT_TURN:
        detail.update(_short_turn(sim, t, info["short_turn"][0], int(info["node"][0])))
    elif a == C.A_YARD:
        detail.update(_yard(sim, info["yard"][0]))
    sim.counts[a] += 1
    sim._pl_stats = None
    return _event(sim, t, a, station, True, detail)


# --------------------------------------------------------------------------------------
# HOLD_IN_STATION
# --------------------------------------------------------------------------------------
def _hold(sim: "NYCSubwaySim", t: int) -> None:
    from sim.engine import M_DWELL

    sim.tr_mode[t] = M_DWELL
    sim.tr_timer[t] = C.HOLD_STEP_S
    sim.tr_held[t] = True
    sim.tr_hold_total[t] += C.HOLD_STEP_S
    sim.tr_next_dec[t] = np.inf


# --------------------------------------------------------------------------------------
# Path rewrites
# --------------------------------------------------------------------------------------
def _next_stops(sim: "NYCSubwaySim", t: int, count: int = C.N_DEST - 1) -> list[int]:
    a = int(sim.tr_sptr[t])
    b = min(int(sim.tr_nstops[t]), a + count)
    return [int(s) for s in sim.s_station[t, a:b].tolist()]


def remap_buckets(ob: np.ndarray, old: Sequence[int], new: Sequence[int]) -> tuple[np.ndarray, float]:
    """Remap destination buckets from the old next-stop list to the new one.

    Returns the new buckets and the riders whose destination is no longer served
    (they overshoot to the next served stop of their old sequence).
    """
    nd = C.N_DEST
    out = np.zeros(nd, dtype=np.float64)
    near = {s: i for i, s in reversed(list(enumerate(list(new)[: nd - 1])))}
    far = set(list(new)[nd - 1 :])
    displaced = 0.0
    for k in range(nd - 1):
        b = float(ob[k])
        if b <= 0.0:
            continue
        if k >= len(old):
            out[nd - 1] += b
            continue
        s = old[k]
        if s in near:
            out[near[s]] += b
        elif s in far:
            out[nd - 1] += b
        else:
            displaced += b
            dest = None
            for s2 in old[k + 1 :]:
                if s2 in near:
                    dest = near[s2]
                    break
                if s2 in far:
                    dest = nd - 1
                    break
            if dest is None:
                dest = 0 if len(new) else nd - 1
            out[dest] += b
    out[nd - 1] += float(ob[nd - 1])
    return out, displaced


def _rewrite(
    sim: "NYCSubwaySim",
    t: int,
    edges: Sequence[int],
    stops: Sequence[bool],
    tpos: Sequence[int],
    sched: Sequence[float],
    keep_ptr: int,
) -> None:
    """Replace train ``t``'s path; positions ``< keep_ptr`` are unchanged (past + current)."""
    P = sim._PC
    trim = max(0, len(edges) - P)
    if trim:
        trim = min(trim, max(keep_ptr - 1, 0))
        edges, stops, tpos, sched = edges[trim:], stops[trim:], tpos[trim:], sched[trim:]
    cp = compile_path(sim.g, list(edges), list(stops), tpos=list(tpos), sched=list(sched))
    sim._write_path(t, cp)
    sim.tr_ptr[t] = int(sim.tr_ptr[t]) - trim
    pos_now = int(sim.tr_ptr[t]) - (1 if sim.tr_at_start[t] else 0)
    s_pos = cp.s_pos.astype(np.int64)
    sim.tr_sptr[t] = int(np.searchsorted(s_pos, pos_now, side="right"))


def _skip(sim: "NYCSubwaySim", t: int) -> dict[str, Any]:
    j = int(sim.tr_sptr[t])
    pos = int(sim.s_pos[t, j])
    skipped = int(sim.s_station[t, j])
    L = int(sim.tr_plen[t])
    old_next = _next_stops(sim, t)
    edges = sim.p_edge[t, :L].astype(np.int64).tolist()
    stops = sim.p_stop[t, :L].copy()
    stops[pos] = False
    tpos = sim.p_tpos[t, :L].tolist()
    sched = sim.p_sched[t, :L].tolist()
    _rewrite(sim, t, edges, stops.tolist(), tpos, sched, keep_ptr=int(sim.tr_ptr[t]))
    new_next = _next_stops(sim, t)
    ob, displaced = remap_buckets(sim.tr_onboard[t], old_next, new_next)
    sim.tr_onboard[t] = ob
    sim.tr_skip_next[t] = True
    sim.charge("skip", C.W_SKIP * displaced)
    return {"skipped_station_idx": skipped, "riders_overshot": displaced}


def _detour(sim: "NYCSubwaySim", t: int, det: Any, kind: str) -> dict[str, Any]:
    g = sim.g
    p = int(sim.tr_pattern[t])
    at_start = bool(sim.tr_at_start[t])
    nxt = int(sim.tr_ptr[t]) + (0 if at_start else 1)
    L = int(sim.tr_plen[t])
    old_next = _next_stops(sim, t)
    pre_e = sim.p_edge[t, :nxt].astype(np.int64).tolist()
    pre_s = sim.p_stop[t, :nxt].tolist()
    pre_tp = sim.p_tpos[t, :nxt].tolist()
    pre_sc = sim.p_sched[t, :nxt].astype(np.float64).tolist()
    tp_next = int(sim.p_tpos[t, nxt])
    delta = float(sim.p_sched[t, nxt]) - float(g.tmpl_sched[p, tp_next])
    base = 0.0
    if nxt >= 1:
        base = float(sim.p_sched[t, nxt - 1]) + (C.DWELL_SCHED_S if bool(sim.p_stop[t, nxt - 1]) else 0.0)
    d_edges = list(det.edges)
    d_stops = list(det.stops)
    d_sc = (nominal_schedule(g, d_edges, d_stops).astype(np.float64) + base).tolist()
    edges = pre_e + d_edges
    stops = pre_s + d_stops
    tpos = pre_tp + [-1] * len(d_edges)
    sched = pre_sc + d_sc
    if det.rejoin_pos >= 0:
        r = int(det.rejoin_pos)
        Lp = int(g.pat_len[p])
        edges += g.tmpl_edge[p, r:Lp].astype(np.int64).tolist()
        stops += g.tmpl_stop[p, r:Lp].tolist()
        tpos += list(range(r, Lp))
        sched += (g.tmpl_sched[p, r:Lp].astype(np.float64) + delta).tolist()
    del L
    _rewrite(sim, t, edges, stops, tpos, sched, keep_ptr=nxt)
    new_next = _next_stops(sim, t)
    ob, displaced = remap_buckets(sim.tr_onboard[t], old_next, new_next)
    sim.tr_onboard[t] = ob
    sim.tr_deviated[t] = True
    if kind == "reroute":
        sim.tr_rerouted[t] = True
        sim.charge("reroute", C.W_REROUTE + C.W_REROUTE_RIDER * displaced)
    else:
        sim.tr_switch[t] = True
        sim.charge("skip", C.W_SKIP * displaced)
    return {
        "via_station_idx": int(det.via_station),
        "rejoin_station_idx": int(det.rejoin_station),
        "bypassed_station_idx": [int(s) for s in det.bypassed_stations],
        "riders_displaced": displaced,
    }


# --------------------------------------------------------------------------------------
# SHORT_TURN_RELAY
# --------------------------------------------------------------------------------------
def _short_turn(sim: "NYCSubwaySim", t: int, opt: Any, node: int) -> dict[str, Any]:
    from sim.engine import M_TIMED, TK_EDGE

    g = sim.g
    S = sim._S
    riders = float(sim.tr_onboard[t].sum())
    pl = int(S.ndirpl[node])
    if pl >= 0 and not sim.pl_closed[pl]:
        sim.waiting[pl, 0] += riders
    else:
        sim.riders_displaced += riders
    sim.tr_onboard[t] = 0.0
    sim.charge("short_turn", C.W_ST * riders)
    latency = float(sim._rng_action.uniform(C.TURNBACK_MIN_S, C.TURNBACK_MAX_S))
    q, j = int(opt.pattern), int(opt.position)
    Lq = int(g.pat_len[q])
    tb = int(opt.turnback_edge)
    tmpl = g.tmpl_sched[q].astype(np.float64)
    base_q = float(tmpl[j - 1]) + (C.DWELL_SCHED_S if (j >= 1 and bool(g.tmpl_stop[q, j - 1])) else 0.0) if j >= 1 else 0.0
    rest_sc = (tmpl[j:Lq] - base_q + latency + C.DWELL_SCHED_S).tolist()
    edges = [tb] + g.tmpl_edge[q, j:Lq].astype(np.int64).tolist()
    stops = [True] + g.tmpl_stop[q, j:Lq].tolist()
    tpos = [-1] + list(range(j, Lq))
    sched = [latency] + rest_sc
    cp = compile_path(g, edges, stops, tpos=tpos, sched=sched)
    sim._write_path(t, cp)
    sim.tr_pattern[t] = q
    sim.tr_ptr[t] = 0
    sim.tr_at_start[t] = False
    sim.tr_off[t] = 0.0
    sim.tr_v[t] = 0.0
    sim.tr_mode[t] = M_TIMED
    sim.tr_tkind[t] = TK_EDGE
    sim.tr_timer[t] = latency
    sim.tr_sptr[t] = 0
    sim.tr_sched_dep[t] = sim.clock_s
    sim.tr_late[t] = 0.0
    sim.tr_hold_total[t] = 0.0
    sim.tr_held[t] = False
    sim.tr_skip_next[t] = False
    sim.tr_skipped_prev[t] = False
    sim.tr_deviated[t] = False
    sim.tr_switch[t] = False
    sim.tr_rerouted[t] = False
    sim.tr_next_dec[t] = np.inf
    sim._e_timed_occ[tb] = True
    sim._st_pocket_busy[int(S.nst[node])] = True
    sim._mask_epoch += 1
    return {"riders_forced_off": riders, "turnback_latency_s": latency}


# --------------------------------------------------------------------------------------
# DISPATCH_YARD_GAP_TRAIN
# --------------------------------------------------------------------------------------
def _yard(sim: "NYCSubwaySim", lead: Any) -> dict[str, Any]:
    g = sim.g
    y = int(lead.yard)
    sim.yard_stock[y] -= 1
    sim.yard_cool[y] = sim.clock_s + C.YARD_COOLDOWN_S
    t_enter = sim.clock_s + float(g.yard_deadhead_s[y])
    sim._gap_queue.append((t_enter, y, int(lead.pattern), int(lead.position), int(lead.node)))
    sim.charge("yard", C.W_YARD)
    sim._mask_epoch += 1
    return {"yard_idx": y, "lead_station_idx": int(lead.station), "deadhead_s": float(g.yard_deadhead_s[y])}
