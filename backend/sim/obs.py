"""Observation builder (ARCHITECTURE §4.3): the only observation code (env and server).

``OBS_DIM = 216 = 10 corridor slots x 18 + 28 ego features + 8 mask bits``. Slots are the
stations of the train's *actual* path: three behind (most recent last), the current
station, and the next six. Every value is clipped to [-1, 1]; invalid slots are zeros.

The layout is GUIDANCE in the contract; this module is its single definition and
:data:`FEATURE_NAMES` documents it (exported for the RL model card and debugging).
Gap features are headway ratios scaled by 2: ``gap / (2 * h)`` with ``h`` the scheduled
headway where the gap is measured (the same reference the masks use), so the HOLD
thresholds at 0.75 and 1.5 headways sit at 0.375 and 0.75, below the clip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from sim import constants as C
from sim.masks import edge_headway

if TYPE_CHECKING:  # pragma: no cover
    from sim.engine import NYCSubwaySim

SLOT_NAMES: tuple[str, ...] = ("behind3", "behind2", "behind1", "current", "ahead1", "ahead2", "ahead3",
                               "ahead4", "ahead5", "ahead6")
SLOT_FEATURES: tuple[str, ...] = (
    "valid",
    "is_stop",
    "has_platform",
    "is_hub",
    "is_terminal",
    "waiting_log",
    "mean_wait_over_15min",
    "crowd_ratio",
    "next_edge_blocked",
    "next_edge_signal_delay",
    "node_closed",
    "stopped_trains_next_edge_over_3",
    "time_since_last_departure_over_2h",
    "has_track_switch",
    "has_pocket_turnback",
    "is_diverge",
    "run_time_from_ego_over_1800s",
    "onboard_alighting_share",
)
EGO_FEATURES: tuple[str, ...] = (
    "load",
    "leader_gap_over_2h",
    "follower_gap_over_2h",
    "lateness_over_600s",
    "hold_total_frac",
    "skipped_prev",
    "on_express_track",
    "cbtc",
    "division_b",
    "profile_am_peak",
    "profile_midday",
    "profile_pm_rush",
    "profile_late_night",
    "route_avg_wait_over_15min",
    "route_tunnel_stuck_over_5",
    "active_disruptions_over_4",
    "disruption_distance_slots_over_6",
    "yard_stock_frac",
    "yard_cooldown_frac",
    "opposite_dir_gap_over_2h",
    "is_gap_train",
    "is_deviated",
    "sched_headway_over_900s",
    "episode_time_frac",
    "demand_scale_minus_1",
    "reserved_0",
    "reserved_1",
    "reserved_2",
)
MASK_FEATURES: tuple[str, ...] = tuple(f"mask_{name}" for name in C.ACTION_NAMES)
FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"{s}.{f}" for s in SLOT_NAMES for f in SLOT_FEATURES) + tuple(f"ego.{f}" for f in EGO_FEATURES) + MASK_FEATURES
)
assert len(SLOT_FEATURES) == C.OBS_SLOT_FEATURES
assert len(EGO_FEATURES) == C.OBS_EGO_FEATURES
assert len(FEATURE_NAMES) == C.OBS_DIM
_LOG_W = float(np.log1p(C.OBS_WAITING_LOG_SCALE))


def edge_headway_at_nodes(sim: "NYCSubwaySim", nodes: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Scheduled headway of the service leaving ``nodes`` (sentinel/unserved: ``fallback``)."""
    return edge_headway(sim, sim._P.node_out_edge[np.asarray(nodes, dtype=np.intp)], fallback)


def _route_stats(sim: "NYCSubwaySim") -> tuple[np.ndarray, np.ndarray]:
    """(rider-weighted mean wait per route [min], tunnel-stuck trains per route)."""
    cache = sim.__dict__.get("_route_stats_cache")
    if cache is not None and cache[0] == (sim.tick, sim.clock_s):
        return cache[1], cache[2]
    S = sim._S
    tot, mean_w, _ = sim._platform_stats()
    wsum = tot * mean_w
    num = S.route_pl @ wsum
    den = S.route_pl @ tot
    avg = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
    stuck = sim._tunnel_stuck_mask()
    r = sim.tr_route[stuck].astype(np.int64)
    cnt = np.bincount(r[r >= 0], minlength=sim.g.n_routes).astype(np.float64)
    sim.__dict__["_route_stats_cache"] = ((sim.tick, sim.clock_s), avg, cnt)
    return avg, cnt


def build_obs(sim: "NYCSubwaySim", idx: np.ndarray, masks: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    """``float32[n, OBS_DIM]`` for decision trains ``idx`` (with their masks and mask facts)."""
    S = sim._S
    g = sim.g
    n = idx.size
    out = np.zeros((n, C.OBS_DIM), dtype=np.float32)
    if n == 0:
        return out
    VC = sim._VC
    rows = idx[:, None]
    cv = info["cv"].astype(np.int64)
    sched_h = info["sched_h"]

    # ---------------------------------------------------------------- corridor slots
    vis = cv[:, None] + np.arange(-C.K_BEHIND, C.K_AHEAD + 1)[None, :]
    valid = (vis >= 0) & (vis < sim.tr_nvis[idx][:, None])
    vc = np.clip(vis, 0, VC - 1)
    vst = sim.v_station[rows, vc].astype(np.intp)
    vnode = sim.v_node[rows, vc].astype(np.intp)
    vnode = np.where(valid, vnode, S.SN)
    vst = np.where(valid & (vst >= 0), vst, -1)
    vout = np.where(valid, sim.v_out[rows, vc].astype(np.intp), S.SE)
    vtime = sim.v_time[rows, vc].astype(np.float64)
    vsidx = sim.v_sidx[rows, vc].astype(np.int64)
    tot, mean_w, _ = sim._platform_stats()
    pl = S.ndirpl[vnode]
    has_pl = pl >= 0
    p0 = np.maximum(pl, 0)
    w = np.where(has_pl, tot[p0], 0.0)
    st0 = np.maximum(vst, 0)
    F = np.zeros((n, C.OBS_SLOTS, C.OBS_SLOT_FEATURES), dtype=np.float64)
    F[..., 0] = valid
    F[..., 1] = sim.v_stop[rows, vc] & valid
    F[..., 2] = (S.npl[vnode] >= 0) & valid
    F[..., 3] = S.st_hub[st0] & (vst >= 0)
    F[..., 4] = S.st_term[st0] & (vst >= 0)
    F[..., 5] = np.log1p(w) / _LOG_W
    F[..., 6] = np.where(has_pl, mean_w[p0], 0.0) / 15.0
    F[..., 7] = np.where(has_pl, w / S.pl_cap[p0], 0.0)
    F[..., 8] = sim.e_blocked[vout]
    F[..., 9] = sim.e_sig[vout]
    F[..., 10] = sim.n_closed[vnode] & valid
    F[..., 11] = sim._e_stopped[vout] / 3.0
    node_h = edge_headway_at_nodes(sim, vnode, sched_h[:, None])
    F[..., 12] = np.where(valid, (sim.clock_s - sim.nd_last_dep[vnode]) / (2.0 * node_h), 0.0)
    trk = S.ntrack[vnode]
    F[..., 13] = ((trk == 0) & (S.nl2x[vnode] >= 0)) | ((trk == 1) & (S.nx2l[vnode] >= 0))
    F[..., 14] = S.npocket[vnode]
    F[..., 15] = S.ndiverge[vnode]
    cur_time = vtime[:, C.K_BEHIND]
    F[..., 16] = np.where(valid, (vtime - cur_time[:, None]) / C.OBS_RUN_TIME_SCALE_S, 0.0)
    ob = sim.tr_onboard[idx]
    ob_tot = ob.sum(axis=1)
    k = vsidx - sim.tr_sptr[idx][:, None]
    kin = valid & (vsidx >= 0) & (k >= 0) & (k < C.N_DEST - 1)
    share = np.take_along_axis(ob, np.clip(k, 0, C.N_DEST - 1), axis=1) / np.maximum(ob_tot, 1e-9)[:, None]
    F[..., 17] = np.where(kin, share, 0.0)
    F *= valid[..., None]
    out[:, : C.OBS_SLOTS * C.OBS_SLOT_FEATURES] = F.reshape(n, -1)

    # ---------------------------------------------------------------- ego
    base = C.OBS_SLOTS * C.OBS_SLOT_FEATURES
    node = info["node"]
    e_next = info["e_next"]
    E = np.zeros((n, C.OBS_EGO_FEATURES), dtype=np.float64)
    E[:, 0] = ob_tot / sim.tr_cap[idx]
    E[:, 1] = np.minimum(info["lead_gap"] / (2.0 * info["lead_h"]), 1.0)
    E[:, 2] = np.minimum(info["foll_gap"] / (2.0 * info["foll_h"]), 1.0)
    E[:, 3] = info["late"] / 600.0
    E[:, 4] = sim.tr_hold_total[idx] / C.MAX_HOLD_S
    E[:, 5] = sim.tr_skipped_prev[idx]
    E[:, 6] = S.ntrack[node] == 1
    E[:, 7] = S.ecbtc[e_next]
    E[:, 8] = sim.tr_div[idx] == 1
    E[:, 9 + C.PROFILES.index(sim.profile)] = 1.0
    r_wait, r_stuck = _route_stats(sim)
    routes = sim.tr_route[idx].astype(np.int64)
    E[:, 13] = r_wait[routes] / 15.0
    E[:, 14] = r_stuck[routes] / 5.0
    E[:, 15] = len(sim._dis_active) / 4.0
    dis_slot = (F[:, C.K_BEHIND:, 8] > 0) | (F[:, C.K_BEHIND:, 10] > 0) | (F[:, C.K_BEHIND:, 9] > 0)
    first = np.where(dis_slot.any(axis=1), np.argmax(dis_slot, axis=1), C.K_AHEAD)
    E[:, 16] = first / float(C.K_AHEAD)
    for i in range(n):
        lead = info["yard"][i]
        if lead is None:
            leads = sim._P.route_leads.get(int(routes[i]))
            lead = leads[0] if leads else None
        if lead is not None:
            y = int(lead.yard)
            E[i, 17] = sim.yard_stock[y] / max(float(g.yard_fleet_cap[y]), 1.0)
            E[i, 18] = max(sim.yard_cool[y] - sim.clock_s, 0.0) / C.YARD_COOLDOWN_S
    opp = S.nopp[node]
    opp_h = edge_headway_at_nodes(sim, opp, sched_h)
    E[:, 19] = np.where(opp < S.SN, (sim.clock_s - sim.nd_last_dep[opp]) / (2.0 * opp_h), 0.0)
    E[:, 20] = sim.tr_gap[idx]
    E[:, 21] = sim.tr_deviated[idx]
    E[:, 22] = sched_h / 900.0
    E[:, 23] = sim.episode_time_s / max(sim.episode_s, 1.0)
    E[:, 24] = sim.demand_scale - 1.0
    out[:, base : base + C.OBS_EGO_FEATURES] = E
    out[:, base + C.OBS_EGO_FEATURES :] = masks
    np.clip(out, -1.0, 1.0, out=out)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    return out
