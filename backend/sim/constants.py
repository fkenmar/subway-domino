"""Physical constants, operating rules and cost coefficients for Subway Domino.

This module is the single source of truth (ARCHITECTURE §3.2, §4.2-§4.4). Every other
module, including ``rl/`` and ``app/``, reads its constants from here. Values marked
CONTRACT in the architecture document must not change without updating that document.
"""

from __future__ import annotations

import hashlib
import json
import math

# --------------------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------------------
DT_S: float = 10.0  # sim seconds per tick, used everywhere (training, eval, server)
WARMUP_S: float = 900.0  # warm-up simulated at reset (not part of episodes)

PROFILES: tuple[str, ...] = ("am_peak", "midday", "pm_rush", "late_night")
PROFILE_START_S: dict[str, float] = {
    "am_peak": 28800.0,
    "midday": 43200.0,
    "pm_rush": 61200.0,
    "late_night": 7200.0,
}
# Demand / service windows [start_h, end_h) used by the data build (ARCHITECTURE §2.3).
PROFILE_WINDOWS_H: dict[str, tuple[int, int]] = {
    "am_peak": (7, 10),
    "midday": (10, 16),
    "pm_rush": (16, 19),
    "late_night": (0, 5),
}

# --------------------------------------------------------------------------------------
# Fleet & rolling stock
# --------------------------------------------------------------------------------------
MAX_TRAINS: int = 700  # train slot capacity
MAX_PATH: int = 96  # minimum per-train path capacity (edges); grown to fit the network
DIVISIONS: tuple[str, ...] = ("A", "B")
TRAIN_CAPACITY: dict[str, float] = {"A": 1100.0, "B": 1450.0}  # riders at 100 % load
TRAIN_LENGTH_M: dict[str, float] = {"A": 156.0, "B": 183.0}
ACCEL: float = 1.1  # m/s^2
BRAKE: float = 1.2  # m/s^2 (service brake used by the braking curve)
V_MAX_CAP: float = 24.6  # m/s, upper clamp used by the data build
ACCEL_DECEL_LOSS_S: float = 13.0  # time lost to one acceleration + one braking (data build)

# --------------------------------------------------------------------------------------
# Signalling
# --------------------------------------------------------------------------------------
BLOCK_M: float = 250.0  # fixed-block length (boundaries every BLOCK_M along each edge)
CBTC_MARGIN_M: float = 60.0  # CBTC safety margin behind the leader's rear
FIXED_BLOCK_STOP_MARGIN_M: float = 10.0  # stop this far before the protecting signal
# Safety overlap beyond each fixed-block stop signal: the signal protecting the block in
# which the leader's rear sits is the one at the start of the block that contains the
# point (rear - overlap). Guarantees queue spacing >= train length + 0.5 * BLOCK_M.
FIXED_BLOCK_OVERLAP_M: float = 0.5 * BLOCK_M
CLOSED_NODE_MARGIN_M: float = 40.0  # home signal outside a closed station (train stays in tunnel)
LOOKAHEAD_EDGES: int = 3  # movement authority looks ahead this many edges past the current one
SIGNAL_DELAY_VMAX: float = 4.5  # m/s on edges affected by a signal_delay
SIGNAL_DELAY_KEYBY_S: float = 45.0  # key-by stop at the entry of a signal-delayed edge
STOPPED_V: float = 0.5  # m/s; slower than this counts as stopped

# --------------------------------------------------------------------------------------
# Stations & dwell
# --------------------------------------------------------------------------------------
DWELL_MIN_S: float = 20.0
BOARD_S_PER_RIDER: float = 0.03  # (board + alight) door time per rider
CROWD_DWELL_K: float = 6.0  # dwell x (1 + K * max(0, load - CROWD_DWELL_LOAD0)^2)
CROWD_DWELL_LOAD0: float = 0.6
DWELL_MAX_S: float = 180.0
DWELL_SCHED_S: float = 30.0  # nominal dwell used by the timetable (lateness reference)
SCHEDULE_PADDING: float = 1.03  # timetable recovery margin on running times
LATE_DEPARTURE_MAX_FRAC: float = 0.5  # a missed departure waits up to this x headway for a train
ORIGIN_DWELL_S: float = 30.0  # a dispatched train opens its doors this long before its departure
AT_PLATFORM_M: float = 60.0  # front within this distance of a platform node => "in the station"
# Interlocking route locking (crossovers, pocket turnbacks, spawns at a node):
# * approach locking: a route into a node is set only when no train front is within
#   NODE_CLEAR_M of it on a running in-edge (an approaching train can still stop) and no
#   train body straddles it;
# * a crossover occupant has reserved both of its nodes; it leaves as soon as its exit
#   node is physically free (no train front within NODE_OCCUPIED_M of it, no body over it).
NODE_CLEAR_M: float = 260.0
NODE_OCCUPIED_M: float = 5.0
POOL_MAX: int = 24  # terminal pool slots (trains laid over at one (route family, station))

# --------------------------------------------------------------------------------------
# Passengers (fluid, wait-age histogram)
# --------------------------------------------------------------------------------------
AGE_BIN_S: float = 60.0
N_AGE_BINS: int = 31  # last bin = 30+ min
N_DEST: int = 12  # onboard destination buckets (k-th next stop; last = 12+)
DEST_DECAY_STOPS: float = 8.0  # destination weight decay(k) = exp(-k / 8)
TRANSFER_FRAC: float = 0.25
TRANSFER_MIN_STATIONS: int = 2  # complex with >= 2 stations, or
TRANSFER_MIN_ROUTES: int = 3  # station with >= 3 routes
SPILLOVER_MIN_AGE_MIN: int = 12  # riders waiting longer than this may spill over
SPILLOVER_HEADWAY_MULT: float = 3.0  # ... when the platform's gap exceeds 3 x scheduled headway
SPILLOVER_RATE_PER_MIN: float = 0.05
OVERCROWD_FRAC: float = 0.80
CRITICAL_MEAN_WAIT_MIN: float = 10.0
CRITICAL_OLDEST_WAIT_MIN: float = 20.0


def age_bin_centers_min() -> list[float]:
    """Bin centres in minutes (b + 0.5); the last (30+) bin counts as 30 (ARCHITECTURE §3.8)."""
    centers = [b + 0.5 for b in range(N_AGE_BINS)]
    centers[-1] = 30.0
    return centers


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
TUNNEL_STUCK_S: float = 60.0  # stopped outside a platform at least this long => tunnel-stuck

# --------------------------------------------------------------------------------------
# Actions & operating rules
# --------------------------------------------------------------------------------------
HOLD_STEP_S: float = 30.0
MAX_HOLD_S: float = 180.0
TURNBACK_S: float = 240.0  # nominal short-turn latency (discharge + cab walk + brake test)
TURNBACK_MIN_S: float = 180.0  # per-event draw U[TURNBACK_MIN_S, TURNBACK_MAX_S]
TURNBACK_MAX_S: float = 300.0
CROSSOVER_S: float = 20.0
LAYOVER_S: float = 180.0  # terminal minimum layover
YARD_COOLDOWN_S: float = 600.0  # per yard
K_AHEAD: int = 6  # ego corridor: stations ahead
K_BEHIND: int = 3  # ego corridor: stations behind

N_ACTIONS: int = 8
ACTION_NAMES: tuple[str, ...] = (
    "CONTINUE_SCHEDULED",
    "HOLD_IN_STATION",
    "RUN_EXPRESS",
    "RUN_LOCAL",
    "SKIP_STOP",
    "SHORT_TURN_RELAY",
    "REROUTE_VIA_COMPATIBLE_TRUNK",
    "DISPATCH_YARD_GAP_TRAIN",
)
A_CONTINUE: int = 0
A_HOLD: int = 1
A_EXPRESS: int = 2
A_LOCAL: int = 3
A_SKIP: int = 4
A_SHORT_TURN: int = 5
A_REROUTE: int = 6
A_YARD: int = 7

# Mask thresholds (ARCHITECTURE §4.2)
HOLD_GAP_WIDE_MULT: float = 1.5  # leader_gap < 1.5 * sched_h (headway not already wide)
HOLD_GAP_BUNCHED_MULT: float = 0.75  # leader_gap < 0.75 * sched_h (bunched)
SKIP_LATENESS_S: float = 120.0
SKIP_FOLLOWER_MULT: float = 0.5
SHORT_TURN_OPPOSITE_GAP_MULT: float = 2.0
SHORT_TURN_FOLLOWER_MULT: float = 0.5
SHORT_TURN_BLOCK_HORIZON_S: float = 900.0  # a blockage counts for a relay if reached within this (nominal)
BLOCKED_STOPPED_TRAINS: int = 2  # >= 2 trains stopped outside platforms => blocked ahead
REROUTE_MAX_STATIONS: int = 30
REROUTE_MAX_TIME_RATIO: float = 2.5
YARD_STARVED_MULT: float = 2.0
YARD_STARVED_MIN_S: float = 480.0
YARD_SEARCH_STATIONS: int = 30  # yard lead nodes considered along the remaining path
GAP_PLAN_MIN_S: float = 1.0  # floor of the planned (timetable) gap used as a mask reference
# Revision of the mask/obs *semantics* (bumped when a predicate's meaning changes; part of
# constants_hash so a model trained against older masks warns when served).
MASK_REVISION: int = 2  # 2: timetable-relative gap reference, multi-pattern yard starvation
SCHED_H_MIN_S: float = 60.0  # clamp of the scheduled headway used by masks/obs
SCHED_H_MAX_S: float = 3600.0
SCHED_H_FALLBACK_S: float = 600.0
# Leader/follower gaps are headways at the other train's location: nominal time-to-reach
# plus the time that train has stood at its platform, capped at a normal dwell (a train
# standing longer is treated as stuck, so a stuck leader still gives a small gap).
GAP_DWELL_CAP_S: float = 1.5 * DWELL_SCHED_S

# Frame status bits (ARCHITECTURE §3.9)
ST_AT_PLATFORM: int = 1
ST_TUNNEL_STUCK: int = 2
ST_HELD: int = 4
ST_SHORT_TURNING: int = 8
ST_TRACK_SWITCH: int = 16
ST_REROUTED: int = 32
ST_GAP_TRAIN: int = 64
ST_SKIPPING_NEXT: int = 128

# --------------------------------------------------------------------------------------
# Observation (ARCHITECTURE §4.3)
# --------------------------------------------------------------------------------------
OBS_SLOTS: int = K_BEHIND + 1 + K_AHEAD  # 10
OBS_SLOT_FEATURES: int = 18
OBS_EGO_FEATURES: int = 28
OBS_DIM: int = OBS_SLOTS * OBS_SLOT_FEATURES + OBS_EGO_FEATURES + N_ACTIONS  # 216
OBS_WAITING_LOG_SCALE: float = 1000.0  # waiting feature = log1p(w) / log1p(1000) (clipped)
OBS_RUN_TIME_SCALE_S: float = 1800.0

# --------------------------------------------------------------------------------------
# Cost / reward (ARCHITECTURE §4.4, PRD §4.3)
# --------------------------------------------------------------------------------------
W_WAIT: float = 1.0
WAIT_AGE_EXPONENT: float = 1.5
WAIT_AGE_CAP_MIN: float = 30.0  # last wait-age bin centre; also caps the trapped-rider age
# Riders trapped in a tunnel-stopped train weigh a^TUNNEL_RIDER_AGE_EXPONENT per rider-minute,
# a = minutes stopped clipped to [1, WAIT_AGE_CAP_MIN]: the platform wait kernel, so a trapped
# rider is never cheaper than a waiting one (0.0 restores the flat 1 per rider-minute).
TUNNEL_RIDER_AGE_EXPONENT: float = 1.5
W_TUNNEL_TRAIN: float = 200.0
W_TUNNEL_RIDER: float = 1.0
W_HOLD_TRAIN: float = 10.0
W_HOLD_RIDER: float = 0.05
W_CROWD: float = 500.0
CROWD_THRESHOLD: float = 0.8
CROWD_SPAN: float = 0.2
W_ST: float = 10.0
W_SKIP: float = 5.0
W_REROUTE: float = 300.0
W_REROUTE_RIDER: float = 2.0
W_YARD: float = 400.0
REWARD_SCALE: float = 1.0 / 2000.0
# WP5 recalibration (docs/progress/WP5.md): 10 -> 100. A step's reward is the AI-vs-twin cost
# difference over the ticks it advances, so on sparse Stage-1 trunks it carries the divergence
# accumulated over ~1.4 ticks. At 10, 13 % of Stage-1 steps saturated (27 % in the best am_peak
# meltdown), which erased the signal exactly where actions pay off most. At 100: 1.6 % in Stage 1,
# 0 % in Stage 2. Training normalizes returns (rl.ppo.ReturnNormalizer), so only the tail cap moves.
REWARD_CLIP: float = 100.0

COST_KEYS: tuple[str, ...] = (
    "wait",
    "tunnel",
    "hold",
    "crowd",
    "short_turn",
    "skip",
    "reroute",
    "yard",
    "total",
)
METRIC_KEYS: tuple[str, ...] = (
    "clock_s",
    "avg_wait_min",
    "overcrowded_platforms",
    "tunnel_stuck_trains",
    "riders_served",
    "riders_waiting",
    "riders_onboard",
    "trains_active",
)

# --------------------------------------------------------------------------------------
# Server playback speeds (sim-seconds per real second)
# --------------------------------------------------------------------------------------
SPEEDS: dict[str, int] = {"pause": 0, "1x": 60, "2x": 120, "5x": 300}

# --------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------
EVAL_SEED_BASE: int = 1_000_000  # eval seeds are EVAL_SEED_BASE + i; never used in training
VALIDATION_SEED_BASE: int = 2_000_000  # validation seeds (training-time model selection)
TRAINING_SEED_MAX: int = EVAL_SEED_BASE  # training scenario seeds are drawn from [0, this)

# Stage-1 trunk groups (ARCHITECTURE §4.6)
TRUNK_GROUPS: dict[str, tuple[str, ...]] = {
    "lexington": ("4", "5", "6", "6X"),
    "seventh_av": ("1", "2", "3"),
    "eighth_av": ("A", "C", "E"),
    "sixth_av": ("B", "D", "F", "FX", "M"),
    "broadway": ("N", "Q", "R", "W"),
    "queens_blvd": ("E", "F", "M", "R"),
    "flushing": ("7", "7X"),
    "canarsie": ("L",),
    "nassau": ("J", "Z", "M"),
    "crosstown": ("G",),
}

# Scenario sampling (ARCHITECTURE §4.6-§4.7)
STAGE1_EPISODE_S: float = 3600.0
STAGE2_EPISODE_S: float = 5400.0
DISRUPTION_START_RANGE_S: tuple[float, float] = (300.0, 1500.0)
DISRUPTION_DURATION_RANGE_S: tuple[float, float] = (600.0, 2400.0)
STAGE1_JITTER_RANGE: tuple[float, float] = (0.1, 0.45)
STAGE2_JITTER_RANGE: tuple[float, float] = (0.0, 0.25)
DR_DEMAND_SCALE: tuple[float, float] = (0.8, 1.3)
DR_DWELL_SCALE: tuple[float, float] = (0.85, 1.25)
DR_SPEED_SCALE: tuple[float, float] = (0.9, 1.05)
EVAL_DISRUPTION_START_RANGE_S: tuple[float, float] = (300.0, 900.0)
EVAL_DISRUPTION_DURATION_RANGE_S: tuple[float, float] = (900.0, 2400.0)
EVAL_DEMAND_SCALE: tuple[float, float] = (0.9, 1.15)
EVAL_JITTER_RANGE: tuple[float, float] = (0.0, 0.2)

DISRUPTION_KINDS: tuple[str, ...] = (
    "local_block",
    "express_block",
    "full_station_closure",
    "signal_delay",
)
DISRUPTION_TARGETS: dict[str, tuple[str, ...]] = {
    "local_block": ("segment", "station", "interlocking"),
    "express_block": ("segment", "station", "interlocking"),
    "full_station_closure": ("station", "complex"),
    "signal_delay": ("segment", "interlocking"),
}


def constants_hash() -> str:
    """SHA-1 of the reward/observation-relevant constants (stored in ONNX metadata).

    A model trained against one set of observation/mask/cost semantics should warn when
    served against another; this hash is the shared fingerprint.
    """
    relevant = {
        "DT_S": DT_S,
        "OBS_DIM": OBS_DIM,
        "N_ACTIONS": N_ACTIONS,
        "ACTION_NAMES": list(ACTION_NAMES),
        "K_AHEAD": K_AHEAD,
        "K_BEHIND": K_BEHIND,
        "N_AGE_BINS": N_AGE_BINS,
        "N_DEST": N_DEST,
        "HOLD_STEP_S": HOLD_STEP_S,
        "MAX_HOLD_S": MAX_HOLD_S,
        "TURNBACK": [TURNBACK_MIN_S, TURNBACK_MAX_S],
        "LAYOVER_S": LAYOVER_S,
        "YARD_COOLDOWN_S": YARD_COOLDOWN_S,
        "mask_revision": MASK_REVISION,
        "mask": [
            GAP_PLAN_MIN_S,
            HOLD_GAP_WIDE_MULT,
            HOLD_GAP_BUNCHED_MULT,
            SKIP_LATENESS_S,
            SKIP_FOLLOWER_MULT,
            SHORT_TURN_OPPOSITE_GAP_MULT,
            SHORT_TURN_FOLLOWER_MULT,
            SHORT_TURN_BLOCK_HORIZON_S,
            BLOCKED_STOPPED_TRAINS,
            REROUTE_MAX_STATIONS,
            REROUTE_MAX_TIME_RATIO,
            YARD_STARVED_MULT,
            YARD_STARVED_MIN_S,
        ],
        "cost": [
            W_WAIT,
            WAIT_AGE_EXPONENT,
            W_TUNNEL_TRAIN,
            W_TUNNEL_RIDER,
            TUNNEL_RIDER_AGE_EXPONENT,
            W_HOLD_TRAIN,
            W_HOLD_RIDER,
            W_CROWD,
            W_ST,
            W_SKIP,
            W_REROUTE,
            W_REROUTE_RIDER,
            W_YARD,
            REWARD_SCALE,
            REWARD_CLIP,
        ],
        "obs_scales": [OBS_WAITING_LOG_SCALE, OBS_RUN_TIME_SCALE_S],
    }
    blob = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()


assert OBS_DIM == 216, OBS_DIM
assert len(ACTION_NAMES) == N_ACTIONS
assert math.isclose(FIXED_BLOCK_OVERLAP_M, 125.0)
