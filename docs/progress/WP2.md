# WP2 — Simulation engine (progress)

## Orchestrator notes (2026-11-20 03:55): read first
- **WP1 is complete.** `backend/data/network.json` (graph_version e184af352bc6) has 475 stations, 1,432 nodes, 1,386 run edges, 316 crossovers, 145 turnbacks, 68 yard leads, 158 patterns and 7 presets. Your contract change requests 1 and 3 were accepted into ARCHITECTURE §4.2 and §3.1.
- **CRITICAL BUG found by WP1 on the real network: crossover-merge deadlock in `sim/engine.py`.**
  - A train on a crossover waits for `_node_clear(to_node)`. That check refuses while any train front is within `NODE_CLEAR_M = 260 m` on any in-edge.
  - The train approaching on the other in-edge has stopped short, because the crossover occupant blocks that node. Neither moves.
  - Repro: real network, `Scenario(seed=0, profile='am_peak')`, 90 ticks. An F on crossover edge 1653 (Forest Hills-71 Av, X→L) waits for node 1044 while an R stands 188 m short on edge 1028.
  - Tunnel-stuck trains grow from 30 to 125 in 30 sim-min with **no disruption**, which swamps every preset and any training run.
  - Fix this first, with real merge arbitration: first-come or route-locking reservation of the merge node, not mutual waiting. Add a regression test with no disruptions: citywide am_peak for 2 sim-hours, where tunnel_stuck must stay ≤ 3 on average and no train may be stopped for more than 10 min.
- **Empty `next_patterns`:** 7 low-frequency patterns (3–12 trips/day) have none. The sim must handle this, e.g. by returning the train to a terminal pool of the same route at that station, or laying it up.
- **lex-union-sq preset:** southbound 4/5/6 can only HOLD or get yard gap trains, because there is no verified Manhattan turnback north of 14 St. Northbound trains can short-turn at Brooklyn Bridge or Bowling Green, and the 5 can reroute at Nevins St.
- `test_import_firewall` currently imports `sim.bench`, which doesn't exist yet. Write `sim/bench.py`.
- If you find data defects, note them under "Defects reported by WP2" in `docs/progress/WP1.md`. Don't edit WP1 files; the orchestrator will route the fixes.

## Done
- [x] constants.py, graph.py, testing.py (+ test helpers), kinematics.py, paths.py, cost.py
- [x] engine.py, masks.py, obs.py, actions.py, disruptions.py, scenarios.py, policy.py
- [x] toy fixes: stops beyond a timed edge (infinite dwell loop), pool -inf slot bug, terminal
      arrival-platform demand, leader/follower gap definition, berthing at a platform behind a red
      signal / station close-in, state_hash excludes disruption-id bookkeeping
- [x] tests: conftest, import_firewall, determinism, obs_reward, signaling, short_turn, yard, masks,
      division_isolation (all green on the toy; real-network variants skip until network.json exists)
- [x] REAL NETWORK: crossover-merge deadlock fixed (2026-11-20) with interlocking route locking
      (engine `_try_enter_timed` / `_timed_done` / `_node_clear`): entry = approach locking on the
      exit node + reservation of both nodes (`_node_res` owner, `_node_tail`, `_node_route`); exit =
      exit node physically free only; route node = red home signal for others; a train standing
      at the node to leave over a timed edge never blocks (track swaps). Entries happen at tick
      start (`_timed_entries`) against true positions; a train must have berthed/dwelt first.
      Calm am_peak 1 h: stuck 0-3 (was 30 -> 192), wait 3.1 min, 1 missed departure.
- [x] calibrate.py (`python -m sim.calibrate`): calm profiles + presets vs targets (module constants)
- [x] 2026-11-21 real-network mask fix (MASK_REVISION 2, see CR 4): calm am_peak decisions 7541/h -> 794/h
      (a) gaps "short" vs min(track headway, planned gap to that train) — timetable merges are irregular;
      (b) yard starvation ETA over every in-service pattern on the lead's track (was: the lead's own
      low-frequency variant pattern, e.g. 2 pattern 109 with h=10800 s -> DISPATCH valid 5093x/h calm)
- [x] bench.py (`python -m sim.bench`): tick 2.05 -> 0.83 ms (yard starvation vectorized over all
      leads per tick: `masks.lead_etas`/`yard_choice`, verified equal to the scalar version on 8280
      lead-ticks; detour checks test `_orig_affected` first; SKIP critical-wait stats only for the
      batch's next-stop platforms). pending x32 0.10 ms, reset 56 ms, copy 0.09 ms (all ok)
- [x] test_action_leverage.py (slow, 12 tests green): scripted incident-response policy (test-only)
      beats Normal on cost AND stuck train-min in all 7 presets + stage-2 seeds 11/12/13
      (-0.2 .. -26.9 % cost); track switches add leverage where they bypass the blockage (qbl
      -19.7 -> -59.5 %, manhattan-bridge -9.4 -> -14.9 %); all 8 actions valid somewhere.
      Root causes fixed on the way (CR 5): SHORT_TURN "disruption blocks the path to the
      terminal" read over the K_AHEAD horizon (was: Bronx pockets 20 stations before 14 St ->
      riders stranded, +2.9 M wait); trapped riders weighted with the wait-age kernel
      (`TUNNEL_RIDER_AGE_EXPONENT = 1.5`, 0.0 restores the contract formula)
- [x] test_perf.py (slow; best-of-N bench; tick 0.6 target xfail, guard 1.5 ms), calm-service
      regression + calibration targets (slow, test_signaling.py), real-network determinism +
      rider conservation (test_determinism.py), new-semantics tests (test_masks/yard/obs_reward)
- [x] final (2026-11-21): calibrate all green (below); WP2 suite 101 passed / 1 skipped (rl.env) /
      1 xfailed (tick target); repo fast suite 217 passed; API edge cases re-checked (invalid /
      repeated / non-batch actions -> applied=False, ValueErrors, metrics/cost keys, frame dtypes);
      real-network determinism, rider conservation (rel. err 1e-14) and twin difference = 0

## Calibration (python -m sim.calibrate, seed 0, 60 sim-min calm; presets first 15 min)
| profile | trains (target) | dwell median | avg wait | overcrowded | stuck mean/max | decisions/h |
|---|---|---|---|---|---|---|
| am_peak | 454 (417-615) | 26.6 s | 3.09 min | 0 % | 0.37 / 2 | 865 |
| midday | 393 (340-460) | 26.1 s | 3.01 | 0 % | 0.10 / 1 | 402 |
| pm_rush | 457 (428-580) | 26.7 s | 2.79 | 0 % | 0.30 / 2 | 800 |
| late_night | 94 (71-133) | 25.4 s | 8.18 | 0 % | 0 / 0 | 46 |
Presets (stuck max in 15 min; wait 0 -> 15 min, calm twin): lex-union-sq 14, 3.15 -> 3.56 (3.05);
qbl 18; canarsie 8; manhattan-bridge 15; 8av-59st 10; joralemon 7; times-sq-signals 5 (signal delay,
calm max 2). Max stationary train in calm 2 h am_peak <= 120 s.

## Leverage (tests/test_action_leverage.py, 60 sim-min, Normal -> scripted, stuck train-min)
lex -13.4 % (438 -> 276), qbl -19.7 % (518 -> 442; with switches -59.5 %, 86), times-sq -1.0 %
(126 -> 112), canarsie -20.4 % (192 -> 62), manhattan -16.4 % (607 -> 412; switches -14.9..),
8av -15.6 % (395 -> 286), joralemon -4.1 % (294 -> 276), stage2 seed11 -28.2 % (881 -> 472),
seed12 -1.7 % (95 -> 70), seed13 -0.2 % (0.3 -> 0.3, below noise floor). Decision rows under
Normal (all 10): HOLD valid 61 %, SKIP 34 %, YARD 13 %, LOCAL 3.6 %, SHORT_TURN 2.3 %,
REROUTE 0.7 %, EXPRESS 0.6 %.
- [~] perf: tick 0.83 ms (target 0.6). Remaining cost is numpy per-call overhead spread over
      _move 0.28, _collect_decisions 0.25, _timers 0.09, _occupancy 0.08 ms (per-phase timing,
      scratch phases.py). Further gains need kernel fusion -> documented, moved on

## Next
- (optional) tick 0.83 -> 0.6 ms needs kernel fusion in `_move` / `_collect_decisions` (numpy
  per-call overhead); obs could expose the timetable-relative gap (a reserved ego slot) so the
  policy sees why HOLD/SKIP fire on shared trunks (FEATURE_NAMES change -> coordinate with WP5).

## Known issues
- §3.7 tick target: 0.83 ms median on an idle performance core (0.6 target; xfail in test_perf,
  guard 1.5 ms). Under contention the process runs on an efficiency core: ~1.3 ms, every phase
  uniformly 1.6x slower (identical call counts), so bench reports best-of-N batch medians.
- Citywide avg wait barely moves for local presets in 15 min (+0.05..0.15 min vs calm); only the
  reference meltdown (lex-union-sq) is asserted as "clearly rising" (+0.50 vs calm).
- trains_active am_peak 454 is 7.5 % under the §0 08h low end (491; pool/layover trains are not
  'active'); inside the +-15 % band.

## Contract change requests
5. (2026-11-21) Found by the leverage test on the real network.
   (a) §4.2 SHORT_TURN_RELAY: "a disruption blocks the path between here and the terminal" now
   requires the train to reach the first blocked edge / closed station within
   `SHORT_TURN_BLOCK_HORIZON_S = 900` s of nominal running. Literal reading: lex-union-sq relayed
   southbound 4/5 trains at Bedford Park Blvd / 149 St (~25 min before the closure), stranding
   riders where service still ran (+2.9 M wait cost, 21 relays). (A K_AHEAD-*stations* horizon
   was tried first: it also dropped Bowling Green NB, 8 stations incl. passed local stations but
   ~8 min before 14 St.) Proposed text: "a disruption lies on the path within 15 min of nominal
   running".
   (b) §4.4 C_tunnel rider term: `W_TUNNEL_RIDER * onboard * a^1.5`, a = minutes stopped clipped to
   [1, 30] (first minute = old value). With the flat 1/rider-min a rider trapped 30 min cost 30 vs
   ~1,970 for one waiting 30 min on a platform (65x), so any action that moved riders out of stuck
   trains (relay, hold at a platform) looked far worse than trapping them - contrary to PRD §4.3
   "Tunnel-Trapping Penalty". Knob `constants.TUNNEL_RIDER_AGE_EXPONENT` (hashed; 0.0 = contract
   formula). The leverage test also passes with 0.0 (margins -0.2..-18.8 % instead of -0.2..-26.9 %),
   so (b) is a calibration choice for the RL owner, not a test crutch.
4. (2026-11-21) §4.2 "short gap" clauses on shared trunks. With `sched_h` = combined track headway,
   38.6 % of calm am_peak departures on multi-pattern track have leader_gap < 0.75 sched_h *by
   timetable* (uncoordinated merges of patterns with different headways/phases; single-pattern
   track: 2.6 %), so HOLD fired at ~4000 calm departures/h. Implemented: the bunched-leader clause
   of HOLD and the short-follower clauses of SKIP_STOP / SHORT_TURN_RELAY compare the gap with
   `ref_h = min(sched_h, planned_gap)`, `planned_gap = gap - (late_behind - late_ahead)` (the pair's
   timetable spacing; lateness difference = drift from plan). "Headway not already wide" keeps
   sched_h. Reduces to the contract text on regular unshared track. Proposed §4.2 text: "bunched /
   short gaps are judged against min(sched_h, the timetable gap to that train)". Also §4.2 yard
   clause: "the next train" = any in-service train on the lead's track (all patterns using the
   lead edge), not only the lead's own pattern. Both in `constants.MASK_REVISION = 2` (hashed).
1. §4.2 `leader_gap_s` / `sched_h` (mask definitions). Literal "distance / edge v_max" ignores
   dwells and acceleration: an on-time leader measures ~0.6 headways, so the HOLD need clause
   (< 0.75 h) holds at ~85 % of calm departures (toy measurement), contradicting "need clauses
   keep calm service nearly decision-free". Implemented instead (sim/masks.py docstring):
   gap = nominal time-to-reach the leader's front along the path (timetable profile incl. dwells)
   + the leader's elapsed platform time (capped 45 s); a red signal at a disruption ahead counts as
   a stuck leader. Each gap is compared with the scheduled headway *of the track where it is
   measured* (combined headway of all patterns on that edge), not the ego route's headway (on a
   shared trunk the leader is usually another route). Result: calm toy service 0 % decisions;
   a stuck leader just ahead still gives a small gap. Proposed contract text: "leader_gap_s is the
   headway to the train ahead at its location: nominal time-to-reach its front plus its elapsed
   platform dwell; sched_h is the combined scheduled headway of the track where the gap is
   measured."
2. §4.3 obs gap features use the same per-location reference headway (`gap / (2 h)`), so mask
   thresholds are readable from the observation. (Layout is GUIDANCE; recorded for the model card.)
3. `Scenario.headway_jitter: float = 0.0` (backwards-compatible superset of §3.1) implements the
   Stage-1 "random headway jitter".

### 2026-11-22: orchestrator, speed check (cited in the README)
- `scratchpad/evaltools/speed_check.py` on calm am_peak, 10 sim-min, seed 0:
  - mean train speed 27.6 km/h including dwells (real NYC subway ≈ 27 km/h);
  - moving trains: median 39.6 km/h, max 64.8 km/h.
- The live frame stream gives a median of 37 km/h for moving trains against the on-screen clock at 1 min/s, and 30 km/h at 5 min/s.
