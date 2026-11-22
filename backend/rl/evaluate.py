"""Normal Schedule vs AI on the fixed evaluation seeds (ARCHITECTURE §4.7, acceptance bar §4.8).

For each seed ``EVAL_SEED_BASE + i`` (i < ``--seeds``) the same ``eval_scenario`` runs twice for
its full ``episode_s``:

* **Normal**: every decision resolves as ``CONTINUE_SCHEDULED``;
* **AI**: ``sim.load_policy`` (``.onnx`` in production, ``.zip`` for local checkpoints). Each
  tick's batch is answered from its one snapshot and applied in batch order, exactly like the
  server and the training env.

Per seed it records the time-averaged ``avg_wait_min``, overcrowded platform-minutes,
tunnel-stuck train-minutes, ``riders_served``, the cost totals and the action counts. It writes
``eval_report.json`` (rows plus aggregates: mean, 95 % bootstrap CI, AI-vs-Normal % change and
the seed win rate), ``eval_report.md`` and ``eval_per_seed.csv``.

The exit code is 0, unless ``--require-win`` is given and the AI does not beat Normal on mean
total cost **and** on at least 3 of the 4 UI metrics.

    .venv/bin/python -m rl.evaluate --policy models/nyc_subway_rcc_ppo.onnx --seeds 50 --out models/
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sim import NYCSubwaySim, Scenario, eval_scenario, load_graph, load_policy
from sim import constants as C
from sim.graph import Graph

#: ``act(obs[n, OBS_DIM], masks[n, N_ACTIONS]) -> actions[n]``
ActFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

#: the four UI metrics and whether lower is better
UI_METRICS: dict[str, bool] = {
    "avg_wait_min": True,
    "overcrowded_platform_min": True,
    "tunnel_stuck_train_min": True,
    "riders_served": False,
}
BOOTSTRAP_RESAMPLES: int = 10_000
#: §4.8 acceptance bar
SHIP_WIN_RATE: float = 0.70
SHIP_STUCK_REDUCTION: float = 0.30
SHIP_WAIT_REDUCTION: float = 0.10  # superseded (2026-11-22): reported for information only, not part of the bar
#: Share of the disruption-induced excess avg wait (Normal vs the same scenario without disruptions)
#: the AI must remove, as a ratio of sums over the seeds. Confirmed with the user 2026-11-22: 20 % =
#: what a rule-based incident dispatcher achieves on the same seeds (evaluation-only benchmark).
SHIP_EXCESS_WAIT_RECOVERY: float = 0.20


# --------------------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------------------
def run_episode(graph: Graph, scenario: Scenario, act: ActFn | None = None) -> dict[str, Any]:
    """Run ``scenario`` for its full ``episode_s``; ``act=None`` is Normal Schedule."""
    sim = NYCSubwaySim(graph, scenario)
    n_ticks = int(math.ceil(scenario.episode_s / C.DT_S - 1e-9))
    cost0 = np.array(sim.cost, dtype=np.float64)
    served0 = float(sim.metrics()["riders_served"])
    wait_sum = over_sum = stuck_sum = 0.0
    decisions = 0
    dt_min = C.DT_S / 60.0
    for _ in range(n_ticks):
        if act is not None:
            b = sim.pending_decisions()
            if len(b):
                actions = np.asarray(act(b.obs, b.masks), dtype=np.int64).reshape(-1)
                sim.apply_actions(b.train_ids, actions)
                decisions += len(b)
        sim.step(1)
        m = sim.metrics()
        wait_sum += m["avg_wait_min"]
        over_sum += m["overcrowded_platforms"]
        stuck_sum += m["tunnel_stuck_trains"]
    m = sim.metrics()
    cost = np.array(sim.cost, dtype=np.float64) - cost0
    return {
        "avg_wait_min": wait_sum / max(n_ticks, 1),
        "overcrowded_platform_min": over_sum * dt_min,
        "tunnel_stuck_train_min": stuck_sum * dt_min,
        "riders_served": float(m["riders_served"]) - served0,
        "cost": dict(zip(C.COST_KEYS, cost.tolist())),
        "actions": sim.action_counts(),
        "decisions": decisions,
        "ticks": n_ticks,
    }


def evaluate_seed(graph: Graph, seed: int, act: ActFn) -> dict[str, Any]:
    """Normal and AI on ``eval_scenario(seed)`` (one row of the report)."""
    sc = eval_scenario(int(seed), graph)
    t0 = time.perf_counter()
    normal = run_episode(graph, sc, None)
    ai = run_episode(graph, sc, act)
    # counterfactual calm day: the same scenario without its disruptions, under Normal Schedule
    calm = normal if not sc.disruptions else run_episode(graph, dataclasses.replace(sc, disruptions=()), None)
    return {"seed": int(seed), "profile": sc.profile, "n_disruptions": len(sc.disruptions),
            "demand_scale": sc.demand_scale, "normal": normal, "ai": ai,
            "calm": {"avg_wait_min": calm["avg_wait_min"]}, "seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def bootstrap_ci(x: Sequence[float], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> tuple[float, float]:
    """95 % percentile-bootstrap CI of the mean."""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return (float("nan"), float("nan"))
    if a.size == 1:
        return (float(a[0]), float(a[0]))
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(resamples, a.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def excess_wait_recovery(rows: list[dict[str, Any]], resamples: int = BOOTSTRAP_RESAMPLES,
                         seed: int = 0) -> dict[str, Any]:
    """Share of the disruption-induced excess avg wait the AI removes: sum(normal - ai) / sum(normal - calm).

    Ratio of sums over seeds (seeds whose disruptions barely move citywide wait carry little weight), with
    a 95 % seed-bootstrap CI. Rows without a ``calm`` entry (older reports) give NaN."""
    if not rows or any("calm" not in r for r in rows):
        nan = float("nan")
        return {"calm_mean": nan, "excess_mean": nan, "recovery": nan, "recovery_ci95": (nan, nan)}
    n = np.array([r["normal"]["avg_wait_min"] for r in rows], dtype=np.float64)
    a = np.array([r["ai"]["avg_wait_min"] for r in rows], dtype=np.float64)
    c = np.array([r["calm"]["avg_wait_min"] for r in rows], dtype=np.float64)
    exc = np.maximum(n - c, 0.0)
    rec = n - a
    ratio = float(rec.sum() / exc.sum()) if exc.sum() > 0 else float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(resamples, len(rows)))
    den = exc[idx].sum(axis=1)
    ok = den > 0
    boots = rec[idx].sum(axis=1)[ok] / den[ok]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots.size else (float("nan"),) * 2
    return {"calm_mean": float(c.mean()), "excess_mean": float(exc.mean()), "recovery": ratio, "recovery_ci95": ci}


def _pct_change(ai: float, normal: float) -> float:
    return float("nan") if normal == 0 else 100.0 * (ai - normal) / abs(normal)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates over seeds: per metric mean / CI for both modes, paired difference, % change."""
    keys = [*UI_METRICS, "total_cost"]

    def col(mode: str, k: str) -> np.ndarray:
        if k == "total_cost":
            return np.array([r[mode]["cost"]["total"] for r in rows], dtype=np.float64)
        return np.array([r[mode][k] for r in rows], dtype=np.float64)

    metrics: dict[str, Any] = {}
    better = 0
    for k in keys:
        n, a = col("normal", k), col("ai", k)
        lower = UI_METRICS.get(k, True)
        diff = a - n
        ai_better = bool(a.mean() < n.mean()) if lower else bool(a.mean() > n.mean())
        metrics[k] = {
            "normal_mean": float(n.mean()), "normal_ci95": bootstrap_ci(n),
            "ai_mean": float(a.mean()), "ai_ci95": bootstrap_ci(a),
            "diff_mean": float(diff.mean()), "diff_ci95": bootstrap_ci(diff),
            "pct_change": _pct_change(float(a.mean()), float(n.mean())),
            "lower_is_better": lower, "ai_better": ai_better,
        }
        if k in UI_METRICS and ai_better:
            better += 1
    wins = [r["ai"]["cost"]["total"] < r["normal"]["cost"]["total"] for r in rows]
    rel = [(r["normal"]["cost"]["total"] - r["ai"]["cost"]["total"]) / r["normal"]["cost"]["total"]
           for r in rows if r["normal"]["cost"]["total"] > 0]
    actions: dict[str, float] = {name: float(np.mean([r["ai"]["actions"].get(name, 0) for r in rows]))
                                 for name in C.ACTION_NAMES[1:]}
    cost_better = metrics["total_cost"]["ai_better"]
    stuck_red = -metrics["tunnel_stuck_train_min"]["pct_change"] / 100.0
    wait_red = -metrics["avg_wait_min"]["pct_change"] / 100.0
    win_rate = float(np.mean(wins)) if wins else 0.0
    exw = excess_wait_recovery(rows)
    rec_ok = bool(math.isfinite(exw["recovery"]) and exw["recovery"] >= SHIP_EXCESS_WAIT_RECOVERY)
    return {
        "n_seeds": len(rows),
        "metrics": metrics,
        "seed_win_rate": win_rate,
        "mean_rel_cost_reduction": float(np.mean(rel)) if rel else 0.0,
        "ui_metrics_better": better,
        "require_win_pass": bool(cost_better and better >= 3),
        "acceptance_bar": {
            "cost_better": cost_better,
            "win_rate_ok": win_rate >= SHIP_WIN_RATE,
            "stuck_reduction": stuck_red, "stuck_ok": stuck_red >= SHIP_STUCK_REDUCTION,
            "wait_reduction": wait_red,  # informational since 2026-11-22
            "excess_wait_recovery": exw["recovery"], "excess_wait_recovery_ci95": exw["recovery_ci95"],
            "excess_wait_recovery_target": SHIP_EXCESS_WAIT_RECOVERY, "wait_ok": rec_ok,
            "pass": bool(cost_better and win_rate >= SHIP_WIN_RATE and stuck_red >= SHIP_STUCK_REDUCTION
                         and rec_ok),
        },
        "excess_wait": exw,
        "mean_actions_per_episode": actions,
    }


# --------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------
def write_reports(out_dir: Path, report: dict[str, Any]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / "eval_report.json", "md": out_dir / "eval_report.md", "csv": out_dir / "eval_per_seed.csv"}
    _atomic_write(paths["json"], json.dumps(report, indent=2))
    _atomic_write(paths["md"], render_markdown(report))
    rows = report["rows"]
    fields = ["seed", "profile", "n_disruptions", "demand_scale"]
    for mode in ("normal", "ai"):
        fields += [f"{mode}_{k}" for k in UI_METRICS] + [f"{mode}_cost_{k}" for k in C.COST_KEYS]
    fields += [f"ai_n_{n}" for n in C.ACTION_NAMES[1:]] + ["ai_decisions", "ai_wins", "calm_avg_wait_min"]
    tmp = paths["csv"].with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            line: list[Any] = [r["seed"], r["profile"], r["n_disruptions"], r["demand_scale"]]
            for mode in ("normal", "ai"):
                line += [round(r[mode][k], 6) for k in UI_METRICS] + [round(r[mode]["cost"][k], 3) for k in C.COST_KEYS]
            line += [r["ai"]["actions"].get(n, 0) for n in C.ACTION_NAMES[1:]]
            line += [r["ai"]["decisions"], int(r["ai"]["cost"]["total"] < r["normal"]["cost"]["total"])]
            line += [round(r["calm"]["avg_wait_min"], 6) if "calm" in r else ""]
            w.writerow(line)
    os.replace(tmp, paths["csv"])
    return paths


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _fmt(x: float) -> str:
    if not math.isfinite(x):
        return "n/a"
    ax = abs(x)
    return f"{x:,.0f}" if ax >= 1000 else (f"{x:.1f}" if ax >= 10 else f"{x:.3f}")


def _excess_line(s: dict[str, Any]) -> str:
    e = s.get("excess_wait")
    if not e or not math.isfinite(e.get("recovery", float("nan"))):
        return "- Disruption excess wait: n/a (no calm counterfactual in this report)"
    lo, hi = e["recovery_ci95"]
    return (f"- Disruption excess wait: Normal {s['metrics']['avg_wait_min']['normal_mean']:.3f} min vs calm "
            f"{e['calm_mean']:.3f} min (+{e['excess_mean']:.3f} min). The AI removes **{100 * e['recovery']:.0f} %** "
            f"of it (95 % CI {100 * lo:.0f}..{100 * hi:.0f} %); §4.8 needs >= {100 * SHIP_EXCESS_WAIT_RECOVERY:.0f} %")


def render_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Subway Domino evaluation: Normal Schedule vs AI",
        "",
        f"- Policy: `{report['policy']}`",
        f"- Seeds: {s['n_seeds']} (`EVAL_SEED_BASE + i`, base {report['seed_base']}), graph `{report['graph_version']}`, "
        f"constants `{report['constants_hash'][:12]}`",
        f"- Seed win rate (AI total cost < Normal): **{100 * s['seed_win_rate']:.0f} %**; mean relative cost "
        f"reduction {100 * s['mean_rel_cost_reduction']:.1f} %",
        f"- `--require-win` gate (cost and >= 3/4 UI metrics): **{'PASS' if s['require_win_pass'] else 'FAIL'}**; "
        f"§4.8 acceptance bar: **{'PASS' if s['acceptance_bar']['pass'] else 'FAIL'}**",
        _excess_line(s),
        "",
        "| Metric | Normal (95 % CI) | AI (95 % CI) | Change | AI better |",
        "|---|---|---|---|---|",
    ]
    names = {"avg_wait_min": "Avg wait (min)", "overcrowded_platform_min": "Overcrowded platform-min",
             "tunnel_stuck_train_min": "Tunnel-stuck train-min", "riders_served": "Riders served",
             "total_cost": "Total cost"}
    for k, v in s["metrics"].items():
        n_ci, a_ci = v["normal_ci95"], v["ai_ci95"]
        lines.append(f"| {names.get(k, k)} | {_fmt(v['normal_mean'])} ({_fmt(n_ci[0])}..{_fmt(n_ci[1])}) | "
                     f"{_fmt(v['ai_mean'])} ({_fmt(a_ci[0])}..{_fmt(a_ci[1])}) | {v['pct_change']:+.1f} % | "
                     f"{'yes' if v['ai_better'] else 'no'} |")
    lines += ["", "Mean actions per AI episode: " + ", ".join(
        f"{k} {v:.1f}" for k, v in s["mean_actions_per_episode"].items()), "",
        "| Seed | Profile | Disruptions | Normal cost | AI cost | Change | Normal stuck min | AI stuck min |",
        "|---|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        nc, ac = r["normal"]["cost"]["total"], r["ai"]["cost"]["total"]
        lines.append(f"| {r['seed']} | {r['profile']} | {r['n_disruptions']} | {_fmt(nc)} | {_fmt(ac)} | "
                     f"{_pct_change(ac, nc):+.1f} % | {_fmt(r['normal']['tunnel_stuck_train_min'])} | "
                     f"{_fmt(r['ai']['tunnel_stuck_train_min'])} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Parallel driver
# --------------------------------------------------------------------------------------
_W: dict[str, Any] = {}


def _init_worker(policy_path: str, graph_path: str | None) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if policy_path.lower().endswith(".zip"):
        import torch  # local checkpoints only

        torch.set_num_threads(1)
    _W["graph"] = load_graph(graph_path)
    _W["policy"] = load_policy(policy_path)


def _run_seed(seed: int) -> dict[str, Any]:
    return evaluate_seed(_W["graph"], seed, _W["policy"].act)


def evaluate(policy_path: str | Path, seeds: Sequence[int], *, workers: int = 1, graph_path: str | None = None,
             progress: bool = True) -> list[dict[str, Any]]:
    """Rows for ``seeds`` (sorted by seed), optionally on ``workers`` spawned processes."""
    path = str(policy_path)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    if workers <= 1:
        _init_worker(path, graph_path)
        it = map(_run_seed, seeds)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(processes=workers, initializer=_init_worker, initargs=(path, graph_path))
        it = pool.imap_unordered(_run_seed, seeds)
    try:
        for r in it:
            rows.append(r)
            if progress:
                nc, ac = r["normal"]["cost"]["total"], r["ai"]["cost"]["total"]
                print(f"[{len(rows)}/{len(seeds)}] seed {r['seed']} {r['profile']:<10} cost {_pct_change(ac, nc):+6.1f} % "
                      f"stuck {r['normal']['tunnel_stuck_train_min']:7.1f} -> {r['ai']['tunnel_stuck_train_min']:7.1f} "
                      f"train-min ({time.perf_counter() - t0:.0f} s)", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return sorted(rows, key=lambda r: r["seed"])


def build_report(policy_path: str | Path, rows: list[dict[str, Any]], seed_base: int, graph: Graph) -> dict[str, Any]:
    pol = load_policy(policy_path)
    return {
        "policy": str(policy_path),
        "policy_metadata": dict(getattr(pol, "metadata", {}) or {}),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed_base": int(seed_base),
        "graph_version": graph.graph_version,
        "constants_hash": C.constants_hash(),
        "summary": summarize(rows),
        "rows": rows,
    }


def evaluation_summary(report: dict[str, Any], report_path: str | Path | None = None) -> dict[str, Any]:
    """The compact evaluation block stored in the model card (§4.9)."""
    s = report.get("summary", {})
    return {
        "n_seeds": s.get("n_seeds"), "seed_base": report.get("seed_base"), "policy": report.get("policy"),
        "created_utc": report.get("created_utc"), "seed_win_rate": s.get("seed_win_rate"),
        "mean_rel_cost_reduction": s.get("mean_rel_cost_reduction"), "require_win_pass": s.get("require_win_pass"),
        "acceptance_bar": s.get("acceptance_bar"),
        "metrics": {k: {kk: v.get(kk) for kk in ("normal_mean", "ai_mean", "pct_change", "ai_better")}
                    for k, v in s.get("metrics", {}).items()},
        "report": None if report_path is None else str(report_path),
    }


def update_model_card(card_path: Path, report: dict[str, Any], report_path: Path) -> None:
    """Merge the evaluation summary into an existing ``<model>.json`` model card."""
    card = json.loads(card_path.read_text())
    card["evaluation"] = evaluation_summary(report, report_path)
    _atomic_write(card_path, json.dumps(card, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help=".onnx (production) or .zip (MaskablePPO checkpoint)")
    ap.add_argument("--seeds", type=int, default=50, help="number of evaluation seeds")
    ap.add_argument("--seed-base", type=int, default=C.EVAL_SEED_BASE)
    ap.add_argument("--out", type=Path, default=Path("models"), help="output directory for the reports")
    ap.add_argument("--workers", default="auto", help="processes (int or 'auto' = cpu_count - 1)")
    ap.add_argument("--graph", default=None, help="network.json (default: data/network.json)")
    ap.add_argument("--require-win", action="store_true", help="exit 1 unless the AI wins (§4.7)")
    ap.add_argument("--no-model-card", action="store_true",
                    help="do not merge the summary into <policy>.json (done by default for .onnx policies)")
    args = ap.parse_args(argv)

    if args.seeds < 1:
        ap.error("--seeds must be >= 1")
    if str(args.workers).lower() == "auto":
        workers = max(1, min(args.seeds, (os.cpu_count() or 2) - 1))
    else:
        workers = max(1, int(args.workers))
    graph = load_graph(args.graph)
    seeds = [args.seed_base + i for i in range(args.seeds)]
    print(f"evaluating {args.policy} on {len(seeds)} seeds with {workers} worker(s)", flush=True)
    rows = evaluate(args.policy, seeds, workers=workers, graph_path=args.graph)
    report = build_report(args.policy, rows, args.seed_base, graph)
    paths = write_reports(args.out, report)
    s = report["summary"]
    print(render_markdown(report).split("\n\nMean actions")[0])
    print(f"wrote {', '.join(str(p) for p in paths.values())}")
    card = Path(args.policy).with_suffix(".json")
    if Path(args.policy).suffix.lower() == ".onnx" and card.is_file() and not args.no_model_card:
        update_model_card(card, report, paths["json"])
        print(f"updated the model card {card}")
    if args.require_win and not s["require_win_pass"]:
        print("--require-win: the AI does not beat Normal on mean total cost and >= 3 of 4 UI metrics", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
