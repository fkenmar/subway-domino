"""Ridership profiles, platform arrival rates, destination attraction, capacity and hubs (§2.3.12-13, §2.5).

``entries[c, h]`` is the mean weekday entries of complex ``c`` in clock hour ``h``: the ``5wq4-mkjj``
aggregate ``r / days`` (one SoQL ``$group`` query, §2.1.3), or, when ridership is unavailable,
``base(c) * hourly_curve(h)`` with ``base = 900 * n_routes^1.3 * (2.5 if CBD) * borough_factor``
and the fixed system curve normalized to sum to 1.

* A profile's entries/h of a complex is the mean over the window's hours. It is split over the
  complex's platforms in proportion to the scheduled trains stopping at each platform in that
  window (a complex without trains in the window injects nobody).
* Attraction uses the mirror-hour rule: am_peak destinations are where people leave from in the
  pm_rush (and vice versa); midday and late_night use the same window. Station weights are the
  source window's complex entries split by trains; the vector is normalized to sum to 1.
* Platform capacity is ``clamp(350 + 0.30 * peak_hour_entries, 350, 3500)`` riders, where the
  peak hour is the platform's busiest hour of the day (entries split by that hour's trains).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.netbuild.constants import (
    ATTRACTION_SOURCE_PROFILE,
    CAPACITY_BASE,
    CAPACITY_MAX,
    CAPACITY_PER_PEAK_ENTRY,
    HOURLY_CURVE,
    HUB_MIN_ROUTES,
    HUB_TOP_N_COMPLEXES,
    PROFILES,
    SYNTH_BASE_ENTRIES,
    SYNTH_BOROUGH_FACTOR,
    SYNTH_CBD_FACTOR,
    SYNTH_ROUTE_EXPONENT,
)
from scripts.netbuild.places import ComplexInfo
from scripts.netbuild.service import window_hours


@dataclass
class Demand:
    entries: np.ndarray  # float64[n_complexes, 24] mean weekday entries per clock hour
    source: str  # "ridership" | "synthetic" | "mixed"
    platform_rate: dict[str, np.ndarray]  # profile -> riders/s per platform
    attraction: dict[str, np.ndarray]  # profile -> share per station (sums to 1)
    capacity: np.ndarray  # riders per platform
    station_entries: np.ndarray  # weekday entries attributed to each station
    unserved_entries_h: dict[str, float]  # profile -> entries/h of complexes without trains
    warnings: list[str] = field(default_factory=list)

    @property
    def weekday_entries(self) -> np.ndarray:
        return self.entries.sum(axis=1)

    def system_entries_h(self, profile: str) -> float:
        """Riders per hour the platforms of the network inject in ``profile``."""
        return float(self.platform_rate[profile].sum() * 3600.0)


def synthetic_entries(complexes: Sequence[ComplexInfo], n_routes: Sequence[int]) -> np.ndarray:
    curve = np.asarray(HOURLY_CURVE, dtype=np.float64)
    curve = curve / curve.sum()
    base = np.asarray([
        SYNTH_BASE_ENTRIES * max(n, 1) ** SYNTH_ROUTE_EXPONENT * (SYNTH_CBD_FACTOR if c.cbd else 1.0)
        * SYNTH_BOROUGH_FACTOR[c.borough]
        for c, n in zip(complexes, n_routes)
    ])
    return base[:, None] * curve[None, :]


def ridership_entries(
    complexes: Sequence[ComplexInfo], rows: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """``(entries[n, 24], found[n])`` from the aggregate rows (``r / days`` per complex and hour)."""
    index = {c.complex_id: i for i, c in enumerate(complexes)}
    entries = np.zeros((len(complexes), 24), dtype=np.float64)
    found = np.zeros(len(complexes), dtype=bool)
    for r in rows:
        ci = index.get(str(r.get("station_complex_id", "")))
        if ci is None:
            continue
        hh = int(float(r["hh"]))
        days = float(r["days"])
        if not 0 <= hh < 24 or days <= 0:
            continue
        entries[ci, hh] += float(r["r"]) / days
        found[ci] = True
    return entries, found


def _shares(weights: np.ndarray, groups: np.ndarray, n_groups: int, fallback_equal: bool) -> np.ndarray:
    """Normalize ``weights`` within each group (``groups[i]`` = group of item i)."""
    totals = np.bincount(groups, weights=weights, minlength=n_groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(totals[groups] > 0, weights / np.where(totals[groups] > 0, totals[groups], 1.0), 0.0)
    if fallback_equal:
        counts = np.bincount(groups, minlength=n_groups).astype(np.float64)
        empty = totals[groups] <= 0
        share[empty] = 1.0 / counts[groups[empty]]
    return share


def build_demand(
    complexes: Sequence[ComplexInfo],
    station_complex: np.ndarray,  # int[n_stations]
    platform_station: np.ndarray,  # int[n_platforms]
    stop_counts: np.ndarray,  # int[n_platforms, 24]
    complex_routes: Sequence[int],
    ridership_rows: Sequence[Mapping[str, Any]] | None,
) -> Demand:
    warnings: list[str] = []
    n_c, n_s, n_p = len(complexes), len(station_complex), len(platform_station)
    synth = synthetic_entries(complexes, complex_routes)
    if ridership_rows:
        entries, found = ridership_entries(complexes, ridership_rows)
        if not found.all():
            missing = [complexes[i].complex_id for i in np.nonzero(~found)[0]]
            warnings.append(f"no ridership rows for complexes {missing}; synthesized their entries")
            entries[~found] = synth[~found]
        source = "ridership" if found.all() else "mixed"
    else:
        entries, source = synth, "synthetic"

    platform_complex = station_complex[platform_station]
    counts = stop_counts.astype(np.float64)

    platform_rate: dict[str, np.ndarray] = {}
    unserved: dict[str, float] = {}
    trains_in: dict[str, np.ndarray] = {}
    for prof in PROFILES:
        hours = window_hours(prof)
        trains = counts[:, hours].sum(axis=1)
        trains_in[prof] = trains
        share = _shares(trains, platform_complex, n_c, fallback_equal=False)
        mean_entries = entries[:, hours].mean(axis=1)
        platform_rate[prof] = mean_entries[platform_complex] * share / 3600.0
        served = np.bincount(platform_complex, weights=trains, minlength=n_c) > 0
        unserved[prof] = float(mean_entries[~served].sum())

    attraction: dict[str, np.ndarray] = {}
    for prof in PROFILES:
        src = ATTRACTION_SOURCE_PROFILE[prof]
        station_trains = np.bincount(platform_station, weights=trains_in[src], minlength=n_s)
        share = _shares(station_trains, station_complex, n_c, fallback_equal=True)
        weight = entries[:, window_hours(src)].mean(axis=1)[station_complex] * share
        total = weight.sum()
        attraction[prof] = weight / total if total > 0 else np.full(n_s, 1.0 / n_s)

    peak = np.zeros(n_p, dtype=np.float64)
    station_entries = np.zeros(n_s, dtype=np.float64)
    for h in range(24):
        pshare = _shares(counts[:, h], platform_complex, n_c, fallback_equal=False)
        peak = np.maximum(peak, entries[platform_complex, h] * pshare)
        st_trains = np.bincount(platform_station, weights=counts[:, h], minlength=n_s)
        sshare = _shares(st_trains, station_complex, n_c, fallback_equal=True)
        station_entries += entries[station_complex, h] * sshare
    capacity = np.clip(CAPACITY_BASE + CAPACITY_PER_PEAK_ENTRY * peak, CAPACITY_BASE, CAPACITY_MAX)

    return Demand(
        entries=entries,
        source=source,
        platform_rate=platform_rate,
        attraction=attraction,
        capacity=capacity,
        station_entries=station_entries,
        unserved_entries_h=unserved,
        warnings=warnings,
    )


def hub_stations(
    weekday_entries: np.ndarray,
    station_complex: np.ndarray,
    complex_route_count: Sequence[int],
    station_gtfs: Sequence[str],
    hubs_extra: Sequence[str],
) -> np.ndarray:
    """§2.5: top-N complex by weekday entries, OR complex serves >= 4 routes, OR listed in hubs_extra."""
    order = sorted(range(len(weekday_entries)), key=lambda c: (-float(weekday_entries[c]), c))
    top = np.zeros(len(weekday_entries), dtype=bool)
    top[order[:HUB_TOP_N_COMPLEXES]] = True
    many = np.asarray(complex_route_count) >= HUB_MIN_ROUTES
    hub = top[station_complex] | many[station_complex]
    extra = set(hubs_extra)
    unknown = extra - set(station_gtfs)
    if unknown:
        raise ValueError(f"hubs_extra references unknown stations {sorted(unknown)}")
    hub |= np.asarray([g in extra for g in station_gtfs], dtype=bool)
    return hub


def label_ranks(station_entries: np.ndarray, complex_entries: np.ndarray) -> np.ndarray:
    """0 = most important station: by the weekday entries of its complex (what the ridership data
    measures), then by the station's own share of them, then by station index.

    Ranking by the station share alone would put a single-station complex (34 St-Penn Station)
    ahead of the five stations that split Times Sq-42 St's entries.
    """
    order = sorted(range(len(station_entries)), key=lambda s: (-round(float(complex_entries[s]), 3),
                                                               -round(float(station_entries[s]), 3), s))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order))
    return ranks
