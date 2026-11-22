"""Loader and schema validation for the hand-curated ``data/infrastructure.json`` (§2.4).

The overlay carries only facts GTFS cannot express. Keys starting with ``_`` and ``note``
fields are documentation and are ignored by the build. Everything is validated here; the
references to stations, segments and interlockings are resolved (and fail the build when they
do not resolve) once the network exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.netbuild.constants import INFRASTRUCTURE_SCHEMA_VERSION, IN_SCOPE_ROUTES, PROFILES

DISRUPTION_KINDS = ("local_block", "express_block", "full_station_closure", "signal_delay")
TARGET_TYPES = ("station", "complex", "segment", "interlocking")
# Contract table §3.5: which target types each disruption kind accepts.
KIND_TARGETS = {
    "local_block": ("segment", "station", "interlocking"),
    "express_block": ("segment", "station", "interlocking"),
    "full_station_closure": ("station", "complex"),
    "signal_delay": ("segment", "interlocking"),
}
PEAK_VALUES = ("am_toward_manhattan", "am_away_from_manhattan")


class InfrastructureError(ValueError):
    """``infrastructure.json`` is malformed or inconsistent with the network."""


@dataclass(frozen=True)
class Corridor:
    """A station range along a route (CBTC territory, middle track, 3-track group)."""

    name: str
    routes: tuple[str, ...]
    from_gtfs: str
    to_gtfs: str
    peak: str = ""


@dataclass(frozen=True)
class TurnbackSpec:
    gtfs: str
    routes: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class YardSpec:
    name: str
    division: str
    lat: float
    lon: float
    lead_gtfs: tuple[str, ...]
    routes: tuple[str, ...]
    fleet_cap: int
    deadhead_s: int


@dataclass(frozen=True)
class RiverCrossing:
    a: str
    b: str
    name: str


@dataclass(frozen=True)
class DisruptionSpec:
    kind: str
    target: tuple[tuple[str, str], ...]  # sorted (key, value) pairs of the target object
    duration_s: float | None

    def target_dict(self) -> dict[str, str]:
        return dict(self.target)


@dataclass(frozen=True)
class PresetSpec:
    id: str
    title: str
    subtitle: str
    profile: str
    disruptions: tuple[DisruptionSpec, ...]
    zoom_m_per_px: float | None


@dataclass(frozen=True)
class Infrastructure:
    cbtc: tuple[Corridor, ...]
    middle_tracks: tuple[Corridor, ...]
    three_track_peak: tuple[Corridor, ...]
    turnbacks: tuple[TurnbackSpec, ...]
    yards: tuple[YardSpec, ...]
    river_crossings: tuple[RiverCrossing, ...]
    hubs_extra: tuple[str, ...]
    presets: tuple[PresetSpec, ...]


def _req(obj: Any, key: str, typ: type | tuple[type, ...], where: str) -> Any:
    if not isinstance(obj, dict):
        raise InfrastructureError(f"{where}: expected an object")
    if key not in obj:
        raise InfrastructureError(f"{where}: missing key {key!r}")
    value = obj[key]
    numeric = typ in (int, float) or (isinstance(typ, tuple) and set(typ) <= {int, float})
    if not isinstance(value, typ) or (numeric and isinstance(value, bool)):
        raise InfrastructureError(f"{where}: {key!r} must be {typ}, got {type(value).__name__}")
    return value


def _routes(obj: dict[str, Any], where: str) -> tuple[str, ...]:
    routes = _req(obj, "routes", list, where)
    if not routes or any(r not in IN_SCOPE_ROUTES for r in routes) or len(set(routes)) != len(routes):
        raise InfrastructureError(f"{where}: routes {routes} must be distinct in-scope route ids")
    return tuple(routes)


def _corridors(doc: dict[str, Any], key: str, *, needs_peak: bool) -> tuple[Corridor, ...]:
    out = []
    for i, c in enumerate(doc.get(key, [])):
        where = f"{key}[{i}]"
        peak = ""
        if needs_peak:
            peak = _req(c, "peak", str, where)
            if peak not in PEAK_VALUES:
                raise InfrastructureError(f"{where}: peak must be one of {PEAK_VALUES}")
        corridor = Corridor(
            name=_req(c, "name", str, where),
            routes=_routes(c, where),
            from_gtfs=_req(c, "from_gtfs", str, where),
            to_gtfs=_req(c, "to_gtfs", str, where),
            peak=peak,
        )
        if corridor.from_gtfs == corridor.to_gtfs:
            raise InfrastructureError(f"{where}: from_gtfs and to_gtfs must differ")
        out.append(corridor)
    return tuple(out)


def _target(t: Any, kind: str, where: str) -> tuple[tuple[str, str], ...]:
    typ = _req(t, "type", str, where)
    if typ not in TARGET_TYPES:
        raise InfrastructureError(f"{where}: target type {typ!r} not in {TARGET_TYPES}")
    if typ not in KIND_TARGETS[kind]:
        raise InfrastructureError(f"{where}: {kind} cannot target a {typ} (allowed: {KIND_TARGETS[kind]})")
    if typ == "segment":
        keys = ("a", "b")
    elif typ == "complex":
        keys = ("complex_id",) if "complex_id" in t else ("gtfs",)
    else:
        keys = ("gtfs",)
    values = {k: _req(t, k, str, where) for k in keys}
    return tuple(sorted({"type": typ, **values}.items()))


def parse_infrastructure(doc: dict[str, Any]) -> Infrastructure:
    if not isinstance(doc, dict) or doc.get("schema_version") != INFRASTRUCTURE_SCHEMA_VERSION:
        raise InfrastructureError(f"schema_version must be {INFRASTRUCTURE_SCHEMA_VERSION}")
    known = {
        "schema_version", "cbtc", "middle_tracks", "three_track_peak", "turnbacks", "yards",
        "river_crossings", "hubs_extra", "presets",
    }
    unknown = sorted(k for k in doc if k not in known and not k.startswith("_"))
    if unknown:
        raise InfrastructureError(f"unknown top-level keys {unknown}")

    turnbacks = []
    for i, t in enumerate(doc.get("turnbacks", [])):
        where = f"turnbacks[{i}]"
        turnbacks.append(TurnbackSpec(_req(t, "gtfs", str, where), _routes(t, where), str(t.get("note", ""))))

    yards = []
    for i, y in enumerate(doc.get("yards", [])):
        where = f"yards[{i}]"
        division = _req(y, "division", str, where)
        if division not in ("A", "B"):
            raise InfrastructureError(f"{where}: division must be 'A' or 'B'")
        fleet_cap = _req(y, "fleet_cap", int, where)
        deadhead = _req(y, "deadhead_s", int, where)
        if not 4 <= fleet_cap <= 8:
            raise InfrastructureError(f"{where}: fleet_cap {fleet_cap} outside 4..8")
        if not 300 <= deadhead <= 720:
            raise InfrastructureError(f"{where}: deadhead_s {deadhead} outside 300..720")
        lead = _req(y, "lead_gtfs", list, where)
        if not lead or not all(isinstance(s, str) for s in lead):
            raise InfrastructureError(f"{where}: lead_gtfs must be a non-empty list of GTFS station ids")
        lat = float(_req(y, "lat", (int, float), where))
        lon = float(_req(y, "lon", (int, float), where))
        if not (40.4 < lat < 41.0 and -74.3 < lon < -73.6):
            raise InfrastructureError(f"{where}: lat/lon ({lat}, {lon}) is outside New York City")
        yards.append(YardSpec(_req(y, "name", str, where), division, lat, lon, tuple(lead), _routes(y, where),
                              fleet_cap, deadhead))
    if len({y.name for y in yards}) != len(yards):
        raise InfrastructureError("yard names must be unique")

    crossings = []
    for i, c in enumerate(doc.get("river_crossings", [])):
        where = f"river_crossings[{i}]"
        crossings.append(RiverCrossing(_req(c, "a", str, where), _req(c, "b", str, where), _req(c, "name", str, where)))

    hubs_extra = doc.get("hubs_extra", [])
    if not isinstance(hubs_extra, list) or not all(isinstance(h, str) for h in hubs_extra):
        raise InfrastructureError("hubs_extra must be a list of GTFS station ids")

    presets = []
    seen_ids: set[str] = set()
    for i, p in enumerate(doc.get("presets", [])):
        where = f"presets[{i}]"
        pid = _req(p, "id", str, where)
        if pid in seen_ids:
            raise InfrastructureError(f"{where}: duplicate preset id {pid!r}")
        seen_ids.add(pid)
        profile = _req(p, "profile", str, where)
        if profile not in PROFILES:
            raise InfrastructureError(f"{where}: profile {profile!r} not in {PROFILES}")
        disruptions = []
        for j, d in enumerate(_req(p, "disruptions", list, where)):
            dwhere = f"{where}.disruptions[{j}]"
            kind = _req(d, "kind", str, dwhere)
            if kind not in DISRUPTION_KINDS:
                raise InfrastructureError(f"{dwhere}: kind {kind!r} not in {DISRUPTION_KINDS}")
            duration = d.get("duration_s")
            if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float))
                                         or duration <= 0):
                raise InfrastructureError(f"{dwhere}: duration_s must be a positive number or null")
            disruptions.append(DisruptionSpec(kind, _target(d.get("target"), kind, dwhere),
                                              None if duration is None else float(duration)))
        if not disruptions:
            raise InfrastructureError(f"{where}: needs at least one disruption")
        zoom = p.get("zoom_m_per_px")
        if zoom is not None and (isinstance(zoom, bool) or not isinstance(zoom, (int, float)) or zoom <= 0):
            raise InfrastructureError(f"{where}: zoom_m_per_px must be a positive number")
        presets.append(PresetSpec(pid, _req(p, "title", str, where), _req(p, "subtitle", str, where), profile,
                                  tuple(disruptions), None if zoom is None else float(zoom)))

    return Infrastructure(
        cbtc=_corridors(doc, "cbtc", needs_peak=False),
        middle_tracks=_corridors(doc, "middle_tracks", needs_peak=True),
        three_track_peak=_corridors(doc, "three_track_peak", needs_peak=True),
        turnbacks=tuple(turnbacks),
        yards=tuple(yards),
        river_crossings=tuple(crossings),
        hubs_extra=tuple(hubs_extra),
        presets=tuple(presets),
    )


def load_infrastructure(path: Path) -> Infrastructure:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InfrastructureError(f"cannot read {path}: {exc}") from exc
    return parse_infrastructure(doc)
