"""Parse the MTA GTFS static feed into weekday trips, parent stations and shapes (§2.2).

Only ``service_id == "Weekday"`` trips of in-scope routes are kept, and trips that start
at or after 24:00 are dropped from the cyclic day. Stop ids are mapped to their parent
station: physical direction is later derived from station order, never from the
``N``/``S`` suffix or ``direction_id`` (CLAUDE.md hard rule).
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from scripts.netbuild.constants import DAY_S, GTFS_REQUIRED_MEMBERS, IN_SCOPE_ROUTES, WEEKDAY_SERVICE_ID


class GtfsError(ValueError):
    """The GTFS archive is missing members or has inconsistent references."""


@dataclass(frozen=True)
class GtfsRoute:
    route_id: str
    short_name: str
    long_name: str
    color: str
    text_color: str
    sort_order: int


@dataclass(frozen=True)
class GtfsStation:
    stop_id: str
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class GtfsTrip:
    trip_id: str
    route_id: str
    shape_id: str
    stations: tuple[str, ...]
    arr: tuple[int, ...]
    dep: tuple[int, ...]

    @property
    def start_s(self) -> int:
        return self.dep[0]


@dataclass
class GtfsFeed:
    routes: dict[str, GtfsRoute]
    stations: dict[str, GtfsStation]
    trips: list[GtfsTrip]
    shapes: dict[str, np.ndarray]
    transfers: list[tuple[str, str]]
    feed_info: dict[str, str]
    stats: dict[str, int] = field(default_factory=dict)


def validate_gtfs_zip(data: bytes) -> list[str]:
    """Return the list of problems that make ``data`` unusable as the subway GTFS zip."""
    if not data.startswith(b"PK\x03\x04"):
        return ["not a zip archive (bad magic number)"]
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            bad = zf.testzip()
    except zipfile.BadZipFile as exc:
        return [f"corrupt zip: {exc}"]
    problems = [f"missing member {m}" for m in GTFS_REQUIRED_MEMBERS if m not in names]
    if bad is not None:
        problems.append(f"corrupt member {bad}")
    return problems


def _read_csv(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with zf.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(text)]


def _hms(value: str) -> int:
    h, m, s = value.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_gtfs(data: bytes) -> GtfsFeed:
    problems = validate_gtfs_zip(data)
    if problems:
        raise GtfsError("; ".join(problems))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        routes = {
            r["route_id"]: GtfsRoute(
                route_id=r["route_id"],
                short_name=r.get("route_short_name", ""),
                long_name=r.get("route_long_name", ""),
                color=(r.get("route_color") or "808183").upper(),
                text_color=(r.get("route_text_color") or "FFFFFF").upper(),
                sort_order=int(r["route_sort_order"]) if r.get("route_sort_order") else 999,
            )
            for r in _read_csv(zf, "routes.txt")
        }
        stops = _read_csv(zf, "stops.txt")
        parent_of: dict[str, str] = {}
        stations: dict[str, GtfsStation] = {}
        for s in stops:
            sid = s["stop_id"]
            parent = s.get("parent_station", "")
            parent_of[sid] = parent or sid
            if s.get("location_type") == "1" or not parent:
                stations[sid] = GtfsStation(sid, s["stop_name"], float(s["stop_lat"]), float(s["stop_lon"]))
        trip_rows = _read_csv(zf, "trips.txt")
        wanted = {
            t["trip_id"]: t
            for t in trip_rows
            if t["service_id"] == WEEKDAY_SERVICE_ID and t["route_id"] in IN_SCOPE_ROUTES
        }
        stop_rows: dict[str, list[tuple[int, str, int, int]]] = defaultdict(list)
        with zf.open("stop_times.txt") as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            header = [h.strip() for h in next(reader)]
            i_trip = header.index("trip_id")
            i_stop = header.index("stop_id")
            i_arr = header.index("arrival_time")
            i_dep = header.index("departure_time")
            i_seq = header.index("stop_sequence")
            for row in reader:
                tid = row[i_trip]
                if tid not in wanted:
                    continue
                arr = row[i_arr].strip() or row[i_dep].strip()
                dep = row[i_dep].strip() or arr
                stop_rows[tid].append((int(row[i_seq]), row[i_stop].strip(), _hms(arr), _hms(dep)))
        trips: list[GtfsTrip] = []
        dropped_late = 0
        for tid in sorted(wanted):
            rows = sorted(stop_rows.get(tid, ()))
            if len(rows) < 2:
                continue
            if rows[0][3] >= DAY_S:
                dropped_late += 1
                continue
            unknown = [r[1] for r in rows if r[1] not in parent_of]
            if unknown:
                raise GtfsError(f"trip {tid} references unknown stops {unknown[:3]}")
            t = wanted[tid]
            trips.append(
                GtfsTrip(
                    trip_id=tid,
                    route_id=t["route_id"],
                    shape_id=t.get("shape_id", ""),
                    stations=tuple(parent_of[r[1]] for r in rows),
                    arr=tuple(r[2] for r in rows),
                    dep=tuple(r[3] for r in rows),
                )
            )
        used_shapes = {t.shape_id for t in trips if t.shape_id}
        shape_pts: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        with zf.open("shapes.txt") as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            header = [h.strip() for h in next(reader)]
            i_id = header.index("shape_id")
            i_seq = header.index("shape_pt_sequence")
            i_lat = header.index("shape_pt_lat")
            i_lon = header.index("shape_pt_lon")
            for row in reader:
                sid = row[i_id].strip()
                if sid in used_shapes:
                    shape_pts[sid].append((int(row[i_seq]), float(row[i_lat]), float(row[i_lon])))
        shapes = {
            sid: np.asarray([(lat, lon) for _, lat, lon in sorted(pts)], dtype=np.float64)
            for sid, pts in sorted(shape_pts.items())
        }
        transfers: list[tuple[str, str]] = []
        if "transfers.txt" in names:
            for r in _read_csv(zf, "transfers.txt"):
                a, b = parent_of.get(r["from_stop_id"]), parent_of.get(r["to_stop_id"])
                if a and b and a != b:
                    transfers.append((a, b))
        feed_info = _read_csv(zf, "feed_info.txt")[0] if "feed_info.txt" in names else {}
    stats = {
        "weekday_trips_in_scope": len(trips),
        "weekday_trips_dropped_after_24h": dropped_late,
        "shapes_used": len(shapes),
    }
    return GtfsFeed(
        routes=routes,
        stations=stations,
        trips=trips,
        shapes=shapes,
        transfers=sorted(set(transfers)),
        feed_info=dict(sorted(feed_info.items())),
        stats=stats,
    )
