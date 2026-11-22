"""Station metadata (division, borough, CBD, structure) and station complexes (§0, §2.1.2).

Complexes come from the ``5f5g-n3cz`` dataset (rows mapped to GTFS parent stations through
``gtfs_stop_ids``). When it is unavailable they are derived deterministically from GTFS parent
stations unioned through ``transfers.txt`` pairs with ``from != to``; a derived complex id is
``G<smallest parent id>`` (§2.1 fallback).

Borough, CBD flag and structure come from the first source that has them: the per-station
``39hk-dx4f`` dataset, the complex row, the basemap borough polygons, and finally a coarse
documented geographic rule (Brooklyn and Queens share a synthetic-ridership factor, so the
rule only has to be exact for Manhattan, the Bronx and the CBD).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.netbuild.geo import point_in_ring, project
from scripts.netbuild.gtfs import GtfsFeed

BOROUGH_CODES = {"Manhattan": "M", "Brooklyn": "Bk", "Queens": "Q", "Bronx": "Bx", "Staten Island": "SI"}

# Coarse Manhattan outline (lat, lon), clockwise from the Battery; includes Roosevelt Island and
# Marble Hill. Used only when neither the station datasets nor the basemap are available.
_MANHATTAN_OUTLINE = (
    (40.700, -74.020), (40.710, -74.020), (40.752, -74.010), (40.770, -73.995), (40.800, -73.975),
    (40.835, -73.951), (40.865, -73.935), (40.880, -73.927), (40.880, -73.905), (40.855, -73.915),
    (40.835, -73.930), (40.810, -73.932), (40.795, -73.927), (40.775, -73.935), (40.757, -73.945),
    (40.745, -73.970), (40.730, -73.970), (40.710, -73.975), (40.705, -73.995), (40.700, -74.010),
)
# Brooklyn/Queens boundary approximated by a polyline (lat, lon) from Newtown Creek southward.
_BK_Q_BOUNDARY = ((40.739, -73.962), (40.700, -73.906), (40.690, -73.866), (40.580, -73.855))
_BRONX_MIN_LAT = 40.790
# CBD (Manhattan south of 61 St): south of the line through these two points.
_CBD_LINE = ((40.7691, -73.9819), (40.7602, -73.9585))


class PlacesError(ValueError):
    """Station or complex data is inconsistent with the GTFS network."""


@dataclass(frozen=True)
class StationMeta:
    gtfs: str
    name: str
    lat: float
    lon: float
    division: str
    borough: str
    cbd: bool
    structure: str


@dataclass
class ComplexInfo:
    complex_id: str
    name: str
    borough: str
    lat: float
    lon: float
    cbd: bool
    stations: list[str]


@dataclass
class Places:
    stations: dict[str, StationMeta]
    complexes: list[ComplexInfo]
    complex_of: dict[str, int]
    complex_source: str  # "dataset" | "derived"
    meta_source: str  # "stations_dataset" | "complexes_dataset" | "basemap" | "coarse"
    warnings: list[str] = field(default_factory=list)


def division_of(stop_id: str) -> str:
    """A division (IRT) stop ids are all numeric (§0)."""
    return "A" if stop_id.isdigit() else "B"


def natural_key(value: str) -> tuple[int, int, str]:
    return (0, int(value), "") if value.isdigit() else (1, 0, value)


def _split_ids(value: Any) -> list[str]:
    return [v.strip() for v in str(value or "").split(";") if v.strip()]


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


# --------------------------------------------------------------------------- geography fallbacks


def _coarse_borough(lat: float, lon: float) -> str:
    ring = np.asarray([(lo, la) for la, lo in _MANHATTAN_OUTLINE], dtype=np.float64)
    if point_in_ring(lon, lat, ring):
        return "M"
    if lat >= _BRONX_MIN_LAT:
        return "Bx"
    pts = _BK_Q_BOUNDARY
    if lat >= pts[0][0]:
        return "Q"
    for (la0, lo0), (la1, lo1) in zip(pts, pts[1:]):
        if la1 <= lat <= la0:
            t = (la0 - lat) / (la0 - la1)
            return "Bk" if lon < lo0 + t * (lo1 - lo0) else "Q"
    return "Bk" if lon < pts[-1][1] else "Q"


def _coarse_cbd(lat: float, lon: float, borough: str) -> bool:
    if borough != "M":
        return False
    (la0, lo0), (la1, lo1) = _CBD_LINE
    lat_line = la0 + (lon - lo0) * (la1 - la0) / (lo1 - lo0)
    return lat < lat_line


class _BasemapBoroughs:
    """Point-in-polygon borough lookup on the ``gthc-hcne`` borough polygons."""

    def __init__(self, basemap: Mapping[str, Any]) -> None:
        self.rings: list[tuple[str, np.ndarray, list[np.ndarray]]] = []
        for feat in basemap.get("features", []):
            props = feat.get("properties") or {}
            code = BOROUGH_CODES.get(str(props.get("boroname", "")))
            geom = feat.get("geometry") or {}
            if code is None or geom.get("type") not in ("Polygon", "MultiPolygon"):
                continue
            polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            for poly in polys:
                rings = [np.asarray(r, dtype=np.float64)[:, :2] for r in poly if len(r) >= 4]
                if rings:
                    self.rings.append((code, rings[0], rings[1:]))

    def borough(self, lat: float, lon: float) -> str | None:
        for code, outer, holes in self.rings:
            if point_in_ring(lon, lat, outer) and not any(point_in_ring(lon, lat, h) for h in holes):
                return code
        return None


# --------------------------------------------------------------------------- builder


def build_places(
    feed: GtfsFeed,
    in_scope: Sequence[str],
    complexes_rows: list[dict[str, Any]] | None,
    stations_rows: list[dict[str, Any]] | None,
    basemap: Mapping[str, Any] | None,
) -> Places:
    """Station metadata and complexes for the in-scope stations (sorted by ``natural_key``)."""
    warnings: list[str] = []
    scope = set(in_scope)
    by_station_row: dict[str, dict[str, Any]] = {}
    for r in stations_rows or []:
        sid = str(r.get("gtfs_stop_id", "")).strip()
        if sid in scope:
            by_station_row[sid] = r
    cx_rows: dict[str, tuple[dict[str, Any], int]] = {}
    for r in complexes_rows or []:
        for k, sid in enumerate(_split_ids(r.get("gtfs_stop_ids"))):
            if sid in scope:
                cx_rows[sid] = (r, k)
    lookup = _BasemapBoroughs(basemap) if basemap else None

    sources_used: set[str] = set()
    stations: dict[str, StationMeta] = {}
    for sid in sorted(scope, key=natural_key):
        st = feed.stations[sid]
        borough = cbd = structure = None
        if sid in by_station_row:
            row = by_station_row[sid]
            borough, cbd, structure = str(row.get("borough", "")), _truthy(row.get("cbd")), str(row.get("structure", ""))
            sources_used.add("stations_dataset")
        elif sid in cx_rows:
            row, k = cx_rows[sid]
            borough, cbd = str(row.get("borough", "")), _truthy(row.get("cbd"))
            parts = [p.strip() for p in str(row.get("structure_type", "")).split(";")]
            structure = parts[k] if len(parts) == len(_split_ids(row.get("gtfs_stop_ids"))) else parts[0]
            sources_used.add("complexes_dataset")
        else:
            b = lookup.borough(st.lat, st.lon) if lookup else None
            if b is not None:
                sources_used.add("basemap")
            else:
                b = _coarse_borough(st.lat, st.lon)
                sources_used.add("coarse")
            borough, cbd, structure = b, _coarse_cbd(st.lat, st.lon, b), "Unknown"
        if borough not in ("M", "Bk", "Q", "Bx"):
            raise PlacesError(f"station {sid} ({st.name}) is in borough {borough!r}, outside the scope")
        stations[sid] = StationMeta(sid, st.name, st.lat, st.lon, division_of(sid), borough, bool(cbd),
                                    structure or "Unknown")
    meta_source = next(s for s in ("coarse", "basemap", "complexes_dataset", "stations_dataset")
                       if s in sources_used) if sources_used else "stations_dataset"

    if complexes_rows:
        complexes, source = _dataset_complexes(complexes_rows, stations, warnings), "dataset"
    else:
        complexes, source = _derived_complexes(feed, stations), "derived"
    complexes.sort(key=lambda c: natural_key(c.complex_id))
    complex_of: dict[str, int] = {}
    for ci, c in enumerate(complexes):
        c.stations.sort(key=natural_key)
        for sid in c.stations:
            if sid in complex_of:
                raise PlacesError(f"station {sid} belongs to complexes {complexes[complex_of[sid]].complex_id} "
                                  f"and {c.complex_id}")
            complex_of[sid] = ci
    missing = sorted(scope - set(complex_of), key=natural_key)
    if missing:
        raise PlacesError(f"stations without a complex: {missing}")
    return Places(stations, complexes, complex_of, source, meta_source, warnings)


def _dataset_complexes(rows: list[dict[str, Any]], stations: Mapping[str, StationMeta],
                       warnings: list[str]) -> list[ComplexInfo]:
    out: list[ComplexInfo] = []
    covered: set[str] = set()
    for r in rows:
        if str(r.get("borough", "")) == "SI":
            continue
        ids = [s for s in _split_ids(r.get("gtfs_stop_ids")) if s in stations]
        if not ids:
            continue
        lat = float(r.get("latitude") or np.mean([stations[s].lat for s in ids]))
        lon = float(r.get("longitude") or np.mean([stations[s].lon for s in ids]))
        out.append(ComplexInfo(
            complex_id=str(r["complex_id"]),
            name=str(r.get("stop_name") or stations[ids[0]].name),
            borough=str(r.get("borough") or stations[ids[0]].borough),
            lat=lat, lon=lon,
            cbd=_truthy(r.get("cbd")),
            stations=ids,
        ))
        covered.update(ids)
    for sid in sorted(set(stations) - covered, key=natural_key):
        st = stations[sid]
        warnings.append(f"station {sid} ({st.name}) is in no 5f5g-n3cz complex; created complex G{sid}")
        out.append(ComplexInfo(f"G{sid}", st.name, st.borough, st.lat, st.lon, st.cbd, [sid]))
    return out


def _derived_complexes(feed: GtfsFeed, stations: Mapping[str, StationMeta]) -> list[ComplexInfo]:
    parent = {s: s for s in stations}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in feed.transfers:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                lo, hi = sorted((ra, rb), key=natural_key)
                parent[hi] = lo
    groups: dict[str, list[str]] = {}
    for s in sorted(stations, key=natural_key):
        groups.setdefault(find(s), []).append(s)
    out = []
    for root, ids in groups.items():
        first = stations[min(ids, key=natural_key)]
        out.append(ComplexInfo(
            complex_id=f"G{first.gtfs}",
            name=first.name,
            borough=first.borough,
            lat=float(np.mean([stations[s].lat for s in ids])),
            lon=float(np.mean([stations[s].lon for s in ids])),
            cbd=any(stations[s].cbd for s in ids),
            stations=ids,
        ))
    return out


def complex_xy(c: ComplexInfo) -> tuple[float, float]:
    return project(c.lat, c.lon)
