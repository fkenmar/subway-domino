"""Segment polylines cut from GTFS shapes, and the borough basemap (§2.1.4, §2.3.10).

Each segment ``(a, b)`` (adjacent stations, ``a`` first in station order) takes its polyline from
the shape of a pattern that traverses the hop: patterns stopping at both ends first (their shapes
pass through both platforms), then by weekday trips. Every station of the pattern's elementary path
is located on the shape in order (forward search, so loops and parallel return tracks never swap
stations), the shape is cut between the two stations, oriented ``a -> b``, its endpoints are
snapped to the station points (so consecutive segments join exactly; the snap displacement is
blended over the last ~120 m so no short kinked leg remains) and it is simplified with
Douglas-Peucker at 2 m. A segment no shape can serve falls back to a straight line (recorded).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.netbuild.constants import (
    BASEMAP_MIN_RING_AREA_M2,
    BASEMAP_SIMPLIFY_EPS_M,
    SEGMENT_SIMPLIFY_EPS_M,
    SNAP_BLEND_M,
    STATION_SHAPE_MAX_DIST_M,
    TERMINAL_SHAPE_MAX_DIST_M,
)
from scripts.netbuild.geo import cumulative_length, cut, locate, polyline_length, project_latlon_array, ring_area, simplify
from scripts.netbuild.patterns import Pattern
from scripts.netbuild.topology import Segment

# A cut whose length exceeds this multiple of the straight distance (+ slack) is a mis-located
# station (e.g. a shape that loops back), never a real track.
_MAX_DETOUR = 3.0
_DETOUR_SLACK_M = 250.0


@dataclass
class SegmentGeometry:
    pts: dict[int, np.ndarray]  # segment idx -> simplified polyline a -> b (meters)
    length_m: dict[int, float]
    source: dict[int, str]  # segment idx -> "shape:<shape_id>" | "straight"
    warnings: list[str] = field(default_factory=list)


class _ShapeIndex:
    def __init__(self, shapes: Mapping[str, np.ndarray]) -> None:
        self.shapes = shapes
        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, shape_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        if shape_id not in self.shapes or len(self.shapes[shape_id]) < 2:
            return None
        if shape_id not in self.cache:
            xy = project_latlon_array(self.shapes[shape_id])
            self.cache[shape_id] = (xy, cumulative_length(xy))
        return self.cache[shape_id]


def _main_shape(p: Pattern, index: _ShapeIndex) -> str | None:
    counts = Counter(t.shape_id for t in p.trips if index.get(t.shape_id) is not None)
    if not counts:
        return None
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _positions(p: Pattern, shape: tuple[np.ndarray, np.ndarray], station_xy: Mapping[str, tuple[float, float]]
               ) -> list[float | None]:
    """Arc position of every station of the pattern's path on the shape (``None`` if off the shape).

    Stations are located in path order with a forward search. A terminal (first or last station)
    whose nearest shape point is the shape's own end may lie up to ``TERMINAL_SHAPE_MAX_DIST_M``
    away: shapes sometimes stop at the bumper block while the station point sits mid-platform.
    """
    pts, cum = shape
    out: list[float | None] = []
    s_min = 0.0
    last = len(p.path) - 1
    for k, s in enumerate(p.path):
        pos, dist = locate(pts, cum, station_xy[s], s_min)
        at_shape_end = (k == 0 and pos <= 1e-6) or (k == last and pos >= float(cum[-1]) - 1e-6)
        if dist <= STATION_SHAPE_MAX_DIST_M or (at_shape_end and dist <= TERMINAL_SHAPE_MAX_DIST_M):
            out.append(pos)
            s_min = pos
        else:
            out.append(None)
    return out


def snap_ends(piece: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    """Move the ends of ``piece`` onto ``a_xy`` / ``b_xy``, blending each displacement along the arc.

    The displacement of an end decays linearly to zero over ``max(SNAP_BLEND_M, 2 * |displacement|)``
    of arc (at most half the polyline), so vertices close to a station move with it instead of
    leaving a short kinked leg. Interior vertices farther away are untouched.
    """
    out = piece.astype(np.float64, copy=True)
    cum = cumulative_length(out)
    total = float(cum[-1])
    for disp, dist_from_end in ((a_xy - out[0], cum), (b_xy - out[-1], total - cum)):
        blend = min(max(SNAP_BLEND_M, 2.0 * float(np.hypot(*disp))), max(total / 2.0, 1e-9))
        weight = np.clip(1.0 - dist_from_end / blend, 0.0, 1.0)
        out += weight[:, None] * disp[None, :]
    out[0], out[-1] = a_xy, b_xy
    return out


def segment_geometry(
    segments: Sequence[Segment],
    kept: Sequence[Pattern],
    shapes: Mapping[str, np.ndarray],
    station_xy: Mapping[str, tuple[float, float]],
) -> SegmentGeometry:
    index = _ShapeIndex(shapes)
    seg_of: dict[tuple[str, str], int] = {}
    for s in segments:
        seg_of[(s.a, s.b)] = s.idx
        seg_of[(s.b, s.a)] = s.idx
    candidates: dict[int, list[tuple[tuple, Pattern, int]]] = {s.idx: [] for s in segments}
    for p in kept:
        stops = set(p.stops)
        for i, (u, v) in enumerate(p.path_hops()):
            si = seg_of.get((u, v))
            if si is not None:
                both = u in stops and v in stops
                candidates[si].append(((0 if both else 1, -p.n_trips, p.pid, i), p, i))
    shape_of: dict[int, str | None] = {}
    pos_cache: dict[int, list[float | None]] = {}
    pts_out: dict[int, np.ndarray] = {}
    length: dict[int, float] = {}
    source: dict[int, str] = {}
    warnings: list[str] = []
    for seg in segments:
        a_xy = np.asarray(station_xy[seg.a], dtype=np.float64)
        b_xy = np.asarray(station_xy[seg.b], dtype=np.float64)
        straight = float(np.hypot(*(b_xy - a_xy)))
        chosen: np.ndarray | None = None
        for _key, p, i in sorted(candidates[seg.idx], key=lambda c: c[0]):
            if p.pid not in shape_of:
                shape_of[p.pid] = _main_shape(p, index)
            sid = shape_of[p.pid]
            if sid is None:
                continue
            shape = index.get(sid)
            assert shape is not None
            if p.pid not in pos_cache:
                pos_cache[p.pid] = _positions(p, shape, station_xy)
            s0, s1 = pos_cache[p.pid][i], pos_cache[p.pid][i + 1]
            if s0 is None or s1 is None or s1 - s0 < 1.0:
                continue
            piece = cut(shape[0], shape[1], s0, s1)
            if polyline_length(piece) > _MAX_DETOUR * straight + _DETOUR_SLACK_M:
                continue
            if p.path[i] != seg.a:
                piece = piece[::-1].copy()
            chosen = snap_ends(piece, a_xy, b_xy)
            source[seg.idx] = f"shape:{sid}"
            break
        if chosen is None:
            chosen = np.vstack([a_xy, b_xy])
            source[seg.idx] = "straight"
            warnings.append(f"segment {seg.a}-{seg.b}: no shape covers it; straight line used")
        simple = simplify(chosen, SEGMENT_SIMPLIFY_EPS_M)
        pts_out[seg.idx] = simple
        length[seg.idx] = max(polyline_length(simple), 1.0)
    return SegmentGeometry(pts_out, length, source, warnings)


def polyline_point_at_half(pts: np.ndarray) -> tuple[float, float]:
    """Midpoint (by arc length) of a polyline."""
    cum = cumulative_length(pts)
    half = cum[-1] / 2.0
    i = int(np.searchsorted(cum, half, side="right") - 1)
    i = min(max(i, 0), len(pts) - 2)
    seg = cum[i + 1] - cum[i]
    t = 0.0 if seg <= 0 else (half - cum[i]) / seg
    return (float(pts[i, 0] + t * (pts[i + 1, 0] - pts[i, 0])), float(pts[i, 1] + t * (pts[i + 1, 1] - pts[i, 1])))


def basemap_land(geojson: Mapping[str, Any] | None) -> list[np.ndarray] | None:
    """Outer rings of the four in-scope boroughs, projected, simplified (25 m), small islands dropped."""
    if not geojson:
        return None
    rings: list[tuple[str, float, np.ndarray]] = []
    for feat in geojson.get("features", []):
        props = feat.get("properties") or {}
        name = str(props.get("boroname", ""))
        code = str(props.get("borocode", ""))
        if name == "Staten Island" or code == "5":
            continue
        geom = feat.get("geometry") or {}
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            if not poly or len(poly[0]) < 4:
                continue
            lonlat = np.asarray(poly[0], dtype=np.float64)[:, :2]
            xy = project_latlon_array(lonlat[:, ::-1])
            simple = simplify(xy, BASEMAP_SIMPLIFY_EPS_M)
            if len(simple) > 1 and np.allclose(simple[0], simple[-1]):
                simple = simple[:-1]
            if len(simple) < 3:
                continue
            area = abs(ring_area(simple))
            if area < BASEMAP_MIN_RING_AREA_M2:
                continue
            rings.append((code, -area, simple))
    rings.sort(key=lambda r: (r[0], r[1], float(r[2][0, 0]), float(r[2][0, 1])))
    return [r[2] for r in rings] or None


def bbox(arrays: Sequence[np.ndarray]) -> tuple[float, float, float, float]:
    allp = np.vstack([a for a in arrays if len(a)])
    return (float(allp[:, 0].min()), float(allp[:, 1].min()), float(allp[:, 0].max()), float(allp[:, 1].max()))


def hypot_len(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
