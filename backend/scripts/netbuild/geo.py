"""Local projection and polyline utilities (NumPy only).

Coordinates are meters in the local equirectangular projection of ARCHITECTURE §2.6:
``x = (lon - LON0) * KX``, ``y = (lat - LAT0) * KY``.
"""

from __future__ import annotations

from typing import overload

import numpy as np

from scripts.netbuild.constants import KX, KY, LAT0, LON0

FloatArray = np.ndarray


@overload
def project(lat: float, lon: float) -> tuple[float, float]: ...
@overload
def project(lat: FloatArray, lon: FloatArray) -> tuple[FloatArray, FloatArray]: ...
def project(lat, lon):  # type: ignore[no-untyped-def]
    """Project WGS84 degrees to local meters."""
    if isinstance(lat, np.ndarray):
        return (lon - LON0) * KX, (lat - LAT0) * KY
    return (float(lon) - LON0) * KX, (float(lat) - LAT0) * KY


def project_latlon_array(latlon: FloatArray) -> FloatArray:
    """``(n, 2)`` array of ``[lat, lon]`` to ``(n, 2)`` array of ``[x, y]``."""
    out = np.empty_like(latlon, dtype=np.float64)
    out[:, 0] = (latlon[:, 1] - LON0) * KX
    out[:, 1] = (latlon[:, 0] - LAT0) * KY
    return out


def cumulative_length(pts: FloatArray) -> FloatArray:
    if len(pts) == 0:
        return np.zeros(0)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    return np.concatenate(([0.0], np.cumsum(seg)))


def polyline_length(pts: FloatArray) -> float:
    return float(cumulative_length(pts)[-1]) if len(pts) > 1 else 0.0


def locate(pts: FloatArray, cum: FloatArray, p: tuple[float, float], s_min: float = 0.0) -> tuple[float, float]:
    """Arc-length position and distance of the point of ``pts`` closest to ``p`` at arc >= ``s_min``.

    Searching only forward of ``s_min`` keeps successive stations in order on shapes that
    pass the same place twice (loops, parallel return tracks).
    """
    a = pts[:-1]
    d = pts[1:] - a
    seg_len = np.hypot(d[:, 0], d[:, 1])
    px, py = p
    safe = np.where(seg_len > 0, seg_len, 1.0)
    t = ((px - a[:, 0]) * d[:, 0] + (py - a[:, 1]) * d[:, 1]) / (safe * safe)
    # Only the part of each segment at arc >= s_min is eligible.
    t_lo = np.clip((s_min - cum[:-1]) / safe, 0.0, 1.0)
    t = np.where(seg_len > 0, np.clip(t, t_lo, 1.0), 0.0)
    eligible = cum[1:] >= s_min
    dist = np.hypot(a[:, 0] + t * d[:, 0] - px, a[:, 1] + t * d[:, 1] - py)
    dist = np.where(eligible, dist, np.inf)
    i = int(np.argmin(dist))
    return float(cum[i] + t[i] * seg_len[i]), float(dist[i])


def point_at(pts: FloatArray, cum: FloatArray, s: float) -> tuple[float, float]:
    s = min(max(s, 0.0), float(cum[-1]))
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = min(max(i, 0), len(pts) - 2)
    seg = cum[i + 1] - cum[i]
    t = 0.0 if seg <= 0 else (s - cum[i]) / seg
    return float(pts[i, 0] + t * (pts[i + 1, 0] - pts[i, 0])), float(pts[i, 1] + t * (pts[i + 1, 1] - pts[i, 1]))


def cut(pts: FloatArray, cum: FloatArray, s0: float, s1: float) -> FloatArray:
    """Sub-polyline between arc positions ``s0 < s1`` (endpoints interpolated)."""
    if s1 < s0:
        raise ValueError("cut requires s0 <= s1")
    inner = np.nonzero((cum > s0) & (cum < s1))[0]
    start = point_at(pts, cum, s0)
    end = point_at(pts, cum, s1)
    out = [start, *[(float(pts[i, 0]), float(pts[i, 1])) for i in inner], end]
    return np.asarray(out, dtype=np.float64)


def _segment_distances(pts: FloatArray, a: FloatArray, b: FloatArray) -> FloatArray:
    d = b - a
    l2 = float(d @ d)
    if l2 == 0.0:
        return np.hypot(pts[:, 0] - a[0], pts[:, 1] - a[1])
    t = np.clip(((pts[:, 0] - a[0]) * d[0] + (pts[:, 1] - a[1]) * d[1]) / l2, 0.0, 1.0)
    return np.hypot(pts[:, 0] - (a[0] + t * d[0]), pts[:, 1] - (a[1] + t * d[1]))


def simplify(pts: FloatArray, eps: float) -> FloatArray:
    """Douglas-Peucker (iterative, segment distance, deterministic). Keeps both endpoints."""
    n = len(pts)
    if n <= 2:
        return pts.copy()
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        dist = _segment_distances(pts[i + 1 : j], pts[i], pts[j])
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return pts[keep]


def ring_area(ring: FloatArray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def point_in_ring(x: float, y: float, ring: FloatArray) -> bool:
    """Even-odd rule point-in-polygon test for one closed ring."""
    xs, ys = ring[:, 0], ring[:, 1]
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    cond = (ys > y) != (yj > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (xj - xs) * (y - ys) / (yj - ys) + xs
    return bool(np.count_nonzero(cond & (x < xint)) % 2 == 1)


def bearing_vector(pts: FloatArray) -> tuple[float, float]:
    d = pts[-1] - pts[0]
    n = float(np.hypot(d[0], d[1]))
    return (0.0, 0.0) if n == 0 else (float(d[0] / n), float(d[1] / n))
