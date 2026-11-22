"""Resolve the curated meltdown presets onto network indices (ARCHITECTURE §2.4).

Targets are written with GTFS ids in ``infrastructure.json`` and become ``{"type", "idx"}``:

* ``station``: ``{"gtfs"}``; ``complex``: ``{"complex_id"}`` or ``{"gtfs"}`` of a member station;
* ``segment``: ``{"a", "b"}`` (adjacent stations, either order);
* ``interlocking``: ``{"gtfs"}`` of the interlocking's station.

An ``express_block`` must hit something with an express track. Any target that does not resolve
fails the build. ``focus`` is the mean of the targets' points with the preset's zoom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from scripts.netbuild.constants import COORD_DECIMALS, FOCUS_ZOOM_DEFAULT_M_PER_PX
from scripts.netbuild.infra import InfrastructureError, PresetSpec
from scripts.netbuild.jsonio import rnd


@dataclass(frozen=True)
class PresetIndex:
    station: Mapping[str, int]  # gtfs -> station idx
    complex_by_id: Mapping[str, int]
    station_complex: Sequence[int]
    segment: Mapping[tuple[int, int], int]  # (station idx, station idx) both orders -> segment idx
    interlocking_at: Mapping[int, int]  # station idx -> interlocking idx
    station_has_x: Sequence[bool]
    segment_has_x: Sequence[bool]
    station_xy: Sequence[tuple[float, float]]
    complex_xy: Sequence[tuple[float, float]]
    segment_mid: Sequence[tuple[float, float]]


def _station(ix: PresetIndex, gtfs: str, where: str) -> int:
    if gtfs not in ix.station:
        raise InfrastructureError(f"{where}: unknown station {gtfs!r}")
    return ix.station[gtfs]


def resolve_target(ix: PresetIndex, kind: str, target: Mapping[str, str], where: str) -> tuple[dict[str, Any], tuple[float, float]]:
    typ = target["type"]
    if typ == "station":
        s = _station(ix, target["gtfs"], where)
        if kind == "express_block" and not ix.station_has_x[s]:
            raise InfrastructureError(f"{where}: express_block on station {target['gtfs']} without an express track")
        return {"type": "station", "idx": s}, ix.station_xy[s]
    if typ == "complex":
        if "complex_id" in target:
            if target["complex_id"] not in ix.complex_by_id:
                raise InfrastructureError(f"{where}: unknown complex {target['complex_id']!r}")
            c = ix.complex_by_id[target["complex_id"]]
        else:
            c = ix.station_complex[_station(ix, target["gtfs"], where)]
        return {"type": "complex", "idx": c}, ix.complex_xy[c]
    if typ == "segment":
        a, b = _station(ix, target["a"], where), _station(ix, target["b"], where)
        if (a, b) not in ix.segment:
            raise InfrastructureError(f"{where}: stations {target['a']} and {target['b']} are not adjacent")
        seg = ix.segment[(a, b)]
        if kind == "express_block" and not ix.segment_has_x[seg]:
            raise InfrastructureError(f"{where}: express_block on segment {target['a']}-{target['b']} without express")
        return {"type": "segment", "idx": seg}, ix.segment_mid[seg]
    if typ == "interlocking":
        s = _station(ix, target["gtfs"], where)
        if s not in ix.interlocking_at:
            raise InfrastructureError(f"{where}: station {target['gtfs']} has no interlocking")
        if kind == "express_block" and not ix.station_has_x[s]:
            raise InfrastructureError(f"{where}: express_block on interlocking {target['gtfs']} without express")
        return {"type": "interlocking", "idx": ix.interlocking_at[s]}, ix.station_xy[s]
    raise InfrastructureError(f"{where}: unsupported target type {typ!r}")


def resolve_presets(presets: Sequence[PresetSpec], ix: PresetIndex) -> list[dict[str, Any]]:
    out = []
    for p in presets:
        disruptions = []
        points = []
        for j, d in enumerate(p.disruptions):
            target, pt = resolve_target(ix, d.kind, d.target_dict(), f"preset {p.id!r} disruption {j}")
            disruptions.append({"kind": d.kind, "target": target, "duration_s": d.duration_s})
            points.append(pt)
        fx = sum(x for x, _ in points) / len(points)
        fy = sum(y for _, y in points) / len(points)
        out.append({
            "id": p.id,
            "title": p.title,
            "subtitle": p.subtitle,
            "profile": p.profile,
            "disruptions": disruptions,
            "focus": {"x": rnd(fx, COORD_DECIMALS), "y": rnd(fy, COORD_DECIMALS),
                      "zoom_m_per_px": rnd(p.zoom_m_per_px or FOCUS_ZOOM_DEFAULT_M_PER_PX, 2)},
        })
    return out
