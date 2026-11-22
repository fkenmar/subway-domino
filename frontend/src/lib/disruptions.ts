/**
 * Disruption targeting rules (§3.5 — CONTRACT semantics) for the UI: which
 * kinds are valid on which target, human labels, and where to anchor overlays.
 */
import { pointAt, type PointOnLine } from "./geometry";
import type { Network } from "./network";
import type { DisruptionInfo, DisruptionKind, DisruptionTarget } from "./protocol";

export const DURATION_CHOICES: readonly { label: string; seconds: number | null }[] = [
  { label: "Until cleared", seconds: null },
  { label: "10 min", seconds: 600 },
  { label: "20 min", seconds: 1200 },
  { label: "30 min", seconds: 1800 },
];

/**
 * Kinds valid for a target, in UI order (most dramatic first):
 * - station: closure, local block, express block (only with X track)
 * - complex: closure
 * - segment: local block, express block (only if the segment has express), signal delay
 * - interlocking: local block, express block (only with X track), signal delay
 */
export function validKinds(net: Network, target: DisruptionTarget): DisruptionKind[] {
  switch (target.type) {
    case "station": {
      if (!net.stations[target.idx]) return [];
      const out: DisruptionKind[] = ["full_station_closure", "local_block"];
      if (net.stationHasExpress[target.idx]) out.push("express_block");
      return out;
    }
    case "complex":
      return net.complexes[target.idx] ? ["full_station_closure"] : [];
    case "segment": {
      const seg = net.segments[target.idx];
      if (!seg) return [];
      const out: DisruptionKind[] = ["local_block"];
      if (seg.has_express || net.segmentHasXEdge[seg.idx]) out.push("express_block");
      out.push("signal_delay");
      return out;
    }
    case "interlocking": {
      const il = net.interlockings[target.idx];
      if (!il) return [];
      const out: DisruptionKind[] = ["local_block"];
      if (net.stationHasExpress[il.station]) out.push("express_block");
      out.push("signal_delay");
      return out;
    }
    default:
      return [];
  }
}

export function isValidDisruption(net: Network, kind: DisruptionKind, target: DisruptionTarget): boolean {
  return validKinds(net, target).includes(kind);
}

/** Short human label for a target. */
export function targetLabel(net: Network, target: DisruptionTarget): string {
  switch (target.type) {
    case "station":
      return net.stations[target.idx]?.name ?? `Station ${target.idx}`;
    case "complex":
      return net.complexes[target.idx]?.name ?? `Complex ${target.idx}`;
    case "segment": {
      const seg = net.segments[target.idx];
      if (!seg) return `Segment ${target.idx}`;
      const a = net.stations[seg.a]?.name;
      const b = net.stations[seg.b]?.name;
      return a && b ? `${a} ↔ ${b}` : seg.name;
    }
    case "interlocking": {
      const il = net.interlockings[target.idx];
      const st = il ? net.stations[il.station] : undefined;
      return st ? `${st.name} interlocking` : `Interlocking ${target.idx}`;
    }
    default:
      return "";
  }
}

export function targetKindLabel(net: Network, target: DisruptionTarget): string {
  switch (target.type) {
    case "station":
      return "Station";
    case "complex":
      return "Station complex";
    case "segment":
      return net.segments[target.idx]?.river ? "River crossing" : "Track segment";
    case "interlocking":
      return "Interlocking";
    default:
      return "";
  }
}

const scratch: PointOnLine = { x: 0, y: 0, tx: 0, ty: 0 };

/** World anchor of a target (overlay chip position, camera focus). */
export function targetAnchor(net: Network, target: DisruptionTarget): { x: number; y: number } | null {
  switch (target.type) {
    case "station": {
      const s = net.stations[target.idx];
      return s ? { x: s.x, y: s.y } : null;
    }
    case "complex": {
      const c = net.complexes[target.idx];
      return c ? { x: c.x, y: c.y } : null;
    }
    case "segment": {
      const seg = net.segments[target.idx];
      if (!seg) return null;
      pointAt(seg.line, 0.5, scratch);
      return { x: scratch.x, y: scratch.y };
    }
    case "interlocking": {
      const il = net.interlockings[target.idx];
      return il ? { x: il.x, y: il.y } : null;
    }
    default:
      return null;
  }
}

/** Remaining sim seconds, or null for "until cleared". */
export function remainingS(d: DisruptionInfo, clockS: number | null): number | null {
  if (d.end_t == null) return null;
  return Math.max(0, d.end_t - (clockS ?? d.start_t));
}

/** Stations touched by a disruption (for crowding emphasis and list badges). */
export function disruptionStations(net: Network, d: DisruptionInfo): number[] {
  const t = d.target;
  switch (t.type) {
    case "station":
      return [t.idx];
    case "complex":
      return net.complexes[t.idx]?.stations ?? [];
    case "segment": {
      const seg = net.segments[t.idx];
      return seg ? [seg.a, seg.b] : [];
    }
    case "interlocking": {
      const il = net.interlockings[t.idx];
      return il ? [il.station] : [];
    }
    default:
      return [];
  }
}
