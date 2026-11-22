/**
 * Disruption targeting rules (§3.5 — CONTRACT semantics) for the UI: which
 * kinds are valid on which target, human labels, and where to anchor overlays.
 *
 * Every disruption the UI starts is a blockage that lasts until the user clears
 * it: there is no duration choice (the client always sends `duration_s: null`)
 * and no `signal_delay` (the sim, protocol and training keep it; the UI doesn't
 * offer it). Presets are served the same way by the server.
 */
import { formatClock } from "./format";
import { pointAt, type PointOnLine } from "./geometry";
import type { Network } from "./network";
import type { DisruptionInfo, DisruptionKind, DisruptionTarget } from "./protocol";

/**
 * Blockage kinds valid for a target, in UI order (most dramatic first):
 * - station: closure, local block, express block (only with X track)
 * - complex: closure
 * - segment: local block, express block (only if the segment has express)
 * - interlocking: local block, express block (only with X track)
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
      return out;
    }
    case "interlocking": {
      const il = net.interlockings[target.idx];
      if (!il) return [];
      const out: DisruptionKind[] = ["local_block"];
      if (net.stationHasExpress[il.station]) out.push("express_block");
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

/** Remaining sim seconds, or null for "until cleared" (only a session restore still needs this). */
export function remainingS(d: DisruptionInfo, clockS: number | null): number | null {
  if (d.end_t == null) return null;
  return Math.max(0, d.end_t - (clockS ?? d.start_t));
}

/** "CLOSED" for a closure, "BLOCKED" for anything else (a short, countdown-free status). */
export function blockedWord(kind: DisruptionKind): "CLOSED" | "BLOCKED" {
  return kind === "full_station_closure" ? "CLOSED" : "BLOCKED";
}

const CHIP_LABEL: Record<DisruptionKind, string> = {
  full_station_closure: "CLOSED",
  local_block: "LOCAL BLOCKED",
  express_block: "EXPRESS BLOCKED",
  signal_delay: "BLOCKED", // never started from the UI; an old restored session could still carry one
};

/**
 * Map chip text for a disruption at zoom tier `tierId`: "CLOSED" / "BLOCKED" when
 * zoomed out, "LOCAL BLOCKED" etc. closer in, plus "since 8:12 AM" when zoomed in
 * far enough. Never a countdown: blockages last until the user clears them.
 */
export function chipText(d: Pick<DisruptionInfo, "kind" | "start_t">, tierId: number): { label: string; since: string } {
  return {
    label: tierId >= 1 ? CHIP_LABEL[d.kind] : blockedWord(d.kind),
    since: tierId >= 2 ? `since ${formatClock(d.start_t)}` : "",
  };
}

/** "Closed since 8:12 AM" / "Blocked since 8:12 AM" (popover and the active list). */
export function sinceText(d: Pick<DisruptionInfo, "kind" | "start_t">): string {
  return `${d.kind === "full_station_closure" ? "Closed" : "Blocked"} since ${formatClock(d.start_t)}`;
}

/**
 * What a "Service restored" notice says for blockages that just ended: "14 St-Union Sq
 * reopened", "Jackson Hts: express track clear again", or for several at once
 * "Service restored at 3 places".
 */
export function restoredText(net: Network | null, ended: readonly Pick<DisruptionInfo, "kind" | "target" | "label">[]): string {
  if (ended.length > 1) return `Service restored at ${ended.length} places`;
  const d = ended[0];
  if (!d) return "Service restored";
  const name = net ? targetLabel(net, d.target) : d.label;
  switch (d.kind) {
    case "full_station_closure":
      return `${name} reopened`;
    case "local_block":
      return `${name}: local track clear again`;
    case "express_block":
      return `${name}: express track clear again`;
    default:
      return `${name}: trains running normally again`;
  }
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
