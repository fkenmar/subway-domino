/**
 * Dynamic layer, redrawn every animation frame (§6): disruption overlays and
 * countdown chips, station crowding, trains (positioned by the §2.7 rule and
 * interpolated between frames), selection / hover / search highlights.
 */
import type { Camera } from "@/lib/camera";
import { disruptionStations, remainingS, targetAnchor } from "@/lib/disruptions";
import { formatCountdown } from "@/lib/format";
import type { InterpolatedTrains } from "@/lib/interp";
import type { Network } from "@/lib/network";
import { locate, makePose } from "@/lib/positioning";
import { type DisruptionInfo, type DisruptionKind, STATION_CLOSED, STATUS } from "@/lib/protocol";
import type { Selection } from "@/lib/store";
import { appendOffsetPath, interlockingOffset, projectLine } from "./renderBase";
import { bundleWidth, CROWD_ALERT, CROWD_CALM, crowdColor, FONT_STACK, mixHex, PALETTE, trainSize, type ZoomTier } from "./style";

export type HoverTarget =
  | { type: "station" | "segment" | "interlocking"; idx: number }
  | { type: "disruption"; id: number }
  | { type: "train"; trainId: number };

export interface DynamicScene {
  net: Network;
  camera: Camera;
  tier: ZoomTier;
  dpr: number;
  now: number;
  trains: InterpolatedTrains;
  /** Trains fading out (left the map, or their slot was reused); optional. */
  ghosts?: InterpolatedTrains | null;
  crowd: Float32Array;
  crowdCount: number;
  disruptions: readonly DisruptionInfo[];
  clockS: number | null;
  selection: Selection | null;
  hover: HoverTarget | null;
  highlight: { stationIdx: number; until: number } | null;
}

export interface ChipRect {
  id: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

const KIND_COLOR: Record<DisruptionKind, string> = {
  full_station_closure: PALETTE.red,
  local_block: PALETTE.block,
  express_block: PALETTE.block,
  signal_delay: PALETTE.signal,
};

const KIND_CHIP: Record<DisruptionKind, string> = {
  full_station_closure: "CLOSED",
  local_block: "LOCAL BLOCKED",
  express_block: "EXPRESS BLOCKED",
  signal_delay: "SIGNALS",
};

type Ctx2D = CanvasRenderingContext2D;

const reducedMotionQuery =
  typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia("(prefers-reduced-motion: reduce)")
    : null;

/** Pulses, flashes and marching dashes hold still for viewers who ask for reduced motion. */
function reducedMotion(): boolean {
  return reducedMotionQuery?.matches === true;
}

/** Status bits that mean a dispatch action is acting on the train. */
const ACTION_BITS = STATUS.HELD | STATUS.SHORT_TURNING | STATUS.TRACK_SWITCH | STATUS.REROUTED | STATUS.GAP_TRAIN | STATUS.SKIPPING_NEXT;
/** A train that an action just touched flashes this long. */
const ACTION_FLASH_MS = 1500;
/** Heading smoothing time constant: pills turn through curves instead of snapping at polyline vertices. */
const HEADING_TAU_MS = 70;
/**
 * Soft drop shadow (light map): two translucent, fattened copies of each pill under it,
 * nudged down: (extra radius px, alpha), outer then inner. Fills, not strokes:
 * the cheapest way to touch only the pixels around the pills.
 */
const HALO = [
  [3.2, 0.07],
  [1.5, 0.14],
] as const;
/** Shadow offset (CSS px, downwards). */
const SHADOW_DY = 1.2;
/** Citywide train bead fill (inset on the ribbon). */
const BEAD_FILL = "#ffffff";
/** Fill intensity by load bucket (empty → crowded): the route colour mixed this far towards white, so fuller = deeper. */
const LOAD_LIGHTEN = [0.24, 0.15, 0.07, 0] as const;
/** Extra lightening on wide tiers (none on the light map: the dark outline separates pill from ribbon). */
const WIDE_LIGHTEN = 0;
/** Glow radius and strength by tier (city → street): small and faint citywide, where the trunks are dense. */
const HALO_RADIUS_K = [0.45, 0.7, 1, 1.25, 1.25] as const;
const HALO_ALPHA_K = [0.55, 0.8, 1, 1, 1] as const;

function loadBucket(load: number): number {
  return load < 25 ? 0 : load < 55 ? 1 : load < 85 ? 2 : 3;
}

/** Append a capsule centred at (x, y), along unit (ux, uy), `h` from centre to cap centre, radius `r`. */
function addPill(p: Path2D, x: number, y: number, ux: number, uy: number, h: number, r: number): void {
  const fx = x + ux * h;
  const fy = y + uy * h;
  const bx = x - ux * h;
  const by = y - uy * h;
  const th = Math.atan2(uy, ux);
  p.moveTo(fx - uy * r, fy + ux * r);
  p.arc(fx, fy, r, th + Math.PI / 2, th - Math.PI / 2, true);
  p.arc(bx, by, r, th - Math.PI / 2, th + Math.PI / 2, true);
  p.closePath();
}

const CHIP_FONT = `700 10.5px ${FONT_STACK}`;
const CHIP_MONO = `600 10.5px ui-monospace, "SF Mono", Menlo, monospace`;

interface ChipLayout {
  rect: ChipRect;
  kind: keyof typeof KIND_COLOR;
  label: string;
  labelWidth: number;
  time: string;
  selected: boolean;
}

export class DynamicRenderer {
  /** Chip rectangles drawn this frame (CSS px), for hit-testing. */
  chips: ChipRect[] = [];
  private chipLayout: ChipLayout[] = [];
  /** Screen positions of trains drawn this frame: [x, y] pairs, with train ids. */
  trainXY = new Float32Array(2048);
  trainIds = new Uint16Array(1024);
  trainIdx = new Int32Array(1024);
  trainCount = 0;
  private hatch: CanvasPattern | null = null;
  private hatchDpr = 0;
  private readonly pose = makePose();
  private sx = new Float64Array(4096);
  private sy = new Float64Array(4096);

  /** Disruption overlays, crowding, highlights and trains (station labels go on top of these). */
  render(ctx: Ctx2D, scene: DynamicScene): void {
    this.drawDisruptions(ctx, scene);
    this.drawCrowding(ctx, scene);
    this.drawHighlights(ctx, scene);
    this.drawTrains(ctx, scene);
  }

  /** Disruption countdown chips: the topmost (interactive) layer, as laid out by `layoutChips`. */
  renderChips(ctx: Ctx2D): void {
    this.paintChips(ctx);
  }

  // ─── helpers ──────────────────────────────────────────────────────────────

  private hatchPattern(ctx: Ctx2D, dpr: number): CanvasPattern | null {
    if (this.hatch && this.hatchDpr === dpr) return this.hatch;
    if (typeof document === "undefined") return null;
    const size = Math.round(8 * dpr);
    const tile = document.createElement("canvas");
    tile.width = size;
    tile.height = size;
    const t = tile.getContext("2d");
    if (!t) return null;
    t.fillStyle = "rgba(215, 38, 30, 0.2)";
    t.fillRect(0, 0, size, size);
    t.strokeStyle = PALETTE.block;
    t.lineWidth = 2.2 * dpr;
    t.beginPath();
    for (let k = -size; k <= size; k += size / 2) {
      t.moveTo(k, size);
      t.lineTo(k + size, 0);
    }
    t.stroke();
    const p = ctx.createPattern(tile, "repeat");
    if (p && typeof DOMMatrix !== "undefined") p.setTransform(new DOMMatrix().scale(1 / dpr));
    this.hatch = p;
    this.hatchDpr = dpr;
    return p;
  }

  /** Project a segment's polyline (bundle-oriented) into the scratch buffers. */
  private projectSegment(net: Network, cam: Camera, segIdx: number): number {
    const seg = net.segments[segIdx];
    if (!seg) return 0;
    const n = seg.line.xs.length;
    if (this.sx.length < n) {
      this.sx = new Float64Array(n * 2);
      this.sy = new Float64Array(n * 2);
    }
    return projectLine(
      seg.line.xs,
      seg.line.ys,
      seg.bundleDir === -1,
      { sx: (x) => cam.toScreenX(x), sy: (y) => cam.toScreenY(y) },
      this.sx,
      this.sy,
      0.5,
    );
  }

  private segmentOverlay(ctx: Ctx2D, scene: DynamicScene, segIdx: number, kind: DisruptionKind): void {
    const { net, camera, tier, now, dpr } = scene;
    const seg = net.segments[segIdx];
    if (!seg) return;
    const n = this.projectSegment(net, camera, segIdx);
    if (n < 2) return;
    const half = bundleWidth(tier, Math.max(1, seg.colors.length)) / 2;
    ctx.save();
    ctx.lineCap = "butt";
    ctx.lineJoin = "round";
    if (kind === "signal_delay") {
      ctx.setLineDash([7, 5]);
      ctx.lineDashOffset = reducedMotion() ? 0 : -(now / 40) % 12;
      ctx.strokeStyle = PALETTE.signal;
      ctx.lineWidth = Math.max(2.5, half * 2 + 3);
      ctx.globalAlpha = 0.95;
      ctx.beginPath();
      appendOffsetPath(ctx, this.sx, this.sy, n, 0);
      ctx.stroke();
    } else {
      const pattern = this.hatchPattern(ctx, dpr);
      const band = Math.max(3, tier.lineWidth + 1);
      ctx.strokeStyle = pattern ?? PALETTE.block;
      if (kind === "local_block") {
        // Local tracks run outside: hatch both outer sides of the bundle.
        ctx.lineWidth = band;
        ctx.beginPath();
        appendOffsetPath(ctx, this.sx, this.sy, n, half + band / 2);
        appendOffsetPath(ctx, this.sx, this.sy, n, -(half + band / 2));
        ctx.stroke();
        ctx.strokeStyle = PALETTE.block;
        ctx.lineWidth = 1;
        ctx.beginPath();
        appendOffsetPath(ctx, this.sx, this.sy, n, half + band);
        appendOffsetPath(ctx, this.sx, this.sy, n, -(half + band));
        ctx.stroke();
      } else {
        // Express tracks run inside: hatch over the centre of the bundle.
        ctx.lineWidth = Math.max(band, half * 2 + 2);
        ctx.beginPath();
        appendOffsetPath(ctx, this.sx, this.sy, n, 0);
        ctx.stroke();
        ctx.strokeStyle = PALETTE.block;
        ctx.lineWidth = 1.2;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        appendOffsetPath(ctx, this.sx, this.sy, n, 0);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  private ring(ctx: Ctx2D, x: number, y: number, r: number, color: string, width: number, dash?: number[]): void {
    ctx.save();
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
    ctx.restore();
  }

  private stationRadius(scene: DynamicScene, s: number): number {
    const st = scene.net.stations[s];
    return st?.hub ? scene.tier.hubRadius : scene.tier.stationRadius;
  }

  // ─── layers ───────────────────────────────────────────────────────────────

  private drawDisruptions(ctx: Ctx2D, scene: DynamicScene): void {
    const { net, camera, tier } = scene;
    for (const d of scene.disruptions) {
      const t = d.target;
      if (t.type === "segment") {
        this.segmentOverlay(ctx, scene, t.idx, d.kind);
        continue;
      }
      if (t.type === "station" || t.type === "interlocking") {
        const s = t.type === "station" ? t.idx : (net.interlockings[t.idx]?.station ?? -1);
        const st = net.stations[s];
        if (!st) continue;
        if (d.kind !== "full_station_closure") {
          // Blocks / signal problems affect the edges entering and leaving the station.
          for (const seg of net.stationSegments[s] ?? []) this.segmentOverlay(ctx, scene, seg, d.kind);
        }
        const x = camera.toScreenX(st.x);
        const y = camera.toScreenY(st.y);
        const r = this.stationRadius(scene, s) + 5;
        if (d.kind === "full_station_closure") {
          this.ring(ctx, x, y, r + 1, PALETTE.red, 1.5);
          this.ring(ctx, x, y, r - 1, PALETTE.closed, 3);
        } else if (d.kind === "signal_delay") {
          this.ring(ctx, x, y, r, PALETTE.signal, 2, [3, 2]);
        } else {
          this.ring(ctx, x, y, r, PALETTE.block, 2, [3, 2]);
        }
        if (t.type === "interlocking" && tier.showInterlockings) {
          const off = interlockingOffset(tier);
          this.ring(ctx, x + off.dx, y + off.dy, 6, KIND_COLOR[d.kind], 1.8);
        }
        continue;
      }
      if (t.type === "complex") {
        for (const s of net.complexes[t.idx]?.stations ?? []) {
          const st = net.stations[s];
          if (!st) continue;
          const x = camera.toScreenX(st.x);
          const y = camera.toScreenY(st.y);
          const r = this.stationRadius(scene, s) + 5;
          this.ring(ctx, x, y, r + 1, PALETTE.red, 1.5);
          this.ring(ctx, x, y, r - 1, PALETTE.closed, 3);
        }
      }
    }
  }

  private drawCrowding(ctx: Ctx2D, scene: DynamicScene): void {
    const { net, camera, tier, crowd, crowdCount, now } = scene;
    const m = Math.min(crowdCount, net.stations.length);
    const W = camera.width;
    const H = camera.height;
    const glow = tier.id <= 1;
    for (let i = 0; i < m; i++) {
      const c = crowd[i];
      const closed = c >= STATION_CLOSED - 0.5;
      const color = closed ? null : crowdColor(c);
      if (!closed && !color) continue;
      const st = net.stations[i];
      const x = camera.toScreenX(st.x);
      const y = camera.toScreenY(st.y);
      if (x < -30 || y < -30 || x > W + 30 || y > H + 30) continue;
      const r = st.hub ? tier.hubRadius : tier.stationRadius;
      if (closed) {
        const s = Math.max(3, r + 1.5);
        ctx.beginPath();
        ctx.arc(x, y, s + 1, 0, Math.PI * 2);
        ctx.fillStyle = PALETTE.closed;
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(x - s * 0.62, y - s * 0.62);
        ctx.lineTo(x + s * 0.62, y + s * 0.62);
        ctx.moveTo(x + s * 0.62, y - s * 0.62);
        ctx.lineTo(x - s * 0.62, y + s * 0.62);
        ctx.strokeStyle = PALETTE.closedX;
        ctx.lineWidth = Math.max(1.4, r * 0.45);
        ctx.lineCap = "round";
        ctx.stroke();
        continue;
      }
      const ratio = Math.min(1.6, c / 100);
      if (glow && c >= CROWD_CALM + 10) {
        const gr = 3 + ratio * 7;
        const g = ctx.createRadialGradient(x, y, 0, x, y, gr);
        g.addColorStop(0, color as string);
        g.addColorStop(1, "rgba(215, 38, 30, 0)");
        ctx.globalAlpha = 0.55;
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(x, y, gr, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      ctx.beginPath();
      ctx.arc(x, y, Math.max(1.4, r - (tier.id >= 2 ? 0.6 : 0)), 0, Math.PI * 2);
      ctx.fillStyle = color as string;
      ctx.fill();
      if (c >= CROWD_ALERT) {
        const phase = reducedMotion() ? 0.35 : (now / 1300 + i * 0.137) % 1;
        ctx.globalAlpha = (1 - phase) * 0.85;
        this.ring(ctx, x, y, r + 1.5 + phase * (tier.id >= 2 ? 9 : 6), PALETTE.red, 1.5);
        ctx.globalAlpha = 1;
      }
    }
  }

  private drawHighlights(ctx: Ctx2D, scene: DynamicScene): void {
    const { net, camera, tier, now } = scene;
    const outline = (target: { type: string; idx: number }, alpha: number) => {
      ctx.save();
      ctx.globalAlpha = alpha;
      if (target.type === "station") {
        const st = net.stations[target.idx];
        if (st) this.ring(ctx, camera.toScreenX(st.x), camera.toScreenY(st.y), this.stationRadius(scene, target.idx) + 3.5, PALETTE.selection, 2);
      } else if (target.type === "complex") {
        for (const s of net.complexes[target.idx]?.stations ?? []) {
          const st = net.stations[s];
          if (st) this.ring(ctx, camera.toScreenX(st.x), camera.toScreenY(st.y), this.stationRadius(scene, s) + 3.5, PALETTE.selection, 2);
        }
      } else if (target.type === "segment") {
        const seg = net.segments[target.idx];
        const n = this.projectSegment(net, camera, target.idx);
        if (seg && n >= 2) {
          ctx.strokeStyle = PALETTE.selection;
          ctx.lineWidth = bundleWidth(tier, Math.max(1, seg.colors.length)) + 7;
          ctx.globalAlpha = alpha * 0.28;
          ctx.lineCap = "round";
          ctx.beginPath();
          appendOffsetPath(ctx, this.sx, this.sy, n, 0);
          ctx.stroke();
        }
      } else if (target.type === "interlocking") {
        const il = net.interlockings[target.idx];
        if (il) {
          const off = tier.showInterlockings ? interlockingOffset(tier) : { dx: 0, dy: 0 };
          const x = camera.toScreenX(il.x) + off.dx;
          const y = camera.toScreenY(il.y) + off.dy;
          ctx.strokeStyle = PALETTE.selection;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(x, y - 7);
          ctx.lineTo(x + 7, y);
          ctx.lineTo(x, y + 7);
          ctx.lineTo(x - 7, y);
          ctx.closePath();
          ctx.stroke();
        }
      }
      ctx.restore();
    };
    if (scene.hover && scene.hover.type !== "disruption" && scene.hover.type !== "train") outline(scene.hover, 0.55);
    if (scene.selection) outline(scene.selection.target, 1);
    const h = scene.highlight;
    if (h && now < h.until) {
      const st = net.stations[h.stationIdx];
      if (st) {
        const phase = (now / 900) % 1;
        ctx.save();
        ctx.globalAlpha = 1 - phase;
        this.ring(ctx, camera.toScreenX(st.x), camera.toScreenY(st.y), 6 + phase * 22, PALETTE.highlight, 2.5);
        ctx.restore();
      }
    }
  }

  // Per-train state by train_id: smoothed heading, last status, when an action last touched it.
  private headX = new Float32Array(1024);
  private headY = new Float32Array(1024);
  private headAt = new Float64Array(1024);
  private lastStatus = new Uint8Array(1024);
  private actionAt = new Float64Array(1024).fill(Number.NEGATIVE_INFINITY);
  /** Pill fill per route colour and load bucket (and wide-tier boost). */
  private fills = new Map<string, string[]>();
  // Per-frame scratch (screen geometry of the drawn trains).
  private tx = new Float32Array(1024);
  private ty = new Float32Array(1024);
  private tux = new Float32Array(1024);
  private tuy = new Float32Array(1024);

  private ensureTrainState(id: number): void {
    if (id < this.headX.length) return;
    const cap = Math.max(id + 1, this.headX.length * 2);
    const grow = <T extends Float32Array | Float64Array | Uint8Array>(a: T, b: T, fill?: number): T => {
      if (fill !== undefined) b.fill(fill);
      b.set(a);
      return b;
    };
    this.headX = grow(this.headX, new Float32Array(cap));
    this.headY = grow(this.headY, new Float32Array(cap));
    this.headAt = grow(this.headAt, new Float64Array(cap));
    this.lastStatus = grow(this.lastStatus, new Uint8Array(cap));
    this.actionAt = grow(this.actionAt, new Float64Array(cap), Number.NEGATIVE_INFINITY);
  }

  private fillFor(color: string, load: number, wide = false): string {
    const key = wide ? `${color}+` : color;
    let v = this.fills.get(key);
    if (!v) {
      v = LOAD_LIGHTEN.map((t) => mixHex(color, "#ffffff", wide ? t + WIDE_LIGHTEN : t));
      this.fills.set(key, v);
    }
    return v[loadBucket(load)];
  }

  /**
   * Screen pose of train `i` of `set` (lane-offset position + smoothed heading)
   * into slot `k` of the scratch arrays. Returns false when it can't be placed or is off screen.
   */
  private placeTrain(scene: DynamicScene, set: InterpolatedTrains, i: number, k: number, smooth: boolean, margin: number): boolean {
    const { net, camera, tier, now } = scene;
    const pose = this.pose;
    if (!locate(net, set.edge[i], set.frac[i], pose)) return false;
    let x = camera.toScreenX(pose.x);
    let y = camera.toScreenY(pose.y);
    const dx = pose.dx;
    const dy = -pose.dy; // world y up → screen y down
    if (pose.lane !== 0 && (dx !== 0 || dy !== 0)) {
      const off = pose.lane * tier.laneWidth;
      x += -dy * off;
      y += dx * off;
    }
    if (x < -margin || y < -margin || x > camera.width + margin || y > camera.height + margin) return false;
    const id = set.trainId[i];
    this.ensureTrainState(id);
    let ux = this.headX[id];
    let uy = this.headY[id];
    const fresh = now - this.headAt[id] > 400 || (ux === 0 && uy === 0);
    if (dx !== 0 || dy !== 0) {
      if (!smooth || fresh || ux * dx + uy * dy < -0.2) {
        ux = dx;
        uy = dy;
      } else {
        const a = 1 - Math.exp(-(now - this.headAt[id]) / HEADING_TAU_MS);
        ux += (dx - ux) * a;
        uy += (dy - uy) * a;
        const l = Math.hypot(ux, uy) || 1;
        ux /= l;
        uy /= l;
      }
    } else if (fresh) {
      ux = 1;
      uy = 0;
    }
    if (smooth) {
      this.headX[id] = ux;
      this.headY[id] = uy;
      this.headAt[id] = now;
    }
    this.tx[k] = x;
    this.ty[k] = y;
    this.tux[k] = ux;
    this.tuy[k] = uy;
    return true;
  }

  private drawTrains(ctx: Ctx2D, scene: DynamicScene): void {
    const { net, camera, tier, trains, now } = scene;
    const ghosts = scene.ghosts ?? null;
    const n = trains.count;
    const total = n + (ghosts?.count ?? 0);
    if (this.trainIds.length < n) {
      this.trainIds = new Uint16Array(n * 2);
      this.trainIdx = new Int32Array(n * 2);
      this.trainXY = new Float32Array(n * 4);
    }
    if (this.tx.length < total) {
      const cap = total * 2;
      this.tx = new Float32Array(cap);
      this.ty = new Float32Array(cap);
      this.tux = new Float32Array(cap);
      this.tuy = new Float32Array(cap);
    }
    // Citywide: trains are small white beads inset on the full-colour ribbons, so the lines stay
    // solid (no "sausage chain") and the printed-map look holds; pills from the borough tier on.
    const bead = tier.id === 0;
    const size = trainSize(camera.mPerPx);
    const len = bead ? 4.4 : size.length;
    const wid = bead ? tier.lineWidth * 0.5 : size.width;
    const r = wid / 2;
    const h = Math.max(0, len / 2 - r);
    const close = tier.id >= 2;
    const margin = len + 24;

    // 1) Place live trains (k = 0..count-1 are hit-testable), then fading ghosts.
    let k = 0;
    for (let i = 0; i < n; i++) {
      if (!this.placeTrain(scene, trains, i, k, true, margin)) continue;
      const id = trains.trainId[i];
      const status = trains.status[i];
      const gained = status & ACTION_BITS & ~this.lastStatus[id];
      if (gained) this.actionAt[id] = now;
      this.lastStatus[id] = status;
      this.trainXY[2 * k] = this.tx[k];
      this.trainXY[2 * k + 1] = this.ty[k];
      this.trainIds[k] = id;
      this.trainIdx[k] = i;
      k++;
    }
    this.trainCount = k;
    const ghostIdx: number[] = [];
    let kg = k;
    if (ghosts) {
      for (let i = 0; i < ghosts.count; i++) {
        if (ghosts.fade[i] <= 0.02 || !this.placeTrain(scene, ghosts, i, kg, false, margin)) continue;
        ghostIdx.push(i);
        kg++;
      }
    }

    const colorOf = (set: InterpolatedTrains, i: number) => net.routes[set.route[i]]?.mapColor ?? "#9aa8bc";

    // 2) Pill paths of fully visible trains, batched by fill (route colour × load) for the bodies
    //    and by route colour for the halos; status outlines alongside.
    const byFill = new Map<string, Path2D>();
    const halos = new Map<string, [Path2D, Path2D]>();
    const tierIdx = Math.min(HALO_RADIUS_K.length - 1, Math.max(0, tier.id));
    const haloK = HALO_RADIUS_K[tierIdx];
    const haloA = HALO_ALPHA_K[tierIdx];
    const wide = tier.id <= 1;
    // Tunnel-stuck: a white gap, then a red ring (and the red pulse) — legible on the red lines too.
    const stuck = new Path2D();
    const stuckRing = new Path2D();
    const stuckHalo: [Path2D, Path2D] = [new Path2D(), new Path2D()];
    let anyStuck = false;
    const casing = new Path2D();
    const heldRing = new Path2D();
    const rerouted = new Path2D();
    const lights = new Path2D();
    // Wide tiers: a light core along each pill, so trains read as objects, not as dashes in the line.
    const cores = new Path2D();
    const still = reducedMotion();
    const pulse = still ? 0.6 : 0.5 + 0.5 * Math.sin(now / 210);
    for (let j = 0; j < k; j++) {
      const i = this.trainIdx[j];
      if (trains.fade[i] < 1) continue;
      const x = this.tx[j];
      const y = this.ty[j];
      const ux = this.tux[j];
      const uy = this.tuy[j];
      const color = colorOf(trains, i);
      const fill = bead ? BEAD_FILL : this.fillFor(color, trains.load[i], wide);
      let p = byFill.get(fill);
      if (!p) {
        p = new Path2D();
        byFill.set(fill, p);
      }
      addPill(p, x, y, ux, uy, h, r);
      const status = trains.status[i];
      if (status & STATUS.TUNNEL_STUCK) {
        anyStuck = true;
        addPill(stuck, x, y, ux, uy, h, r + 1.4);
        addPill(stuckRing, x, y, ux, uy, h, r + 3);
        const grow = 1.3 + 0.9 * pulse;
        addPill(stuckHalo[0], x, y, ux, uy, h, r + HALO[0][0] * haloK * grow);
        addPill(stuckHalo[1], x, y, ux, uy, h, r + HALO[1][0] * haloK * grow);
      } else if (!bead) {
        let g = halos.get(color);
        if (!g) {
          g = [new Path2D(), new Path2D()];
          halos.set(color, g);
        }
        addPill(g[0], x, y, ux, uy, h, r + HALO[0][0] * haloK);
        addPill(g[1], x, y, ux, uy, h, r + HALO[1][0] * haloK);
      }
      if (status & STATUS.HELD) addPill(heldRing, x, y, ux, uy, h, r + 2.4);
      // A dark casing separates the pill from the line under it (beads sit inside the ribbon instead).
      if (!bead) addPill(casing, x, y, ux, uy, h, r + (close ? 1 : 0.9));
      if (!close && !bead) {
        cores.moveTo(x - ux * h, y - uy * h);
        cores.lineTo(x + ux * h, y + uy * h);
      }
      if (close) {
        if (status & STATUS.REROUTED) addPill(rerouted, x, y, ux, uy, h, r + 2.4);
        const fx = x + ux * (h + r * 0.3);
        const fy = y + uy * (h + r * 0.3);
        lights.moveTo(fx + r * 0.34, fy);
        lights.arc(fx, fy, r * 0.34, 0, Math.PI * 2);
      }
    }

    // 3) Soft drop shadow so pills lift off the light map; tunnel-stuck trains glow red and pulse.
    ctx.save();
    ctx.translate(0, SHADOW_DY);
    ctx.fillStyle = PALETTE.trainShadow;
    for (let q = 0; q < 2; q++) {
      ctx.globalAlpha = HALO[q][1] * haloA;
      for (const g of halos.values()) ctx.fill(g[q]);
    }
    ctx.restore();
    for (let q = 0; q < 2; q++) {
      if (anyStuck) {
        ctx.globalAlpha = Math.min(1, HALO[q][1] * (3.4 + 2.4 * pulse));
        ctx.fillStyle = PALETTE.red;
        ctx.fill(stuckHalo[q]);
      }
    }
    ctx.globalAlpha = 1;

    // 4) Bodies: route colour (deeper = fuller) inside a thin dark outline.
    ctx.lineJoin = "round";
    ctx.fillStyle = PALETTE.trainEdge;
    ctx.fill(casing);
    for (const [fill, p] of byFill) {
      ctx.fillStyle = fill;
      ctx.fill(p);
    }
    if (!close) {
      ctx.lineCap = "round";
      ctx.lineWidth = Math.max(0.9, r * 0.55);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
      ctx.stroke(cores);
    }
    if (anyStuck) {
      ctx.lineWidth = close ? 2.2 : 1.8;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke(stuck);
      ctx.lineWidth = close ? 2 : 1.6;
      ctx.strokeStyle = PALETTE.red;
      ctx.stroke(stuckRing);
    }
    if (close) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
      ctx.fill(lights);
    }

    // 5) Fading trains (new trips appearing, ghosts leaving): drawn one by one with their opacity.
    for (let j = 0; j < kg; j++) {
      const ghost = j >= k;
      const set = ghost ? (ghosts as InterpolatedTrains) : trains;
      const i = ghost ? ghostIdx[j - k] : this.trainIdx[j];
      const fade = set.fade[i];
      if (fade >= 1) continue;
      const p = new Path2D();
      addPill(p, this.tx[j], this.ty[j], this.tux[j], this.tuy[j], h, r);
      ctx.globalAlpha = fade;
      ctx.fillStyle = bead ? BEAD_FILL : this.fillFor(colorOf(set, i), set.load[i], wide);
      ctx.fill(p);
    }
    ctx.globalAlpha = 1;

    // 6) Status cues that don't depend on the route colour: held = white ring with a black
    //    outline (reads on yellow, red and blue alike); rerouted = dashed ring (close zoom).
    ctx.lineWidth = close ? 3.6 : 3;
    ctx.strokeStyle = PALETTE.heldEdge;
    ctx.stroke(heldRing);
    ctx.lineWidth = close ? 1.8 : 1.4;
    ctx.strokeStyle = PALETTE.heldRing;
    ctx.stroke(heldRing);
    if (close) {
      ctx.save();
      ctx.setLineDash([2.5, 2]);
      ctx.strokeStyle = PALETTE.glyph;
      ctx.lineWidth = 1.1;
      ctx.stroke(rerouted);
      ctx.restore();
    }

    // 7) A short flash when an action starts acting on a train (AI holds, short turns, gap trains…).
    for (let j = 0; j < k; j++) {
      const i = this.trainIdx[j];
      const id = trains.trainId[i];
      const t = (now - this.actionAt[id]) / ACTION_FLASH_MS;
      if (!(t >= 0 && t < 1)) continue;
      const e = still ? 0.35 : 1 - (1 - t) ** 3;
      const p = new Path2D();
      addPill(p, this.tx[j], this.ty[j], this.tux[j], this.tuy[j], h, r + 2 + e * (close ? 9 : 6));
      // Black-and-white double ring: visible on every route colour (blue A/C/E included).
      ctx.globalAlpha = 1 - t;
      ctx.lineWidth = 4.2 - 1.2 * t;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke(p);
      ctx.lineWidth = 2 - t;
      ctx.strokeStyle = trains.status[i] & STATUS.GAP_TRAIN ? PALETTE.gap : PALETTE.aiRing;
      ctx.stroke(p);
    }
    ctx.globalAlpha = 1;

    // 8) Route letter inside pills at close zoom.
    if (tier.showTrainLabels) {
      ctx.save();
      ctx.font = `800 ${Math.max(8, Math.round(wid * 0.86))}px ${FONT_STACK}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let j = 0; j < k; j++) {
        const i = this.trainIdx[j];
        const route = net.routes[trains.route[i]];
        if (!route || trains.fade[i] < 1) continue;
        ctx.fillStyle = route.textColor;
        ctx.fillText(route.bullet, this.tx[j], this.ty[j] + 0.5);
      }
      ctx.restore();
    }

    // 9) Status glyphs above the pill: short-turn ↺, gap train ✚, skipping », track switch ⇄.
    if (tier.showStatusIcons) {
      ctx.save();
      const size = Math.max(10, wid + 4);
      ctx.font = `700 ${size}px ${FONT_STACK}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.lineWidth = 3;
      ctx.strokeStyle = PALETTE.labelHalo;
      for (let j = 0; j < k; j++) {
        const i = this.trainIdx[j];
        const s = trains.status[i];
        if (!(s & (STATUS.SHORT_TURNING | STATUS.GAP_TRAIN | STATUS.SKIPPING_NEXT | STATUS.TRACK_SWITCH))) continue;
        const glyph = s & STATUS.SHORT_TURNING ? "↺" : s & STATUS.GAP_TRAIN ? "✚" : s & STATUS.SKIPPING_NEXT ? "»" : "⇄";
        const color = s & STATUS.GAP_TRAIN ? PALETTE.gap : s & STATUS.SHORT_TURNING ? PALETTE.highlight : PALETTE.glyph;
        // Just above the pill's own vertical extent (a horizontal pill is short, a vertical one tall).
        const oy = h * Math.abs(this.tuy[j]) + r + size * 0.5;
        ctx.strokeText(glyph, this.tx[j], this.ty[j] - oy);
        ctx.fillStyle = color;
        ctx.fillText(glyph, this.tx[j], this.ty[j] - oy);
      }
      ctx.restore();
    }
  }

  /**
   * Lay the countdown chips out for this frame (stacking ones that would overlap)
   * and record their rectangles in `chips` for hit-testing and for station labels
   * to yield to. Call before drawing labels; `renderChips` paints the result.
   */
  layoutChips(ctx: Ctx2D, scene: DynamicScene): readonly ChipRect[] {
    const { net, camera, tier, clockS } = scene;
    const out: ChipLayout[] = [];
    this.chipLayout = out;
    this.chips = [];
    if (scene.disruptions.length === 0) return this.chips;
    ctx.save();
    for (const d of scene.disruptions) {
      const a = targetAnchor(net, d.target);
      if (!a) continue;
      let x = camera.toScreenX(a.x);
      let y = camera.toScreenY(a.y);
      const stations = disruptionStations(net, d);
      const lift = stations.length && d.target.type !== "segment" ? this.stationRadius(scene, stations[0]) + 16 : 15;
      y -= lift;
      const left = remainingS(d, clockS);
      const label = tier.id >= 1 ? KIND_CHIP[d.kind] : "";
      const time = left == null ? "∞" : formatCountdown(left);
      ctx.font = CHIP_FONT;
      const lw = label ? ctx.measureText(label).width + 6 : 0;
      ctx.font = CHIP_MONO;
      const tw = ctx.measureText(time).width;
      const w = 8 + 8 + lw + tw + 8;
      const h = 18;
      x -= w / 2;
      for (let tries = 0; tries < 5; tries++) {
        const hit = this.chips.some((p) => x < p.x1 && x + w > p.x0 && y - h / 2 < p.y1 && y + h / 2 > p.y0);
        if (!hit) break;
        y -= h + 3;
      }
      if (x + w < 0 || y + h < 0 || x > camera.width || y - h > camera.height) continue;
      const rect = { id: d.id, x0: x, y0: y - h / 2, x1: x + w, y1: y + h / 2 };
      this.chips.push(rect);
      out.push({ rect, kind: d.kind, label, labelWidth: lw, time, selected: scene.selection?.disruptionId === d.id });
    }
    ctx.restore();
    return this.chips;
  }

  private paintChips(ctx: Ctx2D): void {
    if (this.chipLayout.length === 0) return;
    ctx.save();
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    for (const { rect, kind, label, labelWidth, time, selected } of this.chipLayout) {
      const w = rect.x1 - rect.x0;
      const h = rect.y1 - rect.y0;
      const y = (rect.y0 + rect.y1) / 2;
      const color = KIND_COLOR[kind];
      ctx.beginPath();
      ctx.roundRect(rect.x0, rect.y0, w, h, h / 2);
      ctx.fillStyle = PALETTE.chipBg;
      ctx.fill();
      ctx.lineWidth = selected ? 2 : 1.2;
      ctx.strokeStyle = selected ? PALETTE.selection : color;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(rect.x0 + 9, y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      let tx = rect.x0 + 16;
      if (label) {
        ctx.font = CHIP_FONT;
        ctx.fillStyle = color;
        ctx.fillText(label, tx, y + 0.5);
        tx += labelWidth;
      }
      ctx.font = CHIP_MONO;
      ctx.fillStyle = PALETTE.chipText;
      ctx.fillText(time, tx, y + 0.5);
    }
    ctx.restore();
  }
}
