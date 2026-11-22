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
import { bundleWidth, CROWD_ALERT, CROWD_CALM, crowdColor, FONT_STACK, PALETTE, type ZoomTier } from "./style";

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
    t.fillStyle = "rgba(255, 69, 58, 0.25)";
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
      ctx.lineDashOffset = -(now / 40) % 12;
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
        g.addColorStop(1, "rgba(255, 69, 58, 0)");
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
        const phase = (now / 1300 + i * 0.137) % 1;
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

  private drawTrains(ctx: Ctx2D, scene: DynamicScene): void {
    const { net, camera, tier, trains, now } = scene;
    const n = trains.count;
    if (this.trainIds.length < n) {
      this.trainIds = new Uint16Array(n * 2);
      this.trainIdx = new Int32Array(n * 2);
      this.trainXY = new Float32Array(n * 4);
    }
    const W = camera.width;
    const H = camera.height;
    const pose = this.pose;
    const capsule = tier.capsules;
    const len = tier.trainLength;
    const wid = tier.trainWidth;
    const r = wid / 2;
    const bodies = new Map<string, Path2D>();
    const outline = new Path2D();
    const stuck: number[] = [];
    const held: number[] = [];
    const icons: number[] = [];
    const dashed = new Path2D();
    let k = 0;

    // Positions (and per-train screen geometry) first; drawing is batched by colour.
    const geo: { x: number; y: number; dx: number; dy: number }[] = [];
    for (let i = 0; i < n; i++) {
      if (!locate(net, trains.edge[i], trains.frac[i], pose)) continue;
      let x = camera.toScreenX(pose.x);
      let y = camera.toScreenY(pose.y);
      const dx = pose.dx;
      const dy = -pose.dy; // world y up → screen y down
      if (pose.lane !== 0 && (dx !== 0 || dy !== 0)) {
        const off = pose.lane * tier.laneWidth;
        x += -dy * off;
        y += dx * off;
      }
      if (x < -len || y < -len || x > W + len || y > H + len) continue;
      const route = net.routes[trains.route[i]];
      const color = route?.mapColor ?? "#9aa8bc";
      let path = bodies.get(color);
      if (!path) {
        path = new Path2D();
        bodies.set(color, path);
      }
      if (capsule && (dx !== 0 || dy !== 0)) {
        const half = Math.max(0, len / 2 - r);
        const fx = x + dx * half;
        const fy = y + dy * half;
        const bx = x - dx * half;
        const by = y - dy * half;
        const th = Math.atan2(dy, dx);
        for (const p of [path, outline]) {
          p.moveTo(fx + Math.cos(th - Math.PI / 2) * r, fy + Math.sin(th - Math.PI / 2) * r);
          p.arc(fx, fy, r, th - Math.PI / 2, th + Math.PI / 2);
          p.arc(bx, by, r, th + Math.PI / 2, th + (3 * Math.PI) / 2);
          p.closePath();
        }
      } else {
        const rr = capsule ? r * 1.15 : r;
        path.moveTo(x + rr, y);
        path.arc(x, y, rr, 0, Math.PI * 2);
        outline.moveTo(x + rr, y);
        outline.arc(x, y, rr, 0, Math.PI * 2);
      }
      const status = trains.status[i];
      if (status & STATUS.TUNNEL_STUCK) stuck.push(k);
      if (status & STATUS.HELD) held.push(k);
      if (status & (STATUS.SHORT_TURNING | STATUS.GAP_TRAIN | STATUS.SKIPPING_NEXT | STATUS.TRACK_SWITCH)) icons.push(k);
      if (status & STATUS.REROUTED && capsule) {
        dashed.moveTo(x + len / 2 + 2, y);
        dashed.arc(x, y, len / 2 + 2, 0, Math.PI * 2);
      }
      geo.push({ x, y, dx, dy });
      this.trainXY[2 * k] = x;
      this.trainXY[2 * k + 1] = y;
      this.trainIds[k] = trains.trainId[i];
      this.trainIdx[k] = i;
      k++;
    }
    this.trainCount = k;

    // Tunnel-stuck red halo (pulsing) under the bodies.
    if (stuck.length) {
      const pulse = 0.55 + 0.45 * Math.sin(now / 260);
      const hr = (capsule ? len / 2 : r) + (capsule ? 5 : 3.2);
      ctx.save();
      for (const j of stuck) {
        const g = geo[j];
        const grad = ctx.createRadialGradient(g.x, g.y, 0, g.x, g.y, hr);
        grad.addColorStop(0, `rgba(255, 69, 58, ${0.75 * pulse})`);
        grad.addColorStop(1, "rgba(255, 69, 58, 0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(g.x, g.y, hr, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // Bodies: dark outline, then colour fills.
    ctx.lineWidth = capsule ? 2.2 : 1.6;
    ctx.strokeStyle = "rgba(5, 8, 12, 0.95)";
    ctx.stroke(outline);
    for (const [color, path] of bodies) {
      ctx.fillStyle = color;
      ctx.fill(path);
    }

    if (capsule) {
      // Rerouted: dashed outline.
      ctx.save();
      ctx.setLineDash([2.5, 2]);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.2;
      ctx.stroke(dashed);
      ctx.restore();
    }

    // Held: amber ring.
    if (held.length) {
      ctx.save();
      ctx.strokeStyle = PALETTE.heldRing;
      ctx.lineWidth = capsule ? 1.8 : 1.2;
      ctx.beginPath();
      const hr = (capsule ? len / 2 : r) + 2.2;
      for (const j of held) {
        const g = geo[j];
        ctx.moveTo(g.x + hr, g.y);
        ctx.arc(g.x, g.y, hr, 0, Math.PI * 2);
      }
      ctx.stroke();
      ctx.restore();
    }

    // Route letter inside capsules at close zoom.
    if (tier.showTrainLabels) {
      ctx.save();
      ctx.font = `800 ${Math.max(7, Math.round(wid * 0.82))}px ${FONT_STACK}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let j = 0; j < k; j++) {
        const i = this.trainIdx[j];
        const route = net.routes[trains.route[i]];
        if (!route) continue;
        ctx.fillStyle = route.textColor;
        ctx.fillText(route.bullet, geo[j].x, geo[j].y + 0.5);
      }
      ctx.restore();
    }

    // Status glyphs: short-turn ↺, gap train ✚, skipping », track switch ⇄.
    if (tier.showStatusIcons && icons.length) {
      ctx.save();
      const size = Math.max(10, wid + 5);
      ctx.font = `700 ${size}px ${FONT_STACK}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.lineWidth = 3;
      ctx.strokeStyle = PALETTE.labelHalo;
      for (const j of icons) {
        const i = this.trainIdx[j];
        const s = trains.status[i];
        const glyph =
          s & STATUS.SHORT_TURNING ? "↺" : s & STATUS.GAP_TRAIN ? "✚" : s & STATUS.SKIPPING_NEXT ? "»" : "⇄";
        const color = s & STATUS.GAP_TRAIN ? PALETTE.gap : s & STATUS.SHORT_TURNING ? PALETTE.highlight : "#ffffff";
        const g = geo[j];
        const oy = (capsule ? len / 2 : r) + size * 0.55;
        ctx.strokeText(glyph, g.x, g.y - oy);
        ctx.fillStyle = color;
        ctx.fillText(glyph, g.x, g.y - oy);
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
