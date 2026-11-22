/**
 * Cached base layer (§6): basemap, segments and stations, rendered offscreen for
 * a snapshot camera and blitted with an affine transform while the camera moves.
 * It is re-rendered when the zoom tier, viewport or DPR changes, when the
 * transform drifts too far (blur / uncovered area), and once the camera settles.
 */
import type { Camera } from "@/lib/camera";
import { currentOccluders, type ScreenRect } from "@/lib/layout";
import { EDGE_YARD_LEAD, type GeoStation, type Network, projectLatLon } from "@/lib/network";
import { BOROUGH_LABELS, bundleOffset, bundleWidth, FONT_STACK, PALETTE, type ZoomTier, tierFor } from "./style";

/** Extra rendered margin around the viewport (CSS px) so pans reveal pre-drawn map. */
const PAD_PX = 192;
/** Re-render once the camera has been still this long (ms) and the blit is not 1:1. */
const SETTLE_MS = 140;
/** Scale drift tolerated before re-rendering immediately. */
const MAX_SCALE_DRIFT = 1.35;

interface Snapshot {
  cx: number;
  cy: number;
  mPerPx: number;
  width: number;
  height: number;
  dpr: number;
  tier: number;
  net: Network;
  /** Version of the HUD occluders the labels were placed around. */
  occVersion: number;
}

export interface LabelBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * A station label placed by the latest base render. Text is drawn in a final
 * pass above the trains (see {@link BaseLayer.drawLabels}), anchored to the
 * station's world point with a fixed screen-space offset.
 */
export interface PlacedLabel {
  text: string;
  font: string;
  color: string;
  align: CanvasTextAlign;
  wx: number;
  wy: number;
  /** Offset (CSS px) from the projected station point to the text anchor. */
  ox: number;
  oy: number;
  /** Label box relative to the projected station point (CSS px). */
  box: LabelBox;
}

type Ctx2D = CanvasRenderingContext2D;

/** Screen-space interlocking marker offset from its station, so both stay clickable. */
/** "72 St", "86 St", "1 Av": names repeated across the city. */
const BARE_NUMBERED_STREET = /^\d+(?:st|nd|rd|th)? (?:St|Av)$/;

/** Half the side of a yard's square marker (CSS px). */
export function yardHalfSize(tier: Pick<ZoomTier, "stationRadius">): number {
  return Math.max(2.5, tier.stationRadius);
}

export function interlockingOffset(tier: ZoomTier): { dx: number; dy: number } {
  const d = tier.hubRadius + 5;
  return { dx: d * 0.75, dy: -d * 0.75 };
}

/**
 * Stations that may carry a label: the main station of every complex and, on
 * dense tiers, each other station of a complex with a different name (one per
 * distinct name, best-ranked first).
 */
export function labelCandidates(net: Network, tier: Pick<ZoomTier, "labelAllStations">): GeoStation[] {
  const out: GeoStation[] = [];
  for (const c of net.complexes) {
    const main = net.stations[net.complexRep[c.idx]];
    if (!main) continue;
    out.push(main);
    if (!tier.labelAllStations || c.stations.length < 2) continue;
    const named = new Set([main.name]);
    const others = c.stations
      .map((s) => net.stations[s])
      .filter((s): s is GeoStation => s !== undefined)
      .sort((a, b) => a.label_rank - b.label_rank);
    for (const st of others) {
      if (named.has(st.name)) continue;
      named.add(st.name);
      out.push(st);
    }
  }
  return out;
}

interface SegmentBounds {
  x0: Float64Array;
  y0: Float64Array;
  x1: Float64Array;
  y1: Float64Array;
}

const boundsCache = new WeakMap<Network, SegmentBounds>();

function segmentBounds(net: Network): SegmentBounds {
  let b = boundsCache.get(net);
  if (b) return b;
  const n = net.segments.length;
  b = { x0: new Float64Array(n), y0: new Float64Array(n), x1: new Float64Array(n), y1: new Float64Array(n) };
  for (let i = 0; i < n; i++) {
    const { xs, ys } = net.segments[i].line;
    let a = Infinity;
    let c = Infinity;
    let d = -Infinity;
    let e = -Infinity;
    for (let k = 0; k < xs.length; k++) {
      if (xs[k] < a) a = xs[k];
      if (xs[k] > d) d = xs[k];
      if (ys[k] < c) c = ys[k];
      if (ys[k] > e) e = ys[k];
    }
    b.x0[i] = a;
    b.y0[i] = c;
    b.x1[i] = d;
    b.y1[i] = e;
  }
  boundsCache.set(net, b);
  return b;
}

/** Screen-space projection for the snapshot (includes the padding offset). */
interface Proj {
  sx(x: number): number;
  sy(y: number): number;
}

/**
 * Append `pts` (screen space) offset laterally by `offset` px to `ctx`'s path.
 * Positive offsets lie to the right of the direction of travel. Joins are mitred
 * (limited) so parallel strokes stay parallel through curves.
 */
export function appendOffsetPath(ctx: Pick<Ctx2D, "moveTo" | "lineTo">, xs: Float64Array, ys: Float64Array, n: number, offset: number): void {
  if (n < 2) return;
  for (let i = 0; i < n; i++) {
    let nx = 0;
    let ny = 0;
    if (offset !== 0) {
      // Right-hand normals of the incoming and outgoing pieces (screen y points down).
      let ax = 0;
      let ay = 0;
      let bx = 0;
      let by = 0;
      if (i > 0) {
        const dx = xs[i] - xs[i - 1];
        const dy = ys[i] - ys[i - 1];
        const l = Math.hypot(dx, dy) || 1;
        ax = -dy / l;
        ay = dx / l;
      }
      if (i < n - 1) {
        const dx = xs[i + 1] - xs[i];
        const dy = ys[i + 1] - ys[i];
        const l = Math.hypot(dx, dy) || 1;
        bx = -dy / l;
        by = dx / l;
      }
      if (i === 0) {
        nx = bx;
        ny = by;
      } else if (i === n - 1) {
        nx = ax;
        ny = ay;
      } else {
        const mx = ax + bx;
        const my = ay + by;
        const ml = Math.hypot(mx, my);
        if (ml < 1e-6) {
          nx = bx;
          ny = by;
        } else {
          nx = mx / ml;
          ny = my / ml;
          const cos = nx * ax + ny * ay;
          const scale = Math.min(2, 1 / Math.max(0.5, cos));
          nx *= scale;
          ny *= scale;
        }
      }
    }
    const x = xs[i] + nx * offset;
    const y = ys[i] + ny * offset;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
}

/** Project a world polyline to screen space, dropping sub-pixel vertices. */
export function projectLine(
  lineXs: Float64Array,
  lineYs: Float64Array,
  reverse: boolean,
  proj: Proj,
  outX: Float64Array,
  outY: Float64Array,
  minStepPx = 0.75,
): number {
  const n = lineXs.length;
  let k = 0;
  let lastX = 0;
  let lastY = 0;
  for (let j = 0; j < n; j++) {
    const i = reverse ? n - 1 - j : j;
    const x = proj.sx(lineXs[i]);
    const y = proj.sy(lineYs[i]);
    if (k > 0 && j < n - 1 && Math.abs(x - lastX) + Math.abs(y - lastY) < minStepPx) continue;
    outX[k] = x;
    outY[k] = y;
    lastX = x;
    lastY = y;
    k++;
  }
  return k;
}

export class BaseLayer {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: Ctx2D | null = null;
  private snap: Snapshot | null = null;
  private lastCameraVersion = -1;
  private lastMoveAt = 0;
  private scratchX = new Float64Array(4096);
  private scratchY = new Float64Array(4096);
  private textWidths = new Map<string, number>();
  /** Labels placed in the latest render (drawn above the dynamic layer). */
  labels: PlacedLabel[] = [];

  constructor(private readonly createCanvas: () => HTMLCanvasElement) {}

  invalidate(): void {
    this.snap = null;
  }

  /** Decide whether to re-render, re-render if so, then blit onto `ctx` (CSS-px transform applied). */
  drawTo(ctx: Ctx2D, net: Network, camera: Camera, dpr: number, now: number): void {
    if (camera.version !== this.lastCameraVersion) {
      this.lastCameraVersion = camera.version;
      this.lastMoveAt = now;
    }
    if (this.needsRender(net, camera, dpr, now)) this.render(net, camera, dpr);
    const s = this.snap;
    const c = this.canvas;
    if (!s || !c) return;
    const k = s.mPerPx / camera.mPerPx;
    const W = camera.width;
    const H = camera.height;
    const dx = (-W / 2 - PAD_PX) * k + (s.cx - camera.cx) / camera.mPerPx + W / 2;
    const dy = (-H / 2 - PAD_PX) * k + (camera.cy - s.cy) / camera.mPerPx + H / 2;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(c, dx, dy, (s.width + 2 * PAD_PX) * k, (s.height + 2 * PAD_PX) * k);
  }

  private needsRender(net: Network, camera: Camera, dpr: number, now: number): boolean {
    const s = this.snap;
    if (!s || s.net !== net || s.dpr !== dpr || s.width !== camera.width || s.height !== camera.height) return true;
    if (tierFor(camera.mPerPx).id !== s.tier) return true;
    if (currentOccluders().version !== s.occVersion) return true;
    const k = s.mPerPx / camera.mPerPx;
    if (k > MAX_SCALE_DRIFT || k < 1 / MAX_SCALE_DRIFT) return true;
    // Uncovered area: the viewport must lie inside the padded snapshot.
    const halfW = (camera.width / 2) * camera.mPerPx;
    const halfH = (camera.height / 2) * camera.mPerPx;
    const coverW = (s.width / 2 + PAD_PX) * s.mPerPx;
    const coverH = (s.height / 2 + PAD_PX) * s.mPerPx;
    if (
      camera.cx - halfW < s.cx - coverW ||
      camera.cx + halfW > s.cx + coverW ||
      camera.cy - halfH < s.cy - coverH ||
      camera.cy + halfH > s.cy + coverH
    ) {
      return true;
    }
    const exact = k === 1 && s.cx === camera.cx && s.cy === camera.cy;
    return !exact && !camera.animating && now - this.lastMoveAt >= SETTLE_MS;
  }

  private ensureCanvas(w: number, h: number): Ctx2D | null {
    if (!this.canvas) {
      this.canvas = this.createCanvas();
      this.ctx = this.canvas.getContext("2d");
    }
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    return this.ctx;
  }

  private measure(ctx: Ctx2D, font: string, text: string): number {
    const key = `${font}|${text}`;
    let w = this.textWidths.get(key);
    if (w === undefined) {
      ctx.font = font;
      w = ctx.measureText(text).width;
      this.textWidths.set(key, w);
    }
    return w;
  }

  render(net: Network, camera: Camera, dpr: number): void {
    const W = camera.width;
    const H = camera.height;
    const pw = Math.ceil((W + 2 * PAD_PX) * dpr);
    const ph = Math.ceil((H + 2 * PAD_PX) * dpr);
    const ctx = this.ensureCanvas(pw, ph);
    if (!ctx) return;
    const tier = tierFor(camera.mPerPx);
    const m = camera.mPerPx;
    const cx = camera.cx;
    const cy = camera.cy;
    const proj: Proj = {
      sx: (x) => (x - cx) / m + W / 2 + PAD_PX,
      sy: (y) => (cy - y) / m + H / 2 + PAD_PX,
    };
    // World extent of this snapshot (for culling).
    const wx0 = cx - (W / 2 + PAD_PX) * m;
    const wx1 = cx + (W / 2 + PAD_PX) * m;
    const wy0 = cy - (H / 2 + PAD_PX) * m;
    const wy1 = cy + (H / 2 + PAD_PX) * m;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, pw, ph);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    const occ = currentOccluders();
    // HUD rectangles in snapshot coordinates (the snapshot origin is the viewport origin − PAD).
    const occluders: LabelBox[] = occ.rects.map((r: ScreenRect) => ({
      x0: r.x0 + PAD_PX,
      y0: r.y0 + PAD_PX,
      x1: r.x1 + PAD_PX,
      y1: r.y1 + PAD_PX,
    }));

    this.drawLand(ctx, net, proj);
    this.drawGrid(ctx, proj, wx0, wy0, wx1, wy1, m);
    const boroughBoxes = this.drawBoroughs(ctx, net, proj, tier);
    this.drawYardLeads(ctx, net, proj, tier);
    this.drawSegments(ctx, net, proj, tier, wx0, wy0, wx1, wy1);
    this.drawTransfers(ctx, net, proj, tier);
    this.drawStations(ctx, net, proj, tier, wx0, wy0, wx1, wy1);
    this.drawInterlockings(ctx, net, proj, tier);
    this.drawYards(ctx, net, proj, tier);
    this.labels = this.placeLabels(ctx, net, proj, tier, W + 2 * PAD_PX, H + 2 * PAD_PX, [...occluders, ...boroughBoxes]);

    this.snap = { cx, cy, mPerPx: m, width: W, height: H, dpr, tier: tier.id, net, occVersion: occ.version };
  }

  /** Letter-spaced borough names under the lines (wide tiers); returns their boxes as label obstacles. */
  private drawBoroughs(ctx: Ctx2D, net: Network, proj: Proj, tier: ZoomTier): LabelBox[] {
    if (!tier.showBoroughs) return [];
    const size = tier.id === 0 ? 13 : 16;
    const spacing = size * 0.42;
    const font = `700 ${size}px ${FONT_STACK}`;
    const boxes: LabelBox[] = [];
    ctx.save();
    ctx.font = font;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillStyle = PALETTE.borough;
    for (const b of BOROUGH_LABELS) {
      const p = projectLatLon(net.projection, b.lat, b.lon);
      const x = proj.sx(p.x);
      const y = proj.sy(p.y);
      const widths = [...b.name].map((ch) => this.measure(ctx, font, ch));
      const total = widths.reduce((a, w) => a + w, 0) + spacing * (widths.length - 1);
      const angle = ((b.angleDeg ?? 0) * Math.PI) / 180;
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(angle);
      let cx = -total / 2;
      [...b.name].forEach((ch, i) => {
        ctx.fillText(ch, cx, 0);
        cx += widths[i] + spacing;
      });
      ctx.restore();
      // Axis-aligned bounds of the (possibly rotated) text box.
      const hw = Math.abs(Math.cos(angle)) * (total / 2) + Math.abs(Math.sin(angle)) * (size / 2);
      const hh = Math.abs(Math.sin(angle)) * (total / 2) + Math.abs(Math.cos(angle)) * (size / 2);
      if (angle === 0) boxes.push({ x0: x - hw - 4, y0: y - hh - 3, x1: x + hw + 4, y1: y + hh + 3 });
    }
    ctx.restore();
    return boxes;
  }

  private drawLand(ctx: Ctx2D, net: Network, proj: Proj): void {
    if (net.land.length === 0) return;
    ctx.beginPath();
    for (const ring of net.land) {
      const n = ring.xs.length;
      for (let i = 0; i < n; i++) {
        const x = proj.sx(ring.xs[i]);
        const y = proj.sy(ring.ys[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
    }
    ctx.fillStyle = PALETTE.land;
    ctx.fill("evenodd");
    ctx.strokeStyle = PALETTE.landEdge;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  /** Faint 1 km / 5 km reference grid — the "control room" texture. */
  private drawGrid(ctx: Ctx2D, proj: Proj, x0: number, y0: number, x1: number, y1: number, m: number): void {
    const step = m > 30 ? 5000 : m > 6 ? 1000 : 500;
    ctx.beginPath();
    for (let x = Math.ceil(x0 / step) * step; x <= x1; x += step) {
      const sx = Math.round(proj.sx(x)) + 0.5;
      ctx.moveTo(sx, proj.sy(y1));
      ctx.lineTo(sx, proj.sy(y0));
    }
    for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) {
      const sy = Math.round(proj.sy(y)) + 0.5;
      ctx.moveTo(proj.sx(x0), sy);
      ctx.lineTo(proj.sx(x1), sy);
    }
    ctx.strokeStyle = PALETTE.grid;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  private ensureScratch(n: number): void {
    if (this.scratchX.length < n) {
      this.scratchX = new Float64Array(n * 2);
      this.scratchY = new Float64Array(n * 2);
    }
  }

  private drawSegments(ctx: Ctx2D, net: Network, proj: Proj, tier: ZoomTier, x0: number, y0: number, x1: number, y1: number): void {
    const b = segmentBounds(net);
    const visible: number[] = [];
    for (let i = 0; i < net.segments.length; i++) {
      if (b.x1[i] < x0 || b.x0[i] > x1 || b.y1[i] < y0 || b.y0[i] > y1) continue;
      visible.push(i);
    }
    // Project each visible segment once (oriented by bundleDir for consistent offsets).
    const projected = new Map<number, { xs: Float64Array; ys: Float64Array; n: number }>();
    for (const i of visible) {
      const seg = net.segments[i];
      this.ensureScratch(seg.line.xs.length);
      const n = projectLine(seg.line.xs, seg.line.ys, seg.bundleDir === -1, proj, this.scratchX, this.scratchY);
      projected.set(i, { xs: this.scratchX.slice(0, n), ys: this.scratchY.slice(0, n), n });
    }

    // 1) River-crossing casing: subtle dashed outline.
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = PALETTE.riverCasing;
    ctx.lineWidth = 1;
    for (const i of visible) {
      const seg = net.segments[i];
      if (!seg.river) continue;
      const p = projected.get(i);
      if (!p) continue;
      const half = bundleWidth(tier, Math.max(1, seg.colors.length)) / 2 + tier.casingExtra + 2.5;
      ctx.beginPath();
      appendOffsetPath(ctx, p.xs, p.ys, p.n, half);
      appendOffsetPath(ctx, p.xs, p.ys, p.n, -half);
      ctx.stroke();
    }
    ctx.restore();

    // 2) Dark casing under each bundle, batched by bundle size.
    const bySize = new Map<number, number[]>();
    for (const i of visible) {
      const n = Math.max(1, net.segments[i].colors.length);
      const list = bySize.get(n) ?? [];
      list.push(i);
      bySize.set(n, list);
    }
    ctx.strokeStyle = PALETTE.casing;
    for (const [n, list] of bySize) {
      ctx.lineWidth = bundleWidth(tier, n) + 2 * tier.casingExtra;
      ctx.beginPath();
      for (const i of list) {
        const p = projected.get(i);
        if (p) appendOffsetPath(ctx, p.xs, p.ys, p.n, 0);
      }
      ctx.stroke();
    }

    // 3) Route-colour strokes, batched by colour.
    const byColor = new Map<string, { seg: number; offset: number }[]>();
    for (const i of visible) {
      const seg = net.segments[i];
      const n = seg.colors.length;
      seg.colors.forEach((color, k) => {
        const list = byColor.get(color) ?? [];
        list.push({ seg: i, offset: bundleOffset(tier, k, n) });
        byColor.set(color, list);
      });
    }
    ctx.lineWidth = tier.lineWidth;
    for (const [color, list] of byColor) {
      ctx.strokeStyle = color;
      ctx.beginPath();
      for (const { seg, offset } of list) {
        const p = projected.get(seg);
        if (p) appendOffsetPath(ctx, p.xs, p.ys, p.n, offset);
      }
      ctx.stroke();
    }
  }

  /**
   * Transfer passages between the stations of one complex: slim cased links, so
   * long passages (Times Sq ↔ Port Authority is ~300 m) read as walkways rather
   * than as another track.
   */
  private drawTransfers(ctx: Ctx2D, net: Network, proj: Proj, tier: ZoomTier): void {
    if (tier.id < 2) return;
    const links = new Path2D();
    for (const c of net.complexes) {
      if (c.stations.length < 2) continue;
      const rep = net.complexRep[c.idx];
      const r = net.stations[rep];
      if (!r) continue;
      for (const s of c.stations) {
        if (s === rep) continue;
        const st = net.stations[s];
        if (!st) continue;
        links.moveTo(proj.sx(r.x), proj.sy(r.y));
        links.lineTo(proj.sx(st.x), proj.sy(st.y));
      }
    }
    const core = Math.max(1.8, tier.stationRadius * 0.7);
    ctx.strokeStyle = PALETTE.stationStroke;
    ctx.lineWidth = core + 2 * tier.stationStroke + 1;
    ctx.stroke(links);
    ctx.strokeStyle = PALETTE.transfer;
    ctx.lineWidth = core;
    ctx.stroke(links);
  }

  private drawStations(ctx: Ctx2D, net: Network, proj: Proj, tier: ZoomTier, x0: number, y0: number, x1: number, y1: number): void {
    const local = new Path2D();
    const hubs = new Path2D();
    for (const st of net.stations) {
      if (st.x < x0 || st.x > x1 || st.y < y0 || st.y > y1) continue;
      if (!st.hub && !tier.showLocalStations) continue;
      const x = proj.sx(st.x);
      const y = proj.sy(st.y);
      const r = st.hub ? tier.hubRadius : tier.stationRadius;
      const path = st.hub ? hubs : local;
      path.moveTo(x + r, y);
      path.arc(x, y, r, 0, Math.PI * 2);
    }
    ctx.lineWidth = tier.stationStroke * 2;
    ctx.strokeStyle = PALETTE.stationStroke;
    ctx.stroke(local);
    ctx.fillStyle = PALETTE.stationFill;
    ctx.fill(local);
    ctx.lineWidth = tier.stationStroke * 2.2;
    ctx.strokeStyle = PALETTE.hubStroke;
    ctx.stroke(hubs);
    ctx.fillStyle = PALETTE.hubFill;
    ctx.fill(hubs);
  }

  private drawInterlockings(ctx: Ctx2D, net: Network, proj: Proj, tier: ZoomTier): void {
    if (!tier.showInterlockings) return;
    const { dx, dy } = interlockingOffset(tier);
    const subtle = tier.id <= 2;
    const s = subtle ? 2.3 : Math.max(2.8, tier.stationRadius * 0.85);
    const path = new Path2D();
    const leaders = new Path2D();
    for (const il of net.interlockings) {
      const st = net.stations[il.station];
      const bx = proj.sx(il.x);
      const by = proj.sy(il.y);
      const atStation = st ? Math.hypot(st.x - il.x, st.y - il.y) < 25 : false;
      const x = atStation ? bx + dx : bx;
      const y = atStation ? by + dy : by;
      if (atStation) {
        leaders.moveTo(bx, by);
        leaders.lineTo(x, y);
      }
      path.moveTo(x, y - s);
      path.lineTo(x + s, y);
      path.lineTo(x, y + s);
      path.lineTo(x - s, y);
      path.closePath();
    }
    ctx.save();
    ctx.globalAlpha = subtle ? 0.6 : 1;
    ctx.strokeStyle = "rgba(143, 179, 255, 0.45)";
    ctx.lineWidth = 1;
    ctx.stroke(leaders);
    ctx.fillStyle = PALETTE.interlocking;
    ctx.strokeStyle = PALETTE.stationStroke;
    ctx.lineWidth = 1.5;
    ctx.stroke(path);
    ctx.fill(path);
    ctx.restore();
  }

  private drawYardLeads(ctx: Ctx2D, net: Network, proj: Proj, tier: ZoomTier): void {
    if (!tier.showYards) return;
    const E = net.edges;
    ctx.save();
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = PALETTE.yardLead;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let e = 0; e < E.count; e++) {
      if (E.kind[e] !== EDGE_YARD_LEAD) continue;
      const yard = net.yards[E.yard[e]];
      const st = net.stations[E.station[e]];
      if (!yard || !st) continue;
      ctx.moveTo(proj.sx(yard.x), proj.sy(yard.y));
      ctx.lineTo(proj.sx(st.x), proj.sy(st.y));
    }
    ctx.stroke();
    ctx.restore();
  }

  private drawYards(ctx: Ctx2D, net: Network, proj: Proj, tier: ZoomTier): void {
    if (!tier.showYards) return;
    const s = yardHalfSize(tier);
    ctx.lineWidth = 1.5;
    ctx.fillStyle = PALETTE.yard;
    ctx.strokeStyle = PALETTE.stationStroke;
    for (const y of net.yards) {
      const x = proj.sx(y.x);
      const yy = proj.sy(y.y);
      ctx.strokeRect(x - s, yy - s, 2 * s, 2 * s);
      ctx.fillRect(x - s, yy - s, 2 * s, 2 * s);
    }
  }

  /**
   * Greedy label placement with collision avoidance (grid hash). Draws nothing.
   *
   * 1. Stations by label_rank up to the tier's rank cut-off.
   * 2. Fill pass (tier.labelFillGap): any remaining station whose label keeps that
   *    clearance from every placed label — names the sparse outer-borough lines
   *    without cluttering Midtown.
   * 3. Yard names (mid zoom on), muted, wherever they fit.
   *
   * HUD panels and borough names are obstacles; on dense tiers so are station and
   * yard markers. A label never straddles the viewport edge: an on-screen station
   * keeps its label on screen, and an off-screen one (in the pan padding) keeps its
   * label off screen.
   */
  private placeLabels(
    ctx: Ctx2D,
    net: Network,
    proj: Proj,
    tier: ZoomTier,
    areaW: number,
    areaH: number,
    obstacles: readonly LabelBox[],
  ): PlacedLabel[] {
    const out: PlacedLabel[] = [];
    const cell = 64;
    // Each grid entry remembers whether it is a label (clearance applies) or an obstacle.
    const grid = new Map<number, { b: LabelBox; label: boolean }[]>();
    const key = (gx: number, gy: number) => gx * 100003 + gy;
    const collides = (b: LabelBox, gap: number): boolean => {
      for (let gx = Math.floor((b.x0 - gap) / cell); gx <= Math.floor((b.x1 + gap) / cell); gx++) {
        for (let gy = Math.floor((b.y0 - gap) / cell); gy <= Math.floor((b.y1 + gap) / cell); gy++) {
          for (const { b: o, label } of grid.get(key(gx, gy)) ?? []) {
            const g = label ? gap : 0;
            if (b.x0 - g < o.x1 && b.x1 + g > o.x0 && b.y0 - g < o.y1 && b.y1 + g > o.y0) return true;
          }
        }
      }
      return false;
    };
    const insert = (b: LabelBox, label: boolean): void => {
      for (let gx = Math.floor(b.x0 / cell); gx <= Math.floor(b.x1 / cell); gx++) {
        for (let gy = Math.floor(b.y0 / cell); gy <= Math.floor(b.y1 / cell); gy++) {
          const k = key(gx, gy);
          const list = grid.get(k);
          if (list) list.push({ b, label });
          else grid.set(k, [{ b, label }]);
        }
      }
    };

    for (const o of obstacles) insert(o, false);
    if (tier.labelsAvoidMarkers) {
      for (const st of net.stations) {
        const x = proj.sx(st.x);
        const y = proj.sy(st.y);
        if (x < -20 || y < -20 || x > areaW + 20 || y > areaH + 20) continue;
        const r = (st.hub ? tier.hubRadius : tier.stationRadius) + 1;
        insert({ x0: x - r, y0: y - r, x1: x + r, y1: y + r }, false);
      }
    }
    if (tier.showYards && tier.labelsAvoidMarkers) {
      const s = yardHalfSize(tier) + 1;
      for (const yd of net.yards) {
        const x = proj.sx(yd.x);
        const y = proj.sy(yd.y);
        insert({ x0: x - s, y0: y - s, x1: x + s, y1: y + s }, false);
      }
    }

    // The visible viewport inside the padded snapshot.
    const vx0 = PAD_PX;
    const vy0 = PAD_PX;
    const vx1 = areaW - PAD_PX;
    const vy1 = areaH - PAD_PX;

    /** Try the six anchor positions around a marker of radius r; true when placed. */
    const tryPlace = (wx: number, wy: number, r: number, text: string, font: string, size: number, color: string, gap: number): boolean => {
      const x = proj.sx(wx);
      const y = proj.sy(wy);
      if (x < 0 || y < 0 || x > areaW || y > areaH) return false;
      const onScreen = x >= vx0 && x <= vx1 && y >= vy0 && y <= vy1;
      const w = this.measure(ctx, font, text);
      const h = size + 2;
      const options: [number, number, CanvasTextAlign][] = [
        [x + r, y, "left"],
        [x - r, y, "right"],
        [x + r * 0.7, y - r - h / 2, "left"],
        [x + r * 0.7, y + r + h / 2, "left"],
        [x - r * 0.7, y - r - h / 2, "right"],
        [x - r * 0.7, y + r + h / 2, "right"],
      ];
      for (const [lx, ly, align] of options) {
        const bx0 = align === "left" ? lx - 1 : lx - w - 1;
        const box = { x0: bx0, y0: ly - h / 2, x1: bx0 + w + 2, y1: ly + h / 2 };
        if (box.x0 < 0 || box.y0 < 0 || box.x1 > areaW || box.y1 > areaH) continue;
        const inside = box.x0 >= vx0 && box.x1 <= vx1 && box.y0 >= vy0 && box.y1 <= vy1;
        const outside = box.x1 <= vx0 || box.x0 >= vx1 || box.y1 <= vy0 || box.y0 >= vy1;
        if (onScreen ? !inside : !outside) continue;
        if (collides(box, gap)) continue;
        insert(box, true);
        out.push({ text, font, color, align, wx, wy, ox: lx - x, oy: ly - y, box: { x0: box.x0 - x, y0: box.y0 - y, x1: box.x1 - x, y1: box.y1 - y } });
        return true;
      }
      return false;
    };

    const placeStation = (st: GeoStation, gap: number): boolean => {
      const major = st.hub || st.label_rank < 12;
      const size = major ? tier.hubLabelSize : tier.labelSize;
      const font = `${major ? 700 : 500} ${size}px ${FONT_STACK}`;
      const r = (st.hub ? tier.hubRadius : tier.stationRadius) + 3;
      return tryPlace(st.x, st.y, r, st.name, font, size, major ? PALETTE.label : PALETTE.labelMuted, gap);
    };

    const candidates = labelCandidates(net, tier).sort((a, b) => a.label_rank - b.label_rank);
    const rest: GeoStation[] = [];
    for (const st of candidates) {
      if (st.label_rank < tier.labelMaxRank) placeStation(st, 0);
      else rest.push(st);
    }
    if (tier.labelFillGap > 0) {
      for (const st of rest) {
        if (st.label_rank >= tier.labelFillMaxRank) continue;
        if (tier.labelFillNamedOnly && BARE_NUMBERED_STREET.test(st.name)) continue;
        placeStation(st, tier.labelFillGap);
      }
    }

    if (tier.showYards && tier.id >= 2) {
      const size = tier.labelSize - 1;
      const font = `600 ${size}px ${FONT_STACK}`;
      for (const yd of net.yards) tryPlace(yd.x, yd.y, yardHalfSize(tier) + 3, yd.name, font, size, PALETTE.yardLabel, 0);
    }
    return out;
  }

  /** Draw the placed station labels at the current camera (called after the dynamic layer). */
  drawLabels(ctx: Ctx2D, camera: Camera, hide: readonly LabelBox[] = []): void {
    const labels = this.labels;
    if (labels.length === 0) return;
    const W = camera.width;
    const H = camera.height;
    ctx.save();
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.lineWidth = 3.4;
    ctx.strokeStyle = PALETTE.labelHalo;
    let font = "";
    for (const l of labels) {
      const px = camera.toScreenX(l.wx);
      const py = camera.toScreenY(l.wy);
      const x = px + l.ox;
      const y = py + l.oy;
      if (y < -16 || y > H + 16 || x < -320 || x > W + 320) continue;
      // A disruption chip on top of this label hides it (the chip names the place anyway).
      if (hide.some((c) => px + l.box.x0 < c.x1 && px + l.box.x1 > c.x0 && py + l.box.y0 < c.y1 && py + l.box.y1 > c.y0)) continue;
      if (l.font !== font) {
        font = l.font;
        ctx.font = font;
      }
      ctx.textAlign = l.align;
      ctx.strokeText(l.text, x, y);
      ctx.fillStyle = l.color;
      ctx.fillText(l.text, x, y);
    }
    ctx.restore();
  }
}
