/**
 * Imperative map engine: owns the canvas, camera, cached base layer, dynamic
 * renderer and hit-tester; runs the requestAnimationFrame loop and handles
 * pan (drag + inertia), zoom (wheel / pinch / double-click / keys) about the
 * cursor, hover tooltips and clicks that open the DisruptionPopover.
 */
import { Camera } from "@/lib/camera";
import { targetAnchor } from "@/lib/disruptions";
import { clamp, nearestOnPolyline, pointAt } from "@/lib/geometry";
import { InterpolatedTrains } from "@/lib/interp";
import { centerWithInsets, currentInsets, currentOccluders, fitWithInsets, homeView } from "@/lib/layout";
import type { Network } from "@/lib/network";
import { STATUS } from "@/lib/protocol";
import { getSession } from "@/lib/session";
import { type CameraRequest, useStore } from "@/lib/store";
import { parseViewHash, viewToCamera } from "@/lib/viewHash";
import { HitTester } from "./hitTest";
import { mapView } from "./mapView";
import { BaseLayer } from "./renderBase";
import { DynamicRenderer, type HoverTarget } from "./renderDynamic";
import { PALETTE, tierFor } from "./style";

const CLICK_SLOP_PX = 5;
const INERTIA_FRICTION = 0.9;
const WHEEL_SENSITIVITY = 0.0022;
const KEY_PAN_PX = 90;

interface DragState {
  id: number;
  lastX: number;
  lastY: number;
  startX: number;
  startY: number;
  moved: boolean;
  samples: { x: number; y: number; t: number }[];
  touch: boolean;
}

export interface MapEngineElements {
  canvas: HTMLCanvasElement;
  container: HTMLElement;
  tooltip: HTMLElement | null;
}

export class MapEngine {
  private readonly camera = new Camera();
  private readonly base: BaseLayer;
  private readonly dyn = new DynamicRenderer();
  private readonly hit: HitTester;
  private readonly trains = new InterpolatedTrains();
  private readonly ghosts = new InterpolatedTrains(256);
  private readonly crowd: Float32Array;
  private readonly ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private raf = 0;
  private sized = false;
  private readonly pointers = new Map<number, { x: number; y: number }>();
  private drag: DragState | null = null;
  private pinch: { dist: number; cx: number; cy: number } | null = null;
  private vx = 0;
  private vy = 0;
  private lastInertiaAt = 0;
  private hover: HoverTarget | null = null;
  private hoverPoint: { x: number; y: number } | null = null;
  private hoverDirty = false;
  private lastNotified = -1;
  private readonly cleanups: (() => void)[] = [];
  private resizeObserver: ResizeObserver | null = null;

  constructor(
    private readonly net: Network,
    private readonly el: MapEngineElements,
  ) {
    const ctx = el.canvas.getContext("2d", { alpha: false });
    if (!ctx) throw new Error("Canvas 2D is not available");
    this.ctx = ctx;
    this.base = new BaseLayer(() => document.createElement("canvas"));
    this.hit = new HitTester(net);
    this.crowd = new Float32Array(net.stations.length);
    this.camera.setBounds(net.bbox);
    this.camera.setContentProbe((x, y, r) => this.hit.nearestContent(x, y, r));
  }

  start(): void {
    const { canvas, container } = this.el;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    mapView.attach(this.camera);

    const on = <K extends keyof HTMLElementEventMap>(
      target: HTMLElement,
      type: K,
      fn: (e: HTMLElementEventMap[K]) => void,
      opts?: AddEventListenerOptions,
    ) => {
      target.addEventListener(type, fn, opts);
      this.cleanups.push(() => target.removeEventListener(type, fn, opts));
    };
    on(canvas, "pointerdown", this.onPointerDown);
    on(canvas, "pointermove", this.onPointerMove);
    on(canvas, "pointerup", this.onPointerUp);
    on(canvas, "pointercancel", this.onPointerUp);
    on(canvas, "pointerleave", this.onPointerLeave);
    on(canvas, "wheel", this.onWheel, { passive: false });
    on(canvas, "dblclick", this.onDoubleClick);
    on(canvas, "keydown", this.onKeyDown);
    on(canvas, "contextmenu", (e) => e.preventDefault());

    this.cleanups.push(
      useStore.subscribe((s, prev) => {
        if (s.cameraRequest && s.cameraRequest !== prev.cameraRequest) this.onCameraRequest(s.cameraRequest);
      }),
    );
    this.raf = requestAnimationFrame(this.loop);
  }

  destroy(): void {
    cancelAnimationFrame(this.raf);
    this.resizeObserver?.disconnect();
    for (const c of this.cleanups.splice(0)) c();
    mapView.attach(null);
  }

  // ─── sizing ──────────────────────────────────────────────────────────────

  private resize(): void {
    const { canvas, container } = this.el;
    const rect = container.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    const dpr = Math.min(3, Math.max(1, window.devicePixelRatio || 1));
    this.dpr = dpr;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    this.camera.setViewport(w, h);
    this.camera.configureForExtent(this.net.bbox);
    if (!this.sized) {
      this.sized = true;
      const insets = currentInsets(w, h);
      // A `#map=m/lat/lon` deep link opens on that view; otherwise fit the whole city.
      const linked = parseViewHash(window.location.hash);
      if (linked) {
        const v = viewToCamera(linked, this.net.projection);
        this.camera.set(centerWithInsets(v.cx, v.cy, v.mPerPx, insets));
      } else {
        this.camera.set(this.homeView(w, h));
      }
    }
  }

  // ─── frame loop ──────────────────────────────────────────────────────────

  private loop = (now: number): void => {
    this.raf = requestAnimationFrame(this.loop);
    if ((window.devicePixelRatio || 1) !== this.dpr) this.resize();
    const cam = this.camera;
    cam.step(now);
    this.applyInertia(now);

    const s = useStore.getState();
    const frames = getSession().frames;
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = PALETTE.background;
    ctx.fillRect(0, 0, cam.width, cam.height);
    this.base.drawTo(ctx, this.net, cam, this.dpr, now);

    frames.sample(now, this.net.edges, this.trains, this.ghosts);
    const crowdCount = frames.sampleStationCrowd(now, this.crowd);
    const tier = tierFor(cam.mPerPx);
    if (this.hoverDirty) this.updateHover(tier);
    const scene = {
      net: this.net,
      camera: cam,
      tier,
      dpr: this.dpr,
      now,
      trains: this.trains,
      ghosts: this.ghosts,
      crowd: this.crowd,
      crowdCount,
      disruptions: s.disruptions,
      // The latest frame's clock (not the interpolated one) so map chips and the list agree.
      clockS: s.clockS ?? frames.clockAt(now),
      selection: s.selection,
      hover: this.hover,
      highlight: s.highlight,
    };
    this.dyn.render(ctx, scene);
    // Chips are laid out first so station labels they would cover give way to them.
    const chips = this.dyn.layoutChips(ctx, scene);
    this.base.drawLabels(ctx, cam, chips);
    this.dyn.renderChips(ctx);
    if (cam.version !== this.lastNotified) {
      this.lastNotified = cam.version;
      this.hoverDirty = this.hoverPoint !== null;
      mapView.notify();
    }
    this.updateTooltip();
  };

  private applyInertia(now: number): void {
    if (this.drag || (this.vx === 0 && this.vy === 0)) {
      this.lastInertiaAt = now;
      return;
    }
    const dt = Math.min(48, now - this.lastInertiaAt);
    this.lastInertiaAt = now;
    this.camera.panBy(this.vx * dt, this.vy * dt);
    const decay = INERTIA_FRICTION ** (dt / 16.7);
    this.vx *= decay;
    this.vy *= decay;
    if (Math.hypot(this.vx, this.vy) < 0.02) {
      this.vx = 0;
      this.vy = 0;
    }
  }

  // ─── camera requests (presets, search, feed) ─────────────────────────────

  private onCameraRequest(req: CameraRequest): void {
    const cam = this.camera;
    const insets = currentInsets(cam.width, cam.height);
    const now = performance.now();
    this.vx = 0;
    this.vy = 0;
    if (req.kind === "view") {
      cam.flyTo(centerWithInsets(req.view.cx, req.view.cy, req.view.mPerPx, insets), now);
    } else if (req.kind === "bbox") {
      const view = fitWithInsets(req.bbox, cam.width, cam.height, insets, 32);
      cam.flyTo({ ...view, mPerPx: Math.max(view.mPerPx, 2.2) }, now);
    } else if (req.kind === "zoom") {
      // Keep the world point at the centre of the visible map area fixed.
      const sx = insets.left + (cam.width - insets.left - insets.right) / 2;
      const sy = insets.top + (cam.height - insets.top - insets.bottom) / 2;
      const target = clamp(cam.mPerPx / req.factor, cam.minMPerPx, cam.maxMPerPx);
      const wx = cam.toWorldX(sx);
      const wy = cam.toWorldY(sy);
      cam.flyTo(
        { cx: wx - (sx - cam.width / 2) * target, cy: wy + (sy - cam.height / 2) * target, mPerPx: target },
        now,
        { minDurationMs: 260, maxDurationMs: 420 },
      );
    } else {
      cam.flyTo(this.homeView(cam.width, cam.height), now);
    }
  }

  /** The whole network, centred on the visible map (see `homeView`). */
  private homeView(w: number, h: number) {
    return homeView(this.net.bbox, this.net.stations, w, h, currentInsets(w, h), currentOccluders().rects);
  }

  // ─── input ───────────────────────────────────────────────────────────────

  private local(e: { clientX: number; clientY: number }): { x: number; y: number } {
    const r = this.el.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  private onPointerDown = (e: PointerEvent): void => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    getSession().activity();
    const p = this.local(e);
    this.el.canvas.setPointerCapture(e.pointerId);
    this.pointers.set(e.pointerId, p);
    this.camera.cancelFlight();
    this.vx = 0;
    this.vy = 0;
    if (this.pointers.size === 1) {
      this.drag = {
        id: e.pointerId,
        lastX: p.x,
        lastY: p.y,
        startX: p.x,
        startY: p.y,
        moved: false,
        samples: [{ x: p.x, y: p.y, t: e.timeStamp }],
        touch: e.pointerType !== "mouse",
      };
      this.el.canvas.style.cursor = "grabbing";
    } else if (this.pointers.size === 2) {
      this.drag = null;
      this.pinch = this.pinchState();
    }
  };

  private pinchState(): { dist: number; cx: number; cy: number } | null {
    const [a, b] = [...this.pointers.values()];
    if (!a || !b) return null;
    return { dist: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)), cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2 };
  }

  private onPointerMove = (e: PointerEvent): void => {
    const p = this.local(e);
    if (this.pointers.has(e.pointerId)) this.pointers.set(e.pointerId, p);
    if (this.pinch && this.pointers.size >= 2) {
      const next = this.pinchState();
      if (next) {
        this.camera.zoomAt(next.cx, next.cy, next.dist / this.pinch.dist);
        this.camera.panBy(next.cx - this.pinch.cx, next.cy - this.pinch.cy);
        this.pinch = next;
      }
      return;
    }
    const d = this.drag;
    if (d && d.id === e.pointerId) {
      const dx = p.x - d.lastX;
      const dy = p.y - d.lastY;
      if (!d.moved && Math.hypot(p.x - d.startX, p.y - d.startY) > CLICK_SLOP_PX) d.moved = true;
      if (d.moved) this.camera.panBy(dx, dy);
      d.lastX = p.x;
      d.lastY = p.y;
      d.samples.push({ x: p.x, y: p.y, t: e.timeStamp });
      if (d.samples.length > 8) d.samples.shift();
      return;
    }
    if (e.pointerType === "mouse") {
      getSession().activity();
      this.hoverPoint = p;
      this.hoverDirty = true;
    }
  };

  private onPointerUp = (e: PointerEvent): void => {
    if (!this.pointers.has(e.pointerId)) return;
    this.pointers.delete(e.pointerId);
    if (this.el.canvas.hasPointerCapture(e.pointerId)) this.el.canvas.releasePointerCapture(e.pointerId);
    if (this.pinch) {
      if (this.pointers.size < 2) this.pinch = null;
      if (this.pointers.size === 1) {
        const [id, q] = [...this.pointers.entries()][0];
        this.drag = { id, lastX: q.x, lastY: q.y, startX: q.x, startY: q.y, moved: true, samples: [], touch: true };
      }
      return;
    }
    const d = this.drag;
    this.drag = null;
    this.el.canvas.style.cursor = "";
    if (!d || d.id !== e.pointerId || e.type === "pointercancel") return;
    if (!d.moved) {
      this.click(d.startX, d.startY, d.touch);
      return;
    }
    // Release velocity from the last ~80 ms of samples.
    const last = d.samples[d.samples.length - 1];
    const first = d.samples.find((s) => last && last.t - s.t <= 80) ?? d.samples[0];
    if (first && last && last.t > first.t) {
      this.vx = (last.x - first.x) / (last.t - first.t);
      this.vy = (last.y - first.y) / (last.t - first.t);
      this.lastInertiaAt = performance.now();
    }
  };

  private onPointerLeave = (): void => {
    this.hoverPoint = null;
    this.hover = null;
    this.hoverDirty = false;
    this.el.canvas.style.cursor = "";
  };

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    getSession().activity();
    const p = this.local(e);
    const unit = e.deltaMode === 1 ? 18 : e.deltaMode === 2 ? 400 : 1;
    const k = e.ctrlKey ? WHEEL_SENSITIVITY * 4 : WHEEL_SENSITIVITY; // trackpad pinch arrives as ctrl+wheel
    const factor = Math.min(2, Math.max(0.5, Math.exp(-e.deltaY * unit * k)));
    this.vx = 0;
    this.vy = 0;
    this.camera.zoomAt(p.x, p.y, factor);
  };

  private onDoubleClick = (e: MouseEvent): void => {
    const p = this.local(e);
    const cam = this.camera;
    const target = cam.mPerPx / 2.2;
    const wx = cam.toWorldX(p.x);
    const wy = cam.toWorldY(p.y);
    // Keep the double-clicked point under the cursor at the new scale.
    cam.flyTo(
      { cx: wx - (p.x - cam.width / 2) * target, cy: wy + (p.y - cam.height / 2) * target, mPerPx: target },
      performance.now(),
      { minDurationMs: 320, maxDurationMs: 520 },
    );
  };

  private onKeyDown = (e: KeyboardEvent): void => {
    const cam = this.camera;
    getSession().activity();
    switch (e.key) {
      case "ArrowLeft":
        cam.panBy(KEY_PAN_PX, 0);
        break;
      case "ArrowRight":
        cam.panBy(-KEY_PAN_PX, 0);
        break;
      case "ArrowUp":
        cam.panBy(0, KEY_PAN_PX);
        break;
      case "ArrowDown":
        cam.panBy(0, -KEY_PAN_PX);
        break;
      case "+":
      case "=":
        cam.zoomAt(cam.width / 2, cam.height / 2, 1.5);
        break;
      case "-":
      case "_":
        cam.zoomAt(cam.width / 2, cam.height / 2, 1 / 1.5);
        break;
      case "0":
        useStore.getState().requestCamera({ kind: "home" });
        break;
      case "Escape":
        useStore.getState().select(null);
        break;
      default:
        return;
    }
    e.preventDefault();
  };

  private click(x: number, y: number, touch: boolean): void {
    const s = useStore.getState();
    const target = this.hit.hit(this.camera, tierFor(this.camera.mPerPx), x, y, { coarse: touch, chips: this.dyn.chips });
    if (!target || target.type === "train") {
      s.select(null);
      return;
    }
    if (target.type === "disruption") {
      const d = s.disruptions.find((item) => item.id === target.id);
      const a = d ? targetAnchor(this.net, d.target) : null;
      if (d && a) s.select({ target: d.target, wx: a.x, wy: a.y, disruptionId: d.id });
      return;
    }
    const sel = s.selection;
    if (sel && !sel.disruptionId && sel.target.type === target.type && sel.target.idx === target.idx) {
      s.select(null); // second click on the same target closes the popover
      return;
    }
    let wx: number;
    let wy: number;
    if (target.type === "station") {
      const st = this.net.stations[target.idx];
      wx = st.x;
      wy = st.y;
    } else if (target.type === "interlocking") {
      const il = this.net.interlockings[target.idx];
      wx = il.x;
      wy = il.y;
    } else {
      const line = this.net.segments[target.idx].line;
      const near = nearestOnPolyline(line, this.camera.toWorldX(x), this.camera.toWorldY(y));
      const pt = pointAt(line, near.t, { x: 0, y: 0, tx: 0, ty: 0 });
      wx = pt.x;
      wy = pt.y;
    }
    s.select({ target: { type: target.type, idx: target.idx }, wx, wy });
  }

  private updateHover(tier: ReturnType<typeof tierFor>): void {
    this.hoverDirty = false;
    const p = this.hoverPoint;
    if (!p || this.drag) return;
    this.hover = this.hit.hit(this.camera, tier, p.x, p.y, {
      coarse: false,
      chips: this.dyn.chips,
      includeTrains: true,
      trainXY: this.dyn.trainXY,
      trainIds: this.dyn.trainIds,
      trainCount: this.dyn.trainCount,
    });
    this.el.canvas.style.cursor = this.hover ? "pointer" : "";
  }

  private updateTooltip(): void {
    const tip = this.el.tooltip;
    if (!tip) return;
    const h = this.hover;
    const p = this.hoverPoint;
    if (!h || !p || this.drag) {
      if (!tip.hidden) tip.hidden = true;
      return;
    }
    const text = this.describe(h);
    if (!text) {
      if (!tip.hidden) tip.hidden = true;
      return;
    }
    if (tip.textContent !== text) tip.textContent = text;
    tip.hidden = false;
    const x = Math.min(p.x + 14, this.camera.width - tip.offsetWidth - 8);
    const y = Math.max(8, p.y - tip.offsetHeight - 10);
    tip.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  }

  private describe(h: HoverTarget): string | null {
    const net = this.net;
    switch (h.type) {
      case "station": {
        const st = net.stations[h.idx];
        if (!st) return null;
        const routes = st.routes.map((r) => net.routes[r]?.bullet).filter(Boolean).join(" ");
        return `${st.name}${routes ? ` · ${routes}` : ""} — click to disrupt`;
      }
      case "segment": {
        const seg = net.segments[h.idx];
        if (!seg) return null;
        const a = net.stations[seg.a]?.name ?? "";
        const b = net.stations[seg.b]?.name ?? "";
        return `${seg.river ? "River crossing: " : ""}${a} ↔ ${b} — click to disrupt`;
      }
      case "interlocking": {
        const il = net.interlockings[h.idx];
        const st = il ? net.stations[il.station] : undefined;
        return st ? `Interlocking at ${st.name} (${il?.kinds.join(", ")}) — click to disrupt` : null;
      }
      case "disruption": {
        const d = useStore.getState().disruptions.find((x) => x.id === h.id);
        return d ? `${d.label} — click to clear` : null;
      }
      case "train": {
        const t = this.trains;
        for (let i = 0; i < t.count; i++) {
          if (t.trainId[i] !== h.trainId) continue;
          const route = net.routes[t.route[i]];
          const s = t.status[i];
          const flags: string[] = [];
          if (s & STATUS.TUNNEL_STUCK) flags.push("stuck in tunnel");
          if (s & STATUS.HELD) flags.push("held at platform");
          if (s & STATUS.SHORT_TURNING) flags.push("short-turning");
          if (s & STATUS.REROUTED) flags.push("rerouted");
          if (s & STATUS.GAP_TRAIN) flags.push("gap train");
          if (s & STATUS.SKIPPING_NEXT) flags.push("skipping next stop");
          if (s & STATUS.TRACK_SWITCH) flags.push("switching tracks");
          if (!flags.length && s & STATUS.AT_PLATFORM) flags.push("at platform");
          return `${route ? `${route.bullet} train` : "Train"} · ${t.load[i]}% full${flags.length ? ` · ${flags.join(", ")}` : ""}`;
        }
        return null;
      }
      default:
        return null;
    }
  }
}
