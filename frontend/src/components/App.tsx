"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { BREAKPOINT_DESKTOP, clampInsets, type ScreenRect, setMeasuredInsets, setOccluders } from "@/lib/layout";
import { loadNetwork } from "@/lib/network";
import { getSession } from "@/lib/session";
import { useStore } from "@/lib/store";
import MapCanvas from "@/map/MapCanvas";
import ActivityFeed, { ActivityFeedToggle } from "./ActivityFeed";
import styles from "./App.module.css";
import CalloutLayer from "./CalloutLayer";
import ConnectionBanner from "./ConnectionBanner";
import DisruptionList from "./DisruptionList";
import DisruptionPopover from "./DisruptionPopover";
import Header from "./Header";
import InfoModal from "./InfoModal";
import MapControls from "./MapControls";
import MetricsHUD from "./MetricsHUD";
import ModeToggle from "./ModeToggle";
import Onboarding from "./Onboarding";
import PresetBar from "./PresetBar";
import ProfileSelect from "./ProfileSelect";
import ResetButton from "./ResetButton";
import SpeedControl from "./SpeedControl";

function describeError(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  return "The network geometry could not be loaded.";
}

/**
 * Measure the HUD chrome and publish (a) the map area it leaves uncovered
 * (elements tagged `data-inset`), so camera fits and fly-tos centre on visible
 * map, and (b) the panels covering the map (`data-occlude`), which station
 * labels avoid.
 */
function useMeasuredInsets(root: React.RefObject<HTMLDivElement | null>): void {
  useLayoutEffect(() => {
    const el = root.current;
    if (!el) return;
    const measure = () => {
      const W = window.innerWidth;
      const H = window.innerHeight;
      const desktop = W >= BREAKPOINT_DESKTOP;
      let top = 0;
      let bottom = 0;
      let right = 0;
      for (const node of el.querySelectorAll<HTMLElement>("[data-inset]")) {
        const r = node.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) continue;
        const role = node.dataset.inset;
        if (role === "top") top = Math.max(top, r.bottom);
        else if (role === "bottom" || (role === "side" && !desktop)) bottom = Math.max(bottom, H - r.top);
        else if (role === "side") right = Math.max(right, W - r.left);
      }
      const insets = { top: top + 8, right: right + (desktop ? 12 : 6), bottom: bottom + 8, left: desktop ? 20 : 6 };
      setMeasuredInsets(clampInsets(insets, W, H), W, H);
      const occluders: ScreenRect[] = [];
      for (const node of el.querySelectorAll<HTMLElement>("[data-occlude]")) {
        const r = node.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) occluders.push({ x0: r.left, y0: r.top, x1: r.right, y1: r.bottom });
      }
      setOccluders(occluders);
    };
    // Measure at most once per frame, however many resizes / mutations arrive.
    let frame = 0;
    const schedule = () => {
      if (frame === 0) {
        frame = requestAnimationFrame(() => {
          frame = 0;
          measure();
        });
      }
    };
    const ro = new ResizeObserver(schedule);
    const observed = new WeakSet<Element>();
    const observeTagged = () => {
      for (const node of el.querySelectorAll<HTMLElement>("[data-inset], [data-occlude]")) {
        if (observed.has(node)) continue;
        observed.add(node);
        ro.observe(node);
      }
    };
    ro.observe(el);
    observeTagged();
    measure();
    // Panels mount late (map controls after the geometry, the disruption list on the
    // first disruption): pick up newly tagged elements as they appear.
    const mo = new MutationObserver(() => {
      observeTagged();
      schedule();
    });
    mo.observe(el, { childList: true, subtree: true });
    return () => {
      cancelAnimationFrame(frame);
      mo.disconnect();
      ro.disconnect();
      setMeasuredInsets(null);
      setOccluders([]);
    };
  }, [root]);
}

/** The whole page: full-bleed map with the control-room HUD layered on top. */
export default function App() {
  const setNetwork = useStore((s) => s.setNetwork);
  const setNetworkError = useStore((s) => s.setNetworkError);
  const [loadAttempt, setLoadAttempt] = useState(0);
  const overlayRef = useRef<HTMLDivElement>(null);
  useMeasuredInsets(overlayRef);

  // Map geometry: the static map paints from it before the socket connects (§6 UX flow 1).
  useEffect(() => {
    const ctrl = new AbortController();
    loadNetwork({ signal: ctrl.signal })
      .then((net) => {
        if (!ctrl.signal.aborted) setNetwork(net);
      })
      .catch((err: unknown) => {
        if (ctrl.signal.aborted) return;
        console.error("[subway-domino] failed to load network_geo.json", err);
        setNetworkError(describeError(err));
      });
    return () => ctrl.abort();
  }, [loadAttempt, setNetwork, setNetworkError]);

  // Live session over the WebSocket (connects in parallel with the geometry download).
  useEffect(() => {
    const session = getSession();
    session.start();
    return () => session.stop();
  }, []);

  // Any interaction with the page counts as activity (the client throttles the heartbeat).
  useEffect(() => {
    const onActivity = () => getSession().activity();
    const opts = { capture: true, passive: true } as const;
    window.addEventListener("pointerdown", onActivity, opts);
    window.addEventListener("keydown", onActivity, opts);
    return () => {
      window.removeEventListener("pointerdown", onActivity, opts);
      window.removeEventListener("keydown", onActivity, opts);
    };
  }, []);

  // The dispatch log starts open where there is room for it.
  useEffect(() => {
    if (window.innerWidth >= BREAKPOINT_DESKTOP) useStore.getState().setFeedOpen(true);
  }, []);

  return (
    <div className={styles.app}>
      <MapCanvas onRetry={() => setLoadAttempt((n) => n + 1)} />

      <div ref={overlayRef} className={styles.overlay}>
        <div className={styles.header} data-inset="top" data-occlude>
          <Header />
        </div>
        <div className={styles.presets} data-inset="top" data-occlude>
          <PresetBar />
        </div>
        <div className={styles.mapArea}>
          <ConnectionBanner />
          <CalloutLayer />
          <MapControls />
        </div>
        <aside className={styles.side} data-inset="side" aria-label="Live status">
          <div className={styles.sideHud} data-occlude>
            <MetricsHUD />
          </div>
          <div className={styles.sideDisruptions} data-occlude>
            <DisruptionList />
          </div>
          <div className={styles.sideFeed} data-occlude>
            <ActivityFeed />
          </div>
        </aside>
        <div className={styles.dock} data-inset="bottom" data-occlude role="toolbar" aria-label="Simulation controls">
          <div className={styles.dockSpeed}>
            <SpeedControl />
          </div>
          <div className={styles.dockToggle}>
            <ModeToggle />
          </div>
          <div className={styles.dockMore}>
            <ProfileSelect />
            <ResetButton />
            <ActivityFeedToggle />
          </div>
        </div>
      </div>

      <DisruptionPopover />
      <Onboarding />
      <InfoModal />
    </div>
  );
}
