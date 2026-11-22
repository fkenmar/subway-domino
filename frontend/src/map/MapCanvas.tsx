"use client";

import { useEffect, useRef } from "react";
import { useStore } from "@/lib/store";
import { MapEngine } from "./engine";
import styles from "./MapCanvas.module.css";

/** Full-bleed Canvas 2D map. The static map paints as soon as the geo file loads. */
export default function MapCanvas({ onRetry }: { onRetry?: () => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const network = useStore((s) => s.network);
  const networkError = useStore((s) => s.networkError);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!network || !canvas || !container) return;
    let engine: MapEngine;
    try {
      engine = new MapEngine(network, { canvas, container, tooltip: tooltipRef.current });
    } catch (err) {
      console.error("[subway-domino] map engine failed to start", err);
      return;
    }
    engine.start();
    return () => engine.destroy();
  }, [network]);

  return (
    <div ref={containerRef} className={styles.root} data-ready={network ? "true" : "false"}>
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        tabIndex={0}
        role="application"
        aria-roledescription="interactive subway map"
        aria-label="NYC subway map. Drag to pan, scroll or pinch to zoom, click a station or track to disrupt it. Arrow keys pan, plus and minus zoom, 0 recentres."
      />
      <div ref={tooltipRef} className={styles.tooltip} role="tooltip" hidden />
      {!network && (
        <div className={styles.skeleton} aria-live="polite">
          {networkError ? (
            <div className={styles.skeletonCard}>
              <strong>Couldn’t load the subway map.</strong>
              <span>{networkError}</span>
              {onRetry && (
                <button type="button" className={styles.retry} onClick={onRetry}>
                  Try again
                </button>
              )}
            </div>
          ) : (
            <div className={styles.skeletonCard}>
              <span className={styles.spinner} aria-hidden="true" />
              <span>Loading the NYC subway map…</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
