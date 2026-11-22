/**
 * A tiny observable handle on the live map camera, so DOM overlays (popover,
 * onboarding) can anchor to world points without re-rendering on every frame
 * unless they are mounted and the camera actually moved.
 */
import { useSyncExternalStore } from "react";
import type { Camera } from "@/lib/camera";

type Listener = () => void;

class MapView {
  camera: Camera | null = null;
  private version = 0;
  private readonly listeners = new Set<Listener>();

  attach(camera: Camera | null): void {
    this.camera = camera;
    this.notify();
  }

  notify(): void {
    this.version++;
    for (const l of this.listeners) l();
  }

  subscribe = (l: Listener): (() => void) => {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  };

  getVersion = (): number => this.version;

  /** World → screen (CSS px relative to the map), or null before the map mounts. */
  project(x: number, y: number): { x: number; y: number } | null {
    const c = this.camera;
    return c ? { x: c.toScreenX(x), y: c.toScreenY(y) } : null;
  }
}

export const mapView = new MapView();

/** Re-render the calling component whenever the camera moves. */
export function useMapViewVersion(): number {
  return useSyncExternalStore(mapView.subscribe, mapView.getVersion, () => 0);
}
