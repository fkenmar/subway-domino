/**
 * Placement of a popover anchored to a map point: above the anchor when it
 * fits, otherwise below, otherwise beside it — always clamped to the viewport
 * and kept clear of the HUD bands (`insets`). Pure, so it is unit-tested.
 */
import type { Insets } from "./layout";

export type ArrowSide = "top" | "bottom" | "left" | "right";

export interface PopoverPlacement {
  left: number;
  top: number;
  /** Which edge of the popover the arrow sits on, and its offset along that edge (px). */
  arrow: { side: ArrowSide; offset: number };
}

export interface PlacementInput {
  anchor: { x: number; y: number };
  size: { w: number; h: number };
  viewport: { w: number; h: number };
  /** HUD bands to avoid (top header, bottom dock, side column). */
  insets: Insets;
  gap?: number;
  edge?: number;
}

const clampTo = (v: number, lo: number, hi: number): number => (hi < lo ? lo : v < lo ? lo : v > hi ? hi : v);

export function placePopover({ anchor, size, viewport, insets, gap = 14, edge = 8 }: PlacementInput): PopoverPlacement {
  const { w, h } = size;
  const minTop = Math.max(edge, insets.top - 48);
  const maxBottom = viewport.h - Math.max(edge, insets.bottom - 24);
  const minLeft = edge;
  const maxRight = viewport.w - edge;
  const arrowInset = 16;

  // Above / below: horizontally centred on the anchor, clamped.
  const left = clampTo(anchor.x - w / 2, minLeft, maxRight - w);
  const arrowX = clampTo(anchor.x - left, arrowInset, w - arrowInset);
  if (anchor.y - gap - h >= minTop) {
    return { left, top: anchor.y - gap - h, arrow: { side: "bottom", offset: arrowX } };
  }
  if (anchor.y + gap + h <= maxBottom) {
    return { left, top: anchor.y + gap, arrow: { side: "top", offset: arrowX } };
  }

  // Beside: on the roomier side, vertically centred on the anchor, clamped.
  const roomRight = maxRight - (anchor.x + gap);
  const roomLeft = anchor.x - gap - minLeft;
  const right = roomRight >= w || roomRight >= roomLeft;
  const sideLeft = right ? Math.min(anchor.x + gap, maxRight - w) : Math.max(minLeft, anchor.x - gap - w);
  const top = clampTo(anchor.y - h / 2, minTop, maxBottom - h);
  const arrowY = clampTo(anchor.y - top, arrowInset, h - arrowInset);
  return { left: sideLeft, top, arrow: { side: right ? "left" : "right", offset: arrowY } };
}
