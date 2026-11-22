/**
 * WP3d: every disruption is a blockage that lasts until the user clears it. The popover
 * offers no duration and no signal problems, the client always sends `duration_s: null`,
 * the active list and popover say "since …" (never a countdown) with an obvious restore
 * button, preset copy says "blocked", and clearing a block posts a "Service restored" notice.
 */
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import DisruptionList from "@/components/DisruptionList";
import DisruptionPopover, { choicesFor } from "@/components/DisruptionPopover";
import { PRESET_TITLE_OVERRIDES, presetTitle } from "@/lib/copy";
import type { ClientMessage, DisruptionInfo, ServerMessage } from "@/lib/protocol";
import { SessionController } from "@/lib/session";
import { useStore } from "@/lib/store";
import { miniNetwork } from "../helpers/geo";

const net = miniNetwork();

const BLOCK: DisruptionInfo = {
  id: 4,
  kind: "local_block",
  target: { type: "segment", idx: 0 },
  label: "Local track blocked: Times Sq-42 St ↔ 34 St-Penn Station",
  start_t: 29_520, // 8:12 AM
  end_t: null,
};

// Server rendering reads zustand's *initial* state, so the SSR tests set it there and put it back.
const initial = { ...useStore.getInitialState() };
function withState(patch: Partial<ReturnType<typeof useStore.getState>>): void {
  Object.assign(useStore.getInitialState(), patch);
  useStore.setState(patch);
}
beforeEach(() => {
  withState({ network: net, selection: null, disruptions: [], toasts: [], clockS: 30_000 });
});
afterEach(() => {
  Object.assign(useStore.getInitialState(), initial);
  useStore.setState(initial, true);
});

const render = (el: Parameters<typeof renderToStaticMarkup>[0]) => renderToStaticMarkup(el);
const text = (html: string) => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");

describe("disruption popover", () => {
  it("offers blockage kinds only, with no duration choice", () => {
    for (const target of [
      { type: "segment", idx: 0 },
      { type: "interlocking", idx: 0 },
      { type: "station", idx: 0 },
    ] as const) {
      expect(choicesFor(net, target).map((c) => c.kind)).not.toContain("signal_delay");
      withState({ selection: { target, wx: 0, wy: 0 } });
      const html = render(createElement(DisruptionPopover));
      const t = text(html);
      expect(t).toContain("What breaks?");
      expect(t).not.toMatch(/For how long|Until cleared|\d+ min\b|Signal problems/);
      expect(html).not.toContain('aria-label="Duration"');
      expect(t).toMatch(/It stays (blocked|closed) until you restore service\./);
    }
  });

  it("leads with a clear Restore service button on a blocked target, and no countdown", () => {
    withState({ selection: { target: BLOCK.target, wx: 0, wy: 0 }, disruptions: [{ ...BLOCK, end_t: 31_320 }] });
    const t = text(render(createElement(DisruptionPopover)));
    expect(t).toContain("Restore service");
    expect(t).toContain("Blocked since 8:12 AM");
    expect(t).not.toMatch(/left|∞|\d+:\d\d(?! [AP]M)/);
    expect(t).toContain("Block something else here…");
    expect(t).not.toContain("What breaks?"); // the add form waits behind "Block something else here…"
  });
});

describe("active blockages list", () => {
  it("lists each block with since-when, a Restore button and Clear all", () => {
    withState({ disruptions: [BLOCK] });
    const html = render(createElement(DisruptionList));
    const t = text(html);
    expect(t).toContain("Blocked now");
    expect(t).toContain("since 8:12 AM");
    expect(t).toContain("Restore");
    expect(t).toContain("Clear all");
    expect(html).toContain(`aria-label="Restore service: ${BLOCK.label}"`);
    expect(t).not.toMatch(/∞|left|\d+:\d\d(?! [AP]M)/);
  });

  it("renders nothing with no blockages", () => {
    expect(render(createElement(DisruptionList))).toBe("");
  });
});

describe("preset copy", () => {
  it("calls the Times Sq preset a blockage and leaves the others as the data says", () => {
    expect(presetTitle({ id: "times-sq-signals", title: "Signal problems at Times Sq–42 St" })).toBe(
      "Tracks blocked at Times Sq–42 St",
    );
    expect(presetTitle({ id: "canarsie-tube", title: "Canarsie Tube blocked" })).toBe("Canarsie Tube blocked");
    for (const title of Object.values(PRESET_TITLE_OVERRIDES)) expect(title).not.toMatch(/signal/i);
  });
});

describe("session controller", () => {
  type Internals = { outbox: ClientMessage[]; onMessage(m: ServerMessage): void };
  const internals = (c: SessionController) => c as unknown as Internals;
  const ended = (kind: "disruption_cleared" | "disruption_expired", d: DisruptionInfo): ServerMessage =>
    ({ type: "events", items: [{ kind, disruption: d }] }) as ServerMessage;

  it("always sends duration_s: null", () => {
    const c = new SessionController();
    c.addDisruption("full_station_closure", { type: "station", idx: 0 });
    expect(internals(c).outbox).toEqual([
      { type: "add_disruption", kind: "full_station_closure", target: { type: "station", idx: 0 }, duration_s: null },
    ]);
  });

  it("posts Service restored when the user clears a block, and stays quiet for a reset", () => {
    const c = new SessionController();
    c.clearDisruption(BLOCK.id);
    internals(c).onMessage(ended("disruption_cleared", BLOCK));
    const toasts = useStore.getState().toasts;
    expect(toasts).toHaveLength(1);
    expect(toasts[0]).toMatchObject({ kind: "notice", kicker: "Service restored", tone: "restored" });
    expect(toasts[0].text).toBe("Times Sq-42 St ↔ 34 St-Penn Station: local track clear again · 8:20 AM");
    expect("cta" in toasts[0]).toBe(false); // no "Slow to 1 min/s" hints for timed crises any more

    useStore.setState({ toasts: [] });
    c.reset(); // a reset ends blocks too: no "restored" claim for those
    internals(c).onMessage(ended("disruption_cleared", { ...BLOCK, id: 9 }));
    expect(useStore.getState().toasts).toEqual([]);
  });

  it("posts one notice for Clear all", () => {
    const c = new SessionController();
    const other = { ...BLOCK, id: 5, kind: "full_station_closure" as const, target: { type: "station" as const, idx: 0 } };
    useStore.setState({ disruptions: [BLOCK, other] });
    c.clearAll();
    internals(c).onMessage({
      type: "events",
      items: [
        { kind: "disruption_cleared", disruption: BLOCK },
        { kind: "disruption_cleared", disruption: other },
      ],
    } as ServerMessage);
    expect(useStore.getState().toasts.map((t) => t.text)).toEqual(["Service restored at 2 places · 8:20 AM"]);
  });
});
