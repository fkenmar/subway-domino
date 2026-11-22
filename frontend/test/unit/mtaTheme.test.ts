/** WP3c: subway design-language helpers (service-change copy, express-stop markers, palette). */
import { describe, expect, it } from "vitest";
import { presetSubtitle } from "@/lib/copy";
import { parseGapTrain, parseSkip, posterCopy, serviceChangeText } from "@/lib/calloutPacer";
import { contrastRatio, MAP_BACKGROUND } from "@/lib/network";
import { transferStations } from "@/map/renderBase";
import { PALETTE } from "@/map/style";
import { miniNetwork } from "../helpers/geo";

describe("serviceChangeText", () => {
  it("phrases each action as a plain-language instruction", () => {
    expect(serviceChangeText("RUN_LOCAL", "125 St")).toBe("Runs local from 125 St");
    expect(serviceChangeText("RUN_EXPRESS", "96 St")).toBe("Runs express from 96 St");
    expect(serviceChangeText("SHORT_TURN_RELAY", "Bowling Green")).toBe("Turns back at Bowling Green");
    expect(serviceChangeText("HOLD_IN_STATION", undefined)).toBe("Held at the platform");
    expect(serviceChangeText("SKIP_STOP", "Canal St")).toBe("Skips its next stop");
    expect(serviceChangeText("DISPATCH_YARD_GAP_TRAIN", undefined)).toBe("Extra train enters service");
  });
});

describe("posterCopy", () => {
  // Server text as app/callouts.py writes it; the event's station is the gap being filled.
  const GAP = "Sending a gap train out of Jamaica Yard — it enters service as an E train at Forest Hills-71 Av in about 7 min.";

  it("names the gap train's entry station in the headline, never the gap it fills", () => {
    const c = posterCopy("DISPATCH_YARD_GAP_TRAIN", "42 St-Port Authority Bus Terminal", GAP);
    const entry = parseGapTrain(GAP)?.entry;
    expect(entry).toBe("Forest Hills-71 Av");
    expect(c.headline).toBe(`Extra train enters service at ${entry}`);
    expect(c.headline).not.toContain("42 St-Port Authority");
    expect(c.reason).toBe("Fills a gap near 42 St-Port Authority Bus Terminal · from Jamaica Yard · in ~7 min");
    expect(c.header).toBe("Extra service");
    expect(c.major).toBe(true);
  });

  it("keeps the entry station when the text is grouped or has no ETA", () => {
    for (const [text, entry] of [
      ["Sending a gap train out of Concourse Yard — it enters service as a B train at Bedford Park Blvd. (+3 more dispatch actions)", "Bedford Park Blvd"],
      ["Sending a gap train out of the yard — it enters service as an A train at Euclid Av in about 45 s.", "Euclid Av"],
      ["Sending a gap train out of Corona Yard — it enters service as a 7 express train at Flushing-Main St in about 12 min.", "Flushing-Main St"],
    ] as const) {
      const c = posterCopy("DISPATCH_YARD_GAP_TRAIN", "Nostrand Av", text);
      expect(c.headline).toBe(`Extra train enters service at ${entry}`);
      expect(c.headline).not.toContain("Nostrand Av");
    }
  });

  it("never invents a place when the gap-train text can't be read", () => {
    const c = posterCopy("DISPATCH_YARD_GAP_TRAIN", "Grand St", "Dispatching a B train at Grand St.");
    expect(c.headline).toBe("Extra train enters service");
    expect(c.reason).toBe("Fills a gap near Grand St");
  });

  it("leads a skip with the station that gets no stop", () => {
    const text = "A 4 train at Bowling Green will pass Wall St without stopping; 14 riders bound there ride on to the next stop.";
    expect(parseSkip(text)).toEqual({ skipped: "Wall St", closed: false, riders: "14 riders" });
    const c = posterCopy("SKIP_STOP", "Bowling Green", text);
    expect(c.headline).toBe("No stop at Wall St");
    expect(c.reason).toBe("Runs through after Bowling Green · 14 riders ride on to the next stop");
    expect(c.header).toBe("Skip stop");
    expect(c.major).toBe(false);
    const closed = posterCopy("SKIP_STOP", "23 St", "Running a 6 train through 14 St-Union Sq without stopping — the station is closed.");
    expect(closed.headline).toBe("No stop at 14 St-Union Sq");
    expect(closed.reason).toBe("14 St-Union Sq is closed · Runs through after 23 St");
  });

  it("gives each action its own header, the black bar only for big changes", () => {
    expect(posterCopy("HOLD_IN_STATION", "23 St", "Holding a 6 train at 23 St with doors open.")).toMatchObject({
      header: "Holding",
      major: false,
      headline: "Held at 23 St",
    });
    const local = posterCopy("RUN_LOCAL", "96 St", "Moving a 2 train to the local track at 96 St. It makes all local stops until Times Sq-42 St.");
    expect(local).toMatchObject({ header: "Service change", major: true, headline: "Runs local · 96 St → Times Sq-42 St" });
    expect(posterCopy("SHORT_TURN_RELAY", "Bowling Green", "Short-turning a 4 train at Bowling Green.").headline).toBe(
      "Turns back at Bowling Green",
    );
  });
});

describe("transferStations", () => {
  it("rings only the stations of multi-station complexes: no express-stop guess", () => {
    const net = miniNetwork();
    const marks = transferStations(net);
    expect(marks.length).toBe(net.stations.length);
    for (const st of net.stations) {
      expect(marks[st.idx]).toBe((net.complexes[st.complex]?.stations.length ?? 0) > 1 ? 1 : 0);
    }
    expect(transferStations(net)).toBe(marks); // cached per network
  });
});

describe("light map palette", () => {
  it("keeps labels and the AI accent legible on the land and water", () => {
    for (const bg of [PALETTE.land, MAP_BACKGROUND]) {
      expect(contrastRatio(PALETTE.label, bg)).toBeGreaterThanOrEqual(7);
      expect(contrastRatio(PALETTE.labelMuted, bg)).toBeGreaterThanOrEqual(4.5);
      expect(contrastRatio(PALETTE.highlight, bg)).toBeGreaterThanOrEqual(3);
    }
    expect(contrastRatio(PALETTE.borough, PALETTE.land)).toBeGreaterThanOrEqual(4.5);
  });
});

describe("presetSubtitle", () => {
  it("names trunks as lines so a bare avenue can't read as a route", () => {
    expect(presetSubtitle("1 2 3 · 7 Av")).toBe("1 2 3 · Broadway–7 Av Line");
    expect(presetSubtitle("4 5 6 · Lexington Av")).toBe("4 5 6 · Lexington Av Line");
    expect(presetSubtitle("L · 1 Av – Bedford Av")).toBe("L · 1 Av – Bedford Av");
    expect(presetSubtitle("E F")).toBe("E F");
  });
});
