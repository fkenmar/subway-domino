/** Display formatting. */
import { describe, expect, it } from "vitest";
import {
  DASH,
  formatClock,
  formatCompact,
  formatCountdown,
  formatInt,
  formatMinutes,
  formatSigned,
  formatSpan,
  percentChange,
  pluralize,
} from "@/lib/format";

describe("format", () => {
  it("turns the sim clock into a time of day", () => {
    expect(formatClock(30_120)).toBe("8:22 AM");
    expect(formatClock(0)).toBe("12:00 AM");
    expect(formatClock(43_200)).toBe("12:00 PM");
    expect(formatClock(61_200)).toBe("5:00 PM");
    expect(formatClock(86_400 + 7_200)).toBe("2:00 AM"); // the clock keeps increasing past midnight
    expect(formatClock(null)).toBe("--:--");
    expect(formatClock(Number.NaN)).toBe("--:--");
    expect(formatClock(30_125, { seconds: true })).toBe("8:22:05 AM");
    expect(formatClock(null, { seconds: true })).toBe("--:--:--");
  });

  it("formats countdown chips", () => {
    expect(formatCountdown(65)).toBe("1:05");
    expect(formatCountdown(59.2)).toBe("1:00");
    expect(formatCountdown(3_661)).toBe("1:01:01");
    expect(formatCountdown(-5)).toBe("0:00");
  });

  it("formats spans for prose", () => {
    expect(formatSpan(45)).toBe("45 s");
    expect(formatSpan(720)).toBe("12 min");
    expect(formatSpan(3_900)).toBe("1 h 05 min");
  });

  it("formats numbers for the HUD", () => {
    expect(formatInt(123_456.4)).toBe("123,456");
    expect(formatInt(null)).toBe(DASH);
    expect(formatMinutes(4.25)).toBe("4.3");
    expect(formatMinutes(Number.POSITIVE_INFINITY)).toBe(DASH);
    expect(formatCompact(999)).toBe("999");
    expect(formatCompact(1_234)).toBe("1.2k");
    expect(formatCompact(12_345)).toBe("12k");
    expect(formatCompact(1_234_567)).toBe("1.23M");
    expect(formatCompact(12_345_678)).toBe("12.3M");
  });

  it("formats signed deltas with a true minus sign", () => {
    expect(formatSigned(3, formatInt)).toBe("+3");
    expect(formatSigned(-1.24, formatMinutes)).toBe("−1.2");
    expect(formatSigned(0, formatInt)).toBe("±0");
    expect(formatSigned(Number.NaN, formatInt)).toBe(DASH);
    // Deltas that round to zero at the display precision never read "−0.0".
    expect(formatSigned(-0.04, formatMinutes)).toBe("±0.0");
    expect(formatSigned(0.3, formatInt)).toBe("±0");
    expect(formatSigned(0.05, formatMinutes)).toBe("+0.1");
  });

  it("guards percentage change against tiny baselines", () => {
    expect(percentChange(90, 100)).toBeCloseTo(-10);
    expect(percentChange(5, 0)).toBeNull();
    expect(percentChange(Number.NaN, 3)).toBeNull();
  });

  it("pluralizes", () => {
    expect(pluralize(1, "train")).toBe("1 train");
    expect(pluralize(1_200, "rider")).toBe("1,200 riders");
  });
});

describe("splitMoreSuffix", () => {
  it("splits the server's aggregated-callout suffix into a count", async () => {
    const { splitMoreSuffix } = await import("@/lib/format");
    expect(splitMoreSuffix("Held 6 at 14 St to keep it out of the tunnel (+50 more dispatch actions)")).toEqual({
      text: "Held 6 at 14 St to keep it out of the tunnel",
      more: 50,
    });
    expect(splitMoreSuffix("Short-turned L at 8 Av (+1 more dispatch action)")).toEqual({ text: "Short-turned L at 8 Av", more: 1 });
    expect(splitMoreSuffix("Plain sentence.")).toEqual({ text: "Plain sentence.", more: 0 });
  });
});
