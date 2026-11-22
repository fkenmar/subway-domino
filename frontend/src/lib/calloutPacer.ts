/**
 * Paces the AI callout toast. At 5 min/s the AI can act a dozen times per sim
 * minute; showing each one replaces the toast faster than anyone can read it. The
 * pacer keeps at most one callout on screen, swaps it at most every
 * {@link CALLOUT_GAP_MS} of wall time, prefers an action inside the current
 * viewport, and counts how many actions arrived recently so the toast can say
 * "14 actions in the last 10 s". Every action still goes to the dispatch log.
 */
import type { ActionName, CalloutEvent } from "./protocol";

/** Minimum wall time between two callout toasts. */
export const CALLOUT_GAP_MS = 4_500;
/** Window of the "N actions in the last 10 s" count. */
export const CALLOUT_RATE_WINDOW_MS = 10_000;
/** Pending callouts older than this are stale (never shown, still counted and logged). */
const STALE_MS = 6_000;

export interface PacedCallout {
  event: CalloutEvent;
  /** Actions received in the last {@link CALLOUT_RATE_WINDOW_MS} (including this one). */
  recent: number;
}

export class CalloutPacer {
  private pending: { ev: CalloutEvent; at: number }[] = [];
  private arrivals: number[] = [];
  private lastShownAt = Number.NEGATIVE_INFINITY;

  constructor(private readonly gapMs = CALLOUT_GAP_MS) {}

  offer(ev: CalloutEvent, nowMs: number): void {
    this.pending.push({ ev, at: nowMs });
    this.arrivals.push(nowMs);
    if (this.pending.length > 64) this.pending.splice(0, this.pending.length - 64);
  }

  /** Actions received in the last 10 s. */
  recent(nowMs: number): number {
    const cut = nowMs - CALLOUT_RATE_WINDOW_MS;
    while (this.arrivals.length && this.arrivals[0] < cut) this.arrivals.shift();
    return this.arrivals.length;
  }

  /**
   * The callout to show now, or null (nothing new, or the current one hasn't had
   * its {@link CALLOUT_GAP_MS} yet). `inView` says whether a station is on screen.
   */
  next(nowMs: number, inView: (stationIdx: number) => boolean): PacedCallout | null {
    this.pending = this.pending.filter((p) => nowMs - p.at <= STALE_MS);
    if (this.pending.length === 0 || nowMs - this.lastShownAt < this.gapMs) return null;
    let pick = this.pending[this.pending.length - 1];
    for (let i = this.pending.length - 1; i >= 0; i--) {
      if (inView(this.pending[i].ev.station_idx)) {
        pick = this.pending[i];
        break;
      }
    }
    this.pending = [];
    this.lastShownAt = nowMs;
    return { event: pick.ev, recent: this.recent(nowMs) };
  }

  /** Forget everything queued (mode switched back to Normal, reset). */
  clear(): void {
    this.pending = [];
    this.arrivals = [];
    this.lastShownAt = Number.NEGATIVE_INFINITY;
  }
}

/** Short verb phrase per action for the one-line toast. */
const VERB: Partial<Record<ActionName, string>> = {
  HOLD_IN_STATION: "Hold",
  RUN_EXPRESS: "To express",
  RUN_LOCAL: "To local",
  SKIP_STOP: "Skip next stop",
  SHORT_TURN_RELAY: "Short-turn",
  REROUTE_VIA_COMPATIBLE_TRUNK: "Reroute",
  DISPATCH_YARD_GAP_TRAIN: "Gap train",
};

/**
 * Plain-language service-change instruction for one train (poster style; the route bullet
 * sits beside it): "Runs local from 125 St", "Turns back at Bowling Green". Station-only
 * fallback; {@link posterCopy} reads the server text for the precise place.
 */
export function serviceChangeText(action: ActionName, stationName: string | undefined): string {
  const at = (pre: string) => (stationName ? `${pre} ${stationName}` : "");
  switch (action) {
    case "HOLD_IN_STATION":
      return stationName ? `Held at ${stationName}` : "Held at the platform";
    case "RUN_EXPRESS":
      return `Runs express${at(" from")}`;
    case "RUN_LOCAL":
      return `Runs local${at(" from")}`;
    case "SKIP_STOP":
      return "Skips its next stop";
    case "SHORT_TURN_RELAY":
      return stationName ? `Turns back at ${stationName}` : "Turns back early";
    case "REROUTE_VIA_COMPATIBLE_TRUNK":
      return `Rerouted via another line${at(" at")}`;
    case "DISPATCH_YARD_GAP_TRAIN":
      // The event's station is the gap being filled, not where the train enters service:
      // never put it in the headline (see posterCopy).
      return "Extra train enters service";
    default:
      return calloutHeadline(action, stationName);
  }
}

/** One service-change poster: its header tab, headline (the observable fact) and why. */
export interface PosterCopy {
  /** Header tab: "Service change", "Extra service", "Holding", "Skip stop". */
  header: string;
  /** Big changes get the black poster bar; holds and skips a lighter grey tab. */
  major: boolean;
  headline: string;
  reason: string | null;
}

const HEADER: Partial<Record<ActionName, [string, boolean]>> = {
  HOLD_IN_STATION: ["Holding", false],
  SKIP_STOP: ["Skip stop", false],
  RUN_EXPRESS: ["Service change", true],
  RUN_LOCAL: ["Service change", true],
  SHORT_TURN_RELAY: ["Service change", true],
  REROUTE_VIA_COMPATIBLE_TRUNK: ["Service change", true],
  DISPATCH_YARD_GAP_TRAIN: ["Extra service", true],
};

/** Server text without the "(+N more dispatch actions)" suffix and the final full stop. */
function plain(text: string): string {
  return text
    .replace(/\s*\(\+\d+ more dispatch actions?\)\s*$/, "")
    .trim()
    .replace(/\.$/, "");
}

/** "Sending a gap train out of Jamaica Yard — it enters service as an E train at Forest Hills-71 Av in about 7 min." */
export function parseGapTrain(text: string): { yard: string | null; entry: string; eta: string | null } | null {
  const t = plain(text);
  const m = /enters service as .*? train at (.+)$/.exec(t);
  if (!m) return null;
  let entry = m[1].trim();
  let eta: string | null = null;
  const when = / in about (.+)$/.exec(entry);
  if (when) {
    eta = when[1].trim();
    entry = entry.slice(0, when.index).trim();
  }
  const y = /out of (.+?) — /.exec(t);
  return entry ? { yard: y ? y[1].trim() : null, entry, eta } : null;
}

/** "… at Bowling Green will pass Wall St without stopping; 14 riders bound there ride on …" */
export function parseSkip(text: string): { skipped: string; closed: boolean; riders: string | null } | null {
  const t = plain(text);
  const closed = /through (.+?) without stopping — the station is closed/.exec(t);
  if (closed) return { skipped: closed[1].trim(), closed: true, riders: null };
  const m = /will pass (.+?) without stopping/.exec(t);
  if (!m) return null;
  const r = /; (.+?) bound there ride/.exec(t);
  return { skipped: m[1].trim(), closed: false, riders: r ? r[1].trim() : null };
}

/**
 * Poster copy for an AI action. The headline is always the observable fact, in the place it
 * happens: a gap train names the station where it enters service (from the server text),
 * never the gap it fills; a skip names the station that gets no stop.
 */
export function posterCopy(action: ActionName, stationName: string | undefined, text: string): PosterCopy {
  const [header, major] = HEADER[action] ?? ["Service change", true];
  const base = { header, major };
  const t = plain(text);
  switch (action) {
    case "DISPATCH_YARD_GAP_TRAIN": {
      const g = parseGapTrain(text);
      const why = [stationName ? `Fills a gap near ${stationName}` : "Fills a gap in service"];
      if (g?.yard) why.push(`from ${g.yard}`);
      if (g?.eta) why.push(`in ~${g.eta}`);
      return { ...base, headline: g ? `Extra train enters service at ${g.entry}` : "Extra train enters service", reason: why.join(" · ") };
    }
    case "SKIP_STOP": {
      const s = parseSkip(text);
      if (!s) return { ...base, headline: serviceChangeText(action, stationName), reason: calloutReason(text) };
      const why = s.closed ? [`${s.skipped} is closed`] : [];
      if (stationName) why.push(`Runs through after ${stationName}`);
      if (s.riders) why.push(`${s.riders} ride on to the next stop`);
      return { ...base, headline: `No stop at ${s.skipped}`, reason: why.length ? why.join(" · ") : null };
    }
    case "RUN_LOCAL": {
      const until = /makes all local stops until (.+?)(?:\. |$)/.exec(t);
      const headline = stationName && until ? `Runs local · ${stationName} → ${until[1].trim()}` : serviceChangeText(action, stationName);
      return { ...base, headline, reason: calloutReason(text) };
    }
    case "RUN_EXPRESS": {
      const rejoin = /rejoins its route at (.+?)(?:, passing|\. |$)/.exec(t);
      const headline = stationName && rejoin ? `Runs express · ${stationName} → ${rejoin[1].trim()}` : serviceChangeText(action, stationName);
      return { ...base, headline, reason: calloutReason(text) };
    }
    default:
      return { ...base, headline: serviceChangeText(action, stationName), reason: calloutReason(text) };
  }
}

/** "To local at 125 St" — the action and where it happened. */
export function calloutHeadline(action: ActionName, stationName: string | undefined): string {
  const verb = VERB[action] ?? "Dispatch";
  return stationName ? `${verb} at ${stationName}` : verb;
}

/**
 * The reason clause of a server callout ("— trains are stopped in the tunnel ahead
 * near 77 St, so it waits…" → "Trains are stopped in the tunnel ahead near 77 St"),
 * or null when the text gives none. The full text stays in the dispatch log.
 */
export function calloutReason(text: string): string | null {
  const dash = text.indexOf(" — ");
  if (dash < 0) return null;
  let rest = text.slice(dash + 3);
  const cut = rest.search(/(, so |, with |\. |\.$)/);
  if (cut > 0) rest = rest.slice(0, cut);
  rest = rest.trim();
  if (!rest) return null;
  return rest[0].toUpperCase() + rest.slice(1);
}
