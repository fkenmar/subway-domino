/**
 * User-facing copy and metadata: profiles, speeds, metrics, the 8 RCC actions
 * with their §4.2 masking rules in plain English, and the §4.4 reward terms.
 */
import type { ActionName, DisruptionKind, MetricValues, Profile, Speed } from "./protocol";

export const PROFILE_INFO: Record<Profile, { label: string; hours: string }> = {
  am_peak: { label: "AM Peak", hours: "7–10 AM" },
  midday: { label: "Midday", hours: "10 AM–4 PM" },
  pm_rush: { label: "PM Rush", hours: "4–7 PM" },
  late_night: { label: "Late Night", hours: "12–5 AM" },
};

export const SPEED_INFO: Record<Speed, { label: string; hint: string }> = {
  pause: { label: "Pause", hint: "Freeze the simulation" },
  "1x": { label: "1 min/s", hint: "1 simulated minute per real second (60× real time)" },
  "2x": { label: "2 min/s", hint: "2 simulated minutes per real second (120× real time)" },
  "5x": { label: "5 min/s", hint: "5 simulated minutes per real second (300× real time)" },
};

export type MetricKey = "avg_wait_min" | "overcrowded_platforms" | "tunnel_stuck_trains" | "riders_served";

export interface MetricInfo {
  key: MetricKey & keyof MetricValues;
  label: string;
  short: string;
  unit: string;
  better: "lower" | "higher";
  help: string;
}

export const HUD_METRICS: readonly MetricInfo[] = [
  {
    key: "avg_wait_min",
    label: "Average wait",
    short: "Avg wait",
    unit: "min",
    better: "lower",
    help: "Rider-weighted mean time everyone on a platform right now has been waiting.",
  },
  {
    key: "tunnel_stuck_trains",
    label: "Tunnel-stuck trains",
    short: "Stuck trains",
    unit: "",
    better: "lower",
    help: "Trains stopped between stations for a minute or more, riders trapped aboard.",
  },
  {
    key: "overcrowded_platforms",
    label: "Overcrowded platforms",
    short: "Crowded",
    unit: "",
    better: "lower",
    help: "Open platforms holding more than 80% of their safe capacity.",
  },
  {
    key: "riders_served",
    label: "Riders served",
    short: "Served",
    unit: "",
    better: "higher",
    help: "Completed trips since the simulation started.",
  },
];

export interface ActionInfo {
  id: number;
  name: ActionName;
  title: string;
  /** What the action does. */
  effect: string;
  /** When the mask allows it (§4.2, plain English). */
  allowedWhen: string;
}

export const ACTION_INFO: readonly ActionInfo[] = [
  {
    id: 0,
    name: "CONTINUE_SCHEDULED",
    title: "Run as scheduled",
    effect: "Depart on schedule as soon as the signals allow. Normal Schedule answers every decision this way.",
    allowedWhen: "Always allowed.",
  },
  {
    id: 1,
    name: "HOLD_IN_STATION",
    title: "Hold in station",
    effect: "Keep the doors open for another 30 s so riders keep boarding, instead of creeping into a queue in the tunnel.",
    allowedWhen:
      "Only at a platform, for at most 3 min per stop, when the gap to the train ahead isn’t already wide (under 1.5× the scheduled headway) and either the train is bunched behind its leader (under 0.75× headway) or the track ahead is blocked.",
  },
  {
    id: 2,
    name: "RUN_EXPRESS",
    title: "Switch to express",
    effect: "Cross over to the express track and rejoin the route later, skipping the local stops in between.",
    allowedWhen:
      "Only on the local track at a real local-to-express crossover, when an express path rejoins the train’s route, that express track is open (including peak-direction rules on 3-track lines), and the local track ahead is blocked or jammed.",
  },
  {
    id: 3,
    name: "RUN_LOCAL",
    title: "Switch to local",
    effect: "Cross over to the local track and make every local stop until rejoining the express route.",
    allowedWhen:
      "Only on the express track at a real express-to-local crossover, when a local path rejoins the route, the local track is open, and the express track ahead is blocked or jammed.",
  },
  {
    id: 4,
    name: "SKIP_STOP",
    title: "Skip the next stop",
    effect: "Run through the next scheduled stop without stopping to close a gap.",
    allowedWhen:
      "Never at a terminal or a major hub, never two stops in a row, and never where riders have already waited critically long (unless the station is closed) — and only when the train is over 2 min late or the train behind is closer than half a headway.",
  },
  {
    id: 5,
    name: "SHORT_TURN_RELAY",
    title: "Short-turn",
    effect:
      "Discharge everyone, reverse on a pocket track (3–5 min for the crew to walk to the other cab and test the brakes) and run back the other way.",
    allowedWhen:
      "Only at a station with a free physical pocket/center turnback, not at the line’s terminal, and only if a disruption blocks the line ahead or the opposite direction has a big gap (over 2× headway) while this train is bunched.",
  },
  {
    id: 6,
    name: "REROUTE_VIA_COMPATIBLE_TRUNK",
    title: "Reroute via another trunk",
    effect: "At a junction, take another line that rejoins the route or reaches one of its terminals.",
    allowedWhen:
      "Only at a real same-division junction (A-division and B-division cars never mix), when the alternate path is open, rejoins within 30 stations at ≤ 2.5× the time, and the original path is blocked.",
  },
  {
    id: 7,
    name: "DISPATCH_YARD_GAP_TRAIN",
    title: "Send a gap train",
    effect: "Put a spare train into service from a nearby yard after its deadhead run.",
    allowedWhen:
      "Only from a compatible yard with spare trains left, after its 10-min cooldown, when a stretch downstream is starved (next train more than 2× headway or 8 min away).",
  },
];

export const ACTION_TITLE: Record<ActionName, string> = Object.fromEntries(
  ACTION_INFO.map((a) => [a.name, a.title]),
) as Record<ActionName, string>;

export interface RewardTerm {
  name: string;
  formula: string;
  why: string;
}

export const REWARD_TERMS: readonly RewardTerm[] = [
  {
    name: "Wait age",
    formula: "Σ waiting riders × (minutes waited)^1.5",
    why: "Long waits cost more than linearly, so starving a quiet outer-borough local is heavily punished.",
  },
  {
    name: "Tunnel trapping",
    formula: "200 per train-minute + 1 per rider-minute stopped outside a platform",
    why: "Stalling riders in a tunnel is the worst outcome.",
  },
  {
    name: "Station holds",
    formula: "10 per train-minute + 0.05 per rider-minute held at a platform",
    why: "A controlled open-door hold is far cheaper than a tunnel stop — but not free.",
  },
  {
    name: "Platform crowding",
    formula: "500 × ((load − 80%) / 20%)² per platform-minute above 80% capacity",
    why: "Dangerous crowding is penalised steeply.",
  },
  {
    name: "Displacement honesty",
    formula: "short-turn 10 per rider forced off · skip 5 per rider overshot · reroute 300 + 2 per rider · yard 400",
    why: "Every maneuver pays for the riders it inconveniences.",
  },
];

export const DISRUPTION_INFO: Record<DisruptionKind, { title: string; verb: string; effect: string }> = {
  full_station_closure: {
    title: "Station closed",
    verb: "Close",
    effect: "No train may enter; waiting riders are turned away.",
  },
  local_block: {
    title: "Local track blocked",
    verb: "Block local track",
    effect: "Local-track trains stop at a red signal before it.",
  },
  express_block: {
    title: "Express track blocked",
    verb: "Block express track",
    effect: "Express-track trains stop at a red signal before it.",
  },
  signal_delay: {
    title: "Signal problems",
    verb: "Signal problems",
    effect: "Trains crawl through one at a time (10 mph) after a 45 s key-by stop.",
  },
};

export const DATASETS = [
  {
    id: "fgm6-ccue",
    name: "MTA Subway GTFS Static",
    use: "Routes, stations, stop patterns, shapes and the weekday timetable.",
  },
  {
    id: "5f5g-n3cz",
    name: "MTA Subway Station Complexes",
    use: "Which stations form one transfer complex.",
  },
  {
    id: "5wq4-mkjj",
    name: "MTA Subway Hourly Ridership: Beginning 2025",
    use: "Weekday entries per complex and hour (one server-side SoQL aggregation).",
  },
] as const;

export function datasetUrl(id: string): string {
  return `https://data.ny.gov/d/${id}`;
}

/**
 * Trunk names in preset sublabels, spelled as lines: right after the route bullets, a bare
 * "7 Av" reads like the 7 route. Station spans ("1 Av – Bedford Av") are left as they are.
 */
const PRESET_LINE_NAMES: Readonly<Record<string, string>> = {
  "Lexington Av": "Lexington Av Line",
  "Queens Blvd": "Queens Blvd Line",
  "7 Av": "Broadway–7 Av Line",
  "8 Av / Central Park West": "8 Av / Central Park West Line",
};

/** "1 2 3 · 7 Av" → "1 2 3 · Broadway–7 Av Line" (display only; the data file is unchanged). */
export function presetSubtitle(subtitle: string): string {
  const [head, ...rest] = subtitle.split(" · ");
  if (rest.length === 0) return subtitle;
  return [head, ...rest.map((part) => PRESET_LINE_NAMES[part.trim()] ?? part)].join(" · ");
}
