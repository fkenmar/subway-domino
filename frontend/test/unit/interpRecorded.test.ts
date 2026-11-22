/**
 * Replays frames recorded from the live server at 5 min/s (75 sim-s per frame)
 * through the client interpolator, with the real network: how often does a
 * train teleport, and does any continuing train jump on screen?
 * Skipped when the geo file is missing or was rebuilt since the recording.
 */
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { FrameBuffer, InterpolatedTrains, plausibleDistanceM } from "@/lib/interp";
import { buildNetwork, validateGeo } from "@/lib/network";
import { locate, makePose } from "@/lib/positioning";
import { decodeFrame, type Frame } from "@/lib/protocol";

const fixture = (name: string) => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
const GEO_PATH = fileURLToPath(new URL("../../public/data/network_geo.json", import.meta.url));
const MANIFEST = JSON.parse(readFileSync(fixture("frames_5x.json"), "utf8")) as { graph_version: string; files: string[] };
const geoText = existsSync(GEO_PATH) ? readFileSync(GEO_PATH, "utf8") : null;
const geoRaw = geoText ? (JSON.parse(geoText) as { graph_version?: string }) : null;
const USABLE = geoRaw !== null && geoRaw.graph_version === MANIFEST.graph_version;

function readFrames(name: string): Frame[] {
  const buf = gunzipSync(readFileSync(fixture(name)));
  const frames: Frame[] = [];
  for (let o = 0; o < buf.length; ) {
    const n = buf.readUInt32LE(o);
    const bytes = buf.subarray(o + 4, o + 4 + n);
    frames.push(decodeFrame(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + n)));
    o += 4 + n;
  }
  return frames;
}

describe.skipIf(!USABLE)("interpolation over recorded 5x frames", () => {
  const net = USABLE ? buildNetwork(validateGeo(geoRaw)) : (null as never);

  for (const name of MANIFEST.files) {
    it(`${name}: snaps ≤ 0.5 % of updates and no continuing train jumps`, () => {
      const frames = readFrames(name);
      const fb = new FrameBuffer(4);
      const trains = new InterpolatedTrains();
      const ghosts = new InterpolatedTrains();
      const pose = makePose();
      const SUB = 10;
      let worstStep = 0;
      let worstLimit = 0;
      let last = new Map<number, [number, number]>();
      frames.forEach((f, k) => {
        const t0 = k * 250;
        fb.push(f, t0);
        const limit = k > 0 ? plausibleDistanceM(f.clockS - frames[k - 1].clockS) : 0;
        for (let s = 0; s < SUB; s++) {
          fb.sample(t0 + (250 * s) / SUB, net.edges, trains, ghosts);
          const now = new Map<number, [number, number]>();
          for (let i = 0; i < trains.count; i++) {
            if (trains.fade[i] < 1 || !locate(net, trains.edge[i], trains.frac[i], pose)) continue;
            const id = trains.trainId[i];
            now.set(id, [pose.x, pose.y]);
            const p = last.get(id);
            if (p && k > 0) {
              const step = Math.hypot(pose.x - p[0], pose.y - p[1]);
              // Eased speed peaks below 2.8× the interval mean; the mean is capped by `limit`.
              if (step / limit > worstStep / Math.max(worstLimit, 1)) {
                worstStep = step;
                worstLimit = limit;
              }
            }
          }
          last = now;
        }
      });
      const st = fb.stats;
      const snapRate = st.snap / st.updates;
      const reuseRate = st.newTrip / st.updates;
      console.info(
        `${name}: ${frames.length} frames, ${st.updates} updates — same edge ${st.sameEdge}, path ${st.path}, ` +
          `snap ${st.snap} (${(snapRate * 100).toFixed(3)} %), new trip in reused slot ${st.newTrip} ` +
          `(${(reuseRate * 100).toFixed(2)} %, cross-faded), appeared ${st.appeared}, vanished ${st.vanished}; ` +
          `largest on-screen step ${worstStep.toFixed(0)} m per 1/${SUB} frame (limit/frame ${worstLimit.toFixed(0)} m)`,
      );
      expect(st.updates).toBeGreaterThan(10_000);
      expect(snapRate).toBeLessThanOrEqual(0.005);
      expect(st.sameEdge + st.path + st.snap + st.newTrip).toBe(st.updates);
      expect(worstStep).toBeLessThanOrEqual((2.8 * worstLimit) / SUB + 1);
    });
  }
});
