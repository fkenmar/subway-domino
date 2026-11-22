/**
 * Test-only encoder for the §5.3 binary frame, written straight from the spec
 * table (literal offsets and codes) rather than from the decoder's constants,
 * so decoder tests are not circular.
 *
 * | 0 u16 magic 0x5344 | 2 u8 version 1 | 3 u8 kind 1 | 4 u32 seq | 8 f64 clock_s |
 * | 16 u16 n | 18 u16 m | 20 u8 mode | 21 u8 speed | 22 u16 flags |
 * | 24 u16[n] train_id | 24+2n u16[n] edge | 24+4n u16[n] frac |
 * | 24+6n u8[n] route | 24+7n u8[n] status | 24+8n u8[n] load |
 * | 24+9n u8[m] station_crowd | 24+9n+m u8[m] station_wait |   (little-endian)
 */

export const MODE_CODE = { normal: 0, ai: 1 } as const;
export const SPEED_CODE = { pause: 0, "1x": 1, "2x": 2, "5x": 3 } as const;

export interface TrainRow {
  id: number;
  edge: number;
  /** Raw u16 fraction (0..65535). */
  frac: number;
  route: number;
  status: number;
  load: number;
}

export interface FrameInput {
  seq: number;
  clockS: number;
  mode: keyof typeof MODE_CODE;
  speed: keyof typeof SPEED_CODE;
  flags?: number;
  trains: TrainRow[];
  stationCrowd: number[];
  stationWait: number[];
  /** Overrides for negative tests. */
  magic?: number;
  version?: number;
  kind?: number;
  modeCode?: number;
  speedCode?: number;
  /** Lie about n / m in the header. */
  nOverride?: number;
  mOverride?: number;
}

/**
 * Encode a frame. `byteOffset` places it inside a larger buffer (unaligned-view
 * tests); `extraBytes` appends trailing bytes.
 */
export function encodeFrame(
  f: FrameInput,
  opts: { byteOffset?: number; extraBytes?: number } = {},
): Uint8Array<ArrayBuffer> {
  const n = f.trains.length;
  const m = f.stationCrowd.length;
  if (f.stationWait.length !== m) throw new Error("stationWait length must equal stationCrowd length");
  const off = opts.byteOffset ?? 0;
  const len = 24 + 9 * n + 2 * m + (opts.extraBytes ?? 0);
  const buf = new ArrayBuffer(off + len);
  const v = new DataView(buf, off, len);
  v.setUint16(0, f.magic ?? 0x5344, true);
  v.setUint8(2, f.version ?? 1);
  v.setUint8(3, f.kind ?? 1);
  v.setUint32(4, f.seq, true);
  v.setFloat64(8, f.clockS, true);
  v.setUint16(16, f.nOverride ?? n, true);
  v.setUint16(18, f.mOverride ?? m, true);
  v.setUint8(20, f.modeCode ?? MODE_CODE[f.mode]);
  v.setUint8(21, f.speedCode ?? SPEED_CODE[f.speed]);
  v.setUint16(22, f.flags ?? 0, true);
  f.trains.forEach((t, i) => {
    v.setUint16(24 + 2 * i, t.id, true);
    v.setUint16(24 + 2 * n + 2 * i, t.edge, true);
    v.setUint16(24 + 4 * n + 2 * i, t.frac, true);
    v.setUint8(24 + 6 * n + i, t.route);
    v.setUint8(24 + 7 * n + i, t.status);
    v.setUint8(24 + 8 * n + i, t.load);
  });
  f.stationCrowd.forEach((c, j) => v.setUint8(24 + 9 * n + j, c));
  f.stationWait.forEach((w, j) => v.setUint8(24 + 9 * n + m + j, w));
  return new Uint8Array(buf, off, len);
}

/** A small, varied frame used across suites. */
export function sampleFrame(overrides: Partial<FrameInput> = {}): FrameInput {
  return {
    seq: 41,
    clockS: 30120.5,
    mode: "ai",
    speed: "2x",
    flags: 1,
    trains: [
      { id: 3, edge: 12, frac: 0, route: 4, status: 1, load: 37 },
      { id: 17, edge: 640, frac: 32768, route: 27, status: 2 | 4, load: 102 },
      { id: 699, edge: 65535, frac: 65535, route: 255, status: 255, load: 255 },
    ],
    stationCrowd: [0, 45, 80, 254, 255],
    stationWait: [0, 4, 60, 120, 255],
    ...overrides,
  };
}
