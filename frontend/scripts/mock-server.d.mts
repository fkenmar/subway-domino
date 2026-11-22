/** Types for the dev-only protocol v1 mock (scripts/mock-server.mjs). */

export const PROTOCOL: 1;

export interface MockServerOptions {
  port?: number | string;
  host?: string;
  /** Geo file path (CLI `--geo`); defaults to public/data/network_geo.json, else the mini fixture. */
  geo?: string;
  geoPath?: string;
  maxSessions?: number | string;
  /** Reject every connection with `at_capacity` + close 1013. */
  atCapacity?: boolean | string;
  idleTimeout?: number | string;
  idleTimeoutS?: number | string;
  noAi?: boolean | string;
  aiAvailable?: boolean;
  frameHz?: number | string;
  /** Effective-speed factor reported in metrics (simulates the CPU governor). */
  governor?: number | string;
  seed?: number | string;
  quiet?: boolean | string;
}

export interface MockServerHandle {
  port: number;
  url: string;
  wsUrl: string;
  close(): Promise<void>;
}

export function startMockServer(options?: MockServerOptions): Promise<MockServerHandle>;

export function defaultGeoPath(): string;
