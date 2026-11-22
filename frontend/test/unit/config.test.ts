/**
 * basePath helpers (§6 "Env & deploy"): every public/ URL goes through asset(),
 * and BASE_PATH must agree with next.config.ts.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { asset, BASE_PATH, backendGeoUrl, DEFAULT_DEV_WS_URL, GEO_PATH, resolveWsUrl } from "@/lib/config";
import { defaultGeoUrls } from "@/lib/network";

describe("BASE_PATH", () => {
  it("matches basePath in next.config.ts (CONTRACT: /subway)", () => {
    const configText = readFileSync(fileURLToPath(new URL("../../next.config.ts", import.meta.url)), "utf8");
    const match = /basePath:\s*"([^"]*)"/.exec(configText);
    expect(match?.[1]).toBe("/subway");
    expect(BASE_PATH).toBe(match?.[1]);
  });

  it("redirects the bare origin to the base path", () => {
    const configText = readFileSync(fileURLToPath(new URL("../../next.config.ts", import.meta.url)), "utf8");
    expect(configText).toMatch(/source:\s*"\/",\s*destination:\s*"\/subway",\s*permanent:\s*false,\s*basePath:\s*false/);
  });
});

describe("asset()", () => {
  it.each([
    ["/data/network_geo.json", "/subway/data/network_geo.json"],
    ["data/network_geo.json", "/subway/data/network_geo.json"],
    ["/icon.svg", "/subway/icon.svg"],
    ["/", "/subway/"],
    ["", "/subway/"],
  ])("prefixes %j → %j", (input, expected) => {
    expect(asset(input)).toBe(expected);
  });

  it("is idempotent", () => {
    for (const p of ["/data/network_geo.json", "x.png", "/subway", "/subway/a/b"]) {
      expect(asset(asset(p))).toBe(asset(p));
    }
    expect(asset("/subway")).toBe("/subway");
    expect(asset("/subway/data/x.json")).toBe("/subway/data/x.json");
  });

  it("does not mistake a sibling path for the base path", () => {
    expect(asset("/subwayx/data.json")).toBe("/subway/subwayx/data.json");
    expect(asset("/subway-maps")).toBe("/subway/subway-maps");
  });

  it.each([
    "https://kenmar.dev/",
    "http://localhost:8000/network_geo.json",
    "//cdn.example.com/x.js",
    "data:image/svg+xml;base64,AAAA",
    "blob:https://kenmar.dev/1234",
    "wss://example.up.railway.app/ws",
  ])("leaves absolute URL %j unchanged", (url) => {
    expect(asset(url)).toBe(url);
  });

  it("resolves the §2.7 geo file under the base path", () => {
    expect(asset(GEO_PATH)).toBe("/subway/data/network_geo.json");
  });
});

describe("resolveWsUrl()", () => {
  it("uses NEXT_PUBLIC_WS_URL when it is a ws:// or wss:// URL", () => {
    expect(resolveWsUrl("wss://subway.up.railway.app/ws", "production")).toBe("wss://subway.up.railway.app/ws");
    expect(resolveWsUrl("  ws://localhost:9000/ws  ", "development")).toBe("ws://localhost:9000/ws");
  });

  it("rejects non-WebSocket schemes instead of guessing", () => {
    expect(resolveWsUrl("https://subway.up.railway.app/ws", "production")).toBeNull();
    expect(resolveWsUrl("subway.up.railway.app/ws", "development")).toBeNull();
  });

  it("defaults to the local backend in development and requires the variable in production", () => {
    expect(resolveWsUrl(undefined, "development")).toBe(DEFAULT_DEV_WS_URL);
    expect(resolveWsUrl("", "test")).toBe(DEFAULT_DEV_WS_URL);
    expect(DEFAULT_DEV_WS_URL).toBe("ws://localhost:8000/ws");
    expect(resolveWsUrl(undefined, "production")).toBeNull();
    expect(resolveWsUrl("   ", "production")).toBeNull();
  });
});

describe("backendGeoUrl()", () => {
  it("derives the backend's GET /network_geo.json from the socket URL", () => {
    expect(backendGeoUrl("wss://subway.up.railway.app/ws")).toBe("https://subway.up.railway.app/network_geo.json");
    expect(backendGeoUrl("ws://localhost:8000/ws")).toBe("http://localhost:8000/network_geo.json");
    expect(backendGeoUrl("ws://localhost:8000/api/ws/?x=1#y")).toBe("http://localhost:8000/api/network_geo.json");
  });

  it("returns null without a usable URL", () => {
    expect(backendGeoUrl(null)).toBeNull();
    expect(backendGeoUrl("not a url")).toBeNull();
  });

  it("is tried after the static asset when loading the network", () => {
    expect(defaultGeoUrls()).toEqual(["/subway/data/network_geo.json", "http://localhost:8000/network_geo.json"]);
  });
});
