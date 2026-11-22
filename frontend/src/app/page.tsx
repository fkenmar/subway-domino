import { preload } from "react-dom";
import App from "@/components/App";
import { asset, GEO_PATH } from "@/lib/config";

export default function Page() {
  // Start downloading the map geometry with the HTML so the static map paints
  // before the WebSocket connects (§6 UX flow 1).
  preload(asset(GEO_PATH), { as: "fetch", crossOrigin: "anonymous" });
  return <App />;
}
