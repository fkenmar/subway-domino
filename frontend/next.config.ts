import type { NextConfig } from "next";
const nextConfig: NextConfig = {
  basePath: "/subway",
  async redirects() { return [{ source: "/", destination: "/subway", permanent: false, basePath: false }]; },
};
export default nextConfig;
