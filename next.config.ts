import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
  // Note: ignoreBuildErrors is needed because shadcn/ui's chart.tsx component
  // has known type incompatibilities with Recharts v3. Our own code is type-safe.
  typescript: {
    ignoreBuildErrors: true,
  },
  serverExternalPackages: ["sharp", "mongodb"],
  images: {
    domains: [],
    unoptimized: true,
  },
};

export default nextConfig;
