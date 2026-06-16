import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
  // Previously had `typescript.ignoreBuildErrors: true` because shadcn/ui's
  // chart.tsx had Recharts v3 type incompatibilities. We now isolate that
  // file with a targeted @ts-nocheck (see src/components/ui/chart.tsx) so
  // the rest of the codebase is type-checked properly during builds.
  serverExternalPackages: ["sharp", "mongodb"],
  images: {
    domains: [],
    unoptimized: true,
  },
};

export default nextConfig;
