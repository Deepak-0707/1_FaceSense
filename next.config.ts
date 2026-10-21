import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        encoding: false,
      };
    }

    if (isServer) {
      config.externals = [
        ...(Array.isArray(config.externals) ? config.externals : []),
        "@vladmandic/face-api",
      ];
    }

    return config;
  },

  turbopack: {
    root: path.resolve(__dirname),
  },

  async headers() {
    return [
      {
        source: "/models/:path*",
        headers: [
          { key: "Cache-Control", value: "public, max-age=86400, immutable" },
          { key: "Access-Control-Allow-Origin", value: "*" },
        ],
      },
    ];
  },
};

export default nextConfig;
