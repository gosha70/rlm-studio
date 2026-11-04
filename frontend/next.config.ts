import type { NextConfig } from "next";

// Two build modes:
//
// * Dev (`next dev`, `next build`): the Next server runs and proxies
//   `/api/*`, `/health`, `/ws/*` to the backend on :8000 via rewrites.
//   This is the path used by `npm run dev`, frontend CI, and any
//   developer who wants the UI on a separate port.
//
// * Bundle (`npm run build:bundle`, BUNDLE=1): produces a fully static
//   export under `frontend/out/`, intended to be copied into
//   `src/rlmstudio/_ui/` and served by FastAPI at `/studio` in the same
//   process as the API. No dev server, no rewrites — every request
//   is same-origin, so the rewrites are unnecessary.
//
// `basePath: "/studio"` makes Next emit asset URLs and client-side
// router paths under `/studio`, matching the FastAPI mount.
// `trailingSlash: true` produces one `index.html` per route directory,
// which is what FastAPI's `StaticFiles(html=True)` resolves cleanly.
const isBundle = process.env.BUNDLE === "1";
const BUNDLE_BASE_PATH = "/studio";

const nextConfig: NextConfig = isBundle
  ? {
      output: "export",
      basePath: BUNDLE_BASE_PATH,
      trailingSlash: true,
      // Next prefixes its own links and assets with basePath, but not a plain
      // fetch() of a file in public/; code reads this to build those URLs.
      env: { NEXT_PUBLIC_BASE_PATH: BUNDLE_BASE_PATH },
    }
  : {
      async rewrites() {
        // INTERNAL_API_URL is the address the *Next server* uses to reach the
        // backend, which is not always the address the *browser* uses. Under
        // Docker Compose the browser talks to http://localhost:8000 (published
        // port) while this process must use http://backend:8000 (compose
        // network). NEXT_PUBLIC_* cannot express that difference: Next inlines
        // those at build time, so a runtime override is silently ignored and
        // the container ends up proxying to itself (ECONNREFUSED 127.0.0.1:8000).
        const backend =
          process.env.INTERNAL_API_URL ||
          process.env.NEXT_PUBLIC_API_URL ||
          "http://localhost:8000";
        return [
          { source: "/api/:path*", destination: `${backend}/api/:path*` },
          { source: "/health", destination: `${backend}/health` },
          { source: "/ws/:path*", destination: `${backend}/ws/:path*` },
        ];
      },
    };

export default nextConfig;
