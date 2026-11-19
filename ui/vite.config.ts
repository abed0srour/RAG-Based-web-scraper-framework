import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * The dev server proxies /v1 and /health to the FastAPI service.
 *
 * Going through the proxy rather than calling http://localhost:8080 directly
 * means the browser sees one origin, so no preflight and no CORS
 * configuration is needed while developing. The API does send permissive
 * CORS headers (API_CORS_ORIGINS), so a direct call works too - set
 * VITE_API_BASE to point at it, which is what a built bundle served from
 * somewhere else does.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
      "/health": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
