import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs beside Vite (`uv run uvicorn atlas.app:app --port 8010` in ../api);
// in production one container serves both, so the front end only ever says "/api".
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8010" } },
});
