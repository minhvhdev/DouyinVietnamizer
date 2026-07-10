import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    open: false,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./tests/setup.ts"
  }
});
