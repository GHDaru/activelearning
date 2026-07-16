import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Em dev o Vite proxya /api para o FastAPI local (evita CORS e URL hardcoded).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
