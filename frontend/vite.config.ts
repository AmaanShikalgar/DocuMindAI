import { defineConfig, loadEnv, type Plugin, type ViteDevServer } from "vite";
import react from "@vitejs/plugin-react";

/** In development, serve /api/chat from api/chat.ts (the same handler Vercel deploys), so the
 * browser build can be tried locally. Put GROQ_API_KEY (or FAKE_LLM=1) in frontend/.env.local. */
function chatFunction(env: Record<string, string>): Plugin {
  return {
    name: "documind-chat-function",
    configureServer(server: ViteDevServer) {
      Object.assign(process.env, env);
      server.middlewares.use("/api/chat", async (req, res) => {
        const { default: handler } = await server.ssrLoadModule("/api/chat.ts");
        const chunks: Buffer[] = [];
        for await (const c of req) chunks.push(c as Buffer);
        const request = new Request(`http://localhost${req.url ?? ""}`, {
          method: req.method,
          headers: { "content-type": "application/json", "x-forwarded-for": req.socket.remoteAddress ?? "local" },
          body: req.method === "POST" ? Buffer.concat(chunks) : undefined,
        });
        const response: Response = await handler(request);
        res.statusCode = response.status;
        response.headers.forEach((v, k) => res.setHeader(k, v));
        if (!response.body) return res.end();
        const reader = response.body.getReader();
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          res.write(value);
        }
        res.end();
      });
    },
  };
}

// Cross-origin isolation lets ONNX Runtime use multiple threads in the browser.
const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "credentialless" };

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const server = env.VITE_ENGINE === "server";
  return {
    plugins: [react(), ...(server ? [] : [chatFunction(env)])],
    server: {
      port: 5173,
      headers: isolation,
      proxy: server ? { "/api": "http://localhost:8000" } : undefined,
    },
    preview: { headers: isolation },
    build: { outDir: "dist" },
    worker: { format: "es" },
  };
});
