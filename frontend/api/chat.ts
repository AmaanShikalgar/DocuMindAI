/** POST /api/chat — the only server-side piece of the in-browser build (a Vercel Edge Function).
 *
 * The browser does retrieval and sends the question plus the passages it found; this function
 * holds GROQ_API_KEY, builds the prompt (same rules as backend/documind/llm.py) and streams the
 * answer back as server-sent events: `token` deltas, then `done` with token usage and cost.
 * The prompt is fixed here and inputs are capped, so the endpoint can't be used as a
 * general-purpose LLM proxy. */
export const config = { runtime: "edge" };

const NOT_FOUND = "I cannot find this in the document.";
const SYSTEM = `You answer questions using only the numbered sources provided.

Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- Cite every claim with the source number in square brackets, e.g. [2] or [1][3].
- If the sources do not contain the answer, reply exactly: ${NOT_FOUND}
- Be concise: 1-3 sentences unless the question asks for a list or steps.
- Do not mention "the sources" or "the context" in the answer.`;

const MAX_QUESTION = 2000;
const MAX_SOURCES = 10;
const MAX_SOURCE_CHARS = 2500;

type Src = { n: number; title: string; page: number; section?: string; text: string };

// Best-effort per-IP limit. Edge instances don't share memory, so this caps bursts, not totals:
// also set a spending limit in the Groq console.
const hits = new Map<string, number[]>();
function allow(ip: string, limit: number, windowMs: number) {
  const now = Date.now();
  const q = (hits.get(ip) ?? []).filter((t) => t > now - windowMs);
  if (q.length >= limit) return false;
  q.push(now);
  hits.set(ip, q);
  if (hits.size > 5000) hits.clear();
  return true;
}

const env = (k: string, d = "") => (globalThis as any).process?.env?.[k] ?? d;
const json = (status: number, body: object) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

export default async function handler(req: Request): Promise<Response> {
  if (req.method !== "POST") return json(405, { error: "POST only" });
  const ip = (req.headers.get("x-forwarded-for") ?? "local").split(",")[0].trim();
  if (!allow(ip, Number(env("RATE_LIMIT_QUERIES", "30")), 10 * 60 * 1000)) return json(429, { error: "Too many questions, try again in a few minutes" });

  let body: { question?: unknown; sources?: unknown };
  try {
    body = await req.json();
  } catch {
    return json(400, { error: "Invalid JSON" });
  }
  const question = typeof body.question === "string" ? body.question.trim() : "";
  if (!question || question.length > MAX_QUESTION) return json(422, { error: "Question must be 1-2000 characters" });
  const raw = Array.isArray(body.sources) ? body.sources.slice(0, MAX_SOURCES) : [];
  const sources: Src[] = raw.map((s: any, i: number) => ({
    n: i + 1,
    title: String(s?.title ?? "document").slice(0, 200),
    page: Number.isFinite(s?.page) ? Number(s.page) : 0,
    section: String(s?.section ?? "").slice(0, 200),
    text: String(s?.text ?? "").slice(0, MAX_SOURCE_CHARS),
  }));
  if (!sources.length) return json(422, { error: "No sources" });

  const model = env("LLM_MODEL", "openai/gpt-oss-120b");
  const priceIn = Number(env("LLM_PRICE_IN_PER_M", "0.15"));
  const priceOut = Number(env("LLM_PRICE_OUT_PER_M", "0.60"));
  const blocks = sources.map((s) => `[${s.n}] (${s.title}, page ${s.page}${s.section ? `, section: ${s.section}` : ""})\n${s.text.trim()}`);
  const messages = [
    { role: "system", content: SYSTEM },
    { role: "user", content: `Sources:\n\n${blocks.join("\n\n")}\n\nQuestion: ${question}` },
  ];

  // Offline mode for local development without a key: a canned, cited answer.
  const fake = env("FAKE_LLM") === "1";
  const key = env("GROQ_API_KEY");
  if (!fake && !key) return json(500, { error: "GROQ_API_KEY is not set on the server" });

  const enc = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (e: string, d: unknown) => controller.enqueue(enc.encode(sse(e, d)));
      try {
        if (fake) {
          for (const w of `This is a stand-in answer (FAKE_LLM=1); the top passage is [1].`.split(" ")) {
            send("token", w + " ");
            await new Promise((r) => setTimeout(r, 30));
          }
          send("done", { prompt_tokens: 600, completion_tokens: 20, cost_usd: (600 * priceIn + 20 * priceOut) / 1e6, model: "fake" });
          return;
        }
        const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
          method: "POST",
          headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            messages,
            temperature: 0.1,
            stream: true,
            stream_options: { include_usage: true },
            ...(model.includes("gpt-oss") ? { reasoning_effort: env("LLM_REASONING_EFFORT", "low") } : {}),
          }),
        });
        if (!r.ok || !r.body) {
          send("error", r.status === 429 ? "The model is busy, try again in a minute" : "The model call failed");
          return;
        }
        let usage = { prompt_tokens: 0, completion_tokens: 0 };
        const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
        let buf = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += value;
          let i;
          while ((i = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, i).trim();
            buf = buf.slice(i + 1);
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (data === "[DONE]") continue;
            const chunk = JSON.parse(data);
            const u = chunk.usage ?? chunk.x_groq?.usage;
            if (u) usage = { prompt_tokens: u.prompt_tokens ?? 0, completion_tokens: u.completion_tokens ?? 0 };
            const delta = chunk.choices?.[0]?.delta?.content;
            if (delta) send("token", delta);
          }
        }
        send("done", { ...usage, cost_usd: (usage.prompt_tokens * priceIn + usage.completion_tokens * priceOut) / 1e6, model });
      } catch {
        send("error", "The model call failed");
      } finally {
        controller.close();
      }
    },
  });
  return new Response(stream, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache, no-transform" } });
}
