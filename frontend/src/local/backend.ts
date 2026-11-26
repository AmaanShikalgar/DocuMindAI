/** The browser engine: implements the same Api as the FastAPI backend, but retrieval runs on
 * the visitor's device. Demo documents come pre-indexed from /demo (built by
 * deploy/build_static.py); uploaded PDFs are parsed, chunked and embedded here and never leave
 * the browser. Only the question and the passages used for an answer go to /api/chat, a small
 * serverless function that holds the Groq key. */
import type { Api, AskBody, AskHandlers, Config, Doc, Done, EngineStatus, LoggedQuery, Source, Summary } from "../api";
import { fixedChunks, structureChunks } from "../engine/chunking";
import { DEFAULT_DTYPES, embed, preload } from "../engine/models";
import { parsePdf } from "../engine/pdf";
import { forgetPdf, pdfjs } from "../engine/pdfjs-browser";
import { Chunk, Hit, Index, indexText, Timings } from "../engine/retrieval";
import { readSse } from "../sse";

type Strategy = "fixed" | "structure";
type DocMeta = Doc & { page_sizes: Record<string, [number, number]> };
type Entry = {
  meta: DocMeta;
  source: string | Uint8Array; // URL for demo PDFs, bytes for uploads
  blobUrl?: string;
  parsed?: Awaited<ReturnType<typeof parsePdf>>;
  chunks: Partial<Record<Strategy, Chunk[]>>;
  vectors: Partial<Record<Strategy, Float32Array>>;
};

const NOT_FOUND = "I cannot find this in the document.";
const LOG_KEY = "documind-queries";
const MAX_PAGES = 200;
const MODEL = import.meta.env.VITE_LLM_MODEL ?? "openai/gpt-oss-120b";

// ---------------- state ----------------
let demoIndex: Promise<{ entries: Entry[]; questions: string[]; strategy: Record<Strategy, Index> }> | null = null;
const uploads: Entry[] = [];
const merged = new Map<Strategy, Index>();
let status: EngineStatus = { state: "idle" };
const listeners = new Set<(s: EngineStatus) => void>();
const setStatus = (s: EngineStatus) => {
  status = s;
  listeners.forEach((l) => l(s));
};

function loadDemo() {
  demoIndex ??= (async () => {
    const [index, fixed, structure] = await Promise.all([
      fetch("/demo/index.json").then((r) => r.json()),
      fetch("/demo/vectors_fixed.f32").then((r) => r.arrayBuffer()),
      fetch("/demo/vectors_structure.f32").then((r) => r.arrayBuffer()),
    ]);
    const entries: Entry[] = index.docs.map((d: DocMeta) => ({ meta: { ...d, created_at: 0 }, source: `/demo/${d.id}.pdf`, chunks: {}, vectors: {} }));
    return {
      entries,
      questions: index.questions,
      strategy: {
        fixed: new Index(index.chunks.fixed, new Float32Array(fixed)),
        structure: new Index(index.chunks.structure, new Float32Array(structure)),
      },
    };
  })();
  return demoIndex;
}

/** Start downloading the two models (~39 MB, cached by the browser after the first visit). */
function warmModels() {
  if (status.state !== "idle") return;
  setStatus({ state: "loading", progress: 0, message: "Loading search models" });
  const files = new Map<string, [number, number]>();
  preload(DEFAULT_DTYPES, ({ file, loaded, total }) => {
    files.set(file, [loaded, total]);
    let l = 0;
    let t = 0;
    for (const [a, b] of files.values()) {
      l += a;
      t += b;
    }
    setStatus({ state: "loading", progress: t ? l / t : 0, message: "Loading search models" });
  })
    .then(() => setStatus({ state: "ready", message: "Search runs on this device" }))
    .catch((e) => setStatus({ state: "error", message: `Couldn't load the search models: ${e.message ?? e}` }));
}

async function allEntries() {
  return [...(await loadDemo()).entries, ...uploads];
}

async function indexFor(strategy: Strategy, onProgress?: (m: string) => void): Promise<Index> {
  if (merged.has(strategy)) return merged.get(strategy)!;
  const demo = await loadDemo();
  for (const u of uploads) if (!u.vectors[strategy]) await embedUpload(u, strategy, onProgress);
  const parts = [demo.strategy[strategy], ...uploads.map((u) => new Index(u.chunks[strategy]!, u.vectors[strategy]!))];
  const idx = parts.length === 1 ? parts[0] : Index.merge(parts);
  merged.set(strategy, idx);
  return idx;
}

async function embedUpload(u: Entry, strategy: Strategy, onProgress?: (m: string) => void) {
  const chunks = (u.chunks[strategy] ??= strategy === "fixed" ? fixedChunks(u.parsed!, u.meta.id) : structureChunks(u.parsed!, u.meta.id));
  const out = new Float32Array(chunks.length * 384);
  const batch = 16;
  for (let s = 0; s < chunks.length; s += batch) {
    onProgress?.(`Indexing ${u.meta.title}: ${Math.round((s / chunks.length) * 100)}%`);
    const vecs = await embed(chunks.slice(s, s + batch).map(indexText), DEFAULT_DTYPES.embed, batch);
    vecs.forEach((v, j) => out.set(v, (s + j) * 384));
  }
  u.vectors[strategy] = out;
}

async function sha12(data: ArrayBuffer) {
  const h = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 12);
}

const publicDoc = (e: Entry): Doc => {
  const { page_sizes: _, ...d } = e.meta;
  return d;
};

// ---------------- query log (this browser only) ----------------
function readLog(): LoggedQuery[] {
  try {
    return JSON.parse(localStorage.getItem(LOG_KEY) ?? "[]");
  } catch {
    return [];
  }
}
function writeLog(q: LoggedQuery) {
  const log = [q, ...readLog()].slice(0, 200);
  localStorage.setItem(LOG_KEY, JSON.stringify(log));
}

function pct(values: (number | null)[], p: number): number | null {
  const v = values.filter((x): x is number => x != null).sort((a, b) => a - b);
  if (!v.length) return null;
  return Math.round(v[Math.min(v.length - 1, Math.max(0, Math.round((p / 100) * (v.length - 1))))] * 10) / 10;
}

function summarize(days: number): Summary {
  const since = Date.now() / 1000 - days * 86400;
  const rows = readLog().filter((r) => r.ts >= since);
  const ok = rows.filter((r) => !r.error);
  const cost = ok.reduce((a, r) => a + (r.cost_usd ?? 0), 0);
  return {
    queries: rows.length,
    errors: rows.length - ok.length,
    not_found_rate: ok.length ? ok.filter((r) => r.not_found).length / ok.length : null,
    latency_ms: { p50: pct(ok.map((r) => r.total_ms), 50), p95: pct(ok.map((r) => r.total_ms), 95) },
    ttft_ms: { p50: pct(ok.map((r) => r.ttft_ms), 50), p95: pct(ok.map((r) => r.ttft_ms), 95) },
    retrieval_ms_p50: pct(ok.map((r) => r.retrieval_ms), 50),
    rerank_ms_p50: pct(ok.map((r) => r.rerank_ms), 50),
    tokens: { prompt: ok.reduce((a, r) => a + (r.prompt_tokens ?? 0), 0), completion: ok.reduce((a, r) => a + (r.completion_tokens ?? 0), 0) },
    cost_usd: { total: cost, per_query: ok.length ? cost / ok.length : null },
    series: [],
  };
}

// ---------------- Api ----------------
export const browserApi: Api = {
  async config(): Promise<Config> {
    const demo = await loadDemo();
    if (typeof window !== "undefined") (window.requestIdleCallback ?? setTimeout)(() => warmModels());
    return {
      model: MODEL,
      defaults: { chunking: "structure", retrieval: "hybrid", rerank: true, top_k: 5 },
      options: { chunking: ["fixed", "structure"], retrieval: ["dense", "bm25", "hybrid"] },
      max_upload_mb: 25,
      sample_questions: demo.questions,
      private_uploads: true,
      engine: "browser",
    };
  },
  docs: async () => (await allEntries()).map(publicDoc),
  async upload(file, onProgress) {
    if (file.size > 25 * 1024 * 1024) throw new Error("PDF larger than 25 MB");
    const buf = await file.arrayBuffer();
    if (new TextDecoder().decode(buf.slice(0, 4)) !== "%PDF") throw new Error("Not a PDF file");
    const id = await sha12(buf);
    const existing = (await allEntries()).find((e) => e.meta.id === id);
    if (existing) return publicDoc(existing);
    onProgress?.("Reading the PDF");
    const bytes = new Uint8Array(buf);
    const title = file.name.replace(/\.pdf$/i, "");
    const parsed = await parsePdf(pdfjs, bytes.slice(), title);
    if (parsed.pages.length > MAX_PAGES) throw new Error(`PDFs up to ${MAX_PAGES} pages are supported here`);
    if (!parsed.pages.some((p) => p.text.trim())) throw new Error("No extractable text in this PDF (is it a scan?)");
    const entry: Entry = {
      meta: {
        id,
        title,
        filename: file.name,
        num_pages: parsed.pages.length,
        num_chunks: {},
        created_at: Date.now() / 1000,
        demo: false,
        page_sizes: Object.fromEntries(parsed.pages.map((p) => [String(p.number), [p.width, p.height]])),
      },
      source: bytes,
      blobUrl: URL.createObjectURL(file),
      parsed,
      chunks: {},
      vectors: {},
    };
    if (status.state === "idle") warmModels();
    await embedUpload(entry, "structure", onProgress);
    entry.meta.num_chunks = { structure: entry.chunks.structure!.length };
    uploads.push(entry);
    merged.clear();
    return publicDoc(entry);
  },
  async remove(id) {
    const i = uploads.findIndex((u) => u.meta.id === id);
    if (i < 0) throw new Error("Demo documents can't be removed");
    const [u] = uploads.splice(i, 1);
    if (u.blobUrl) URL.revokeObjectURL(u.blobUrl);
    forgetPdf(id);
    merged.clear();
  },
  summary: async (days = 30) => summarize(days),
  queries: async (limit = 50) => readLog().slice(0, limit),
  pageUrl: () => "",
  fileUrl: (doc, page) => {
    const u = uploads.find((x) => x.meta.id === doc);
    return `${u?.blobUrl ?? `/demo/${doc}.pdf`}#page=${page}`;
  },
  pdfSource: (doc) => uploads.find((x) => x.meta.id === doc)?.source ?? `/demo/${doc}.pdf`,
  onStatus(cb) {
    listeners.add(cb);
    cb(status);
    return () => listeners.delete(cb);
  },
};

export async function browserAsk(body: AskBody, on: AskHandlers, signal?: AbortSignal) {
  const t0 = performance.now();
  const timings: Timings = { retrieval_ms: 0, rerank_ms: 0 };
  const strategy = (body.chunking === "fixed" ? "fixed" : "structure") as Strategy;
  let hits: Hit[] = [];
  let answer = "";
  let ttft: number | null = null;
  let usage = { prompt_tokens: 0, completion_tokens: 0, cost_usd: 0 };
  let error: string | null = null;
  const entries = await allEntries();
  const titles = new Map(entries.map((e) => [e.meta.id, e.meta]));
  try {
    if (status.state !== "ready") warmModels();
    const index = await indexFor(strategy, (m) => setStatus({ state: "loading", message: m }));
    if (status.state === "loading" && status.message?.startsWith("Indexing")) setStatus({ state: "ready", message: "Search runs on this device" });
    const allowed = body.doc_ids?.length ? new Set(body.doc_ids) : undefined;
    hits = await index.search(body.question, { k: body.top_k, mode: body.retrieval as any, rerank: body.rerank, allowed, timings });
    const sources: Source[] = hits.map((h) => {
      const meta = titles.get(h.chunk.doc_id)!;
      return {
        ...h.chunk,
        n: h.rank,
        title: meta.title,
        page_size: meta.page_sizes[String(h.chunk.page)],
        score: h.score,
        dense_rank: h.dense_rank,
        bm25_rank: h.bm25_rank,
        rerank_score: h.rerank_score,
      };
    });
    on.sources(sources);
    if (!hits.length) {
      answer = NOT_FOUND;
      on.token(NOT_FOUND);
    } else {
      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: body.question,
          sources: sources.map((s) => ({ n: s.n, title: s.title, page: s.page, section: s.section, text: s.text })),
        }),
        signal,
      });
      if (!r.ok || !r.body) {
        let msg = `The answer service returned ${r.status}`;
        try {
          msg = (await r.json()).error ?? msg;
        } catch {
          /* not json */
        }
        throw new Error(msg);
      }
      await readSse(r.body, (event, data) => {
        if (event === "token") {
          ttft ??= performance.now() - t0;
          answer += data as string;
          on.token(data as string);
        } else if (event === "done") usage = data as typeof usage;
        else if (event === "error") throw new Error(data as string);
      });
    }
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    error = (e as Error).message;
    on.error(error);
  }
  const total = performance.now() - t0;
  const cited = [...new Set([...answer.matchAll(/\[(\d+)\]/g)].map((m) => +m[1]))].filter((n) => n >= 1 && n <= hits.length);
  const notFound = answer.toLowerCase().includes(NOT_FOUND.toLowerCase().replace(/\.$/, ""));
  const done: Done = {
    query_id: Date.now(),
    cited,
    not_found: notFound,
    retrieval_ms: Math.round(timings.retrieval_ms * 10) / 10,
    rerank_ms: Math.round(timings.rerank_ms * 10) / 10,
    ttft_ms: ttft == null ? null : Math.round(ttft),
    total_ms: Math.round(total),
    prompt_tokens: usage.prompt_tokens,
    completion_tokens: usage.completion_tokens,
    cost_usd: usage.cost_usd,
  };
  writeLog({
    id: done.query_id,
    ts: Date.now() / 1000,
    question: body.question,
    answer: answer.trim(),
    chunking: strategy,
    retrieval: body.retrieval,
    rerank: body.rerank ? 1 : 0,
    top_k: body.top_k,
    model: MODEL,
    retrieved: hits.map((h) => ({
      n: h.rank,
      id: h.chunk.id,
      doc_id: h.chunk.doc_id,
      page: h.chunk.page,
      section: h.chunk.section,
      score: Math.round(h.score * 1e4) / 1e4,
      dense_rank: h.dense_rank,
      bm25_rank: h.bm25_rank,
      preview: h.chunk.text.slice(0, 240),
    })),
    retrieval_ms: done.retrieval_ms,
    rerank_ms: done.rerank_ms,
    ttft_ms: done.ttft_ms,
    total_ms: done.total_ms,
    prompt_tokens: done.prompt_tokens,
    completion_tokens: done.completion_tokens,
    cost_usd: done.cost_usd,
    not_found: notFound ? 1 : 0,
    error,
  });
  if (!error) on.done(done);
}
