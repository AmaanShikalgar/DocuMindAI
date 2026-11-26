import { browserApi, browserAsk } from "./local/backend";
import { readSse } from "./sse";

export type Doc = {
  id: string;
  title: string;
  filename: string;
  num_pages: number;
  num_chunks: Record<string, number>;
  created_at: number;
  demo: boolean;
};

export type Source = {
  n: number;
  id: string;
  doc_id: string;
  title: string;
  page: number;
  start: number;
  end: number;
  text: string;
  section: string;
  rects: number[][];
  page_size: [number, number];
  score: number;
  dense_rank: number | null;
  bm25_rank: number | null;
  rerank_score: number | null;
};

export type Done = {
  query_id: number;
  cited: number[];
  not_found: boolean;
  retrieval_ms: number;
  rerank_ms: number;
  ttft_ms: number | null;
  total_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
};

export type QueryOpts = { chunking: string; retrieval: string; rerank: boolean; top_k: number };

export type Config = {
  model: string;
  defaults: QueryOpts;
  options: { chunking: string[]; retrieval: string[] };
  max_upload_mb: number;
  sample_questions: string[];
  private_uploads: boolean;
  engine?: "browser" | "server";
};

/** A random id per browser. On the public demo it keeps each visitor's uploads and query log
 * private to them; it is not an account and identifies nothing. */
function clientId(): string {
  let id = localStorage.getItem("documind-client");
  if (!id) {
    id = crypto.randomUUID?.() ?? Array.from({ length: 4 }, () => Math.random().toString(36).slice(2, 10)).join("-");
    localStorage.setItem("documind-client", id);
  }
  return id;
}
const H = (): Record<string, string> => ({ "X-Client-Id": clientId() });

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = r.statusText;
    try {
      msg = (await r.json()).detail ?? msg;
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  return r.json();
}

const serverApi: Api = {
  config: () => fetch("/api/config").then((r) => json<Config>(r)),
  docs: () => fetch("/api/documents", { headers: H() }).then((r) => json<Doc[]>(r)),
  upload: (f: File) => {
    const body = new FormData();
    body.append("file", f);
    return fetch("/api/documents", { method: "POST", body, headers: H() }).then((r) => json<Doc>(r));
  },
  remove: (id: string) => fetch(`/api/documents/${id}`, { method: "DELETE", headers: H() }),
  summary: (days = 30) => fetch(`/api/metrics/summary?days=${days}`).then((r) => json<Summary>(r)),
  queries: (limit = 50) => fetch(`/api/queries?limit=${limit}`, { headers: H() }).then((r) => json<LoggedQuery[]>(r)),
  pageUrl: (doc: string, page: number) => `/api/documents/${doc}/pages/${page}.png`,
  fileUrl: (doc: string, page: number) => `/api/documents/${doc}/file#page=${page}`,
  pdfSource: (doc: string) => `/api/documents/${doc}/file`,
};

export type Api = {
  config: () => Promise<Config>;
  docs: () => Promise<Doc[]>;
  upload: (f: File, onProgress?: (msg: string) => void) => Promise<Doc>;
  remove: (id: string) => Promise<unknown>;
  summary: (days?: number) => Promise<Summary>;
  queries: (limit?: number) => Promise<LoggedQuery[]>;
  pageUrl: (doc: string, page: number) => string;
  fileUrl: (doc: string, page: number) => string;
  /** What pdf.js should open to render a page: a URL, or the bytes of a local upload. */
  pdfSource: (doc: string) => string | Uint8Array;
  /** Browser engine only: model download / readiness, for the status line. */
  onStatus?: (cb: (s: EngineStatus) => void) => () => void;
};

export type EngineStatus = { state: "idle" | "loading" | "ready" | "error"; progress?: number; message?: string };

/** POST a question and consume the server-sent event stream. */
async function serverAsk(body: AskBody, on: AskHandlers, signal?: AbortSignal) {
  const r = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...H() },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) {
    on.error(await json(r).then(() => "Request failed", (e: Error) => e.message));
    return;
  }
  await readSse(r.body, (event, payload) => {
    if (event === "sources") on.sources(payload as Source[]);
    else if (event === "token") on.token(payload as string);
    else if (event === "done") on.done(payload as Done);
    else if (event === "error") on.error(payload as string);
  });
}

export type AskBody = { question: string; doc_ids?: string[] } & QueryOpts;
export type AskHandlers = { sources: (s: Source[]) => void; token: (t: string) => void; done: (d: Done) => void; error: (e: string) => void };

/** Where the search runs: "browser" (static site; retrieval on the visitor's device) or
 * "server" (the FastAPI backend, as in Docker). Set with VITE_ENGINE at build time. */
export const ENGINE: "browser" | "server" = import.meta.env.VITE_ENGINE === "server" ? "server" : "browser";
export const api: Api = ENGINE === "server" ? serverApi : browserApi;
export const ask: (body: AskBody, on: AskHandlers, signal?: AbortSignal) => Promise<void> = ENGINE === "server" ? serverAsk : browserAsk;

export type Summary = {
  queries: number;
  errors: number;
  not_found_rate: number | null;
  latency_ms: { p50: number | null; p95: number | null };
  ttft_ms: { p50: number | null; p95: number | null };
  retrieval_ms_p50: number | null;
  rerank_ms_p50: number | null;
  tokens: { prompt: number; completion: number };
  cost_usd: { total: number; per_query: number | null };
  series: { day: string; queries: number; cost_usd: number; p50_ms: number | null }[];
};

export type LoggedQuery = {
  id: number;
  ts: number;
  question: string;
  answer: string;
  chunking: string;
  retrieval: string;
  rerank: number;
  top_k: number;
  model: string;
  retrieved: { n: number; id: string; doc_id: string; page: number; section: string; score: number; dense_rank: number | null; bm25_rank: number | null; preview: string }[];
  retrieval_ms: number;
  rerank_ms: number;
  ttft_ms: number | null;
  total_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
  not_found: number;
  error: string | null;
};
