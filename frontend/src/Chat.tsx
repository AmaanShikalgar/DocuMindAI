import { useEffect, useRef, useState } from "react";
import { api, ask, Config, Doc, Done, EngineStatus, QueryOpts, Source } from "./api";
import AnswerText from "./AnswerText";
import CitationViewer from "./CitationViewer";
import { fmtMs, fmtUsd, UploadIcon, XIcon } from "./ui";

type Turn = { q: string; answer: string; sources: Source[]; done?: Done; error?: string; streaming: boolean };

const NOT_FOUND = "I cannot find this in the document.";

const STOP = new Set("a an and are as at be by for from how in is it of on or the to was what when where which who why with does do can".split(" "));
const words = (t: string) => (t.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((w) => w.length > 2 && !STOP.has(w));

/** The sentence of a passage that best matches the question, for the source card preview. */
function snippet(text: string, question: string, max = 170): string {
  const flat = text.replace(/-\n(?=[a-z])/g, "").replace(/\s+/g, " ").trim();
  const q = new Set(words(question));
  const sentences = flat.split(/(?<=[.!?])\s+(?=[A-Z0-9"(])/);
  let best = sentences[0] ?? flat;
  let score = -1;
  for (const sent of sentences) {
    const sc = words(sent).filter((w) => q.has(w)).length;
    if (sc > score) [best, score] = [sent, sc];
  }
  const lead = best === sentences[0] ? "" : "…";
  return best.length > max ? `${lead}${best.slice(0, max).replace(/\s+\S*$/, "")}…` : `${lead}${best}`;
}

function Answer({ turn, onCite, active, engine, docCount }: { turn: Turn; onCite: (s: Source) => void; active?: string; engine?: EngineStatus; docCount: number }) {
  const byN = new Map(turn.sources.map((s) => [s.n, s]));
  const question = turn.q;
  const cited = turn.done?.cited ?? [...new Set([...turn.answer.matchAll(/\[(\d+)\]/g)].map((m) => +m[1]))].filter((n) => byN.has(n));
  const [showAll, setShowAll] = useState(false);
  const listed = showAll ? turn.sources : turn.sources.filter((s) => cited.includes(s.n));
  return (
    <div className="answer">
      {turn.streaming && !turn.answer && !turn.error && (
        <p className="pending mono">
          <span className="pulse-dot" aria-hidden="true" />
          {engine?.state === "loading"
            ? `${engine.message ?? "Loading search models"}${engine.progress ? ` ${Math.round(engine.progress * 100)}%` : ""} (first question only)`
            : turn.sources.length
              ? `Found ${turn.sources.length} passages · writing the answer`
              : `Searching ${docCount} document${docCount === 1 ? "" : "s"}`}
        </p>
      )}
      {turn.answer && <AnswerText text={turn.answer} sources={turn.sources} active={active} onCite={onCite} streaming={turn.streaming} />}
      {turn.error && <p className="error">{turn.error}</p>}
      {turn.sources.length > 0 && !turn.streaming && !turn.answer.startsWith(NOT_FOUND) && (
        <div className="sources">
          {listed.map((s) => (
            <button key={s.id} className={`source ${active === s.id ? "on" : ""}`} onClick={() => onCite(s)}>
              <span className="n">{s.n}</span>
              <span className="where">
                {s.title} <span className="page-no">p.{s.page}</span>
                {s.section && <span className="sec"> · {s.section}</span>}
              </span>
              <span className="snippet">{snippet(s.text, question)}</span>
            </button>
          ))}
          {turn.sources.length > cited.length && (
            <button className="link" onClick={() => setShowAll(!showAll)}>
              {showAll ? "Show cited only" : `Show all ${turn.sources.length} retrieved`}
            </button>
          )}
        </div>
      )}
      {turn.done && (
        <p className="meta mono">
          {fmtMs(turn.done.total_ms)} total · first token {fmtMs(turn.done.ttft_ms)} · retrieval {fmtMs(turn.done.retrieval_ms)}
          {turn.done.rerank_ms ? ` · rerank ${fmtMs(turn.done.rerank_ms)}` : ""} ·{" "}
          {(turn.done.prompt_tokens + turn.done.completion_tokens).toLocaleString()} tokens · {fmtUsd(turn.done.cost_usd)}
        </p>
      )}
    </div>
  );
}

export default function Chat() {
  const [cfg, setCfg] = useState<Config>();
  const [opts, setOpts] = useState<QueryOpts>({ chunking: "structure", retrieval: "hybrid", rerank: true, top_k: 5 });
  const [docs, setDocs] = useState<Doc[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState("");
  const [progress, setProgress] = useState("");
  const [engine, setEngine] = useState<EngineStatus>();
  const [viewing, setViewing] = useState<Source>();
  const end = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController>();

  useEffect(() => {
    api.config().then((c) => {
      setCfg(c);
      setOpts(c.defaults);
    });
    api.docs().then(setDocs);
    const off = api.onStatus?.(setEngine);
    return () => {
      off?.();
    };
  }, []);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" }); // newer browsers return a Promise; don't hand it to React
  }, [turns]);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    setNotice("");
    try {
      for (const f of Array.from(files)) {
        const d = await api.upload(f, setProgress);
        setDocs((ds) => (ds.some((x) => x.id === d.id) ? ds : [...ds, d]));
      }
    } catch (e) {
      setNotice((e as Error).message);
    } finally {
      setProgress("");
      setUploading(false);
    }
  }

  async function remove(id: string) {
    await api.remove(id);
    setDocs((ds) => ds.filter((d) => d.id !== id));
    setSelected((s) => new Set([...s].filter((x) => x !== id)));
  }

  async function submit(e?: React.FormEvent, preset?: string) {
    e?.preventDefault();
    const question = (preset ?? q).trim();
    if (!question || busy) return;
    setQ("");
    setBusy(true);
    const idx = turns.length;
    const patch = (f: (t: Turn) => Partial<Turn>) => setTurns((ts) => ts.map((t, i) => (i === idx ? { ...t, ...f(t) } : t)));
    setTurns((ts) => [...ts, { q: question, answer: "", sources: [], streaming: true }]);
    abort.current = new AbortController();
    try {
      await ask(
        { question, doc_ids: selected.size ? [...selected] : undefined, ...opts },
        {
          sources: (sources) => patch(() => ({ sources })),
          token: (t) => patch((x) => ({ answer: x.answer + t })),
          done: (done) => patch(() => ({ done })),
          error: (error) => patch(() => ({ error })),
        },
        abort.current.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") patch(() => ({ error: (err as Error).message }));
    } finally {
      patch(() => ({ streaming: false }));
      setBusy(false);
    }
  }

  return (
    <main className={`chat-layout ${viewing ? "with-viewer" : ""}`}>
      <aside className="sidebar">
        <section>
          <h2>Documents</h2>
          <label className={`btn upload ${uploading ? "busy" : ""}`}>
            <input type="file" accept="application/pdf" multiple onChange={(e) => upload(e.target.files)} disabled={uploading} />
            <UploadIcon />
            {uploading ? "Indexing" : "Upload PDF"}
          </label>
          {progress && <p className="mono muted">{progress}</p>}
          {notice && <p className="error small">{notice}</p>}
          {docs.length === 0 && <p className="muted small">No documents yet</p>}
          <ul className="docs">
            {docs.map((d) => (
              <li key={d.id}>
                <label title="Limit questions to the checked documents (none checked = all)">
                  <input
                    type="checkbox"
                    checked={selected.has(d.id)}
                    onChange={(e) =>
                      setSelected((s) => {
                        const n = new Set(s);
                        e.target.checked ? n.add(d.id) : n.delete(d.id);
                        return n;
                      })
                    }
                  />
                  <span className="doc-title">{d.title}</span>
                </label>
                <span className="mono muted">{d.num_pages} pp</span>
                {d.demo ? (
                  <span className="icon-btn" aria-hidden="true" />
                ) : (
                  <button className="icon-btn" aria-label={`Remove ${d.title}`} onClick={() => remove(d.id)}>
                    <XIcon />
                  </button>
                )}
              </li>
            ))}
          </ul>
        </section>
        <section>
          <h2>Retrieval</h2>
          <label className="field">
            Chunking
            <select value={opts.chunking} onChange={(e) => setOpts({ ...opts, chunking: e.target.value })}>
              {(cfg?.options.chunking ?? ["fixed", "structure"]).map((o) => (
                <option key={o} value={o}>
                  {o === "fixed" ? "Fixed 500 chars (v1)" : "Structure-aware"}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Search
            <select value={opts.retrieval} onChange={(e) => setOpts({ ...opts, retrieval: e.target.value })}>
              {(cfg?.options.retrieval ?? ["dense", "bm25", "hybrid"]).map((o) => (
                <option key={o} value={o}>
                  {{ dense: "Dense (FAISS)", bm25: "Keyword (BM25)", hybrid: "Hybrid (BM25 + FAISS)" }[o] ?? o}
                </option>
              ))}
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={opts.rerank} onChange={(e) => setOpts({ ...opts, rerank: e.target.checked })} />
            Cross-encoder reranker
          </label>
          <label className="field">
            Sources per answer
            <input type="number" min={1} max={10} value={opts.top_k} onChange={(e) => setOpts({ ...opts, top_k: +e.target.value || 5 })} />
          </label>
          {cfg && <p className="mono muted">{cfg.model}</p>}
          {engine && engine.state !== "idle" && (
            <p className={`mono ${engine.state === "error" ? "error" : "muted"}`}>
              {engine.message}
              {engine.state === "loading" && engine.progress != null && engine.progress > 0 ? ` ${Math.round(engine.progress * 100)}%` : ""}
            </p>
          )}
        </section>
      </aside>

      <section className="conversation">
        <div className="turns">
          {turns.length === 0 && (
            <div className="empty">
              <h1>Ask a question about your PDFs</h1>
              <p className="proof mono">95% retrieval hit-rate on a 48-question eval</p>
              <p>
                Every answer cites the page it came from. Select a <span className="cite">1</span> to see the passage highlighted on that page.
              </p>
              {!!cfg?.sample_questions.length && docs.length > 0 && (
                <div className="samples">
                  {cfg.sample_questions.map((s) => (
                    <button key={s} className="sample" onClick={() => submit(undefined, s)} disabled={busy}>
                      {s}
                    </button>
                  ))}
                </div>
              )}
              {cfg?.engine === "browser" ? (
                <p className="small muted">Search runs on your device. PDFs you upload never leave this browser; only the passages used for an answer are sent to the model.</p>
              ) : (
                cfg?.private_uploads && <p className="small muted">PDFs you upload are visible only in this browser</p>
              )}
            </div>
          )}
          {turns.map((t, i) => (
            <div key={i} className="turn">
              <p className="question">{t.q}</p>
              <Answer turn={t} onCite={setViewing} active={viewing?.id} engine={engine} docCount={selected.size || docs.length} />
            </div>
          ))}
          <div ref={end} />
        </div>
        <form className="composer" onSubmit={submit}>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={docs.length ? "Ask a question" : "Upload a PDF first"}
            disabled={!docs.length}
            aria-label="Question"
          />
          {busy ? (
            <button type="button" className="btn" onClick={() => abort.current?.abort()}>
              Stop
            </button>
          ) : (
            <button type="submit" className="btn primary" disabled={!q.trim() || !docs.length}>
              Ask
            </button>
          )}
        </form>
      </section>

      {viewing && <CitationViewer source={viewing} onClose={() => setViewing(undefined)} />}
    </main>
  );
}
