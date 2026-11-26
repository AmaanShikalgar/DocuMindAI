import { Fragment, useEffect, useState } from "react";
import { api, LoggedQuery, Summary } from "./api";
import { fmtMs, fmtUsd } from "./ui";

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

type Tip = { x: number; y: number; lines: [string, string][] } | null;

// Stacked bottom to top. Rerank carries the accent: it is the cost v2 adds, so it's what to watch.
const SERIES = [
  { key: "retrieval", label: "Retrieval" },
  { key: "rerank", label: "Rerank" },
  { key: "generation", label: "Generation" },
] as const;

function Tooltip({ tip }: { tip: Tip }) {
  if (!tip) return null;
  return (
    <div className="tooltip" style={{ left: tip.x, top: tip.y }} role="status">
      {tip.lines.map(([k, v], i) => (
        <div key={i} className={i === 0 ? "tt-title" : "tt-row"}>
          {i === 0 ? (
            k
          ) : (
            <>
              <span className={`tt-key ${SERIES.find((s) => s.label === k)?.key ?? "none"}`} />
              <strong>{v}</strong> <span className="muted">{k}</span>
            </>
          )}
        </div>
      ))}
    </div>
  );
}

/** Stacked bars: where each question's time went. Oldest on the left. */
function LatencyChart({ rows }: { rows: LoggedQuery[] }) {
  const [tip, setTip] = useState<Tip>(null);
  const data = rows
    .filter((r) => !r.error)
    .slice(0, 40)
    .reverse()
    .map((r) => ({
      r,
      retrieval: r.retrieval_ms ?? 0,
      rerank: r.rerank_ms ?? 0,
      generation: Math.max(0, (r.total_ms ?? 0) - (r.retrieval_ms ?? 0) - (r.rerank_ms ?? 0)),
    }));
  if (!data.length) return <p className="muted">No queries yet</p>;
  const W = 960, H = 300, L = 48, B = 24, T = 8;
  const max = Math.max(...data.map((d) => d.retrieval + d.rerank + d.generation), 1);
  const step = niceStep(max);
  const top = Math.ceil(max / step) * step;
  const y = (v: number) => T + (H - T - B) * (1 - v / top);
  const bw = Math.min((W - L) / data.length, 24); // keep bars thin when there are few queries
  const ticks = Array.from({ length: Math.round(top / step) + 1 }, (_, i) => i * step);
  const tipFor = (d: (typeof data)[number]): [string, string][] => [
    [d.r.question.slice(0, 60), ""],
    ["Total", fmtMs(d.r.total_ms)],
    ...SERIES.map((s) => [s.label, fmtMs(d[s.key])] as [string, string]),
  ];
  return (
    <div className="chart" onPointerLeave={() => setTip(null)}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Latency breakdown per question">
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid" x1={L} x2={W} y1={y(t)} y2={y(t)} />
            <text className="axis" x={L - 8} y={y(t) + 4} textAnchor="end">
              {t >= 1000 ? `${t / 1000} s` : `${t} ms`}
            </text>
          </g>
        ))}
        {data.map((d, i) => {
          let acc = 0;
          const x = L + i * bw + 1;
          const w = Math.max(2, bw - 2);
          return (
            <g
              key={d.r.id}
              className="bar"
              tabIndex={0}
              onPointerMove={(e) => {
                const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                setTip({ x: e.clientX - box.left + 12, y: e.clientY - box.top + 12, lines: tipFor(d) });
              }}
              onFocus={() => setTip({ x: 56, y: 8, lines: tipFor(d) })}
              onBlur={() => setTip(null)}
            >
              <rect className="hit" x={L + i * bw} y={T} width={bw} height={H - T - B} />
              {SERIES.map((s) => {
                const v = d[s.key];
                if (v <= 0) return null;
                const y0 = y(acc), y1 = y(acc + v);
                acc += v;
                const hgt = Math.max(0, y0 - y1 - 2); // 2px surface gap between segments
                return <rect key={s.key} className={s.key} x={x} y={y1} width={w} height={hgt} rx={s.key === "generation" ? 2 : 0} />;
              })}
            </g>
          );
        })}
        <line className="baseline" x1={L} x2={W} y1={y(0)} y2={y(0)} />
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

function niceStep(max: number) {
  const raw = max / 4;
  const p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? raw;
}

export default function Dashboard() {
  const [s, setS] = useState<Summary>();
  const [rows, setRows] = useState<LoggedQuery[]>([]);
  const [open, setOpen] = useState<number>();
  const [err, setErr] = useState("");

  const load = () =>
    Promise.all([api.summary(), api.queries(100)])
      .then(([a, b]) => {
        setS(a);
        setRows(b);
      })
      .catch((e) => setErr(e.message));
  useEffect(() => {
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, []);
  const toggle = (id: number) => setOpen(open === id ? undefined : id);

  if (err)
    return (
      <main className="dash">
        <div className="dash-inner">
          <p className="error">{err}</p>
        </div>
      </main>
    );
  if (!s)
    return (
      <main className="dash">
        <div className="dash-inner">
          <p className="muted">Loading</p>
        </div>
      </main>
    );
  return (
    <main className="dash">
      <div className="dash-inner">
        <div className="dash-head">
          <h1>Queries, last 30 days</h1>
          <span className="mono muted">refreshes every 15 s</span>
        </div>
        <div className="tiles">
          <Tile label="Questions answered" value={s.queries.toLocaleString()} sub={s.errors ? `${s.errors} failed` : "none failed"} />
          <Tile label="Latency, median" value={fmtMs(s.latency_ms.p50)} sub={`p95 ${fmtMs(s.latency_ms.p95)}`} />
          <Tile label="Cost per question" value={fmtUsd(s.cost_usd.per_query)} sub={`${fmtUsd(s.cost_usd.total)} total`} />
          <Tile
            label="Not in the documents"
            value={s.not_found_rate == null ? "–" : `${Math.round(s.not_found_rate * 100)}%`}
            sub="answers that abstained"
          />
        </div>
        <p className="secondary-stats mono">
          <span>
            first token <b>{fmtMs(s.ttft_ms.p50)}</b> median
          </span>
          <span>
            retrieval <b>{fmtMs(s.retrieval_ms_p50)}</b>
          </span>
          <span>
            rerank <b>{fmtMs(s.rerank_ms_p50)}</b>
          </span>
          <span>
            tokens <b>{s.tokens.prompt.toLocaleString()}</b> in · <b>{s.tokens.completion.toLocaleString()}</b> out
          </span>
        </p>

        <section className="card">
          <div className="card-head">
            <div>
              <h2>Where the time goes</h2>
              <p className="muted small" style={{ margin: 0 }}>
                Last 40 questions, oldest first
              </p>
            </div>
            <div className="legend">
              {SERIES.map((x) => (
                <span key={x.key}>
                  <i className={`sw ${x.key}`} /> {x.label}
                </span>
              ))}
            </div>
          </div>
          <LatencyChart rows={rows} />
        </section>

        <section className="card">
          <div className="card-head">
            <h2>Query log</h2>
            <span className="muted small">Select a row to see its retrieved chunks</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Question</th>
                  <th>Config</th>
                  <th className="num">Latency</th>
                  <th className="num">Tokens</th>
                  <th className="num">Cost</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <Fragment key={r.id}>
                    <tr
                      className={`row ${open === r.id ? "open" : ""}`}
                      tabIndex={0}
                      onClick={() => toggle(r.id)}
                      onKeyDown={(e) => e.key === "Enter" && toggle(r.id)}
                    >
                      <td className="nowrap mono muted">
                        {new Date(r.ts * 1000).toLocaleString([], { dateStyle: "short", timeStyle: "short" })}
                      </td>
                      <td>
                        {r.question}
                        {r.error && <span className="badge err">error</span>}
                        {!!r.not_found && <span className="badge">not found</span>}
                      </td>
                      <td className="nowrap mono muted">
                        {r.chunking} · {r.retrieval}
                        {r.rerank ? " · rerank" : ""}
                      </td>
                      <td className="num">{fmtMs(r.total_ms)}</td>
                      <td className="num">{((r.prompt_tokens ?? 0) + (r.completion_tokens ?? 0)).toLocaleString()}</td>
                      <td className="num">{fmtUsd(r.cost_usd)}</td>
                    </tr>
                    {open === r.id && (
                      <tr className="detail">
                        <td colSpan={6}>
                          {r.error && <p className="error">{r.error}</p>}
                          <p>{r.answer || "No answer"}</p>
                          <p className="mono muted">
                            retrieval {fmtMs(r.retrieval_ms)} · rerank {fmtMs(r.rerank_ms)} · first token {fmtMs(r.ttft_ms)} · {r.model}
                          </p>
                          <ol className="retrieved">
                            {r.retrieved.map((c) => (
                              <li key={c.id}>
                                <span className="mono">
                                  p.{c.page} · score {c.score}
                                  {c.dense_rank != null && ` · dense #${c.dense_rank}`}
                                  {c.bm25_rank != null && ` · BM25 #${c.bm25_rank}`}
                                </span>
                                {c.section && <span className="small"> · {c.section}</span>}
                                <div className="muted small">{c.preview}…</div>
                              </li>
                            ))}
                          </ol>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  );
}
