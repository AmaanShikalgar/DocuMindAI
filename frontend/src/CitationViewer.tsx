import { useEffect, useRef, useState } from "react";
import { api, Source } from "./api";
import { getPdf } from "./engine/pdfjs-browser";
import { ExternalIcon, XIcon } from "./ui";

/** The cited page, rendered with pdf.js, with the passage's exact rectangles highlighted. */
export default function CitationViewer({ source, onClose }: { source: Source; onClose: () => void }) {
  const [w, h] = source.page_size;
  const canvas = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const first = useRef<HTMLDivElement>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let task: { cancel: () => void; promise: Promise<void> } | null = null;
    setError("");
    (async () => {
      const pdf = await getPdf(source.doc_id, api.pdfSource(source.doc_id));
      const page = await pdf.getPage(source.page);
      const el = canvas.current;
      if (cancelled || !el || !wrap.current) return;
      const base = page.getViewport({ scale: 1 });
      const scale = (wrap.current.clientWidth / base.width) * Math.min(window.devicePixelRatio || 1, 2);
      const vp = page.getViewport({ scale });
      el.width = vp.width;
      el.height = vp.height;
      task = page.render({ canvas: el, viewport: vp });
      await task.promise;
      if (!cancelled) first.current?.scrollIntoView({ block: "center", behavior: "smooth" });
    })().catch((e) => !cancelled && e?.name !== "RenderingCancelledException" && setError("Couldn't render this page"));
    return () => {
      cancelled = true;
      task?.cancel();
    };
  }, [source.doc_id, source.page, source.id]);

  return (
    <aside className="viewer" aria-label="Cited passage">
      <header>
        <div className="title">
          <span className="cite on">{source.n}</span> {source.title} <span className="mono muted">p.{source.page}</span>
          {source.section && <span className="muted"> · {source.section}</span>}
        </div>
        <div className="viewer-actions">
          <a className="open-pdf" href={api.fileUrl(source.doc_id, source.page)} target="_blank" rel="noreferrer">
            Open PDF <ExternalIcon />
          </a>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <XIcon />
          </button>
        </div>
      </header>
      <div className="page-wrap">
        <div className="page" ref={wrap} style={{ aspectRatio: `${w} / ${h}` }}>
          <canvas ref={canvas} aria-label={`Page ${source.page} of ${source.title}`} />
          {source.rects.map((r, i) => (
            <div
              key={i}
              ref={i === 0 ? first : undefined}
              className="hl"
              style={{
                left: `${(r[0] / w) * 100}%`,
                top: `${(r[1] / h) * 100}%`,
                width: `${((r[2] - r[0]) / w) * 100}%`,
                height: `${((r[3] - r[1]) / h) * 100}%`,
              }}
            />
          ))}
          {error && <p className="page-error muted">{error}</p>}
        </div>
      </div>
      <blockquote className="passage">{source.text.replace(/-\n(?=[a-z])/g, "").replace(/\s*\n\s*/g, " ")}</blockquote>
      <p className="meta mono">
        score {source.score.toFixed(3)}
        {source.dense_rank != null && ` · dense #${source.dense_rank}`}
        {source.bm25_rank != null && ` · BM25 #${source.bm25_rank}`}
        {source.rerank_score != null && " · reranked"}
      </p>
    </aside>
  );
}
