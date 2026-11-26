import { useEffect, useRef } from "react";
import { api, Source } from "./api";
import { ExternalIcon, XIcon } from "./ui";

/** The cited page rendered as an image, with the passage's exact rectangles highlighted. */
export default function CitationViewer({ source, onClose }: { source: Source; onClose: () => void }) {
  const [w, h] = source.page_size;
  const first = useRef<HTMLDivElement>(null);
  useEffect(() => {
    first.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [source.id]);
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
        <div className="page" style={{ aspectRatio: `${w} / ${h}` }}>
          <img src={api.pageUrl(source.doc_id, source.page)} alt={`Page ${source.page} of ${source.title}`} />
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
