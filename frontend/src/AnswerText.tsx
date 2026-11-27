import { Fragment, ReactNode } from "react";
import type { Source } from "./api";
import { normalizeCitations } from "./citations";

/** Renders an LLM answer: paragraphs, "-" / "1." lists, **bold**, `code`, and [n] citations as
 * buttons. Deliberately small: anything else stays plain text (never HTML from the model). */
export default function AnswerText({
  text,
  sources,
  active,
  onCite,
  streaming,
}: {
  text: string;
  sources: Source[];
  active?: string;
  onCite: (s: Source) => void;
  streaming: boolean;
}) {
  const byN = new Map(sources.map((s) => [s.n, s]));

  const inline = (line: string, key: string): ReactNode[] =>
    line.split(/(\[\d+\]|\*\*[^*]+\*\*|`[^`]+`)/g).map((p, i) => {
      const k = `${key}-${i}`;
      const m = p.match(/^\[(\d+)\]$/);
      const s = m && byN.get(+m[1]);
      if (s)
        return (
          <button key={k} className={`cite ${active === s.id ? "on" : ""}`} onClick={() => onCite(s)} title={`${s.title}, page ${s.page}`}>
            {s.n}
          </button>
        );
      if (/^\*\*[^*]+\*\*$/.test(p)) return <strong key={k}>{inline(p.slice(2, -2), k)}</strong>;
      if (/^`[^`]+`$/.test(p)) return <code key={k}>{p.slice(1, -1)}</code>;
      return <Fragment key={k}>{p}</Fragment>;
    });

  // group lines into paragraphs and lists
  const blocks: { kind: "p" | "ul" | "ol"; lines: string[] }[] = [];
  for (const raw of normalizeCitations(text).split("\n")) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const kind = bullet ? "ul" : numbered ? "ol" : "p";
    const content = bullet?.[1] ?? numbered?.[1] ?? line;
    const last = blocks[blocks.length - 1];
    if (!line.trim()) {
      blocks.push({ kind: "p", lines: [] });
      continue;
    }
    if (last && last.kind === kind && (kind !== "p" || last.lines.length)) last.lines.push(content);
    else blocks.push({ kind, lines: [content] });
  }
  const shown = blocks.filter((b) => b.lines.length);
  return (
    <div className="answer-text">
      {shown.map((b, i) => {
        const caret = streaming && i === shown.length - 1 ? <span className="caret" /> : null;
        if (b.kind === "p")
          return (
            <p key={i}>
              {b.lines.map((l, j) => (
                <Fragment key={j}>
                  {j > 0 && <br />}
                  {inline(l, `${i}-${j}`)}
                </Fragment>
              ))}
              {caret}
            </p>
          );
        const List = b.kind;
        return (
          <List key={i}>
            {b.lines.map((l, j) => (
              <li key={j}>
                {inline(l, `${i}-${j}`)}
                {j === b.lines.length - 1 && caret}
              </li>
            ))}
          </List>
        );
      })}
    </div>
  );
}
