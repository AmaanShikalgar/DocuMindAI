/** Port of backend/documind/chunking.py: fixed windows and structure-aware chunks, with the same
 * heading rules, running header/footer detection and sentence splitting. */
import { bodySize, Line, Page, ParsedDoc, rects } from "./pdf";
import type { Chunk } from "./retrieval";

export function fixedChunks(doc: ParsedDoc, docId: string, size = 500, overlap = 100): Chunk[] {
  const out: Chunk[] = [];
  const step = size - overlap;
  for (const p of doc.pages) {
    const t = p.text;
    if (!t.trim()) continue;
    for (let i = 0; i < t.length; i += step) {
      const seg = t.slice(i, i + size);
      if (seg.trim()) out.push(mk(p, docId, `f${out.length}`, i, i + seg.length, ""));
      if (i + size >= t.length) break;
    }
  }
  return out;
}

const SENT = /(?<=[.!?])\s+(?=[A-Z0-9"'(\[])/g;
const NUMBERED = /^(\d{1,2}(\.\d{1,2})*\.?|[A-Z]\.|Chapter\s+\d+|Section\s+\d+|Part\s+[IVX\d]+)\s+\S/i;
const SECTION_NO = /^(\d{1,2}(\.\d{1,2})*\.?|[A-Z]\.?|[IVX]+\.)$/;
const BULLETS = [..."•●▪◦‣-–—*·"];

function isHeading(line: Line, text: string, body: number, blockLines: number): boolean {
  const s = text.trim();
  if (!s || s.length > 120 || BULLETS.some((b) => s.startsWith(b)) || (s[0] !== s[0].toUpperCase() && s[0] === s[0].toLowerCase() && /\p{Ll}/u.test(s[0])))
    return false;
  if (s.endsWith(",") || s.endsWith(";") || (s.endsWith(".") && !NUMBERED.test(s))) return false;
  if (s.length < 3 || [...s].filter((ch) => /\p{L}/u.test(ch)).length < 2) return false;
  if (line.size >= body * 1.08) return true;
  if (line.bold && s.length < 90) return true;
  if (blockLines === 1 && NUMBERED.test(s) && s.length < 80) return true;
  return false;
}

const normLine = (s: string) => s.trim().toLowerCase().replace(/\d+/g, "#");

function runningLines(doc: ParsedDoc): Set<string> {
  if (doc.pages.length < 4) return new Set();
  const seen = new Map<string, number>();
  for (const p of doc.pages) {
    const uniq = new Set(p.blocks.flatMap((b) => b.lines.map((ln) => normLine(p.text.slice(ln.start, ln.end)))));
    for (const t of uniq) if (t.length < 80) seen.set(t, (seen.get(t) ?? 0) + 1);
  }
  const min = Math.max(3, 0.3 * doc.pages.length);
  return new Set([...seen].filter(([, n]) => n >= min).map(([t]) => t));
}

function headingLines(page: Page, lines: Line[], body: number, running: Set<string>): number {
  let n = 0;
  while (n < lines.length) {
    const ln = lines[n];
    const t = page.text.slice(ln.start, ln.end).trim();
    if (!SECTION_NO.test(t) && running.has(normLine(t))) break;
    if (SECTION_NO.test(t) && n + 1 < lines.length && (ln.size >= body || ln.bold)) {
      const nxt = lines[n + 1];
      if (isHeading(nxt, page.text.slice(nxt.start, nxt.end), body, lines.length)) {
        n += 2;
        continue;
      }
      break;
    }
    if (!isHeading(ln, t, body, lines.length)) break;
    n++;
  }
  return n;
}

function splitLong(text: string, base: number, limit: number): [number, number][] {
  const bounds = [0, ...[...text.matchAll(SENT)].map((m) => m.index! + m[0].length), text.length];
  const pieces: [number, number][] = [];
  let cur = 0;
  let last = 0;
  for (const b of bounds.slice(1)) {
    if (b - cur > limit && last > cur) {
      pieces.push([cur, last]);
      cur = last;
    }
    while (b - cur > limit * 1.5) {
      let cut = text.lastIndexOf(" ", cur + limit - 1);
      cut = cut > cur ? cut : cur + limit;
      pieces.push([cur, cut]);
      cur = cut;
    }
    last = b;
  }
  if (cur < text.length) pieces.push([cur, text.length]);
  return pieces.filter(([a, b]) => text.slice(a, b).trim()).map(([a, b]) => [base + a, base + b]);
}

export function structureChunks(doc: ParsedDoc, docId: string, target = 900, limit = 1300): Chunk[] {
  const body = bodySize(doc);
  const running = runningLines(doc);
  const out: Chunk[] = [];
  let section = "";
  const emit = (page: Page, a: number, b: number) => {
    const seg = page.text.slice(a, b);
    a += seg.length - seg.trimStart().length;
    b -= seg.length - seg.trimEnd().length;
    if (b > a) out.push(mk(page, docId, `s${out.length}`, a, b, section));
  };
  for (const page of doc.pages) {
    let cur: [number, number] | null = null;
    const flush = () => {
      if (cur) emit(page, cur[0], cur[1]);
      cur = null;
    };
    for (const blk of page.blocks) {
      if (running.has(normLine(page.text.slice(blk.start, blk.end)))) continue;
      let bodyStart = blk.start;
      const nh = headingLines(page, blk.lines, body, running);
      if (nh) {
        flush();
        section = blk.lines
          .slice(0, nh)
          .map((ln) => page.text.slice(ln.start, ln.end).trim())
          .join(" ")
          .slice(0, 150);
        bodyStart = blk.lines[nh - 1].end;
      }
      const para: [number, number] = [bodyStart, blk.end];
      if (!page.text.slice(para[0], para[1]).trim()) continue;
      if (para[1] - para[0] > limit) {
        flush();
        for (const [a, b] of splitLong(page.text.slice(para[0], para[1]), para[0], target)) emit(page, a, b);
        continue;
      }
      if (cur && para[1] - (cur as [number, number])[0] > target) flush();
      cur = cur ? [(cur as [number, number])[0], para[1]] : para;
    }
    flush();
  }
  return out;
}

function mk(page: Page, docId: string, id: string, start: number, end: number, section: string): Chunk {
  return { id: `${docId}:${id}`, doc_id: docId, page: page.number, start, end, text: page.text.slice(start, end), section, rects: rects(page, start, end) };
}
