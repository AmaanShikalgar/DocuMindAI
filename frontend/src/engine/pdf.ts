/** In-browser PDF parsing with pdf.js, producing the same structure as backend/documind/pdf.py:
 * page text built as spans joined within a line, lines joined by "\n", blocks by "\n\n", with
 * every span's rectangle kept so any character range can be highlighted.
 *
 * pdf.js reports text runs, not paragraphs, so lines and blocks are rebuilt from geometry:
 * a new line when the baseline moves, a new block when the vertical gap, the column or the font
 * size changes. pdfjs is passed in so the same code runs in the browser and in Node. */

export type Span = { start: number; end: number; bbox: [number, number, number, number]; size: number; bold: boolean };
export type Line = { start: number; end: number; size: number; bold: boolean };
export type Block = { start: number; end: number; lines: Line[] };
export type Page = { number: number; width: number; height: number; text: string; spans: Span[]; blocks: Block[] };
export type ParsedDoc = { title: string; pages: Page[] };

const BOLD = ["bold", "medi", "semibold", "heavy", "black"];

type Run = { str: string; x0: number; x1: number; top: number; bottom: number; base: number; size: number; bold: boolean };

export async function parsePdf(pdfjs: any, data: ArrayBuffer | Uint8Array, title: string): Promise<ParsedDoc> {
  const doc = await pdfjs.getDocument({ data: data instanceof Uint8Array ? data : new Uint8Array(data) }).promise;
  const pages: Page[] = [];
  for (let p = 1; p <= doc.numPages; p++) {
    const page = await doc.getPage(p);
    const vp = page.getViewport({ scale: 1 });
    await page.getOperatorList(); // loads fonts, so their real names (…-Bold) are known
    const tc = await page.getTextContent();
    const runs: Run[] = [];
    for (const it of tc.items as any[]) {
      if (!("str" in it) || !it.str.trim()) continue;
      const m = pdfjs.Util.transform(vp.transform, it.transform);
      const size = Math.hypot(m[2], m[3]);
      if (!size) continue;
      const style = tc.styles[it.fontName] ?? {};
      const ascent = style.ascent ?? 0.8;
      const descent = Math.abs(style.descent ?? -0.2);
      let fontName = "";
      try {
        fontName = page.commonObjs.get(it.fontName)?.name ?? "";
      } catch {
        /* font not resolved */
      }
      const lower = fontName.toLowerCase();
      runs.push({
        str: it.str.replace(/ /g, " ").replace(/­/g, "").replace(/\t/g, " "),
        x0: m[4],
        x1: m[4] + it.width * (vp.scale ?? 1),
        top: m[5] - size * ascent,
        bottom: m[5] + size * descent,
        base: m[5],
        size,
        bold: BOLD.some((b) => lower.includes(b)),
      });
    }
    pages.push(buildPage(p, vp.width, vp.height, runs));
    page.cleanup();
  }
  await doc.destroy();
  return { title, pages };
}

function buildPage(number: number, width: number, height: number, runs: Run[]): Page {
  // 1. group runs into lines: same baseline (within a third of the font size), moving right
  const lines: Run[][] = [];
  for (const r of runs) {
    const cur = lines[lines.length - 1];
    const last = cur?.[cur.length - 1];
    if (last && Math.abs(r.base - last.base) < 0.35 * Math.max(r.size, last.size) && r.x0 >= last.x0 - 1) cur.push(r);
    else lines.push([r]);
  }
  // 2. group lines into blocks: break on a large vertical gap, a column jump, or a size change
  const blocks: Run[][][] = [];
  const lineSize = (l: Run[]) => weighted(l);
  for (const l of lines) {
    const cur = blocks[blocks.length - 1];
    const prev = cur?.[cur.length - 1];
    if (prev) {
      const ps = lineSize(prev);
      const ls = lineSize(l);
      const gap = l[0].top - prev[0].bottom;
      const sameColumn = Math.abs(l[0].x0 - prev[0].x0) < width * 0.25 && l[0].top >= prev[0].top;
      const sameSize = Math.abs(ls - ps) < 0.5;
      if (sameColumn && sameSize && gap < 0.6 * ls) {
        cur.push(l);
        continue;
      }
    }
    blocks.push([l]);
  }
  // 3. build text with offsets, exactly like pdf.py
  const parts: string[] = [];
  let pos = 0;
  const spans: Span[] = [];
  const outBlocks: Block[] = [];
  for (const b of blocks) {
    if (outBlocks.length) {
      parts.push("\n\n");
      pos += 2;
    }
    const bstart = pos;
    const outLines: Line[] = [];
    b.forEach((l, li) => {
      if (li) {
        parts.push("\n");
        pos += 1;
      }
      const lstart = pos;
      const sizes: [number, number][] = [];
      const bolds: boolean[] = [];
      l.forEach((r, i) => {
        let t = r.str;
        if (i === 0) t = t.trimStart();
        if (i === l.length - 1) t = t.trimEnd();
        if (!t) return;
        if (i > 0 && parts.length && !parts[parts.length - 1].endsWith(" ") && !t.startsWith(" ")) {
          const prev = spans[spans.length - 1];
          if (prev && r.x0 - prev.bbox[2] > r.size * 0.2) {
            parts.push(" ");
            pos += 1;
          }
        }
        spans.push({ start: pos, end: pos + t.length, bbox: [r.x0, r.top, r.x1, r.bottom], size: r.size, bold: r.bold });
        const n = [...t].filter((ch) => /[\p{L}\p{N}]/u.test(ch)).length;
        if (n) {
          sizes.push([r.size, n]);
          bolds.push(r.bold);
        }
        parts.push(t);
        pos += t.length;
      });
      const tot = sizes.reduce((a, [, n]) => a + n, 0) || 1;
      const size = sizes.length ? sizes.reduce((a, [s, n]) => a + s * n, 0) / tot : 0;
      outLines.push({ start: lstart, end: pos, size, bold: bolds.length > 0 && bolds.every(Boolean) });
    });
    outBlocks.push({ start: bstart, end: pos, lines: outLines });
  }
  return { number, width, height, text: parts.join(""), spans, blocks: outBlocks };
}

function weighted(l: Run[]): number {
  let n = 0;
  let s = 0;
  for (const r of l) {
    const c = r.str.replace(/[^\p{L}\p{N}]/gu, "").length;
    n += c;
    s += r.size * c;
  }
  return n ? s / n : l[0].size;
}

/** Rectangles covering text[start:end] on a page (same proportional cut as pdf.py). */
export function rects(page: Page, start: number, end: number): number[][] {
  const out: number[][] = [];
  for (const s of page.spans) {
    if (s.end <= start || s.start >= end) continue;
    const [x0, y0, x1, y1] = s.bbox;
    const n = Math.max(1, s.end - s.start);
    const a = (Math.max(start, s.start) - s.start) / n;
    const b = (Math.min(end, s.end) - s.start) / n;
    const r = [x0 + (x1 - x0) * a, y0, x0 + (x1 - x0) * b, y1];
    const last = out[out.length - 1];
    if (last && Math.abs(last[1] - r[1]) < 1 && Math.abs(last[3] - r[3]) < 1 && r[0] - last[2] < 8) last[2] = Math.max(last[2], r[2]);
    else out.push(r);
  }
  return out.map((r) => r.map((v) => Math.round(v * 10) / 10));
}

/** Most common font size by character count: the body text size. */
export function bodySize(doc: ParsedDoc): number {
  const c = new Map<number, number>();
  for (const p of doc.pages) for (const s of p.spans) c.set(Math.round(s.size * 10) / 10, (c.get(Math.round(s.size * 10) / 10) ?? 0) + s.end - s.start);
  let best = 10;
  let n = -1;
  for (const [k, v] of c) if (v > n) [best, n] = [k, v];
  return best;
}
