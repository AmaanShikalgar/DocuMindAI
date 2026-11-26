/** Upload-path check: parse the eval PDFs with pdf.js, chunk with the TypeScript port, embed
 * in-process, and score the eval questions. Evidence quotes are located in pdf.js's own page text
 * with the same normalisation as eval/run_eval.py.
 *
 *   npx tsx scripts/parity-upload.ts
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as pdfjs from "pdfjs-dist/legacy/build/pdf.mjs";
import { structureChunks, fixedChunks } from "../src/engine/chunking";
import { DEFAULT_DTYPES, embed } from "../src/engine/models";
import { parsePdf, ParsedDoc } from "../src/engine/pdf";
import { Chunk, Index, indexText, Mode } from "../src/engine/retrieval";

const root = resolve(import.meta.dirname, "..", "..");
const docDtype = (process.argv[2] ?? DEFAULT_DTYPES.embed) as "fp32" | "q8";
const strategies = (process.argv[3] ?? "structure,fixed").split(",");
const questions = readFileSync(resolve(root, "eval/questions.jsonl"), "utf-8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l));
const python = JSON.parse(readFileSync(resolve(root, "eval/results/latest.json"), "utf-8")).retrieval;

/** Same as run_eval._norm_map + locate: lowercase, collapse whitespace, drop "-\n" breaks. */
function normMap(s: string): [string, number[]] {
  const out: string[] = [];
  const idx: number[] = [];
  let prevSpace = false;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === "-" && s[i + 1] === "\n") {
      i++;
      continue;
    }
    if (/\s/.test(ch)) {
      if (!prevSpace && out.length) {
        out.push(" ");
        idx.push(i);
      }
      prevSpace = true;
    } else {
      out.push(ch.toLowerCase());
      idx.push(i);
      prevSpace = false;
    }
  }
  return [out.join(""), idx];
}
const quotes = (s: string) => s.replace(/[“”]/g, '"').replace(/[‘’]/g, "'");
function locate(text: string, quote: string): [number, number] | null {
  const [n, idx] = normMap(text);
  const q = quotes(normMap(quote.trim())[0]);
  const j = quotes(n).indexOf(q);
  return j < 0 || !q ? null : [idx[j], idx[j + q.length - 1] + 1];
}

const files = ["nist-csf-2.0.pdf", "irs-p463-2025.pdf", "lost-in-the-middle.pdf"];
const parsed = new Map<string, ParsedDoc>();
const chunks: Record<string, Chunk[]> = { fixed: [], structure: [] };
for (const f of files) {
  const t0 = performance.now();
  const doc = await parsePdf(pdfjs, new Uint8Array(readFileSync(resolve(root, "eval/docs", f))), f);
  parsed.set(f, doc);
  const s = structureChunks(doc, f);
  chunks.structure.push(...s);
  chunks.fixed.push(...fixedChunks(doc, f));
  const sections = new Set(s.map((c) => c.section)).size;
  console.log(`${f}: ${doc.pages.length} pages, ${s.length} structure chunks, ${sections} sections, parsed in ${Math.round(performance.now() - t0)} ms`);
}

const indexes: Record<string, Index> = {};
for (const strat of strategies) {
  const t0 = performance.now();
  const vecs = await embed(chunks[strat].map(indexText), docDtype);
  const flat = new Float32Array(vecs.length * 384);
  vecs.forEach((v, i) => flat.set(v, i * 384));
  indexes[strat] = new Index(chunks[strat], flat);
  console.log(`embedded ${vecs.length} ${strat} chunks in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
}

let missing = 0;
const answerable = questions.filter((q) => q.evidence?.length);
const spans = answerable.map((q) =>
  q.evidence
    .map((e: { page: number; quote: string }) => {
      const page = parsed.get(q.doc)!.pages[e.page - 1];
      const loc = page ? locate(page.text, e.quote) : null;
      if (!loc) missing++;
      return loc && { doc_id: q.doc, page: e.page, start: loc[0], end: loc[1] };
    })
    .filter(Boolean),
);
console.log(`evidence quotes not found verbatim in pdf.js text: ${missing}`);

console.log("\nconfig".padEnd(35) + "hit@5 upload / python-parsed");
for (const [c, m, r] of [
  ["fixed", "dense", false],
  ["structure", "dense", false],
  ["structure", "bm25", false],
  ["structure", "bm25", true],
  ["structure", "hybrid", true],
] as [string, Mode, boolean][]) {
  if (!indexes[c]) continue;
  let hits = 0;
  for (let i = 0; i < answerable.length; i++) {
    const res = await indexes[c].search(answerable[i].question, { k: 5, mode: m, rerank: r, dtypes: { embed: docDtype, rerank: "fp32" } });
    if (res.some((h) => spans[i].some((s: any) => h.chunk.doc_id === s.doc_id && h.chunk.page === s.page && Math.min(s.end, h.chunk.end) - Math.max(s.start, h.chunk.start) >= 0.5 * (s.end - s.start))))
      hits++;
  }
  const name = `${c} + ${m}${r ? " + rerank" : ""}`;
  console.log(name.padEnd(34) + `${(hits / answerable.length).toFixed(3)} / ${python[name]["hit@5"].toFixed(3)}`);
}
