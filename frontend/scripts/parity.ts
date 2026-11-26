/** Parity check: score the in-browser engine (run here in Node) on the eval set with the same
 * hit rule as eval/run_eval.py, and compare with the Python results.
 *
 *   python eval/export_spans.py && python deploy/build_static.py   (once)
 *   npx tsx scripts/parity.ts [fp32 | q8 | <embed>:<rerank>, e.g. q8:fp32]
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Chunk, Index, Mode } from "../src/engine/retrieval";
import type { Dtype, Dtypes } from "../src/engine/models";

const root = resolve(import.meta.dirname, "..", "..");
const arg = process.argv[2] ?? "fp32";
// "fp32", "q8", or "embed:rerank" like "q8:fp32"
const [e, r] = (arg.includes(":") ? arg.split(":") : [arg, arg]) as Dtype[];
const dtypes: Dtypes = { embed: e, rerank: r };
const demo = JSON.parse(readFileSync(resolve(root, "frontend/public/demo/index.json"), "utf-8"));
const questions: { id: string; question: string; answerable: boolean; spans: { doc_id: string; page: number; start: number; end: number }[] }[] =
  JSON.parse(readFileSync(resolve(root, "eval/results/evidence_spans.json"), "utf-8"));
const python = JSON.parse(readFileSync(resolve(root, "eval/results/latest.json"), "utf-8")).retrieval;

function loadIndex(strategy: string): Index {
  const buf = readFileSync(resolve(root, `frontend/public/demo/vectors_${strategy}.f32`));
  const vectors = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  return new Index(demo.chunks[strategy] as Chunk[], vectors);
}

const isHit = (c: Chunk, spans: (typeof questions)[number]["spans"]) =>
  spans.some((s) => c.doc_id === s.doc_id && c.page === s.page && Math.min(s.end, c.end) - Math.max(s.start, c.start) >= 0.5 * (s.end - s.start));

const answerable = questions.filter((q) => q.answerable);
const configs: [string, Mode, boolean][] = [];
for (const c of ["fixed", "structure"]) for (const m of ["dense", "bm25", "hybrid"] as Mode[]) for (const r of [false, true]) configs.push([c, m, r]);

const indexes = { fixed: loadIndex("fixed"), structure: loadIndex("structure") } as Record<string, Index>;
await indexes.structure.search("warm up", { mode: "hybrid", rerank: true, dtypes });

console.log(`embed=${e} rerank=${r}  (python numbers from eval/results/latest.json)\n`);
console.log("config".padEnd(34) + "hit@5 js / py    mrr@5 js / py");
let maxDiff = 0;
for (const [c, m, rr] of configs) {
  let hits = 0;
  let mrrSum = 0;
  for (const q of answerable) {
    const res = await indexes[c].search(q.question, { k: 5, mode: m, rerank: rr, dtypes });
    const ranks = res.filter((h) => isHit(h.chunk, q.spans)).map((h) => h.rank);
    if (ranks.length) {
      hits++;
      mrrSum += 1 / ranks[0];
    }
  }
  const name = `${c} + ${m}${rr ? " + rerank" : ""}`;
  const py = python[name];
  const h = hits / answerable.length;
  const mrr = mrrSum / answerable.length;
  maxDiff = Math.max(maxDiff, Math.abs(h - py["hit@5"]));
  console.log(name.padEnd(34) + `${h.toFixed(3)} / ${py["hit@5"].toFixed(3)}    ${mrr.toFixed(3)} / ${py["mrr@5"].toFixed(3)}`);
}
console.log(`\nlargest hit@5 difference: ${(maxDiff * 100).toFixed(1)} points (one question = ${(100 / answerable.length).toFixed(1)})`);
