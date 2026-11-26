/** Tokenizer and BM25 ported line for line from backend/documind/retrieval.py and rank_bm25's
 * BM25Okapi (k1 = 1.5, b = 0.75, epsilon = 0.25), so keyword scores match the Python eval. */

const TOKEN = /[a-z0-9]+(?:[.'-][a-z0-9]+)*/g;
const STOP = new Set(
  (
    "a an and are as at be but by for from has have in is it its of on or that the this to was were " +
    "will with what which who whom how when where why does do did can could should would i you we they"
  ).split(" "),
);

export function tokenize(text: string): string[] {
  return (text.toLowerCase().match(TOKEN) ?? []).filter((t) => !STOP.has(t));
}

export class BM25 {
  private docFreqs: Map<string, number>[] = [];
  private docLen: number[] = [];
  private idf = new Map<string, number>();
  private avgdl = 0;
  constructor(
    corpus: string[][],
    private k1 = 1.5,
    private b = 0.75,
    epsilon = 0.25,
  ) {
    const nd = new Map<string, number>();
    let total = 0;
    for (const doc of corpus) {
      this.docLen.push(doc.length);
      total += doc.length;
      const f = new Map<string, number>();
      for (const w of doc) f.set(w, (f.get(w) ?? 0) + 1);
      this.docFreqs.push(f);
      for (const w of f.keys()) nd.set(w, (nd.get(w) ?? 0) + 1);
    }
    const n = corpus.length;
    this.avgdl = n ? total / n : 0;
    // idf exactly as rank_bm25: negative idfs are replaced by epsilon * average idf
    let idfSum = 0;
    const negative: string[] = [];
    for (const [w, freq] of nd) {
      const v = Math.log(n - freq + 0.5) - Math.log(freq + 0.5);
      this.idf.set(w, v);
      idfSum += v;
      if (v < 0) negative.push(w);
    }
    const eps = epsilon * (nd.size ? idfSum / nd.size : 0);
    for (const w of negative) this.idf.set(w, eps);
  }

  scores(query: string[]): Float64Array {
    const out = new Float64Array(this.docLen.length);
    for (const q of query) {
      const idf = this.idf.get(q) ?? 0;
      if (!idf) continue;
      for (let i = 0; i < out.length; i++) {
        const tf = this.docFreqs[i].get(q) ?? 0;
        if (!tf) continue;
        out[i] += (idf * (tf * (this.k1 + 1))) / (tf + this.k1 * (1 - this.b + (this.b * this.docLen[i]) / this.avgdl));
      }
    }
    return out;
  }
}
