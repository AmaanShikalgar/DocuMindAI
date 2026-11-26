/** Dense + BM25 + reciprocal-rank fusion + cross-encoder rerank, mirroring
 * backend/documind/retrieval.py (same pools, same tie order, same defaults). */
import { BM25, tokenize } from "./bm25";
import { DEFAULT_DTYPES, Dtypes, embed, rerankScores } from "./models";

export type Chunk = {
  id: string;
  doc_id: string;
  page: number;
  start: number;
  end: number;
  text: string;
  section: string;
  rects: number[][];
};

export type Hit = {
  chunk: Chunk;
  score: number;
  rank: number;
  dense_rank: number | null;
  bm25_rank: number | null;
  rerank_score: number | null;
};

export type Timings = { retrieval_ms: number; rerank_ms: number };
export type Mode = "dense" | "bm25" | "hybrid";

export const indexText = (c: Chunk) => (c.section ? `${c.section}\n${c.text}` : c.text);

export class Index {
  private bm25: BM25;
  constructor(
    public chunks: Chunk[],
    public vectors: Float32Array, // chunks.length x dim, L2-normalised
    public dim = 384,
  ) {
    this.bm25 = new BM25(chunks.map((c) => tokenize(indexText(c)).concat()).map((t) => (t.length ? t : ["_"])));
  }

  /** Concatenate per-document indexes (vectors are kept, BM25 is rebuilt over the union). */
  static merge(parts: Index[]): Index {
    const chunks = parts.flatMap((p) => p.chunks);
    const vectors = new Float32Array(chunks.length * 384);
    let off = 0;
    for (const p of parts) {
      vectors.set(p.vectors, off);
      off += p.vectors.length;
    }
    return new Index(chunks, vectors);
  }

  private dense(q: Float32Array, n: number): [number, number][] {
    const scores: [number, number][] = [];
    for (let i = 0; i < this.chunks.length; i++) {
      let s = 0;
      const off = i * this.dim;
      for (let d = 0; d < this.dim; d++) s += q[d] * this.vectors[off + d];
      scores.push([i, s]);
    }
    scores.sort((a, b) => b[1] - a[1]);
    return scores.slice(0, n);
  }

  private sparse(query: string, n: number): [number, number][] {
    const toks = tokenize(query);
    if (!toks.length || !this.chunks.length) return [];
    const s = this.bm25.scores(toks);
    const order = Array.from(s.keys()).sort((a, b) => s[b] - s[a]);
    return order.slice(0, n).filter((i) => s[i] > 0).map((i) => [i, s[i]]);
  }

  async search(
    query: string,
    opts: { k?: number; mode?: Mode; rerank?: boolean; candidates?: number; allowed?: Set<string>; dtypes?: Dtypes; timings?: Timings } = {},
  ): Promise<Hit[]> {
    const { k = 5, mode = "hybrid", rerank = false, candidates = 20, allowed, dtypes = DEFAULT_DTYPES, timings } = opts;
    if (!this.chunks.length) return [];
    const t0 = performance.now();
    let pool = rerank || mode === "hybrid" ? candidates : k;
    if (allowed) pool = this.chunks.length;
    let dense: [number, number][] = [];
    let sparse: [number, number][] = [];
    if (mode !== "bm25") dense = this.dense((await embed([query], dtypes.embed))[0], pool);
    if (mode !== "dense") sparse = this.sparse(query, pool);
    if (allowed) {
      dense = dense.filter(([i]) => allowed.has(this.chunks[i].doc_id)).slice(0, candidates);
      sparse = sparse.filter(([i]) => allowed.has(this.chunks[i].doc_id)).slice(0, candidates);
    }
    const dRank = new Map(dense.map(([i], r) => [i, r + 1]));
    const sRank = new Map(sparse.map(([i], r) => [i, r + 1]));
    let scored: [number, number][];
    if (mode === "dense") scored = dense;
    else if (mode === "bm25") scored = sparse;
    else {
      const rrf = new Map<number, number>(); // insertion order = dense first, like the Python dict
      for (const ranks of [dRank, sRank]) for (const [i, r] of ranks) rrf.set(i, (rrf.get(i) ?? 0) + 1 / (60 + r));
      scored = [...rrf.entries()].sort((a, b) => b[1] - a[1]);
    }
    let hits: Hit[] = scored.slice(0, rerank ? candidates : k).map(([i, s]) => ({
      chunk: this.chunks[i],
      score: s,
      rank: 0,
      dense_rank: dRank.get(i) ?? null,
      bm25_rank: sRank.get(i) ?? null,
      rerank_score: null,
    }));
    if (timings) timings.retrieval_ms = performance.now() - t0;
    if (rerank && hits.length) {
      const t1 = performance.now();
      const scores = await rerankScores(query, hits.map((h) => indexText(h.chunk)), dtypes.rerank);
      hits.forEach((h, j) => (h.score = h.rerank_score = scores[j]));
      hits = hits.map((h, j) => [h, j] as const).sort((a, b) => b[0].score - a[0].score || a[1] - b[1]).map(([h]) => h);
      if (timings) timings.rerank_ms = performance.now() - t1;
    }
    hits = hits.slice(0, k);
    hits.forEach((h, r) => (h.rank = r + 1));
    return hits;
  }
}
