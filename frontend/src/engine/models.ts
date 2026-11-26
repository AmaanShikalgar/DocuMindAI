/** The same two models as the Python backend, run with transformers.js (ONNX Runtime in the
 * browser, or in Node for the parity check). Tokenisation mirrors sentence-transformers:
 * 256-token truncation, mean pooling and L2 normalisation for embeddings; raw logits for the
 * cross-encoder (ranking is the same as sentence-transformers' sigmoid scores). */
import { AutoModel, AutoModelForSequenceClassification, AutoTokenizer, env } from "@huggingface/transformers";

export const EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";
export const RERANK_MODEL = "Xenova/ms-marco-TinyBERT-L-2-v2";
const MAX_LEN = 256;

export type Dtype = "fp32" | "q8";
export type Dtypes = { embed: Dtype; rerank: Dtype };
/** Compressed embedder (22 MB) + full-precision reranker (17 MB): on the eval set the v2 default
 * config matches Python exactly (scripts/parity.ts). All-q8 loses a question; all-fp32 is 104 MB. */
export const DEFAULT_DTYPES: Dtypes = { embed: "q8", rerank: "fp32" };
export type Progress = (p: { file: string; loaded: number; total: number }) => void;

env.allowLocalModels = false;

type Loaded = { tok: any; model: any };
const cache = new Map<string, Promise<Loaded>>();

function load(kind: "embed" | "rerank", dtype: Dtype, onProgress?: Progress): Promise<Loaded> {
  const key = `${kind}:${dtype}`;
  if (!cache.has(key)) {
    const id = kind === "embed" ? EMBED_MODEL : RERANK_MODEL;
    const progress_callback = (e: any) => {
      if (e.status === "progress" && onProgress) onProgress({ file: `${kind}/${e.file}`, loaded: e.loaded, total: e.total });
    };
    const Model = kind === "embed" ? AutoModel : AutoModelForSequenceClassification;
    cache.set(
      key,
      Promise.all([AutoTokenizer.from_pretrained(id), Model.from_pretrained(id, { dtype, progress_callback })]).then(
        ([tok, model]) => ({ tok, model }),
      ),
    );
  }
  return cache.get(key)!;
}

export function preload(dtypes: Dtypes = DEFAULT_DTYPES, onProgress?: Progress) {
  return Promise.all([load("embed", dtypes.embed, onProgress), load("rerank", dtypes.rerank, onProgress)]);
}

/** L2-normalised mean-pooled sentence embeddings, one Float32Array per text. */
export async function embed(texts: string[], dtype: Dtype = "fp32", batch = 32): Promise<Float32Array[]> {
  const { tok, model } = await load("embed", dtype);
  const out: Float32Array[] = [];
  for (let s = 0; s < texts.length; s += batch) {
    const part = texts.slice(s, s + batch);
    const inputs = tok(part, { padding: true, truncation: true, max_length: MAX_LEN });
    const { last_hidden_state } = await model(inputs);
    const [n, len, dim] = last_hidden_state.dims as number[];
    const h = last_hidden_state.data as Float32Array;
    const mask = inputs.attention_mask.data as BigInt64Array;
    for (let i = 0; i < n; i++) {
      const v = new Float32Array(dim);
      let count = 0;
      for (let t = 0; t < len; t++) {
        if (!mask[i * len + t]) continue;
        count++;
        const off = (i * len + t) * dim;
        for (let d = 0; d < dim; d++) v[d] += h[off + d];
      }
      let norm = 0;
      for (let d = 0; d < dim; d++) {
        v[d] /= Math.max(count, 1);
        norm += v[d] * v[d];
      }
      norm = Math.sqrt(norm) || 1;
      for (let d = 0; d < dim; d++) v[d] /= norm;
      out.push(v);
    }
  }
  return out;
}

/** Cross-encoder relevance logits for (query, passage) pairs. */
export async function rerankScores(query: string, passages: string[], dtype: Dtype = "fp32", batch = 32): Promise<number[]> {
  const { tok, model } = await load("rerank", dtype);
  const out: number[] = [];
  for (let s = 0; s < passages.length; s += batch) {
    const part = passages.slice(s, s + batch);
    const inputs = tok(new Array(part.length).fill(query), { text_pair: part, padding: true, truncation: true, max_length: MAX_LEN });
    const { logits } = await model(inputs);
    out.push(...Array.from(logits.data as Float32Array));
  }
  return out;
}
