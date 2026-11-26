"""Retrieval over a set of chunks: dense (FAISS), sparse (BM25), hybrid (RRF), + reranker."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .chunking import Chunk
from .config import settings

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")
_STOP = frozenset(
    "a an and are as at be but by for from has have in is it its of on or that the this to was were "
    "will with what which who whom how when where why does do did can could should would i you we they".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


_lock = threading.Lock()


@lru_cache(maxsize=2)
def embedder(name: str | None = None):
    from sentence_transformers import SentenceTransformer

    with _lock:
        return SentenceTransformer(name or settings.embed_model, device="cpu")


@lru_cache(maxsize=2)
def reranker(name: str | None = None):
    from sentence_transformers import CrossEncoder

    with _lock:
        return CrossEncoder(name or settings.rerank_model, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    v = embedder(settings.embed_model).encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype="float32")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return {
            **self.chunk.to_dict(),
            "score": round(float(self.score), 4),
            "rank": self.rank,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rerank_score": None if self.rerank_score is None else round(float(self.rerank_score), 4),
        }


@dataclass
class Timings:
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0


@dataclass
class Index:
    """Dense + sparse indexes over one list of chunks (one document or a whole corpus)."""

    chunks: list[Chunk]
    vectors: np.ndarray | None = None
    _faiss: object = field(default=None, repr=False)
    _bm25: object = field(default=None, repr=False)

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: np.ndarray | None = None) -> "Index":
        if vectors is None:
            vectors = embed([c.index_text for c in chunks]) if chunks else np.zeros((0, 384), "float32")
        idx = cls(chunks, vectors)
        idx._init()
        return idx

    def _init(self) -> None:
        import faiss
        from rank_bm25 import BM25Okapi

        self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
        if len(self.chunks):
            self._faiss.add(self.vectors)
        self._bm25 = BM25Okapi([tokenize(c.index_text) or ["_"] for c in self.chunks]) if self.chunks else None

    # -- individual retrievers return [(chunk_idx, score)] best first --
    def dense(self, query: str, n: int) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        q = embed([query])
        scores, ids = self._faiss.search(q, min(n, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def bm25(self, query: str, n: int) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        toks = tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        order = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        rerank: bool = False,
        candidates: int = 30,
        allowed: set[str] | None = None,
        timings: Timings | None = None,
    ) -> list[Hit]:
        t0 = time.perf_counter()
        pool = candidates if (rerank or mode == "hybrid") else k
        if allowed is not None:
            pool = len(self.chunks)  # filter after scoring; corpora here are small
        dense = self.dense(query, pool) if mode in ("dense", "hybrid") else []
        sparse = self.bm25(query, pool) if mode in ("bm25", "hybrid") else []
        if allowed is not None:
            dense = [x for x in dense if self.chunks[x[0]].doc_id in allowed][:candidates]
            sparse = [x for x in sparse if self.chunks[x[0]].doc_id in allowed][:candidates]
        d_rank = {i: r for r, (i, _) in enumerate(dense, 1)}
        s_rank = {i: r for r, (i, _) in enumerate(sparse, 1)}

        if mode == "dense":
            scored = dense
        elif mode == "bm25":
            scored = sparse
        elif mode == "hybrid":
            # Reciprocal Rank Fusion: robust to the two retrievers' incomparable score scales.
            rrf: dict[int, float] = {}
            for ranks in (d_rank, s_rank):
                for i, r in ranks.items():
                    rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + r)
            scored = sorted(rrf.items(), key=lambda x: -x[1])
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        hits = [
            Hit(self.chunks[i], s, 0, d_rank.get(i), s_rank.get(i)) for i, s in scored[: (candidates if rerank else k)]
        ]
        if timings:
            timings.retrieval_ms = (time.perf_counter() - t0) * 1000

        if rerank and hits:
            t1 = time.perf_counter()
            scores = reranker(settings.rerank_model).predict(
                [(query, h.chunk.index_text) for h in hits], batch_size=32, show_progress_bar=False
            )
            for h, s in zip(hits, scores):
                h.rerank_score = float(s)
                h.score = float(s)
            hits.sort(key=lambda h: -h.score)
            if timings:
                timings.rerank_ms = (time.perf_counter() - t1) * 1000

        hits = hits[:k]
        for r, h in enumerate(hits, 1):
            h.rank = r
        return hits
