"""FastAPI app: documents, streaming Q&A with citations, query log, and the built front end."""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import pipeline
from .config import settings
from .llm import LLM
from .obs import QueryLog
from .store import STRATEGIES, DocumentStore

MODES = ("dense", "bm25", "hybrid")


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    doc_ids: list[str] | None = None
    chunking: str = settings.chunking
    retrieval: str = settings.retrieval
    rerank: bool = settings.rerank
    top_k: int = Field(settings.top_k, ge=1, le=10)


class RateLimiter:
    """Per-client sliding window, so a public demo can't burn through the API key."""

    def __init__(self, limit: int, window_s: int):
        self.limit, self.window = limit, window_s
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.time()
        with self.lock:
            q = self.hits[key]
            while q and q[0] < now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


def _client_key(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?")


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(store: DocumentStore | None = None, log: QueryLog | None = None, llm: LLM | None = None) -> FastAPI:
    import os

    from contextlib import asynccontextmanager

    from . import retrieval

    @asynccontextmanager
    async def lifespan(_app):
        # Load the embedding + reranker models in the background at startup; otherwise the
        # first user's question pays ~20 s of model loading.
        if os.getenv("WARMUP", "true").lower() == "true":
            threading.Thread(
                target=lambda: (retrieval.embed(["warm up"]), retrieval.reranker(settings.rerank_model).predict([("a", "b")])),
                daemon=True,
            ).start()
        yield

    app = FastAPI(title="DocuMind API", version="2.0", lifespan=lifespan)
    app.state.store = store or DocumentStore()
    app.state.log = log or QueryLog()
    app.state.llm = llm or LLM()
    query_limit = RateLimiter(int(os.getenv("RATE_LIMIT_QUERIES", "30")), 600)
    upload_limit = RateLimiter(int(os.getenv("RATE_LIMIT_UPLOADS", "10")), 3600)

    @app.get("/api/health")
    def health():
        return {"ok": True, "llm_configured": bool(settings.groq_api_key) or llm is not None}

    @app.get("/api/config")
    def config():
        return {
            "model": app.state.llm.model,
            "defaults": {
                "chunking": settings.chunking,
                "retrieval": settings.retrieval,
                "rerank": settings.rerank,
                "top_k": settings.top_k,
            },
            "options": {"chunking": list(STRATEGIES), "retrieval": list(MODES)},
            "max_upload_mb": settings.max_upload_mb,
        }

    # ---- documents ----
    @app.get("/api/documents")
    def list_documents():
        return app.state.store.list()

    @app.post("/api/documents", status_code=201)
    async def upload(request: Request, file: UploadFile = File(...)):
        if not upload_limit.allow(_client_key(request)):
            raise HTTPException(429, "Upload limit reached, try again later")
        data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"PDF larger than {settings.max_upload_mb} MB")
        if not data.startswith(b"%PDF"):
            raise HTTPException(415, "Not a PDF file")
        try:
            from starlette.concurrency import run_in_threadpool

            meta = await run_in_threadpool(app.state.store.add, data, file.filename or "document.pdf")
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {k: meta[k] for k in ("id", "title", "filename", "num_pages", "num_chunks", "created_at")}

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str):
        try:
            app.state.store.delete(doc_id)
        except KeyError:
            raise HTTPException(404, "Document not found") from None
        return Response(status_code=204)

    @app.get("/api/documents/{doc_id}")
    def document(doc_id: str):
        try:
            return app.state.store.meta(doc_id)
        except KeyError:
            raise HTTPException(404, "Document not found") from None

    @app.get("/api/documents/{doc_id}/file")
    def document_file(doc_id: str):
        try:
            data = app.state.store.pdf_bytes(doc_id)
        except KeyError:
            raise HTTPException(404, "Document not found") from None
        return Response(data, media_type="application/pdf", headers={"Content-Disposition": "inline"})

    @app.get("/api/documents/{doc_id}/pages/{page}.png")
    def page_image(doc_id: str, page: int):
        try:
            meta = app.state.store.meta(doc_id)
            if not 1 <= page <= meta["num_pages"]:
                raise KeyError(page)
            png = app.state.store.page_png(doc_id, page)
        except KeyError:
            raise HTTPException(404, "Page not found") from None
        return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

    # ---- Q&A ----
    @app.post("/api/query")
    def query(q: QueryIn, request: Request):
        if q.chunking not in STRATEGIES or q.retrieval not in MODES:
            raise HTTPException(422, "Unknown chunking or retrieval option")
        if not query_limit.allow(_client_key(request)):
            raise HTTPException(429, "Too many questions, try again in a few minutes")
        store: DocumentStore = app.state.store
        docs = store.list()
        if not docs:
            raise HTTPException(400, "Upload a PDF first")
        titles = {d["id"]: d["title"] for d in docs}
        allowed = set(q.doc_ids) & titles.keys() if q.doc_ids else None
        index = store.index(q.chunking)

        def events():
            res = pipeline.Result()
            error = None
            try:
                for kind, payload in pipeline.run(
                    q.question,
                    index,
                    app.state.llm,
                    mode=q.retrieval,
                    rerank=q.rerank,
                    k=q.top_k,
                    allowed=allowed,
                    titles=titles,
                    res=res,
                ):
                    if kind == "sources":
                        meta = {d: store.meta(d) for d in {h.chunk.doc_id for h in payload}}
                        yield _sse(
                            "sources",
                            [
                                {
                                    **h.to_dict(),
                                    "n": h.rank,
                                    "title": titles.get(h.chunk.doc_id),
                                    "page_size": meta[h.chunk.doc_id]["page_sizes"][str(h.chunk.page)],
                                }
                                for h in payload
                            ],
                        )
                    else:
                        yield _sse("token", payload)
            except Exception as e:  # surface the failure to the client and the log
                error = f"{type(e).__name__}: {e}"
                yield _sse("error", "The model call failed." if "GROQ" not in str(e) else str(e))
            cost = settings.cost_usd(res.usage.prompt_tokens, res.usage.completion_tokens)
            qid = app.state.log.log(
                question=q.question,
                answer=res.answer,
                doc_ids=sorted(allowed) if allowed else [],
                chunking=q.chunking,
                retrieval=q.retrieval,
                rerank=int(q.rerank),
                top_k=q.top_k,
                model=app.state.llm.model,
                retrieved=[
                    {
                        "n": h.rank,
                        "id": h.chunk.id,
                        "doc_id": h.chunk.doc_id,
                        "page": h.chunk.page,
                        "section": h.chunk.section,
                        "score": round(h.score, 4),
                        "dense_rank": h.dense_rank,
                        "bm25_rank": h.bm25_rank,
                        "preview": h.chunk.text[:240],
                    }
                    for h in res.hits
                ],
                retrieval_ms=round(res.timings.retrieval_ms, 1),
                rerank_ms=round(res.timings.rerank_ms, 1),
                ttft_ms=None if res.ttft_ms is None else round(res.ttft_ms, 1),
                total_ms=round(res.total_ms, 1),
                prompt_tokens=res.usage.prompt_tokens,
                completion_tokens=res.usage.completion_tokens,
                cost_usd=cost,
                not_found=int(res.not_found),
                error=error,
            )
            if error is None:
                yield _sse(
                    "done",
                    {
                        "query_id": qid,
                        "cited": res.cited,
                        "not_found": res.not_found,
                        "retrieval_ms": round(res.timings.retrieval_ms, 1),
                        "rerank_ms": round(res.timings.rerank_ms, 1),
                        "ttft_ms": None if res.ttft_ms is None else round(res.ttft_ms, 1),
                        "total_ms": round(res.total_ms, 1),
                        "prompt_tokens": res.usage.prompt_tokens,
                        "completion_tokens": res.usage.completion_tokens,
                        "cost_usd": round(cost, 6),
                    },
                )

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    # ---- observability ----
    @app.get("/api/metrics/summary")
    def metrics(days: int = 30):
        return app.state.log.summary(since=time.time() - days * 86400)

    @app.get("/api/queries")
    def queries(limit: int = 50, offset: int = 0):
        return app.state.log.recent(min(limit, 200), offset)

    # ---- front end (built React app) ----
    static = Path(os.getenv("STATIC_DIR", Path(__file__).resolve().parents[1] / "static")).resolve()
    if (static / "index.html").exists():
        if (static / "assets").exists():
            app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "Not found")
            f = (static / path).resolve()
            if path and f.is_file() and static in f.resolve().parents:
                return FileResponse(f)
            return FileResponse(static / "index.html")

    return app


app = create_app()
