"""On-disk document store. Each PDF is parsed once, chunked with both strategies, and
embedded; the API searches a corpus index assembled from the per-document pieces.

data/docs/<doc_id>/
    original.pdf
    meta.json                 title, pages, page sizes
    chunks_<strategy>.json
    vectors_<strategy>.npy
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from .chunking import Chunk, make_chunks
from .config import settings
from .pdf import parse_pdf, render_page_png
from .retrieval import Index, embed

STRATEGIES = ("fixed", "structure")


class DocumentStore:
    def __init__(self, root: Path | None = None):
        self.root = (root or settings.data_dir) / "docs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._indexes: dict[str, Index] = {}

    # ---- documents ----
    def list(self, viewer: str | None = None) -> list[dict]:
        """All documents, or with `viewer` only the demo documents plus that viewer's uploads."""
        out = []
        for d in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime):
            m = d / "meta.json"
            if m.exists():
                meta = json.loads(m.read_text("utf-8"))
                if viewer is not None and not meta.get("demo") and viewer not in meta.get("owners", []):
                    continue
                out.append(public_meta(meta))
        return out

    def _write_meta(self, meta: dict) -> None:
        (self.root / meta["id"] / "meta.json").write_text(json.dumps(meta), "utf-8")

    def meta(self, doc_id: str) -> dict:
        p = self.root / _safe(doc_id) / "meta.json"
        if not p.exists():
            raise KeyError(doc_id)
        return json.loads(p.read_text("utf-8"))

    def pdf_bytes(self, doc_id: str) -> bytes:
        p = self.root / _safe(doc_id) / "original.pdf"
        if not p.exists():
            raise KeyError(doc_id)
        return p.read_bytes()

    def page_png(self, doc_id: str, page: int) -> bytes:
        cache = self.root / _safe(doc_id) / f"page_{int(page)}.png"
        if cache.exists():
            return cache.read_bytes()
        png = render_page_png(self.pdf_bytes(doc_id), page)
        cache.write_bytes(png)
        return png

    def add(self, data: bytes, filename: str, owner: str | None = None, demo: bool = False, title: str | None = None) -> dict:
        doc_id = hashlib.sha256(data).hexdigest()[:12]
        d = self.root / doc_id
        if (d / "meta.json").exists():  # same file again: just record who else has it
            with self._lock:
                meta = self.meta(doc_id)
                if owner and owner not in meta.setdefault("owners", []):
                    meta["owners"].append(owner)
                if demo and not meta.get("demo"):
                    meta["demo"] = True
                    meta["title"] = title or meta["title"]
                self._write_meta(meta)
            return meta
        title = title or Path(filename).stem
        parsed = parse_pdf(data, title)
        if not any(p.text.strip() for p in parsed.pages):
            raise ValueError("No extractable text in this PDF (is it a scan?)")
        tmp = self.root / f".{doc_id}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        (tmp / "original.pdf").write_bytes(data)
        pages = {p.number: p for p in parsed.pages}
        counts = {}
        for strat in STRATEGIES:
            chunks = make_chunks(parsed, doc_id, strat)
            for c in chunks:
                c.rects = pages[c.page].rects(c.start, c.end)
            vecs = embed([c.index_text for c in chunks])
            (tmp / f"chunks_{strat}.json").write_text(json.dumps([c.to_dict() for c in chunks]), "utf-8")
            np.save(tmp / f"vectors_{strat}.npy", vecs)
            counts[strat] = len(chunks)
        meta = {
            "id": doc_id,
            "title": title,
            "filename": filename,
            "num_pages": len(parsed.pages),
            "num_chunks": counts,
            "page_sizes": {p.number: [p.width, p.height] for p in parsed.pages},
            "created_at": time.time(),
            "demo": demo,
            "owners": [owner] if owner else [],
        }
        (tmp / "meta.json").write_text(json.dumps(meta), "utf-8")
        with self._lock:
            tmp.rename(d)
            self._indexes.clear()
        return meta

    def delete(self, doc_id: str, viewer: str | None = None) -> None:
        """Delete a document. With `viewer`, only that viewer's copy goes; the files go once no
        one else has uploaded the same PDF. Demo documents can't be deleted by viewers."""
        d = self.root / _safe(doc_id)
        if not d.exists():
            raise KeyError(doc_id)
        with self._lock:
            meta = self.meta(doc_id)
            if viewer is not None:
                if meta.get("demo"):
                    raise PermissionError(doc_id)
                if viewer not in meta.get("owners", []):
                    raise KeyError(doc_id)
                meta["owners"].remove(viewer)
                if meta["owners"]:
                    self._write_meta(meta)
                    return
            shutil.rmtree(d)
            self._indexes.clear()

    def load_demo(self, folder: Path) -> None:
        """Index every PDF in `folder` as a demo document (visible to everyone, not deletable).
        Titles and sample questions come from folder/demo.json when present."""
        cfg = self.demo_config(folder)
        for pdf in sorted(folder.glob("*.pdf")):
            self.add(pdf.read_bytes(), pdf.name, demo=True, title=cfg.get("titles", {}).get(pdf.name))

    @staticmethod
    def demo_config(folder: Path | None) -> dict:
        f = folder / "demo.json" if folder else None
        return json.loads(f.read_text("utf-8")) if f and f.exists() else {}

    # ---- search ----
    def index(self, strategy: str) -> Index:
        with self._lock:
            if strategy not in self._indexes:
                chunks: list[Chunk] = []
                vecs: list[np.ndarray] = []
                for doc in self.list():
                    d = self.root / doc["id"]
                    chunks += [Chunk(**c) for c in json.loads((d / f"chunks_{strategy}.json").read_text("utf-8"))]
                    vecs.append(np.load(d / f"vectors_{strategy}.npy"))
                v = np.concatenate(vecs) if vecs else None
                self._indexes[strategy] = Index.build(chunks, v) if chunks else Index.build([], None)
            return self._indexes[strategy]


def public_meta(meta: dict) -> dict:
    out = {k: meta[k] for k in ("id", "title", "filename", "num_pages", "num_chunks", "created_at")}
    out["demo"] = bool(meta.get("demo"))
    return out


def _safe(doc_id: str) -> str:
    if not doc_id.isalnum():
        raise KeyError(doc_id)
    return doc_id
