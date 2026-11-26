"""Export the demo documents for the in-browser build (frontend/public/demo/).

The browser version doesn't re-chunk or re-embed the demo PDFs: it loads exactly what the
Python pipeline (and so the eval) produced. Run this whenever the demo PDFs, the chunkers or
the embedding model change:

    python deploy/build_static.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from documind.chunking import make_chunks  # noqa: E402
from documind.config import settings  # noqa: E402
from documind.pdf import parse_pdf  # noqa: E402
from documind.retrieval import embed  # noqa: E402

DEMO = ROOT / "deploy" / "demo"
DOCS = ROOT / "eval" / "docs"
OUT = ROOT / "frontend" / "public" / "demo"
STRATEGIES = ("fixed", "structure")


def main() -> None:
    cfg = json.loads((DEMO / "demo.json").read_text("utf-8"))
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    docs, chunks = [], {s: [] for s in STRATEGIES}
    vectors = {s: [] for s in STRATEGIES}
    for name, title in cfg["titles"].items():
        data = (DOCS / name).read_bytes()
        doc_id = hashlib.sha256(data).hexdigest()[:12]
        parsed = parse_pdf(data, title)
        pages = {p.number: p for p in parsed.pages}
        counts = {}
        for s in STRATEGIES:
            cs = make_chunks(parsed, doc_id, s)
            for c in cs:
                c.rects = pages[c.page].rects(c.start, c.end)
            chunks[s] += [
                {"id": c.id, "doc_id": doc_id, "page": c.page, "start": c.start, "end": c.end, "text": c.text, "section": c.section, "rects": c.rects}
                for c in cs
            ]
            vectors[s].append(embed([c.index_text for c in cs]))
            counts[s] = len(cs)
        shutil.copyfile(DOCS / name, OUT / f"{doc_id}.pdf")
        docs.append(
            {
                "id": doc_id,
                "title": title,
                "filename": name,
                "num_pages": len(parsed.pages),
                "num_chunks": counts,
                "page_sizes": {str(p.number): [p.width, p.height] for p in parsed.pages},
                "demo": True,
            }
        )
        print(f"{name}: {counts}")
    import numpy as np

    for s in STRATEGIES:
        v = np.concatenate(vectors[s]).astype("<f4")
        v.tofile(OUT / f"vectors_{s}.f32")
    index = {
        "embed_model": settings.embed_model,
        "dim": 384,
        "docs": docs,
        "chunks": chunks,
        "questions": cfg.get("questions", []),
    }
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), "utf-8")
    size = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"wrote {OUT.relative_to(ROOT)} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
