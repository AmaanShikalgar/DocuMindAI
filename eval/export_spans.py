"""Write each eval question's evidence spans (doc id, page, character range) to JSON, so the
in-browser engine can be scored with exactly the same hit rule as run_eval.py.

    python eval/export_spans.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
sys.path.insert(0, str(HERE))

import run_eval as E  # noqa: E402

qs = E.load_questions(HERE / "questions.jsonl")
corpus = E.build_corpus(HERE / "docs", HERE / ".cache")
out = []
for q in qs:
    spans = [
        {"doc_id": d, "page": p, "start": s[0], "end": s[1]} for d, p, s, _ in E.evidence_spans(q, corpus) if s
    ]
    out.append({"id": q["id"], "question": q["question"], "answerable": q["answerable"], "spans": spans})
(HERE / "results" / "evidence_spans.json").write_text(json.dumps(out, indent=1), "utf-8")
print(f"wrote {len(out)} questions")
