"""DocuMind eval: retrieval hit-rate for every chunking x retrieval x rerank config, and
answer accuracy (LLM-judged) for the v1 baseline and the v2 default.

    python eval/run_eval.py --check              # validate questions.jsonl against the PDFs
    python eval/run_eval.py --retrieval-only     # no API key needed
    python eval/run_eval.py --update-readme      # full run, writes results + README table

Dataset (eval/questions.jsonl), one object per line:
    {"id": "q01", "doc": "file.pdf", "question": "...", "answer": "...",
     "evidence": [{"page": 3, "quote": "exact sentence copied from the PDF"}]}
Unanswerable questions have "answerable": false and no evidence; the right answer is to abstain.

A retrieval "hit" means a top-k chunk overlaps at least half of an evidence quote's span
on the right page. Span overlap (not chunk ids) keeps the metric fair across chunkers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "backend"))

from documind import pipeline  # noqa: E402
from documind.chunking import Chunk, make_chunks  # noqa: E402
from documind.config import settings  # noqa: E402
from documind.llm import LLM, NOT_FOUND, Usage  # noqa: E402
from documind.pdf import ParsedDoc, parse_pdf  # noqa: E402
from documind.retrieval import Index, embed  # noqa: E402

BASELINE = ("fixed", "dense", False)  # v1: 500-char windows, FAISS only
V2 = ("structure", "hybrid", True)
ALL_CONFIGS = [(c, r, rr) for c in ("fixed", "structure") for r in ("dense", "bm25", "hybrid") for rr in (False, True)]


def label(cfg) -> str:
    c, r, rr = cfg
    return f"{c} + {r}" + (" + rerank" if rr else "")


# ---------- dataset ----------
def load_questions(path: Path) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip() and not line.startswith("//")]
    for q in qs:
        q.setdefault("answerable", bool(q.get("evidence")))
    return qs


def _norm_map(s: str) -> tuple[str, list[int]]:
    """Lowercased, whitespace-collapsed, de-hyphenated text plus a map back to original offsets."""
    out, idx = [], []
    prev_space = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "-" and i + 1 < len(s) and s[i + 1] == "\n":  # hyphenated line break
            i += 2
            continue
        if ch.isspace():
            if not prev_space and out:
                out.append(" ")
                idx.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            idx.append(i)
            prev_space = False
        i += 1
    return "".join(out), idx


def locate(page_text: str, quote: str) -> tuple[int, int] | None:
    norm, idx = _norm_map(page_text)
    q, _ = _norm_map(quote.strip())
    q = re.sub(r"[‘’]", "'", re.sub(r"[“”]", '"', q))
    norm2 = re.sub(r"[‘’]", "'", re.sub(r"[“”]", '"', norm))
    j = norm2.find(q)
    if j < 0 or not q:
        return None
    return idx[j], idx[j + len(q) - 1] + 1


@dataclass
class Corpus:
    docs: dict[str, ParsedDoc]  # doc_id -> parsed
    names: dict[str, str]  # filename -> doc_id
    indexes: dict[str, Index]  # strategy -> index


def build_corpus(doc_dir: Path, cache: Path) -> Corpus:
    cache.mkdir(parents=True, exist_ok=True)
    docs, names = {}, {}
    chunks: dict[str, list[Chunk]] = {"fixed": [], "structure": []}
    vecs: dict[str, list[np.ndarray]] = {"fixed": [], "structure": []}
    for pdf in sorted(doc_dir.glob("*.pdf")):
        data = pdf.read_bytes()
        doc_id = hashlib.sha256(data).hexdigest()[:12]
        parsed = parse_pdf(data, pdf.stem)
        docs[doc_id], names[pdf.name] = parsed, doc_id
        for strat in chunks:
            cs = make_chunks(parsed, doc_id, strat)
            key = cache / f"{doc_id}_{strat}_{hashlib.md5(settings.embed_model.encode()).hexdigest()[:6]}.npy"
            if key.exists() and np.load(key).shape[0] == len(cs):
                v = np.load(key)
            else:
                v = embed([c.index_text for c in cs])
                np.save(key, v)
            chunks[strat] += cs
            vecs[strat].append(v)
    if not docs:
        sys.exit(f"No PDFs in {doc_dir}")
    indexes = {s: Index.build(chunks[s], np.concatenate(vecs[s])) for s in chunks}
    return Corpus(docs, names, indexes)


def evidence_spans(q: dict, corpus: Corpus) -> list[tuple[str, int, tuple[int, int] | None, str]]:
    doc_id = corpus.names.get(q["doc"])
    out = []
    for ev in q.get("evidence", []):
        page = corpus.docs[doc_id].pages[ev["page"] - 1] if doc_id and 0 < ev["page"] <= len(corpus.docs[doc_id].pages) else None
        out.append((doc_id, ev["page"], locate(page.text, ev["quote"]) if page else None, ev["quote"]))
    return out


def is_hit(chunk: Chunk, spans) -> bool:
    for doc_id, page, span, quote in spans:
        if chunk.doc_id != doc_id or chunk.page != page:
            continue
        if span:
            a, b = span
            overlap = min(b, chunk.end) - max(a, chunk.start)
            if overlap >= 0.5 * (b - a):
                return True
        else:  # quote not found verbatim: fall back to token coverage
            from documind.retrieval import tokenize

            qt = set(tokenize(quote))
            if qt and len(qt & set(tokenize(chunk.text))) / len(qt) >= 0.6:
                return True
    return False


# ---------- runs ----------
def check(qs, corpus) -> int:
    bad = 0
    for q in qs:
        if q["doc"] not in corpus.names:
            print(f"{q['id']}: unknown doc {q['doc']!r}")
            bad += 1
            continue
        for _, page, span, quote in evidence_spans(q, corpus):
            if span is None:
                print(f"{q['id']}: quote not found verbatim on page {page}: {quote[:70]!r}")
                bad += 1
    n_ans = sum(q["answerable"] for q in qs)
    print(f"{len(qs)} questions ({n_ans} answerable, {len(qs) - n_ans} unanswerable) over {len(corpus.docs)} PDFs; {bad} problems")
    return bad


def eval_retrieval(qs, corpus, configs, k: int) -> dict:
    out = {}
    answerable = [q for q in qs if q["answerable"]]
    for idx in corpus.indexes.values():  # load models before timing anything
        idx.search("warm up", k=1, mode="hybrid", rerank=True)
    for cfg in configs:
        strat, mode, rr = cfg
        hits, rr_sum, lat = 0, 0.0, []
        for q in answerable:
            spans = evidence_spans(q, corpus)
            t = time.perf_counter()
            res = corpus.indexes[strat].search(q["question"], k=k, mode=mode, rerank=rr)
            lat.append((time.perf_counter() - t) * 1000)
            ranks = [h.rank for h in res if is_hit(h.chunk, spans)]
            hits += bool(ranks)
            rr_sum += 1 / ranks[0] if ranks else 0
        n = len(answerable)
        out[label(cfg)] = {
            "chunking": strat,
            "retrieval": mode,
            "rerank": rr,
            f"hit@{k}": round(hits / n, 3),
            f"mrr@{k}": round(rr_sum / n, 3),
            "retrieval_ms_p50": round(statistics.median(lat), 1),
        }
        print(f"  {label(cfg):32s} hit@{k}={hits / n:.3f}  mrr@{k}={rr_sum / n:.3f}  p50={statistics.median(lat):.0f}ms")
    return out


JUDGE = """You grade answers from a document Q&A system against a reference answer.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Grade the system answer:
- "correct": states the key facts of the reference; extra correct detail is fine.
- "partial": some key facts right, some missing or wrong.
- "incorrect": wrong, contradicts the reference, or says it cannot find the answer.
Ignore citation markers like [1]. Reply with JSON only: {{"verdict": "correct|partial|incorrect", "reason": "<10 words>"}}"""


def _retry(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # rate limits on the free tier
            if i == tries - 1 or "rate" not in str(e).lower() and "429" not in str(e):
                raise
            time.sleep(min(60, 2 ** (i + 1)))


def judge(llm: LLM, q: dict, answer: str) -> str:
    if not q["answerable"]:
        return "correct" if NOT_FOUND.lower().rstrip(".") in answer.lower() else "incorrect"
    raw = _retry(
        lambda: llm.complete(
            [{"role": "user", "content": JUDGE.format(question=q["question"], reference=q["answer"], answer=answer)}],
            response_format={"type": "json_object"},
        )
    )
    try:
        v = json.loads(raw)["verdict"]
    except Exception:
        m = re.search(r"correct|partial|incorrect", raw)
        v = m.group(0) if m else "incorrect"
    return v if v in ("correct", "partial", "incorrect") else "incorrect"


def eval_answers(qs, corpus, configs, k: int) -> tuple[dict, list[dict]]:
    llm = LLM()
    titles = {i: d.title for i, d in corpus.docs.items()}
    summary, rows = {}, []
    for cfg in configs:
        strat, mode, rr = cfg
        verdicts, lat, ttft, cost, cite_ok, cite_n = [], [], [], 0.0, 0, 0
        for q in qs:
            res = pipeline.Result()

            def go():
                for _ in pipeline.run(q["question"], corpus.indexes[strat], llm, mode=mode, rerank=rr, k=k, titles=titles, res=res):
                    pass

            _retry(go)
            v = judge(llm, q, res.answer)
            verdicts.append(v)
            lat.append(res.total_ms)
            ttft.append(res.ttft_ms or res.total_ms)
            cost += settings.cost_usd(res.usage.prompt_tokens, res.usage.completion_tokens)
            if q["answerable"] and res.cited:
                spans = evidence_spans(q, corpus)
                cite_n += 1
                cite_ok += any(is_hit(res.hits[n - 1].chunk, spans) for n in res.cited)
            rows.append(
                {"config": label(cfg), "id": q["id"], "question": q["question"], "expected": q["answer"] if q["answerable"] else NOT_FOUND,
                 "answer": res.answer, "verdict": v, "cited_pages": [res.hits[n - 1].chunk.page for n in res.cited], "total_ms": round(res.total_ms)}
            )
            print(f"  [{label(cfg)}] {q['id']}: {v}")
        ans = [v for v, q in zip(verdicts, qs) if q["answerable"]]
        una = [v for v, q in zip(verdicts, qs) if not q["answerable"]]
        summary[label(cfg)] = {
            "accuracy": round(sum(v == "correct" for v in ans) / len(ans), 3) if ans else None,
            "accuracy_with_partial": round(sum({"correct": 1, "partial": 0.5}.get(v, 0) for v in ans) / len(ans), 3) if ans else None,
            "abstention_accuracy": round(sum(v == "correct" for v in una) / len(una), 3) if una else None,
            "citation_precision": round(cite_ok / cite_n, 3) if cite_n else None,
            "latency_ms_p50": round(statistics.median(lat)),
            "latency_ms_p95": round(sorted(lat)[max(0, int(0.95 * len(lat)) - 1)]),
            "ttft_ms_p50": round(statistics.median(ttft)),
            "cost_usd_per_query": round(cost / len(qs), 6),
        }
    return summary, rows


# ---------- report ----------
def markdown(meta, retr, ans, k) -> str:
    lines = [
        f"_{meta['n_questions']} questions ({meta['n_answerable']} answerable, {meta['n_unanswerable']} unanswerable) "
        f"over {meta['n_docs']} PDFs · embeddings `{meta['embed_model']}` · reranker `{meta['rerank_model']}` · "
        f"LLM `{meta['llm_model']}` · run {meta['date']}_",
        "",
        "**Retrieval** (answerable questions; hit = a top-k chunk covers the evidence passage)",
        "",
        f"| Config | Hit@{k} | MRR@{k} | Retrieval p50 |",
        "|---|---|---|---|",
    ]
    best = max(r[f"hit@{k}"] for r in retr.values())
    for name, r in retr.items():
        tag = " (v1 baseline)" if name == label(BASELINE) else " (v2 default)" if name == label(V2) else ""
        h = f"**{r[f'hit@{k}']:.0%}**" if r[f"hit@{k}"] == best else f"{r[f'hit@{k}']:.0%}"
        lines.append(f"| {name}{tag} | {h} | {r[f'mrr@{k}']:.2f} | {r['retrieval_ms_p50']:.0f} ms |")
    if ans:
        lines += [
            "",
            "**Answers** (LLM-judged against the reference answers)",
            "",
            "| Config | Accuracy | Abstains when it should | Citation precision | Latency p50 / p95 | First token p50 | Cost / query |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, a in ans.items():
            fmt = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
            lines.append(
                f"| {name} | {fmt(a['accuracy'])} | {fmt(a['abstention_accuracy'])} | {fmt(a['citation_precision'])} | "
                f"{a['latency_ms_p50'] / 1000:.1f} s / {a['latency_ms_p95'] / 1000:.1f} s | {a['ttft_ms_p50'] / 1000:.1f} s | "
                f"${a['cost_usd_per_query']:.5f} |"
            )
    return "\n".join(lines)


def update_readme(md: str) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text("utf-8")
    start, end = "<!-- EVAL:START -->", "<!-- EVAL:END -->"
    if start not in text:
        print("README has no EVAL markers; skipped")
        return
    pre, rest = text.split(start, 1)
    _, post = rest.split(end, 1)
    readme.write_text(f"{pre}{start}\n{md}\n{end}{post}", "utf-8")
    print("README updated")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    ap.add_argument("--docs", type=Path, default=HERE / "docs")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--check", action="store_true", help="only validate the dataset")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--all-answer-configs", action="store_true", help="judge answers for every config (slow, costs more)")
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args()

    qs = load_questions(args.questions)
    print("Indexing PDFs...")
    corpus = build_corpus(args.docs, HERE / ".cache")
    if check(qs, corpus) and args.check:
        sys.exit(1)
    if args.check:
        return

    print("Retrieval:")
    retr = eval_retrieval(qs, corpus, ALL_CONFIGS, args.k)
    ans, rows = {}, []
    if not args.retrieval_only:
        print("Answers:")
        ans, rows = eval_answers(qs, corpus, ALL_CONFIGS if args.all_answer_configs else [BASELINE, V2], args.k)

    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "n_questions": len(qs),
        "n_answerable": sum(q["answerable"] for q in qs),
        "n_unanswerable": sum(not q["answerable"] for q in qs),
        "n_docs": len(corpus.docs),
        "embed_model": settings.embed_model.split("/")[-1],
        "rerank_model": settings.rerank_model.split("/")[-1],
        "llm_model": settings.llm_model,
        "k": args.k,
    }
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "latest.json").write_text(json.dumps({"meta": meta, "retrieval": retr, "answers": ans}, indent=2), "utf-8")
    if rows:
        (out / "answers.jsonl").write_text("\n".join(json.dumps(r) for r in rows), "utf-8")
    md = markdown(meta, retr, ans, args.k)
    (out / "latest.md").write_text(md + "\n", "utf-8")
    print("\n" + md)
    if args.update_readme:
        update_readme(md)


if __name__ == "__main__":
    main()
