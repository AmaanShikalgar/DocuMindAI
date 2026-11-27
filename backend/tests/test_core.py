from documind.chunking import fixed_chunks, structure_chunks
from documind.pdf import parse_pdf
from documind.retrieval import Index


def test_parse_keeps_offsets_and_rects(pdf_bytes):
    doc = parse_pdf(pdf_bytes, "policy")
    assert len(doc.pages) == 2
    text = doc.pages[0].text
    i = text.index("Refunds are issued")
    rects = doc.pages[0].rects(i, i + len("Refunds are issued"))
    assert rects and all(r[2] > r[0] and r[3] > r[1] for r in rects)


def test_structure_chunks_follow_headings(pdf_bytes):
    doc = parse_pdf(pdf_bytes, "policy")
    chunks = structure_chunks(doc, "d1")
    sections = [c.section for c in chunks]
    assert "1. Refund Policy" in sections and "2. Warranty" in sections and "3. Contact" in sections
    refund = next(c for c in chunks if c.section == "1. Refund Policy")
    assert "14 days" in refund.text and "Warranty" not in refund.text
    # offsets point at the exact passage on the page
    page = doc.pages[refund.page - 1]
    assert page.text[refund.start : refund.end] == refund.text


def test_fixed_chunks_overlap(pdf_bytes):
    doc = parse_pdf(pdf_bytes, "policy")
    chunks = fixed_chunks(doc, "d1", size=60, overlap=20)
    p1 = [c for c in chunks if c.page == 1]
    assert p1[1].start == 40
    assert all(doc.pages[c.page - 1].text[c.start : c.end] == c.text for c in chunks)


def test_hybrid_and_rerank(pdf_bytes):
    doc = parse_pdf(pdf_bytes, "policy")
    idx = Index.build(structure_chunks(doc, "d1"))
    for mode in ("dense", "bm25", "hybrid"):
        hits = idx.search("how long do refunds take", k=2, mode=mode)
        assert hits and hits[0].chunk.section == "1. Refund Policy", mode
    hits = idx.search("warranty defects", k=1, mode="hybrid", rerank=True)
    assert hits[0].chunk.section == "2. Warranty" and hits[0].rerank_score is not None
    assert idx.search("warranty", k=3, allowed={"other"}) == []


def test_normalize_citations():
    from documind.llm import normalize_citations

    assert normalize_citations("Tier 4)【1†L1-L4】【3†L4-L7】") == "Tier 4)[1][3]"
    assert normalize_citations("gain [1, 2] and [4†L3] and 【5】") == "gain [1][2] and [4] and [5]"
