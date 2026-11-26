"""PDF parsing that keeps layout: every character offset in a page's text maps back
to a rectangle on the page, so any chunk can be highlighted exactly where it came from.

Page text is built as: spans joined within a line, lines joined by "\n",
blocks (paragraphs) joined by "\n\n".
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pymupdf as fitz


@dataclass
class Span:
    start: int
    end: int
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool


@dataclass
class Line:
    start: int
    end: int
    size: float
    bold: bool
    block: int


@dataclass
class Block:
    start: int
    end: int
    lines: list[Line]


@dataclass
class Page:
    number: int  # 1-based
    width: float
    height: float
    text: str
    spans: list[Span] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)

    def rects(self, start: int, end: int) -> list[list[float]]:
        """Rectangles (x0, y0, x1, y1 in PDF points) covering text[start:end].

        Partially covered spans are cut proportionally by character count, which is
        accurate enough for highlighting with proportional fonts.
        """
        out: list[list[float]] = []
        for s in self.spans:
            if s.end <= start or s.start >= end:
                continue
            x0, y0, x1, y1 = s.bbox
            n = max(1, s.end - s.start)
            a = (max(start, s.start) - s.start) / n
            b = (min(end, s.end) - s.start) / n
            r = [x0 + (x1 - x0) * a, y0, x0 + (x1 - x0) * b, y1]
            # merge with the previous rect when on the same line
            if out and abs(out[-1][1] - r[1]) < 1 and abs(out[-1][3] - r[3]) < 1 and r[0] - out[-1][2] < 8:
                out[-1][2] = max(out[-1][2], r[2])
            else:
                out.append(r)
        return [[round(v, 1) for v in r] for r in out]


@dataclass
class ParsedDoc:
    title: str
    pages: list[Page]

    @property
    def body_size(self) -> float:
        """Most common font size by character count: the body text size."""
        c: Counter[float] = Counter()
        for p in self.pages:
            for s in p.spans:
                c[round(s.size, 1)] += s.end - s.start
        return c.most_common(1)[0][0] if c else 10.0


_BOLD = ("bold", "medi", "semibold", "heavy", "black")  # "-Medi" is how Times/Nimbus names its bold


def _clean(t: str) -> str:
    return t.replace(" ", " ").replace("­", "").replace("\t", " ")


def parse_pdf(data: bytes, title: str = "") -> ParsedDoc:
    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[Page] = []
    for pno, pg in enumerate(doc):
        d = pg.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP)
        parts: list[str] = []
        pos = 0
        spans: list[Span] = []
        blocks: list[Block] = []
        for b in d["blocks"]:
            if b.get("type", 0) != 0:
                continue  # image block
            lines: list[Line] = []
            bstart = None
            for ln in b["lines"]:
                ltext_spans = [(sp, _clean(sp["text"])) for sp in ln["spans"] if sp["text"].strip()]
                if not ltext_spans:
                    continue
                if bstart is None:
                    if blocks:
                        parts.append("\n\n")
                        pos += 2
                    bstart = pos
                elif lines:
                    parts.append("\n")
                    pos += 1
                lstart = pos
                sizes, bolds = [], []
                for i, (sp, t) in enumerate(ltext_spans):
                    if i == 0:
                        t = t.lstrip()
                    if i == len(ltext_spans) - 1:
                        t = t.rstrip()
                    if not t:
                        continue
                    if i > 0 and parts and not parts[-1].endswith(" ") and not t.startswith(" "):
                        # adjacent spans with a visible gap between them
                        prev = spans[-1].bbox if spans else None
                        if prev and sp["bbox"][0] - prev[2] > sp["size"] * 0.2:
                            parts.append(" ")
                            pos += 1
                    bold = bool(sp["flags"] & 16) or any(w in sp.get("font", "").lower() for w in _BOLD)
                    spans.append(Span(pos, pos + len(t), tuple(sp["bbox"]), sp["size"], bold))
                    # Line style comes from its letters only: bullets and symbols are often set
                    # in a bigger or different font and would make a list item look like a heading.
                    n = sum(ch.isalnum() for ch in t)
                    if n:
                        sizes.append((sp["size"], n))
                        bolds.append(bold)
                    parts.append(t)
                    pos += len(t)
                if pos > lstart:
                    tot = sum(n for _, n in sizes) or 1
                    size = sum(s * n for s, n in sizes) / tot if sizes else 0.0
                    lines.append(Line(lstart, pos, size, bool(bolds) and all(bolds), len(blocks)))
            if lines and bstart is not None:
                blocks.append(Block(bstart, pos, lines))
        rect = pg.rect
        pages.append(Page(pno + 1, rect.width, rect.height, "".join(parts), spans, blocks))
    meta_title = (doc.metadata or {}).get("title") or ""
    doc.close()
    return ParsedDoc(title=title or meta_title or "document", pages=pages)


def render_page_png(data: bytes, page_number: int, zoom: float = 1.5) -> bytes:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        pix = doc[page_number - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()
