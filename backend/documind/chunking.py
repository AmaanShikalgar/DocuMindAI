"""Two chunking strategies over the same parsed pages.

fixed:     v1 behaviour. 500-char windows with 100-char overlap, per page, blind to layout.
structure: paragraphs grouped under the nearest heading, never crossing a heading,
           ~900 chars target; oversized paragraphs are split on sentence boundaries.
           The heading path is prepended to the text that gets embedded/indexed.

Each chunk keeps (page, start, end) offsets into Page.text so it can be cited exactly.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .pdf import Line, Page, ParsedDoc


@dataclass
class Chunk:
    id: str
    doc_id: str
    page: int
    start: int
    end: int
    text: str  # the passage as it appears on the page (shown to users / the LLM)
    section: str = ""
    # highlight rectangles on the page, PDF points (x0, y0, x1, y1); filled by the store
    rects: list[list[float]] = field(default_factory=list)

    @property
    def index_text(self) -> str:
        """What retrieval sees: section heading gives the chunk its context."""
        return f"{self.section}\n{self.text}" if self.section else self.text

    def to_dict(self) -> dict:
        return asdict(self)


def fixed_chunks(doc: ParsedDoc, doc_id: str, size: int = 500, overlap: int = 100) -> list[Chunk]:
    out: list[Chunk] = []
    step = size - overlap
    for p in doc.pages:
        t = p.text
        if not t.strip():
            continue
        for i in range(0, len(t), step):
            seg = t[i : i + size]
            if seg.strip():
                out.append(Chunk(f"{doc_id}:f{len(out)}", doc_id, p.number, i, i + len(seg), seg))
            if i + size >= len(t):
                break
    return out


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_NUMBERED = re.compile(r"^(\d{1,2}(\.\d{1,2})*\.?|[A-Z]\.|Chapter\s+\d+|Section\s+\d+|Part\s+[IVX\d]+)\s+\S", re.I)
_SECTION_NO = re.compile(r"^(\d{1,2}(\.\d{1,2})*\.?|[A-Z]\.?|[IVX]+\.)$")  # "2.1" on its own line
_BULLETS = tuple("•●▪◦‣-–—*·")


def _is_heading(line: Line, text: str, body: float, block_lines: int) -> bool:
    s = text.strip()
    if not s or len(s) > 120 or s.startswith(_BULLETS) or s[0].islower():
        return False
    if s.endswith((",", ";")) or s.endswith(".") and not _NUMBERED.match(s):
        return False
    if len(s) < 3 or sum(ch.isalpha() for ch in s) < 2:
        return False
    if line.size >= body * 1.08:  # papers often set headings just 1pt above body text
        return True
    if line.bold and len(s) < 90:
        return True
    if block_lines == 1 and _NUMBERED.match(s) and len(s) < 80:
        return True
    return False


def _norm_line(s: str) -> str:
    return re.sub(r"\d+", "#", s.strip().lower())


def _running_lines(doc: ParsedDoc) -> set[str]:
    """Headers/footers: short lines that repeat on many pages (page numbers normalised away)."""
    if len(doc.pages) < 4:
        return set()
    seen: dict[str, int] = {}
    for p in doc.pages:
        for t in {_norm_line(p.text[ln.start : ln.end]) for b in p.blocks for ln in b.lines}:
            if len(t) < 80:
                seen[t] = seen.get(t, 0) + 1
    return {t for t, n in seen.items() if n >= max(3, 0.3 * len(doc.pages))}


def _heading_lines(page: Page, lines: list[Line], body: float, running: set[str]) -> int:
    """How many leading lines of a block form its heading (0 if none)."""
    n = 0
    while n < len(lines):
        ln = lines[n]
        t = page.text[ln.start : ln.end].strip()
        # (a bare "1" normalises like a page number, so check section numbers first)
        if not _SECTION_NO.match(t) and _norm_line(t) in running:
            break
        if _SECTION_NO.match(t) and n + 1 < len(lines) and (ln.size >= body or ln.bold):
            nxt = lines[n + 1]
            if _is_heading(nxt, page.text[nxt.start : nxt.end], body, len(lines)):
                n += 2
                continue
            break
        if not _is_heading(ln, t, body, len(lines)):
            break
        n += 1
    return n


def _split_long(text: str, base: int, limit: int) -> list[tuple[int, int]]:
    """Split text (starting at offset base) into pieces <= limit on sentence boundaries."""
    bounds = [0] + [m.end() for m in _SENT.finditer(text)] + [len(text)]
    pieces: list[tuple[int, int]] = []
    cur = last = 0
    for b in bounds[1:]:
        if b - cur > limit and last > cur:  # close the piece at the previous sentence end
            pieces.append((cur, last))
            cur = last
        while b - cur > limit * 1.5:  # one sentence that is still far too long
            cut = text.rfind(" ", cur + 1, cur + limit)
            cut = cut if cut > cur else cur + limit
            pieces.append((cur, cut))
            cur = cut
        last = b
    if cur < len(text):
        pieces.append((cur, len(text)))
    return [(base + a, base + b) for a, b in pieces if text[a:b].strip()]


def structure_chunks(doc: ParsedDoc, doc_id: str, target: int = 900, limit: int = 1300) -> list[Chunk]:
    body = doc.body_size
    running = _running_lines(doc)
    out: list[Chunk] = []
    section = ""

    def emit(page: Page, a: int, b: int) -> None:
        # trim whitespace so offsets point at real text
        seg = page.text[a:b]
        a += len(seg) - len(seg.lstrip())
        b -= len(seg) - len(seg.rstrip())
        if b > a:
            out.append(Chunk(f"{doc_id}:s{len(out)}", doc_id, page.number, a, b, page.text[a:b], section))

    for page in doc.pages:
        cur: tuple[int, int] | None = None  # (start, end) of the chunk being built

        def flush() -> None:
            nonlocal cur
            if cur:
                emit(page, *cur)
            cur = None

        for blk in page.blocks:
            lines = blk.lines
            # headings can open a block ("1.2 Scope" followed by its paragraph)
            body_start = blk.start
            if _norm_line(page.text[blk.start : blk.end]) in running:
                continue  # running header/footer: not content
            nh = _heading_lines(page, lines, body, running)
            if nh:
                flush()
                section = " ".join(page.text[ln.start : ln.end].strip() for ln in lines[:nh])[:150]
                body_start = lines[nh - 1].end
            para = (body_start, blk.end)
            if not page.text[para[0] : para[1]].strip():
                continue
            plen = para[1] - para[0]
            if plen > limit:
                flush()
                for a, b in _split_long(page.text[para[0] : para[1]], para[0], target):
                    emit(page, a, b)
                continue
            if cur and (para[1] - cur[0]) > target:
                flush()
            cur = (cur[0], para[1]) if cur else para
        flush()
    return out


def make_chunks(doc: ParsedDoc, doc_id: str, strategy: str) -> list[Chunk]:
    if strategy == "fixed":
        return fixed_chunks(doc, doc_id)
    if strategy == "structure":
        return structure_chunks(doc, doc_id)
    raise ValueError(f"unknown chunking strategy: {strategy}")
