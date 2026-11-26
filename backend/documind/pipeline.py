"""retrieve -> prompt -> stream, shared by the API and the eval harness."""
from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .llm import LLM, NOT_FOUND, Usage, build_messages
from .retrieval import Hit, Index, Timings

_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Result:
    answer: str = ""
    hits: list[Hit] = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)
    usage: Usage = field(default_factory=Usage)
    ttft_ms: float | None = None
    total_ms: float = 0.0

    @property
    def not_found(self) -> bool:
        return NOT_FOUND.lower().rstrip(".") in self.answer.lower()

    @property
    def cited(self) -> list[int]:
        """1-based source numbers the answer actually cites, in order of first use."""
        seen: list[int] = []
        for m in _CITE.finditer(self.answer):
            n = int(m.group(1))
            if 1 <= n <= len(self.hits) and n not in seen:
                seen.append(n)
        return seen


def run(
    question: str,
    index: Index,
    llm: LLM,
    *,
    mode: str = "hybrid",
    rerank: bool = True,
    k: int = 5,
    allowed: set[str] | None = None,
    titles: dict[str, str] | None = None,
    res: Result | None = None,
) -> Iterator[tuple[str, object]]:
    """Yields ("sources", hits) once, then ("token", text) deltas. `res` is filled in as it goes."""
    res = res or Result()
    t0 = time.perf_counter()
    res.hits = index.search(question, k=k, mode=mode, rerank=rerank, allowed=allowed, timings=res.timings)
    yield "sources", res.hits
    if not res.hits:
        res.answer = NOT_FOUND
        yield "token", NOT_FOUND
    else:
        parts: list[str] = []
        for delta in llm.stream(build_messages(question, res.hits, titles), res.usage):
            if res.ttft_ms is None:
                res.ttft_ms = (time.perf_counter() - t0) * 1000
            parts.append(delta)
            yield "token", delta
        res.answer = "".join(parts).strip()
    res.total_ms = (time.perf_counter() - t0) * 1000
