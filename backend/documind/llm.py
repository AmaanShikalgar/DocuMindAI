"""Prompting and streaming generation via Groq, with token usage captured for cost logging."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .config import settings
from .retrieval import Hit

NOT_FOUND = "I cannot find this in the document."

SYSTEM = f"""You answer questions using only the numbered sources provided.

Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- Cite every claim with the source number in square brackets, e.g. [2] or [1][3].
- If the sources do not contain the answer, reply exactly: {NOT_FOUND}
- Be concise: 1-3 sentences unless the question asks for a list or steps.
- Do not mention "the sources" or "the context" in the answer."""


def build_messages(question: str, hits: list[Hit], titles: dict[str, str] | None = None) -> list[dict]:
    titles = titles or {}
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        where = f"{titles.get(c.doc_id, c.doc_id)}, page {c.page}" + (f", section: {c.section}" if c.section else "")
        blocks.append(f"[{i}] ({where})\n{c.text.strip()}")
    user = "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM:
    """Thin streaming wrapper. Subclass and override `stream` for tests/offline use."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.llm_model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set")
            from groq import Groq

            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    def _extra(self) -> dict:
        if "gpt-oss" in self.model and settings.reasoning_effort:
            return {"reasoning_effort": settings.reasoning_effort}
        return {}

    def stream(self, messages: list[dict], usage: Usage) -> Iterator[str]:
        """Yield answer text deltas; fill `usage` when the stream ends."""
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.1, stream=True, **self._extra()
        )
        for chunk in resp:
            u = getattr(chunk, "usage", None) or getattr(getattr(chunk, "x_groq", None), "usage", None)
            if u:
                usage.prompt_tokens = u.prompt_tokens or 0
                usage.completion_tokens = u.completion_tokens or 0
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    def complete(self, messages: list[dict], usage: Usage | None = None, **kw) -> str:
        usage = usage or Usage()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=kw.pop("temperature", 0.0), **self._extra(), **kw
        )
        if resp.usage:
            usage.prompt_tokens = resp.usage.prompt_tokens or 0
            usage.completion_tokens = resp.usage.completion_tokens or 0
        return (resp.choices[0].message.content or "").strip()
