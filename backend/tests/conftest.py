import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DOCUMIND_DATA_DIR", str(Path(__file__).parent / ".data"))
os.environ["WARMUP"] = "false"

import pymupdf as fitz  # noqa: E402

from documind import retrieval, store  # noqa: E402
from documind.llm import LLM  # noqa: E402

DIM = 384


def fake_embed(texts):
    """Deterministic bag-of-words hashing embedder, so tests don't download models."""
    out = np.zeros((len(texts), DIM), "float32")
    for i, t in enumerate(texts):
        for tok in retrieval.tokenize(t):
            out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1, n)


class FakeReranker:
    def predict(self, pairs, **_):
        return [len(set(retrieval.tokenize(q)) & set(retrieval.tokenize(t))) for q, t in pairs]


class FakeLLM(LLM):
    def __init__(self):
        super().__init__(model="fake-model")

    def stream(self, messages, usage):
        usage.prompt_tokens, usage.completion_tokens = 120, 12
        yield from ["Refunds are issued ", "within 14 days [1]."]


@pytest.fixture(autouse=True)
def offline_models(monkeypatch):
    monkeypatch.setattr(retrieval, "embed", fake_embed)
    monkeypatch.setattr(store, "embed", fake_embed)
    monkeypatch.setattr(retrieval, "reranker", lambda *_: FakeReranker())


def make_pdf() -> bytes:
    doc = fitz.open()
    p = doc.new_page()
    y = 72
    p.insert_text((72, y), "1. Refund Policy", fontsize=18, fontname="helv")
    y += 30
    para = (
        "Customers may request a refund for any order. Refunds are issued within 14 days of "
        "receiving the returned item. Shipping fees are not refundable."
    )
    for line in [para[i : i + 80] for i in range(0, len(para), 80)]:
        p.insert_text((72, y), line, fontsize=11, fontname="helv")
        y += 14
    y += 20
    p.insert_text((72, y), "2. Warranty", fontsize=18, fontname="helv")
    y += 30
    p.insert_text((72, y), "All devices carry a two year limited warranty against defects.", fontsize=11)
    p2 = doc.new_page()
    p2.insert_text((72, 72), "3. Contact", fontsize=18, fontname="helv")
    p2.insert_text((72, 102), "Email support at the address printed on your invoice.", fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def pdf_bytes():
    return make_pdf()
