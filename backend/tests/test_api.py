import json

import pytest
from fastapi.testclient import TestClient

from documind.api import create_app
from documind.obs import QueryLog
from documind.store import DocumentStore

from .conftest import FakeLLM


@pytest.fixture
def client(tmp_path):
    app = create_app(DocumentStore(tmp_path), QueryLog(tmp_path / "q.db"), FakeLLM())
    return TestClient(app)


def parse_sse(body: str) -> list[tuple[str, object]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_upload_query_and_log(client, pdf_bytes):
    r = client.post("/api/documents", files={"file": ("policy.pdf", pdf_bytes, "application/pdf")})
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["num_pages"] == 2 and doc["num_chunks"]["structure"] > 0

    r = client.post("/api/query", json={"question": "How long do refunds take?", "top_k": 3})
    assert r.status_code == 200
    events = parse_sse(r.text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and "token" in kinds
    sources = events[0][1]
    assert sources[0]["page"] == 1 and sources[0]["rects"] and sources[0]["page_size"]
    done = events[-1][1]
    assert done["cited"] == [1] and done["prompt_tokens"] == 120 and done["cost_usd"] > 0

    r = client.get(f"/api/documents/{doc['id']}/pages/1.png")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"

    logged = client.get("/api/queries").json()
    assert logged[0]["question"] == "How long do refunds take?"
    assert logged[0]["retrieved"][0]["page"] == 1
    s = client.get("/api/metrics/summary").json()
    assert s["queries"] == 1 and s["tokens"]["prompt"] == 120


def test_rejects_non_pdf_and_bad_options(client, pdf_bytes):
    assert client.post("/api/documents", files={"file": ("x.pdf", b"hello", "application/pdf")}).status_code == 415
    assert client.post("/api/query", json={"question": "hi"}).status_code == 400
    client.post("/api/documents", files={"file": ("p.pdf", pdf_bytes, "application/pdf")})
    assert client.post("/api/query", json={"question": "hi", "retrieval": "magic"}).status_code == 422
