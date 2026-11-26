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


def test_private_uploads_and_demo_docs(tmp_path, monkeypatch, pdf_bytes):
    import pymupdf

    monkeypatch.setenv("PRIVATE_UPLOADS", "true")
    store = DocumentStore(tmp_path)
    demo = tmp_path / "demo"
    demo.mkdir()
    d = pymupdf.open()
    d.new_page().insert_text((72, 72), "Demo handbook: the office opens at nine.", fontsize=11)
    (demo / "demo.pdf").write_bytes(d.tobytes())
    (demo / "demo.json").write_text('{"titles": {"demo.pdf": "Demo handbook"}, "questions": ["When does the office open?"]}')
    store.load_demo(demo)
    monkeypatch.setenv("DEMO_DIR", str(demo))
    c = TestClient(create_app(store, QueryLog(tmp_path / "q.db"), FakeLLM()))
    alice, bob = {"X-Client-Id": "alice-0001"}, {"X-Client-Id": "bob-00002"}

    assert c.get("/api/config").json()["sample_questions"] == ["When does the office open?"]
    r = c.post("/api/documents", files={"file": ("policy.pdf", pdf_bytes, "application/pdf")}, headers=alice)
    mine = r.json()["id"]
    assert {x["title"] for x in c.get("/api/documents", headers=alice).json()} == {"Demo handbook", "policy"}
    assert [x["title"] for x in c.get("/api/documents", headers=bob).json()] == ["Demo handbook"]

    # Bob's questions never reach Alice's PDF, and logs stay per visitor
    r = c.post("/api/query", json={"question": "How long do refunds take?"}, headers=bob)
    assert all(s["title"] == "Demo handbook" for s in parse_sse(r.text)[0][1])
    assert len(c.get("/api/queries", headers=bob).json()) == 1
    assert c.get("/api/queries", headers=alice).json() == []

    demo_id = next(x["id"] for x in c.get("/api/documents", headers=bob).json())
    assert c.delete(f"/api/documents/{demo_id}", headers=bob).status_code == 403
    assert c.delete(f"/api/documents/{mine}", headers=bob).status_code == 404
    assert c.delete(f"/api/documents/{mine}", headers=alice).status_code == 204


def test_rejects_non_pdf_and_bad_options(client, pdf_bytes):
    assert client.post("/api/documents", files={"file": ("x.pdf", b"hello", "application/pdf")}).status_code == 415
    assert client.post("/api/query", json={"question": "hi"}).status_code == 400
    client.post("/api/documents", files={"file": ("p.pdf", pdf_bytes, "application/pdf")})
    assert client.post("/api/query", json={"question": "hi", "retrieval": "magic"}).status_code == 422
