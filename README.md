# DocuMind v2

Ask questions about your PDFs and get answers that cite the exact page and passage. Each
retrieval choice in the pipeline was measured on a fixed eval set before it went in.

**Live demo:** _coming soon_ · v1 (Streamlit): https://rag-pdf-chatbot-amaan.streamlit.app/

## Results

Measured on 40-50 hand-written questions with reference answers and evidence passages,
across 3-4 PDFs. See [eval/README.md](eval/README.md) for how questions were written and how
each metric is defined.

<!-- EVAL:START -->
_Not run yet. Add PDFs to `eval/docs/` and questions to `eval/questions.jsonl`, then run
`python eval/run_eval.py --update-readme`._
<!-- EVAL:END -->

**v1 baseline** is the original app: 500-character windows with 100-character overlap, dense
FAISS search only. **v2 default** is structure-aware chunks, hybrid BM25 + FAISS search, and a
cross-encoder reranker.

## What's in it

- **Two chunkers.** _Fixed_ reproduces v1. _Structure-aware_ reads font sizes and weights from the
  PDF to find headings, groups paragraphs under their heading up to about 900 characters, never
  crosses a heading, and splits long paragraphs at sentence ends. The heading is prepended to the
  text that gets indexed, so a chunk about "Refund Policy" can be found by that phrase.
- **Hybrid search.** BM25 catches exact terms, codes and names. MiniLM embeddings in FAISS catch
  paraphrases. Reciprocal Rank Fusion merges the two lists, so their score scales never need to
  be compared.
- **Reranker.** `ms-marco-MiniLM-L-6-v2` rescores the top 30 fused candidates against the question
  and keeps the best 5.
- **Citations to the passage.** Every chunk stores its character offsets on the page, and every
  offset maps back to a rectangle on the PDF page. The model cites sources as `[n]`. Clicking a
  citation opens the rendered page with the passage highlighted.
- **Streaming.** Answers stream token by token over server-sent events. Sources arrive first, so
  citations work the moment they appear.
- **Observability.** Every query is logged to SQLite with its retrieved chunks (and their dense and
  BM25 ranks), latency split into retrieval, rerank, first token and total, token counts, and cost.
  The `/dashboard` page shows these as p50/p95 tiles, a per-query latency breakdown and a
  query log you can expand to see each query's retrieved chunks.

## Architecture

```
PDF ──► PyMuPDF (text + font sizes + bounding boxes)
          ├─► fixed chunker ──────┐
          └─► structure chunker ──┴─► MiniLM embeddings ─► FAISS
                                  └─► BM25
question ─► FAISS top-30 ┐
          ─► BM25 top-30 ┴─► RRF ─► cross-encoder rerank ─► top-5 ─► Groq LLM (streamed, cites [n])
                                                                        │
                                       SQLite query log ◄───────────────┘  ─► /dashboard
```

| Path | What |
|---|---|
| `backend/documind/pdf.py` | Layout-preserving parsing, offset → rectangle mapping |
| `backend/documind/chunking.py` | Fixed and structure-aware chunkers |
| `backend/documind/retrieval.py` | Dense, BM25, hybrid (RRF), reranker |
| `backend/documind/pipeline.py` | Retrieve → prompt → stream, shared by the API and the eval |
| `backend/documind/api.py` | FastAPI: documents, `/api/query` (SSE), metrics, static front end |
| `backend/documind/obs.py` | SQLite query log and summary stats |
| `frontend/` | React + Vite: chat with citations, page viewer, dashboard |
| `eval/` | Question set, harness, results |

## Run it

### Docker

```powershell
copy .env.example .env   # then put your GROQ_API_KEY in .env
docker compose up --build
```

Open http://localhost:8000. Uploaded PDFs and the query log persist in the `documind-data` volume.

### Local development

```powershell
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r backend\requirements-dev.txt
cd backend; ..\.venv\Scripts\uvicorn documind.api:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend; npm install; npm run dev
```

Open http://localhost:5173. Vite proxies `/api` to the backend.

### Tests

```powershell
cd backend; ..\.venv\Scripts\python -m pytest
```

The tests use a small generated PDF and stand-in models, so they run offline in a few seconds.

### Eval

```powershell
.venv\Scripts\python eval\run_eval.py --check
.venv\Scripts\python eval\run_eval.py --update-readme
```

## Deploy

The image serves both the API and the front end on one port, so it runs on any container host.
The live demo runs on a Hugging Face Docker Space (free CPU tier), deployed by
[a GitHub Action](.github/workflows/deploy-hf-space.yml) on every push to `master`.

One-time setup:

1. On Hugging Face, create a Space with the **Docker** SDK (blank template).
2. In the Space's settings, add the secret `GROQ_API_KEY`.
3. Create a Hugging Face access token with **write** permission.
4. In this GitHub repo's settings, add that token as the Actions secret `HF_TOKEN`. Then add the
   Actions variable `HF_SPACE` with the Space's name, like `your-hf-username/documind`.
5. Run the workflow once from the Actions tab, or push to `master`.

The workflow mirrors the repo to the Space and swaps in [deploy/hf-space/README.md](deploy/hf-space/README.md),
which holds the settings Hugging Face reads.

How the public demo behaves:

- **Preloaded documents.** The three eval PDFs are indexed when the image is built, so the demo
  is ready as soon as the container starts. Visitors can't remove them.
- **Private uploads.** Each browser gets a random ID (`PRIVATE_UPLOADS=true`). Visitors see
  only their own uploads and their own query log; dashboard totals cover everyone.
- **Rate limits.** Each visitor can ask 30 questions per 10 minutes (`RATE_LIMIT_QUERIES`) and
  upload 10 PDFs per hour, so the demo can't drain the API key. Also set a spending limit in the
  Groq console.
- **No persistence.** Free Spaces don't keep files between restarts, so uploads disappear when
  the Space sleeps or restarts.

## Configuration

See [.env.example](.env.example). Retrieval defaults (`CHUNKING`, `RETRIEVAL`, `RERANK`, `TOP_K`)
apply to the API. The front end can switch them per question, so you can compare configurations
live. Token prices for cost tracking are set by `LLM_PRICE_IN_PER_M` and `LLM_PRICE_OUT_PER_M`.

## Author

**Amaan Shikalgar**, Full Stack & AI Developer
