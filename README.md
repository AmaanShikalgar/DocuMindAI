# DocuMind v2

Ask questions about your PDFs and get answers that cite the exact page and passage. Each
retrieval choice in the pipeline was measured on a fixed eval set before it went in.

**Live demo:** _coming soon_ · v1 (Streamlit): https://rag-pdf-chatbot-amaan.streamlit.app/

## Results

**Retrieval hit-rate went from 74% (v1) to 95% (v2)**: the share of questions where the
passage that answers them is among the 5 sources given to the model. Measured on 48
hand-written questions over three public PDFs: a security framework, a tax publication and a
research paper. See [eval/README.md](eval/README.md) for how the questions were written and how
each metric is defined.

<!-- EVAL:START -->
_48 questions (42 answerable, 6 unanswerable) over 3 PDFs · embeddings `all-MiniLM-L6-v2` · reranker `ms-marco-TinyBERT-L-2-v2` · LLM `openai/gpt-oss-120b` · run 2026-11-26_

**Retrieval** (answerable questions; hit = a top-k chunk covers the evidence passage)

| Config | Hit@5 | MRR@5 | Retrieval p50 |
|---|---|---|---|
| fixed + dense (v1 baseline) | 74% | 0.54 | 43 ms |
| fixed + dense + rerank | 83% | 0.56 | 215 ms |
| fixed + bm25 | 79% | 0.60 | 4 ms |
| fixed + bm25 + rerank | 88% | 0.60 | 205 ms |
| fixed + hybrid | 93% | 0.67 | 45 ms |
| fixed + hybrid + rerank | 88% | 0.60 | 196 ms |
| structure + dense | 81% | 0.53 | 36 ms |
| structure + dense + rerank | 90% | 0.67 | 289 ms |
| structure + bm25 | 86% | 0.62 | 2 ms |
| structure + bm25 + rerank | **98%** | 0.70 | 260 ms |
| structure + hybrid | 90% | 0.62 | 42 ms |
| structure + hybrid + rerank (v2 default) | 95% | 0.70 | 308 ms |
<!-- EVAL:END -->

**v1 baseline** is the original app: 500-character windows with 100-character overlap, dense
FAISS search only. **v2 default** is structure-aware chunks, hybrid BM25 + FAISS search, and a
cross-encoder reranker.

What the numbers say:

- **Most changes help on their own.** Structure-aware chunks beat fixed windows with dense
  search (81% vs 74%) and with BM25 (86% vs 79%). BM25 beats dense search on these documents,
  which are full of exact terms like `DE.CM-06`, "Form 2106" and "70 cents". Hybrid search
  beats either alone before reranking.
- **The changes work best together.** Without a reranker, fixed windows with hybrid search
  (93%) edge out structure-aware chunks (90%). With the reranker, structure-aware chunks gain 5
  to 12 points in every search mode. Fixed windows with hybrid search lose 5 points; a likely
  reason is that 500-character windows cut sentences in half and the reranker scores those
  fragments poorly.
- **Why hybrid, when BM25 + rerank scored higher?** Structure + BM25 + rerank scored 98%, one
  question more than the default. Hybrid stays the default because its dense half covers
  questions phrased differently from the text, which this eval set has fewer of than real use.
- **Small sample.** With 42 answerable questions, each one is worth 2.4 points, so treat
  differences of a question or two as noise.

Answer accuracy (judged by an LLM) is measured by the same script with `GROQ_API_KEY` set:
`python eval/run_eval.py --update-readme`.

## What's in it

- **Two chunkers.** _Fixed_ reproduces v1. _Structure-aware_ reads font sizes and weights from the
  PDF to find headings, groups paragraphs under their heading up to about 900 characters, never
  crosses a heading, and splits long paragraphs at sentence ends. The heading is prepended to the
  text that gets indexed, so a chunk about "Refund Policy" can be found by that phrase.
- **Hybrid search.** BM25 catches exact terms, codes and names. MiniLM embeddings in FAISS catch
  paraphrases. Reciprocal Rank Fusion merges the two lists, so their score scales never need to
  be compared.
- **Reranker.** A cross-encoder rescores the top 20 fused candidates against the question and
  keeps the best 5. It uses the 2-layer `ms-marco-TinyBERT-L-2-v2`: on the eval set it finds the
  evidence as often as the 6-layer `ms-marco-MiniLM-L-6-v2` (95% hit@5), with slightly lower MRR
  (0.70 vs 0.73), and takes about 0.3 s on a CPU instead of about 4 s.
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
