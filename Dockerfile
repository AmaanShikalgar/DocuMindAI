# ---- front end ----
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build:server

# ---- API + static site ----
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/models DOCUMIND_DATA_DIR=/data STATIC_DIR=/app/static PORT=8000
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
# bake the embedding and reranker models into the image so the first query isn't a download
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')"
COPY backend/documind ./documind
# Demo documents: indexed at build time so a fresh container (or a Space waking up) is ready
# at once. Visitors' uploads stay private to their browser (PRIVATE_UPLOADS).
COPY eval/docs/*.pdf deploy/demo/demo.json ./demo/
ENV DEMO_DIR=/app/demo PRIVATE_UPLOADS=true
RUN python -c "from pathlib import Path; from documind.store import DocumentStore; DocumentStore().load_demo(Path('/app/demo'))"
COPY --from=web /web/dist ./static
RUN useradd -m -u 1000 app && mkdir -p /data && chown -R app /data /models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/api/health')"
CMD ["sh", "-c", "uvicorn documind.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
