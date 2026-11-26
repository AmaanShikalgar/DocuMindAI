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
