# Evaluation

## Documents

| File | Source | License |
|---|---|---|
| `docs/nist-csf-2.0.pdf` | NIST, *The NIST Cybersecurity Framework (CSF) 2.0*, NIST CSWP 29, 2024. [doi:10.6028/NIST.CSWP.29](https://doi.org/10.6028/NIST.CSWP.29) | US government work, public domain |
| `docs/irs-p463-2025.pdf` | IRS, *Publication 463: Travel, Gift, and Car Expenses* (for 2025 returns). [irs.gov](https://www.irs.gov/pub/irs-prior/p463--2025.pdf) | US government work, public domain |
| `docs/lost-in-the-middle.pdf` | Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang. *Lost in the Middle: How Language Models Use Long Contexts*. TACL 12 (2024). [aclanthology.org/2024.tacl-1.9](https://aclanthology.org/2024.tacl-1.9) | CC BY 4.0, redistributed unchanged |

The questions are generated from `build_questions.py`, which checks that every evidence quote
appears word for word in the PDF and records the page it's on. Edit that file, not
`questions.jsonl`.

The numbers in the main README come from `run_eval.py`, run over the PDFs in `docs/` and the
questions in `questions.jsonl`.

## Writing questions

Put one JSON object on each line:

```json
{"id": "q01", "doc": "employee-handbook.pdf", "question": "How many vacation days do new hires get?",
 "answer": "15 days per year, prorated in the first year.",
 "evidence": [{"page": 12, "quote": "New employees accrue 15 days of paid vacation per year"}]}
```

- `quote` must be copied **word for word** from the PDF page (whitespace and line breaks don't
  matter). Keep it to the sentence or two that answers the question. `--check` tells you about
  any quote it can't find.
- `page` is the 1-based page number in the PDF file, which may differ from the number printed on the page.
- Mix question types: single-fact lookups, numbers and dates, definitions, questions whose answer
  sits under a specific heading, questions phrased differently from the text (to test dense search),
  and questions built around exact terms or codes (to test BM25).
- Add 5-8 questions the documents **cannot** answer, with `"answerable": false` and no evidence.
  The system scores on those only when it abstains.

Target size is 40-50 questions over 3-4 PDFs.

## Running

```powershell
.venv\Scripts\python eval\run_eval.py --check
.venv\Scripts\python eval\run_eval.py --retrieval-only
.venv\Scripts\python eval\run_eval.py --update-readme
```

`--retrieval-only` needs no API key. The full run judges answers for the v1 baseline and the v2
default (about 4 LLM calls per question). Add `--all-answer-configs` to judge every configuration.

## What the metrics mean

| Metric | Definition |
|---|---|
| Hit@5 | Share of answerable questions where one of the top 5 chunks covers at least half of an evidence quote, on the right page. Measured as character-span overlap, so fixed and structure-aware chunks are scored the same way. |
| MRR@5 | Mean of 1 / rank of the first such chunk (0 if none in the top 5). |
| Accuracy | Share of answerable questions the judge grades `correct` against the reference answer. |
| Abstains when it should | Share of unanswerable questions where the system replies that it cannot find the answer. |
| Citation precision | Of answers that cite sources, the share where at least one cited source contains the evidence. |
| Latency / first token / cost | End-to-end time per question (retrieval, rerank and generation), time to the first streamed token, and Groq token cost. |

The judge is the same model that writes the answers, run at temperature 0 with a fixed rubric.
That is a known bias. `results/answers.jsonl` stores every answer next to its verdict, so you
can spot-check the grading.
