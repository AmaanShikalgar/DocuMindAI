"""Query log in SQLite: one row per question with retrieved chunks, latency breakdown,
tokens and cost. Powers the /dashboard page."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    doc_ids TEXT,
    chunking TEXT, retrieval TEXT, rerank INTEGER, top_k INTEGER,
    model TEXT,
    retrieved TEXT,
    retrieval_ms REAL, rerank_ms REAL, ttft_ms REAL, total_ms REAL,
    prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd REAL,
    not_found INTEGER,
    error TEXT,
    client TEXT
);
CREATE INDEX IF NOT EXISTS queries_ts ON queries(ts);
"""

COLS = (
    "ts question answer doc_ids chunking retrieval rerank top_k model retrieved retrieval_ms rerank_ms "
    "ttft_ms total_ms prompt_tokens completion_tokens cost_usd not_found error client"
).split()


class QueryLog:
    def __init__(self, path: Path | None = None):
        self.path = path or settings.data_dir / "documind.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(queries)")}
            if "client" not in cols:  # databases created before per-visitor logs
                c.execute("ALTER TABLE queries ADD COLUMN client TEXT")

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def log(self, **row) -> int:
        row.setdefault("ts", time.time())
        for k in ("doc_ids", "retrieved"):
            if not isinstance(row.get(k), (str, type(None))):
                row[k] = json.dumps(row[k])
        vals = [row.get(k) for k in COLS]
        with self._lock, self._conn() as c:
            cur = c.execute(f"INSERT INTO queries ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})", vals)
            return int(cur.lastrowid)

    def recent(self, limit: int = 50, offset: int = 0, client: str | None = None) -> list[dict]:
        """Latest queries; with `client`, only that visitor's own."""
        where, args = ("WHERE client = ?", [client]) if client is not None else ("", [])
        with self._conn() as c:
            rows = c.execute(f"SELECT * FROM queries {where} ORDER BY id DESC LIMIT ? OFFSET ?", (*args, limit, offset)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("doc_ids", "retrieved"):
                d[k] = json.loads(d[k]) if d[k] else []
            d.pop("client", None)
            out.append(d)
        return out

    def summary(self, since: float | None = None) -> dict:
        since = since or 0
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts, total_ms, ttft_ms, retrieval_ms, rerank_ms, prompt_tokens, completion_tokens, cost_usd, "
                "not_found, error FROM queries WHERE ts >= ? ORDER BY ts",
                (since,),
            ).fetchall()
        ok = [r for r in rows if not r["error"]]

        def pct(vals: list[float], p: float) -> float | None:
            vals = sorted(v for v in vals if v is not None)
            if not vals:
                return None
            i = min(len(vals) - 1, max(0, round(p / 100 * (len(vals) - 1))))
            return round(vals[i], 1)

        tot = [r["total_ms"] for r in ok]
        ttft = [r["ttft_ms"] for r in ok]
        # daily buckets for the chart
        days: dict[str, dict] = {}
        for r in ok:
            day = time.strftime("%Y-%m-%d", time.localtime(r["ts"]))
            b = days.setdefault(day, {"day": day, "queries": 0, "cost_usd": 0.0, "lat": []})
            b["queries"] += 1
            b["cost_usd"] += r["cost_usd"] or 0
            b["lat"].append(r["total_ms"])
        series = [
            {"day": b["day"], "queries": b["queries"], "cost_usd": round(b["cost_usd"], 6), "p50_ms": pct(b["lat"], 50)}
            for b in days.values()
        ]
        return {
            "queries": len(rows),
            "errors": len(rows) - len(ok),
            "not_found_rate": round(sum(1 for r in ok if r["not_found"]) / len(ok), 3) if ok else None,
            "latency_ms": {"p50": pct(tot, 50), "p95": pct(tot, 95)},
            "ttft_ms": {"p50": pct(ttft, 50), "p95": pct(ttft, 95)},
            "retrieval_ms_p50": pct([r["retrieval_ms"] for r in ok], 50),
            "rerank_ms_p50": pct([r["rerank_ms"] for r in ok], 50),
            "tokens": {
                "prompt": sum(r["prompt_tokens"] or 0 for r in ok),
                "completion": sum(r["completion_tokens"] or 0 for r in ok),
            },
            "cost_usd": {
                "total": round(sum(r["cost_usd"] or 0 for r in ok), 6),
                "per_query": round(sum(r["cost_usd"] or 0 for r in ok) / len(ok), 6) if ok else None,
            },
            "series": series,
        }
