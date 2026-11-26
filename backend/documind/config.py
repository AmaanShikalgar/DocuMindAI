"""Runtime settings, read once from the environment (and .env if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DOCUMIND_DATA_DIR", str(ROOT / "data"))))

    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "openai/gpt-oss-120b"))
    # gpt-oss is a reasoning model; "low" keeps latency close to a non-reasoning model.
    reasoning_effort: str = field(default_factory=lambda: _env("LLM_REASONING_EFFORT", "low"))
    # USD per 1M tokens. Defaults are Groq's list price for gpt-oss-120b at the time of
    # writing; override with env vars if the price or model changes.
    price_in_per_m: float = field(default_factory=lambda: float(_env("LLM_PRICE_IN_PER_M", "0.15")))
    price_out_per_m: float = field(default_factory=lambda: float(_env("LLM_PRICE_OUT_PER_M", "0.60")))

    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    # Defaults used by the API; the eval harness sweeps these.
    chunking: str = field(default_factory=lambda: _env("CHUNKING", "structure"))
    retrieval: str = field(default_factory=lambda: _env("RETRIEVAL", "hybrid"))
    rerank: bool = field(default_factory=lambda: _env("RERANK", "true").lower() == "true")
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))

    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "25")))

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.price_in_per_m + completion_tokens * self.price_out_per_m) / 1_000_000


settings = Settings()
