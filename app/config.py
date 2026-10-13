"""Runtime settings, all overridable with environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    db_path: Path = field(default_factory=lambda: Path(_env("DB_PATH", str(ROOT / "data" / "corpus.sqlite"))))
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", str(ROOT / "data" / "index"))))
    # "e5" uses sentence-transformers; "hashing" is an offline lexical fallback.
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "e5"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "intfloat/multilingual-e5-small"))
    # LM Studio exposes an OpenAI-compatible server on port 1234 by default.
    lm_base_url: str = field(default_factory=lambda: _env("LM_BASE_URL", "http://127.0.0.1:1234/v1"))
    lm_model: str = field(default_factory=lambda: _env("LM_MODEL", "qwen3-4b"))
    lm_api_key: str = field(default_factory=lambda: _env("LM_API_KEY", "lm-studio"))
    lm_timeout: float = field(default_factory=lambda: float(_env("LM_TIMEOUT", "120")))
    # Qwen3 "thinking mode". Off by default: the fine-tuning data is non-thinking, so serve it that way.
    thinking: bool = field(default_factory=lambda: _env("THINKING", "0") == "1")
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "6")))
    # When an answer fails the citation check, replace it with "I'm not sure" (raw answer kept separately).
    strict_citations: bool = field(default_factory=lambda: _env("STRICT_CITATIONS", "1") != "0")
    # Semantic similarity (E5 cosine) below which a passage counts as not about the question: it isn't shown
    # to the model, and citing it is flagged. 0 turns this off. Calibrate with `python -m tools.calibrate_relevance`.
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0")))
    # Cross-encoder that rereads the question with each candidate passage and judges relevance, e.g.
    # BAAI/bge-reranker-v2-m3. Off when empty. With it, RERANK_MIN (its 0-1 score) replaces MIN_RELEVANCE.
    reranker: str = field(default_factory=lambda: _env("RERANKER", ""))
    rerank_min: float = field(default_factory=lambda: float(_env("RERANK_MIN", "0")))
    rerank_pool: int = field(default_factory=lambda: int(_env("RERANK_POOL", "30")))
    # Places among the top_k sources given to tafsir in the question's language (0 turns tafsir off).
    tafsir_slots: int = field(default_factory=lambda: int(_env("TAFSIR_SLOTS", "1")))

    @property
    def relevance_floor(self) -> float:
        """The floor for whichever relevance score is in use: the reranker's if one is set, else E5's."""
        return self.rerank_min if self.reranker else self.min_relevance

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


def get_settings() -> Settings:
    return Settings()
