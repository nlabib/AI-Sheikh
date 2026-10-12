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

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


def get_settings() -> Settings:
    return Settings()
