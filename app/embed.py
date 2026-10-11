"""Embedders and the on-disk vector index.

`E5Embedder` is the real multilingual model (English, Bangla and Arabic share one
vector space, so a Bangla question can find an Arabic or English passage).
`HashingEmbedder` needs no download: it hashes character n-grams, which only
matches text in the same language and script. It exists for tests and for
machines that cannot fetch the model; do not judge retrieval quality by it.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

from app.text import tokenize


class Embedder(Protocol):
    name: str

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text, drop_stopwords=True):
            padded = f"<{tok}>"
            grams = [padded] + [padded[i : i + n] for n in (3, 4) for i in range(len(padded) - n + 1)]
            for g in grams:
                h = zlib.crc32(g.encode("utf-8"))
                v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    embed_passages = embed_queries


class E5Embedder:
    """intfloat/multilingual-e5-*: needs the "query: " / "passage: " prefixes."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: str | None = None,
                 batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self.batch_size = batch_size

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        ).astype(np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"query: {t}" for t in texts])

    def embed_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in texts], show_progress)


def make_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "e5":
        return E5Embedder(model_name)
    raise ValueError(f"unknown embedder {kind!r} (use 'e5' or 'hashing')")


class VectorIndex:
    """Dense matrix of unit vectors, row i belongs to passage ids[i]. Exact search."""

    def __init__(self, ids: np.ndarray, vectors: np.ndarray, embedder_name: str):
        self.ids = ids
        self.vectors = vectors
        self.embedder_name = embedder_name

    def search(self, query_vec: np.ndarray, limit: int = 50,
               mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        """mask: optional boolean array over rows; False rows are excluded."""
        if len(self.ids) == 0:
            return []
        scores = self.vectors @ query_vec
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
            limit = min(limit, int(mask.sum()))
            if limit == 0:
                return []
        limit = min(limit, len(scores))
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        return [(int(self.ids[i]), float(scores[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        np.save(directory / "ids.npy", self.ids)
        (directory / "meta.json").write_text(
            json.dumps({"embedder": self.embedder_name, "count": int(len(self.ids)),
                        "dim": int(self.vectors.shape[1]) if len(self.ids) else 0}, indent=2)
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex | None":
        if not (directory / "meta.json").exists():
            return None
        meta = json.loads((directory / "meta.json").read_text())
        return cls(np.load(directory / "ids.npy"), np.load(directory / "vectors.npy"), meta["embedder"])


def build_index(embedder: Embedder, items: list[tuple[int, str]], batch: int = 256,
                progress: bool = False) -> VectorIndex:
    ids = np.array([i for i, _ in items], dtype=np.int64)
    chunks = []
    for start in range(0, len(items), batch):
        texts = [t for _, t in items[start : start + batch]]
        chunks.append(embedder.embed_passages(texts))
        if progress:
            print(f"  embedded {min(start + batch, len(items))}/{len(items)}", flush=True)
    vectors = np.concatenate(chunks) if chunks else np.zeros((0, 1), np.float32)
    return VectorIndex(ids, vectors, embedder.name)
