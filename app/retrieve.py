"""Hybrid retrieval: exact reference lookup, then FTS5 and vector search fused with RRF.

Results are grouped by ref, so each hit is one ayah or one hadith with all the
language editions we hold for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import zip_longest

import numpy as np

from app.embed import Embedder, VectorIndex
from app.refs import HADITH_COLLECTIONS, find_explicit_refs, parse_ref
from app.store import Passage, Store
from app.text import detect_lang

RRF_K = 60


@dataclass
class Hit:
    ref: str
    score: float
    matched_by: list[str] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)

    @property
    def label(self) -> str:
        label = parse_ref(self.ref).label
        meta = next((p.meta for p in self.passages if p.meta), {})
        if meta.get("surah_name"):
            label += f" ({meta['surah_name']})"
        if meta.get("alt_number"):
            label += f" (Abd al-Baqi no. {meta['alt_number']})"
        return label

    def text_in(self, lang: str) -> str | None:
        for p in self.passages:
            if p.lang == lang:
                return p.text
        return None

    def to_dict(self) -> dict:
        meta = next((p.meta for p in self.passages if p.meta), {})
        return {
            "ref": self.ref,
            "label": self.label,
            "score": round(self.score, 5),
            "matched_by": self.matched_by,
            "meta": meta,
            "texts": [
                {"lang": p.lang, "edition": p.edition, "source_id": p.source_id, "text": p.text}
                for p in self.passages
            ],
        }


class Retriever:
    def __init__(self, store: Store, embedder: Embedder | None = None, index: VectorIndex | None = None):
        self.store = store
        self.embedder = embedder
        self.index = index if (index is not None and embedder is not None) else None
        self._row_collection = None
        self._row_of: dict[int, int] = {}
        if self.index is not None:
            by_id = store.collections_by_id()
            self._row_collection = np.array([by_id.get(int(i), "") for i in self.index.ids])
            self._row_of = {int(pid): row for row, pid in enumerate(self.index.ids)}

    def _mask(self, collections: list[str] | None):
        if not collections or self._row_collection is None:
            return None
        return np.isin(self._row_collection, collections)

    def search(self, query: str, k: int = 6, mode: str = "hybrid", collections: list[str] | None = None,
               pool: int = 100) -> list[Hit]:
        """mode: "fts", "vector" or "hybrid"."""
        ranked: dict[str, list[int]] = {}
        if mode in ("fts", "hybrid"):
            ranked["fts"] = [pid for pid, _ in self.store.fts_search(query, limit=pool, collections=collections)]
        if mode in ("vector", "hybrid") and self.index is not None:
            qv = self.embedder.embed_queries([query])[0]
            ranked["vector"] = [pid for pid, _ in self.index.search(qv, limit=pool, mask=self._mask(collections))]
        if mode == "vector" and self.index is None:
            raise RuntimeError("vector search requested but no vector index is loaded (run tools.index)")

        passages = self.store.by_ids(sorted({pid for ids in ranked.values() for pid in ids}))
        scores: dict[str, float] = {}
        matched: dict[str, set[str]] = {}
        for method, ids in ranked.items():
            seen_refs: set[str] = set()
            rank = 0
            for pid in ids:
                p = passages.get(pid)
                if p is None or (collections and p.collection not in collections):
                    continue
                if p.ref in seen_refs:  # only a ref's best row counts per method
                    continue
                seen_refs.add(p.ref)
                rank += 1
                scores[p.ref] = scores.get(p.ref, 0.0) + 1.0 / (RRF_K + rank)
                matched.setdefault(p.ref, set()).add(method)

        # A reference typed explicitly ("2:255", "Bukhari 1", "Surah al-Ikhlas") always ranks first, in the
        # order typed (ayat of a named surah in order).
        explicit = [ref for ref in self.explicit_refs(query)
                    if not collections or parse_ref(ref).collection in collections]
        for i, ref in enumerate(explicit):
            scores[ref] = 2.0 - i / 1000 + scores.get(ref, 0.0) / 1000
            matched.setdefault(ref, set()).add("reference")

        top = sorted(scores, key=scores.get, reverse=True)[:max(k, len(explicit))]
        return [Hit(ref, scores[ref], sorted(matched[ref]), self._ordered(ref, query)) for ref in top]

    def search_balanced(self, query: str, k: int = 6, mode: str = "hybrid",
                        collections: list[str] | None = None) -> list[Hit]:
        """Evidence for answering: the best Quran passages and the best hadith, interleaved.

        Hadith outnumber ayat by about three to one in the corpus, so an unscoped
        search tends to crowd the Quran out. Searching each side separately keeps
        both in front of the model. Explicitly typed references stay first.
        """
        hadith = [c for c in HADITH_COLLECTIONS if not collections or c in collections]
        sides = []
        if not collections or "quran" in collections:
            sides.append(self.search(query, k=k, mode=mode, collections=["quran"]))
        if hadith:
            sides.append(self.search(query, k=k, mode=mode, collections=hadith))
        explicit = [h for side in sides for h in side if "reference" in h.matched_by]
        explicit.sort(key=lambda h: -h.score)
        seen = {h.ref for h in explicit}
        rest: list[Hit] = []
        for group in zip_longest(*sides):
            for h in group:
                if h is not None and h.ref not in seen:
                    rest.append(h)
                    seen.add(h.ref)
        # A whole named surah can fill k by itself; still leave room for passages about the question.
        return explicit + rest[:max(k - len(explicit), k // 2)]

    def explicit_refs(self, query: str) -> list[str]:
        """Typed references that exist in the corpus. "Muslim 394" also finds the hadith whose
        Abd al-Baqi number is 394, the numbering most books and sites use."""
        out: list[str] = []
        for ref in find_explicit_refs(query):
            if ref.startswith("muslim:"):
                out += self.store.refs_by_alt_number("muslim", ref.partition(":")[2])
            if self.store.ref_exists(ref):
                out.append(ref)
        return list(dict.fromkeys(out))

    def relevance(self, query: str, refs: list[str]) -> dict[str, float]:
        """How close each ref is in meaning to the query: the best E5 cosine over its editions.
        Empty without a vector index."""
        if self.index is None or not refs:
            return {}
        qv = self.embedder.embed_queries([query])[0]
        out: dict[str, float] = {}
        for ref in refs:
            rows = [self._row_of[p.id] for p in self.store.by_ref(ref) if p.id in self._row_of]
            if rows:
                out[ref] = float(np.max(self.index.vectors[rows] @ qv))
        return out

    def _ordered(self, ref: str, query: str) -> list[Passage]:
        """All editions of a ref: the question's language first, then Arabic, then the rest."""
        lang = detect_lang(query)
        order = {lang: 0, "ar": 1}
        return sorted(self.store.by_ref(ref), key=lambda p: (order.get(p.lang, 2), p.lang))
