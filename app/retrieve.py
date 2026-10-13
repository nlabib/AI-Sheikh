"""Hybrid retrieval: exact reference lookup, then FTS5 and vector search fused with RRF.

Results are grouped by ref, so each hit is one ayah or one hadith with all the
language editions we hold for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import zip_longest

import numpy as np

from app.embed import Embedder, VectorIndex
from app.rerank import Reranker
from app.refs import HADITH_COLLECTIONS, TAFSIR_COLLECTIONS, TAFSIR_LANGS, find_explicit_refs, parse_ref
from app.store import Passage, Store
from app.text import detect_lang, strip_footnotes

RRF_K = 60
# Candidates per side handed to the reranker (when one is configured); it reorders them and keeps the best k.
RERANK_POOL = 30


@dataclass
class Hit:
    ref: str
    score: float
    matched_by: list[str] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    rerank: float | None = None  # cross-encoder relevance to the question, 0-1, when a reranker is set

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
            "rerank": None if self.rerank is None else round(self.rerank, 4),
            "meta": meta,
            "texts": [
                {"lang": p.lang, "edition": p.edition, "source_id": p.source_id, "text": p.text}
                for p in self.passages
            ],
        }


class Retriever:
    def __init__(self, store: Store, embedder: Embedder | None = None, index: VectorIndex | None = None,
                 reranker: Reranker | None = None, rerank_pool: int = RERANK_POOL, tafsir_slots: int = 1):
        self.store = store
        self.tafsir_slots = tafsir_slots
        self.reranker = reranker
        self.rerank_pool = rerank_pool
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
               pool: int = 100, rerank: bool = True) -> list[Hit]:
        """mode: "fts", "vector" or "hybrid". With a reranker (and rerank=True), the top rerank_pool
        candidates are scored by it and the best k kept."""
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

        use_reranker = self.reranker is not None and rerank
        n = max(k, self.rerank_pool) if use_reranker else k
        top = sorted(scores, key=scores.get, reverse=True)[:max(n, len(explicit))]
        hits = [Hit(ref, scores[ref], sorted(matched[ref]), self._ordered(ref, query)) for ref in top]
        if use_reranker:
            hits = self._rerank(query, hits)
        return hits[:max(k, len(explicit))]

    def search_balanced(self, query: str, k: int = 6, mode: str = "hybrid",
                        collections: list[str] | None = None, min_rerank: float = 0.0) -> list[Hit]:
        """Evidence for answering: the best Quran passages and the best hadith, interleaved.

        Hadith outnumber ayat by about three to one in the corpus, so an unscoped
        search tends to crowd the Quran out. Searching each side separately keeps
        both in front of the model. Explicitly typed references stay first.

        With a reranker, each side is ordered by its score and passages below min_rerank are left out, so a
        side with nothing about the question gives its places to the other. The sides are still interleaved
        rather than ranked together: Pickthall's 1930 English scores low against modern wording ("interest",
        "patience"), and a single ranking pushed the Quran out of Quran questions.
        """
        hadith = [c for c in HADITH_COLLECTIONS if not collections or c in collections]
        sides: list[list[Hit]] = []
        # With a floor, fetch more per side so that what survives it can still fill k.
        per_side = 2 * k if (self.reranker is not None and min_rerank > 0) else k
        if not collections or "quran" in collections:
            sides.append(self.search(query, k=per_side, mode=mode, collections=["quran"]))
        if hadith:
            sides.append(self.search(query, k=per_side, mode=mode, collections=hadith))
        # Tafsir in the question's language: a passage or two of commentary, never crowding out the texts.
        lang = detect_lang(query)
        tafsir = [c for c in TAFSIR_COLLECTIONS
                  if TAFSIR_LANGS[c] == lang and (not collections or c in collections)]
        tafsir_side = (self.search(query, k=max(per_side // 2, self.tafsir_slots), mode=mode, collections=tafsir)
                       if tafsir and self.tafsir_slots > 0 and self.store.has_collections(tafsir) else [])
        if self.reranker is not None and min_rerank > 0:
            sides, tafsir_side = [[h for h in side if "reference" in h.matched_by or (h.rerank or 0.0) >= min_rerank]
                                  for side in (*sides, tafsir_side)][:-1], \
                [h for h in tafsir_side if "reference" in h.matched_by or (h.rerank or 0.0) >= min_rerank]
        explicit = [h for side in (*sides, tafsir_side) for h in side if "reference" in h.matched_by]
        explicit.sort(key=lambda h: -h.score)
        seen = {h.ref for h in explicit}
        rest: list[Hit] = []
        for group in zip_longest(*sides):
            for h in group:
                if h is not None and h.ref not in seen:
                    rest.append(h)
                    seen.add(h.ref)
        commentary = [h for h in tafsir_side if h.ref not in seen][:self.tafsir_slots]
        # A whole named surah can fill k by itself; still leave room for passages about the question.
        room = max(k - len(explicit), k // 2)
        if commentary:
            texts = rest[:room - len(commentary)]
            rest = texts[:2] + commentary + texts[2:]
        return explicit + rest[:room]

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

    def rerank_scores(self, query: str, refs: list[str]) -> dict[str, float]:
        """The reranker's relevance score for each ref: the best over its editions (Arabic, English, Bangla),
        since an archaic translation can hide a match the other editions show. Empty without a reranker."""
        if self.reranker is None or not refs:
            return {}
        owners, texts = [], []
        for ref in dict.fromkeys(refs):
            for p in self.store.by_ref(ref):
                owners.append(ref)
                texts.append(strip_footnotes(p.text))
        out: dict[str, float] = {}
        for ref, s in zip(owners, self.reranker.score(query, texts)):
            out[ref] = max(out.get(ref, 0.0), s)
        return out

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Order hits by reranker score, explicitly typed references first as always."""
        scores = self.rerank_scores(query, [h.ref for h in hits])
        for h in hits:
            h.rerank = scores.get(h.ref, 0.0)
        explicit = [h for h in hits if "reference" in h.matched_by]
        return explicit + sorted((h for h in hits if "reference" not in h.matched_by), key=lambda h: -h.rerank)

    def _ordered(self, ref: str, query: str) -> list[Passage]:
        """The editions shown for a ref, one per language: the question's language first, then Arabic, then the
        rest. Translations kept only for search are left out."""
        lang = detect_lang(query)
        order = {lang: 0, "ar": 1}
        shown, langs = [], set()
        for p in sorted(self.store.by_ref(ref), key=lambda p: (order.get(p.lang, 2), p.lang)):
            if not p.meta.get("search_only") and p.lang not in langs:
                shown.append(p)
                langs.add(p.lang)
        return shown
