"""Cross-encoder reranking: read the question and a passage together and judge whether the passage is about it.

Search (keyword + E5) finds candidates by similarity, but similarity is a poor judge of relevance: for
"what does the Quran say about riba", E5 rates verses that mention "the Qur'an" above 2:275, the verse on
usury (docs/INFERENCE_REVIEW.md). A cross-encoder looks at both texts at once and scores how well the
passage answers the question, which is slower (one model pass per pair) but much sharper. Retrieval
fetches a pool of candidates and the reranker orders them; its scores also drive the relevance floor.

Off unless RERANKER is set, e.g. RERANKER=BAAI/bge-reranker-v2-m3 (multilingual, ~2.3 GB download).
"""

from __future__ import annotations

from typing import Protocol

# Characters of each edition sent to the reranker; its context is 512 tokens anyway.
MAX_CHARS = 1500


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance of each passage to the query, from 0 (not about it) to 1."""
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None,
                 batch_size: int = 32, max_length: int = 512):
        import torch
        from sentence_transformers import CrossEncoder  # heavy import, keep lazy

        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        if self.model.device.type == "cuda":
            self.model.model.half()  # half precision: same rankings, about twice as fast
        self.name = model_name
        self.batch_size = batch_size
        self._torch = torch

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p[:MAX_CHARS]) for p in passages]
        with self._torch.inference_mode():
            # One output logit per pair; the sigmoid turns it into a 0-1 relevance score.
            scores = self.model.predict(pairs, batch_size=self.batch_size, activation_fn=self._torch.nn.Sigmoid(),
                                        convert_to_numpy=True, show_progress_bar=False)
        return [float(s) for s in scores]


def make_reranker(model_name: str) -> Reranker | None:
    return CrossEncoderReranker(model_name) if model_name else None
