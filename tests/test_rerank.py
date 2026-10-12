from fastapi.testclient import TestClient

from app.retrieve import Retriever
from app.server import app
from tests.test_server import FakeLLM


class FakeReranker:
    """Scores a passage 0.9 if it contains `prefer`, else 0.05: enough to see reordering and the floor work."""

    name = "fake"

    def __init__(self, prefer: str):
        self.prefer = prefer.lower()
        self.calls = 0

    def score(self, query, passages):
        self.calls += 1
        return [0.9 if self.prefer in p.lower() else 0.05 for p in passages]


def test_reranker_reorders_candidates_and_records_its_score(store):
    # Keyword search ranks 49:12 seventh for "Allah"; the reranker, told 49:12 is the relevant one, lifts it.
    plain = Retriever(store).search("Allah", k=3, mode="fts")
    reranked = Retriever(store, reranker=FakeReranker("suspicion")).search("Allah", k=3, mode="fts")
    assert all(h.rerank is None for h in plain) and "quran:49:12" not in [h.ref for h in plain]
    assert reranked[0].ref == "quran:49:12"
    assert reranked[0].rerank == 0.9 and reranked[-1].rerank == 0.05
    assert reranked[0].to_dict()["rerank"] == 0.9


def test_rerank_can_be_skipped_per_call(store):
    fake = FakeReranker("suspicion")
    Retriever(store, reranker=fake).search("Allah", k=3, mode="fts", rerank=False)
    assert fake.calls == 0


def test_explicitly_typed_reference_stays_first_whatever_the_reranker_says(store):
    hits = Retriever(store, reranker=FakeReranker("suspicion")).search("2:256 Allah", k=3, mode="fts")
    assert hits[0].ref == "quran:2:256" and "reference" in hits[0].matched_by


def test_reranker_takes_the_best_score_over_editions(store):
    # Only the English edition of 2:256 says "compulsion"; the Arabic one doesn't, and the best counts.
    scores = Retriever(store, reranker=FakeReranker("compulsion")).rerank_scores("q", ["quran:2:256", "quran:1:1"])
    assert scores == {"quran:2:256": 0.9, "quran:1:1": 0.05}


def test_balanced_search_orders_each_side_by_the_reranker(store):
    hits = Retriever(store, reranker=FakeReranker("suspicion")).search_balanced("Allah", k=4, mode="fts")
    assert hits[0].ref == "quran:49:12"
    assert all(h.rerank is not None for h in hits)


def test_balanced_search_leaves_out_passages_below_the_floor(store):
    retriever = Retriever(store, reranker=FakeReranker("suspicion"))
    assert len(retriever.search_balanced("Allah", k=4, mode="fts")) == 4
    hits = retriever.search_balanced("Allah", k=4, mode="fts", min_rerank=0.5)
    assert [h.ref for h in hits] == ["quran:49:12"]
    # A typed reference is kept whatever it scores.
    hits = retriever.search_balanced("2:256 Allah", k=4, mode="fts", min_rerank=0.5)
    assert [h.ref for h in hits] == ["quran:2:256", "quran:49:12"]


def test_rerank_floor_drops_passages_and_reports_scores(store):
    app.state.strict = True
    app.state.retriever = Retriever(store, reranker=FakeReranker("suspicion"))
    app.state.retrieval_mode = "fts"
    app.state.llm = llm = FakeLLM('"Shun much suspicion; for lo! some suspicion is a crime." [quran:49:12]')
    with TestClient(app) as c:
        app.state.min_relevance = 0.5
        body = c.post("/ask", json={"question": "Allah", "mode": "fts"}).json()
        none = c.post("/ask", json={"question": "religion", "mode": "fts"}).json()
        app.state.min_relevance = 0.0
    assert body["sources"] and all(s["relevance"] == 0.9 for s in body["sources"])
    assert [s["ref"] for s in body["sources"]] == ["quran:49:12"]
    assert body["citation_check"]["status"] == "pass", body["citation_check"]
    assert "suspicion" in llm.seen[1]["content"] and "compulsion" not in llm.seen[1]["content"]
    # Nothing about the question scores above the floor: a fixed message, and the model isn't asked.
    assert none["sources"] == [] and none["answer"].startswith("I couldn't find anything")
