from fastapi.testclient import TestClient

from app.llm import LLMError
from app.retrieve import Retriever
from app.server import app


class FakeLLM:
    model = "fake"

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.seen = reply, error, None

    def chat(self, messages, **_):
        self.seen = messages
        if self.error:
            raise LLMError(self.error)
        return self.reply


def client_with(store, llm, strict=True):
    app.state.strict = strict
    app.state.retriever = Retriever(store)
    app.state.retrieval_mode = "fts"
    app.state.llm = llm
    return TestClient(app)


def test_ask_returns_answer_sources_and_citation_check(store):
    llm = FakeLLM('"There is no compulsion in religion." [quran:2:256]')
    with client_with(store, llm) as c:
        body = c.post("/ask", json={"question": "Is there compulsion in religion?"}).json()
    assert body["answer"] == '"There is no compulsion in religion." [quran:2:256]'
    assert body["citation_check"]["status"] == "pass"
    assert "quran:2:256" in [s["ref"] for s in body["sources"]]
    assert "[quran:2:256]" in llm.seen[1]["content"]  # evidence was in the prompt


def test_hallucinated_citation_is_replaced_with_not_sure(store):
    with client_with(store, FakeLLM("Compulsion is forbidden [quran:3:999].")) as c:
        body = c.post("/ask", json={"question": "compulsion in religion"}).json()
    assert body["citation_check"]["status"] == "fail"
    assert body["answer"].startswith("I'm not sure")
    assert body["unverified_answer"] == "Compulsion is forbidden [quran:3:999]."


def test_not_sure_is_in_the_question_language(store):
    with client_with(store, FakeLLM("জোর নেই [quran:3:999]।")) as c:
        body = c.post("/ask", json={"question": "দ্বীনের ব্যাপারে কি জোর জবরদস্তি আছে?"}).json()
    assert body["answer"].startswith("আমি নিশ্চিত নই")


def test_strict_mode_can_be_turned_off(store):
    with client_with(store, FakeLLM("Compulsion is forbidden [quran:3:999]."), strict=False) as c:
        body = c.post("/ask", json={"question": "compulsion in religion"}).json()
    assert body["answer"] == "Compulsion is forbidden [quran:3:999]." and "unverified_answer" not in body


def test_no_sources_means_not_sure(store):
    with client_with(store, FakeLLM("should not be called")) as c:
        body = c.post("/ask", json={"question": "zzzz qqqq"}).json()
    assert body["answer"].startswith("I'm not sure") and body["sources"] == []


def test_ask_still_returns_sources_when_llm_is_down(store):
    with client_with(store, FakeLLM(error="connection refused")) as c:
        body = c.post("/ask", json={"question": "compulsion in religion"}).json()
    assert body["answer"] is None and "connection refused" in body["error"] and body["sources"]


def test_ref_search_and_health(store):
    with client_with(store, FakeLLM("x")) as c:
        assert c.get("/ref/quran:2:255").json()["texts"]
        assert c.get("/ref/quran:2:999").status_code == 404
        assert c.get("/search", params={"q": "إنما الأعمال بالنيات"}).json()["hits"]
        assert c.get("/health").json()["passages"] == store.count()


def test_chat_client_strips_qwen_thinking(monkeypatch):
    import httpx

    from app.config import Settings
    from app.llm import ChatClient

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "<think>plan</think>\nAnswer [quran:1:1]"}}]}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: Resp())
    assert ChatClient(Settings()).chat([]) == "Answer [quran:1:1]"


def test_greeting_gets_an_introduction_without_the_model(store):
    llm = FakeLLM("I am Qasim, a Muslim assistant.")
    with client_with(store, llm) as c:
        body = c.post("/ask", json={"question": "Assalamu alaikum, who are you?"}).json()
    assert body["answer"].startswith("Wa alaikum assalam. I answer questions from the Quran")
    assert llm.seen is None


def test_off_topic_passages_are_dropped_and_off_topic_citations_fail(store):
    from app.embed import HashingEmbedder, build_index

    emb = HashingEmbedder(dim=256)
    retriever = Retriever(store, emb, build_index(emb, store.all_ids_and_texts()))
    question = "Is there compulsion in religion?"
    rel = retriever.relevance(question, ["quran:2:256", "bukhari:1"])
    assert rel["quran:2:256"] > rel["bukhari:1"]
    with client_with(store, FakeLLM("should not be called")) as c:
        app.state.retriever, app.state.min_relevance = retriever, 0.99
        body = c.post("/ask", json={"question": "what is the weather in Dhaka today"}).json()
        app.state.min_relevance = 0.0
    assert body["sources"] == [] and body["answer"].startswith("I couldn't find anything")

    from app.citations import check_answer
    report = check_answer('"Actions are judged by intentions" [bukhari:1].', ["quran:2:256", "bukhari:1"], store,
                          relevance=rel, min_relevance=(rel["quran:2:256"] + rel["bukhari:1"]) / 2)
    assert report.status == "fail" and {"off_topic_citation", "off_topic_answer"} <= {i.kind for i in report.issues}


def test_bangla_full_stop_is_tidied_in_english_answers(store):
    with client_with(store, FakeLLM('"There is no compulsion in religion." [quran:2:256]।')) as c:
        body = c.post("/ask", json={"question": "Is there compulsion in religion?"}).json()
    assert "।" not in body["answer"]


def test_e5_floor_fills_k_from_later_candidates(store):
    from app.retrieve import Hit
    from app.server import answer_question

    refs = ["quran:2:255", "bukhari:1", "quran:2:256", "quran:1:1", "quran:1:2", "quran:1:3"]

    class FloorRetriever(Retriever):
        def search_balanced(self, query, k, **_):
            return [Hit(r, 1.0, ["fts"]) for r in refs[:k]]

        def relevance(self, query, rs):
            return {r: 0.1 if r in refs[:2] else 0.9 for r in rs}

    llm = FakeLLM('"There is no compulsion in religion." [quran:2:256]')
    body = answer_question(FloorRetriever(store), llm, "Is there compulsion in religion?", k=2, min_relevance=0.5)
    assert [s["ref"] for s in body["sources"]] == ["quran:2:256", "quran:1:1"]


def test_nonexistent_reference_is_refused_without_asking_the_model(store):
    llm = FakeLLM("It says something [quran:2:256].")
    with client_with(store, llm) as c:
        body = c.post("/ask", json={"question": "What does Quran 2:999 say?"}).json()
    assert llm.seen is None  # the model was never called
    assert body["missing_refs"] == ["quran:2:999"]
    assert "2:999" in body["answer"] and "won't guess" in body["answer"]


def test_existing_reference_is_not_refused(store):
    llm = FakeLLM('"There is no compulsion in religion." [quran:2:256]')
    with client_with(store, llm) as c:
        body = c.post("/ask", json={"question": "What does 2:256 say?"}).json()
    assert "missing_refs" not in body and llm.seen is not None


def test_out_of_range_chapter_is_refused(store):
    with client_with(store, FakeLLM("x")) as c:
        body = c.post("/ask", json={"question": "What is the text of Quran 115:1?"}).json()
    assert body["missing_refs"] == ["quran:115:1"]


def test_clock_times_are_not_verse_references(store):
    llm = FakeLLM('"There is no compulsion in religion." [quran:2:256]')
    with client_with(store, llm) as c:
        for q in ("Can I pray at 1:30 pm?", "Is there compulsion in religion? Asking at 10:45"):
            body = c.post("/ask", json={"question": q}).json()
            assert "missing_refs" not in body, q
