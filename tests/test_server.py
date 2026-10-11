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
