from fastapi.testclient import TestClient

from app.refs import MissingRef, find_impossible_refs
from app.retrieve import Retriever
from app.server import app
from tests.test_server import FakeLLM


def test_references_that_cannot_exist_in_three_languages():
    assert find_impossible_refs("What is verse 8 of Surah Al-Fatiha?") == [MissingRef("ayah", 1, 8)]
    assert find_impossible_refs("What is the 8th ayah of Surah al-Fatiha?") == [MissingRef("ayah", 1, 8)]
    assert find_impossible_refs("সূরা ফাতিহার ৮ নম্বর আয়াতের অর্থ কী?") == [MissingRef("ayah", 1, 8)]
    assert find_impossible_refs("ما تفسير الآية السابعة من سورة الناس؟") == [MissingRef("ayah", 114, 7)]
    assert find_impossible_refs("ما نص الآية ٢:٣٠٠؟") == [MissingRef("ayah", 2, 300)]
    assert find_impossible_refs("What does Surah 115 say?") == [MissingRef("surah", 115)]


def test_real_references_and_times_of_day_are_left_alone():
    assert find_impossible_refs("What does Quran 2:255 say?") == []
    assert find_impossible_refs("Explain ayah 6 of surah an-nas") == []
    assert find_impossible_refs("ما تفسير الآية السادسة من سورة الناس؟") == []
    assert find_impossible_refs("Is fajr at 5:30 in winter?") == []  # not about the Quran
    assert find_impossible_refs("What did the Prophet say about anger?") == []


def test_ask_answers_impossible_references_without_the_model(store):
    llm = FakeLLM("should not be called")
    app.state.strict, app.state.retriever, app.state.retrieval_mode, app.state.llm = True, Retriever(store), "fts", llm
    with TestClient(app) as c:
        verse = c.post("/ask", json={"question": "What is verse 8 of Surah Al-Fatiha?"}).json()
        hadith = c.post("/ask", json={"question": "Quote Sahih Bukhari hadith number 99999."}).json()
        real = c.post("/ask", json={"question": "What does Quran 2:256 say?"}).json()
    assert verse["answer"].startswith("I'm not sure") and "has 7 ayat, so there is no ayah 8" in verse["answer"]
    assert "no hadith numbered 99999" in hadith["answer"]
    assert verse["sources"] == [] and hadith["sources"] == []
    assert real["sources"]  # a real reference still goes through retrieval and the model
