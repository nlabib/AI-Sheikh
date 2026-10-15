from app.refs import find_impossible_named_refs
from tests.test_server import FakeLLM, client_with


def test_verses_named_in_words_past_the_end_of_the_surah():
    assert find_impossible_named_refs("What is verse 8 of Surah Al-Fatiha?") == ["quran:1:8"]
    assert find_impossible_named_refs("What is the 8th ayah of Surah al-Fatiha?") == ["quran:1:8"]
    assert find_impossible_named_refs("সূরা ফাতিহার ৮ নম্বর আয়াতের অর্থ কী?") == ["quran:1:8"]
    assert find_impossible_named_refs("ما تفسير الآية السابعة من سورة الناس؟") == ["quran:114:7"]
    assert find_impossible_named_refs("What does Surah 115 say?") == ["quran:115"]


def test_real_named_verses_are_left_alone():
    assert find_impossible_named_refs("Explain ayah 6 of surah an-nas") == []
    assert find_impossible_named_refs("ما تفسير الآية السادسة من سورة الناس؟") == []
    assert find_impossible_named_refs("What does Surah Al-Ikhlas say?") == []
    assert find_impossible_named_refs("Is fajr at 5:30 in winter?") == []


def test_ask_explains_why_the_reference_cannot_exist(store):
    llm = FakeLLM("should not be called")
    with client_with(store, llm) as c:
        verse = c.post("/ask", json={"question": "What is verse 8 of Surah Al-Fatiha?"}).json()
        surah = c.post("/ask", json={"question": "What is the text of Quran 115:1?"}).json()
        hadith = c.post("/ask", json={"question": "Quote Sahih Bukhari hadith number 99999."}).json()
        bn = c.post("/ask", json={"question": "সূরা ফাতিহার ৮ নম্বর আয়াতের অর্থ কী?"}).json()
    assert llm.seen is None
    assert verse["missing_refs"] == ["quran:1:8"] and "Surah Al-Faatiha (1) has 7 ayat." in verse["answer"]
    assert "The Quran has 114 surahs." in surah["answer"]
    assert hadith["missing_refs"] == ["bukhari:99999"] and "is numbered up to" in hadith["answer"]
    assert "৭টি আয়াত আছে" in bn["answer"]
