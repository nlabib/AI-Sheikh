from app.refs import MARKER, find_explicit_refs, parse_ref
from app.retrieve import Retriever
from app.store import Passage
from tools.ingest import chunk_text, tafsir_passages
from tools.sources import BY_ID


def test_tafsir_and_new_hadith_markers_and_labels():
    text = "a [quran:2:255] b [ibnkathiren:2:255.2] c [jalalayn:1:1] d [malik:12] e [bukhari:402.2]"
    assert MARKER.findall(text) == ["quran:2:255", "ibnkathiren:2:255.2", "jalalayn:1:1", "malik:12", "bukhari:402.2"]
    assert parse_ref("ibnkathiren:2:255.2").label == "Tafsir Ibn Kathir (English, abridged) on 2:255 (part 2)"
    assert parse_ref("qudsi:3").label == "Forty Hadith Qudsi 3"
    assert find_explicit_refs("Muwatta Malik 12") == ["malik:12"]
    assert find_explicit_refs("What does Ibn Kathir say on 2:255?")[:2] == ["quran:2:255", "ibnkathir:2:255"]


def test_long_commentary_is_chunked_without_losing_text():
    text = "\n".join(["Sentence one is here. " * 40, "Short paragraph.", "Another. " * 300])
    chunks = chunk_text(text, size=500)
    assert all(len(c) <= 1000 for c in chunks) and len(chunks) > 3
    assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")


def test_repeated_group_commentary_is_kept_once():
    src = BY_ID["tafsir-ibnkathir-en"]
    files = {n: [] for n in range(1, 115)}
    files[2] = [{"surah": "2", "ayah": "1", "text": "On 1 to 3 [[note]]."},
                {"surah": "2", "ayah": "2", "text": "On 1 to 3 [[note]]."},
                {"surah": "2", "ayah": "3", "text": "On 1 to 3 [[note]]."},
                {"surah": "2", "ayah": "4", "text": "On 4."}]
    ps = list(tafsir_passages(src, files))
    assert [(p.ref, p.meta["ayah_to"], p.text) for p in ps] == [("ibnkathiren:2:1", 3, "On 1 to 3 ."),
                                                                ("ibnkathiren:2:4", 4, "On 4.")]


def test_search_only_translations_are_not_shown_and_tafsir_gets_a_slot(store):
    store.upsert([
        Passage("quran:2:256", "quran", "en", "eng-ummmuhammad", "There shall be no compulsion in religion",
                "quran-en-sahih", False, {"search_only": True}),
        Passage("ibnkathiren:2:256", "ibnkathiren", "en", "en-tafisr-ibn-kathir",
                "No compulsion in religion means do not force anyone to become Muslim.", "tafsir-ibnkathir-en",
                False, {"surah": 2, "ayah": 256, "ayah_to": 256, "part": 1, "parts": 1}),
    ])
    hits = Retriever(store).search_balanced("Is there compulsion in religion?", k=4, mode="fts")
    refs = [h.ref for h in hits]
    assert "quran:2:256" in refs and "ibnkathiren:2:256" in refs
    verse = next(h for h in hits if h.ref == "quran:2:256")
    assert "eng-ummmuhammad" not in [p.edition for p in verse.passages]
    assert len({p.lang for p in verse.passages}) == len(verse.passages)
