from app.embed import HashingEmbedder, build_index
from app.retrieve import Retriever


def test_fts_finds_english_arabic_and_bangla_digits(store):
    r = Retriever(store)
    assert r.search("Is there compulsion in religion?", mode="fts")[0].ref == "quran:2:256"
    assert r.search("لا إكراه في الدين", mode="fts")[0].ref == "quran:2:256"
    assert r.search("إنما الأعمال بالنيات", mode="fts")[0].ref in {"bukhari:1", "muslim:4927"}


def test_explicit_reference_ranks_first(store):
    hits = Retriever(store).search("tell me about 2:255", mode="fts")
    assert hits[0].ref == "quran:2:255" and "reference" in hits[0].matched_by


def test_hits_group_all_editions_of_a_ref(store):
    hit = Retriever(store).search("2:255", mode="fts")[0]
    assert {p.lang for p in hit.passages} == {"ar", "en"}
    assert hit.label == "Quran 2:255 (Al-Baqara)"


def test_collection_filter(store):
    hits = Retriever(store).search("الله", k=10, mode="fts", collections=["bukhari"])
    assert hits and all(h.ref.startswith("bukhari:") for h in hits)


def test_balanced_search_keeps_quran_and_hadith(store):
    hits = Retriever(store).search_balanced("الله رسول", k=4, mode="fts")
    kinds = {h.ref.split(":")[0] == "quran" for h in hits}
    assert kinds == {True, False}


def test_vector_and_hybrid_modes(store):
    emb = HashingEmbedder(dim=256)
    index = build_index(emb, store.all_ids_and_texts())
    r = Retriever(store, emb, index)
    assert r.search("compulsion in religion", mode="vector")[0].ref == "quran:2:256"
    hybrid = r.search("compulsion in religion", mode="hybrid")[0]
    assert hybrid.ref == "quran:2:256" and set(hybrid.matched_by) == {"fts", "vector"}
    only_hadith = r.search("religion", mode="vector", collections=["bukhari", "muslim"])
    assert all(h.ref.split(":")[0] in {"bukhari", "muslim"} for h in only_hadith)


def test_vector_index_roundtrip(store, tmp_path):
    from app.embed import VectorIndex

    index = build_index(HashingEmbedder(dim=64), store.all_ids_and_texts())
    index.save(tmp_path / "idx")
    loaded = VectorIndex.load(tmp_path / "idx")
    assert loaded.embedder_name == "hashing-64" and (loaded.ids == index.ids).all()


def test_muslim_numbers_also_match_abd_al_baqi_numbering(store):
    from app.store import Passage

    store.upsert([Passage("muslim:874", "muslim", "en", "eng-test", "No prayer without the Opening of the Book.",
                          "hadith-en-test", meta={"alt_number": "394a"})])
    hits = Retriever(store).search_balanced("Sahih Muslim 394", k=6, mode="fts")
    assert hits[0].ref == "muslim:874" and "reference" in hits[0].matched_by
