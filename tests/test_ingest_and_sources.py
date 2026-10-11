import pytest

from app.store import Store
from tools.ingest import _alt_number, hadith_passages, quran_passages
from tools.sources import BY_ID, LICENCES, SHA256, SOURCES


def test_every_source_is_pinned_and_licensed():
    for s in SOURCES:
        assert len(SHA256[s.edition]) == 64
        assert s.licence in LICENCES
        assert s.commit in s.urls[0]


def test_copyrighted_translations_are_never_trainable():
    for s in SOURCES:
        if s.licence in ("modern-translation", "kfgqpc-free"):
            assert not s.allow_training, s.source_id
    assert BY_ID["quran-ar-tanzil"].allow_training and BY_ID["quran-en-pickthall"].allow_training


def test_quran_must_be_complete():
    with pytest.raises(ValueError):
        list(quran_passages(BY_ID["quran-ar-tanzil"], {"quran": [{"chapter": 1, "verse": 1, "text": "x"}]}))


def test_hadith_parsing_skips_empty_and_keeps_grades(tmp_path):
    data = {
        "metadata": {"sections": {"1": "Revelation"}},
        "hadiths": [
            {"hadithnumber": 1, "arabicnumber": 1907.01, "text": "نص", "grades": [{"name": "X", "grade": "Sahih"}],
             "reference": {"book": 1, "hadith": 1}},
            {"hadithnumber": 2, "arabicnumber": 2, "text": "  ", "grades": [], "reference": {"book": 1, "hadith": 2}},
            {"hadithnumber": 402.2, "arabicnumber": 3, "text": "نص آخر", "grades": [], "reference": {}},
        ],
    }
    rows = list(hadith_passages(BY_ID["muslim-ar"], data))
    assert [r.ref for r in rows] == ["muslim:1", "muslim:402.2"]
    assert rows[0].meta["alt_number"] == "1907a" and rows[0].meta["grades"][0]["grade"] == "Sahih"
    store = Store(tmp_path / "db.sqlite")
    store.upsert(rows)
    store.upsert(rows)  # re-ingesting replaces rather than duplicates
    assert store.count() == 2


def test_alt_number_formatting():
    assert _alt_number(1907.01) == "1907a"
    assert _alt_number("8.02") == "8b"
    assert _alt_number(22) == "22"
