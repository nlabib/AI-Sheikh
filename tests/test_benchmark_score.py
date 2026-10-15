from tools.benchmark import score

ITEM = {"expected": "answer", "sources": ["quran:2:255"], "acceptable_sources": [], "must_mention": [], "must_not": ["forbidden"]}


def resp(answer, cited):
    return {"answer": answer, "citation_check": {"status": "pass", "cited": cited}}


def test_expected_source_is_good():
    assert score(ITEM, resp("Ayat al-Kursi [quran:2:255]", ["quran:2:255"]))["verdict"] == "good"


def test_other_cited_source_is_not_called_wrong():
    assert score(ITEM, resp("See [quran:2:256]", ["quran:2:256"]))["verdict"] == "other"


def test_no_citation_is_wrong():
    assert score(ITEM, resp("It is a verse.", []))["verdict"] == "wrong"


def test_must_not_phrase_makes_it_wrong():
    assert score(ITEM, resp("This is forbidden [quran:2:255]", ["quran:2:255"]))["verdict"] == "wrong"


def test_refusal_is_refused():
    assert score(ITEM, resp("I'm not sure.", []))["verdict"] == "refused"
