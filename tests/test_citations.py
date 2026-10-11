from app.citations import check_answer

EVIDENCE = ["quran:2:256", "quran:2:255", "bukhari:1"]


def kinds(report):
    return {i.kind for i in report.issues}


def test_well_cited_answer_passes(store):
    answer = ('The Quran says "There is no compulsion in religion." [quran:2:256] '
              "It adds that the right direction is distinct from error [quran:2:256].")
    report = check_answer(answer, EVIDENCE, store)
    assert report.status == "pass", report.to_dict()
    assert report.cited == ["quran:2:256"]


def test_invented_reference_fails(store):
    report = check_answer("Fasting is good for you [quran:2:999].", EVIDENCE, store)
    assert report.status == "fail" and "nonexistent_reference" in kinds(report)


def test_real_reference_not_in_evidence_fails(store):
    report = check_answer("Purity of heart matters [quran:49:12].", EVIDENCE, store)
    assert report.status == "fail" and "uncited_evidence" in kinds(report)


def test_misquote_fails(store):
    answer = 'The verse says "There is absolutely no force allowed in any faith whatsoever" [quran:2:256].'
    report = check_answer(answer, EVIDENCE, store)
    assert "misquote" in kinds(report)


def test_arabic_quote_matches_without_harakat(store):
    answer = 'قال تعالى: "لا إكراه في الدين" [quran:2:256]'
    assert check_answer(answer, EVIDENCE, store).status == "pass"


def test_uncited_claim_warns(store):
    answer = ("There is no compulsion in religion [quran:2:256]. "
              "Scholars across all four schools unanimously hold that zakat on jewellery is obligatory.")
    report = check_answer(answer, EVIDENCE, store)
    assert report.status in {"warn", "fail"} and "uncited_claim" in kinds(report)


def test_declining_without_sources_passes(store):
    report = check_answer("The sources I was given do not address this question. Please ask a qualified scholar.",
                          EVIDENCE, store)
    assert report.status == "pass"


def test_no_citations_warns(store):
    report = check_answer("Islam teaches many beautiful things about patience, mercy and justice for all people.",
                          EVIDENCE, store)
    assert report.status == "warn" and "no_citations" in kinds(report)


def test_malformed_marker_flagged(store):
    report = check_answer("See [Q 2:256] for this [quran:2:256].", EVIDENCE, store)
    assert "malformed_marker" in kinds(report)


def test_quoted_sentence_with_period_inside_is_not_split(store):
    answer = '"Allah! There is no deity save Him, the Alive, the Eternal. Neither slumber nor sleep overtaketh Him." [quran:2:255]'
    assert check_answer(answer, EVIDENCE, store).status == "pass"


def test_not_sure_answers_pass_and_are_detected(store):
    from app.citations import says_not_sure

    for text in ["I'm not sure. The sources do not cover this.", "আমি নিশ্চিত নই।", "لست متأكدًا."]:
        assert check_answer(text, EVIDENCE, store).status == "pass"
        assert says_not_sure(text)
    assert not says_not_sure("There is no compulsion in religion [quran:2:256].")
