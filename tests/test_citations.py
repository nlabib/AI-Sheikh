from app.citations import check_answer
from app.store import Passage

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


def test_answer_from_memory_fails(store):
    report = check_answer("Islam teaches many beautiful things about patience, mercy and justice for all people.",
                          EVIDENCE, store)
    assert report.status == "fail" and "unsupported_answer" in kinds(report)


def test_uncited_reply_that_does_not_decline_fails(store):
    report = check_answer("Yes.", EVIDENCE, store)
    assert report.status == "fail" and "no_citations" in kinds(report)
    # A claim with a refusal fragment tacked on is not a refusal (seen in Arabic: "wudu is clean, I won't guess").
    report = check_answer("الوضوء هو نظيف، فلن أخمّن", EVIDENCE, store)
    assert report.status == "fail"


def test_refusal_may_explain_what_the_sources_lack(store):
    answer = ("I'm not sure. The sources provided do not include a Sahih Bukhari hadith numbered 9999, "
              "so I cannot quote one for you.")
    assert check_answer(answer, EVIDENCE, store).status == "pass"


def test_hadith_presented_as_quran_fails(store):
    report = check_answer('The Quran answers this: "Actions are judged by intentions" [bukhari:1].', EVIDENCE, store)
    assert report.status == "fail" and "misattributed" in kinds(report)
    ok = check_answer('The Prophet said: "Actions are judged by intentions" [bukhari:1].', EVIDENCE, store)
    assert "misattributed" not in kinds(ok)


def test_verse_presented_as_hadith_fails(store):
    report = check_answer('The Prophet said "There is no compulsion in religion" [quran:2:256].', EVIDENCE, store)
    assert "misattributed" in kinds(report)


def test_bangla_footnote_marks_do_not_break_a_quote(store):
    store.upsert([Passage("quran:2:256", "quran", "bn", "ben-test", "দ্বীন গ্রহণের ব্যাপারে কোন জোর-জবরদস্তি নেই [১]; "
                          "সত্য পথ সুস্পষ্ট হয়েছে ভ্রান্ত পথ থেকে [২]।", "quran-bn-test")])
    answer = 'আল্লাহ বলেন, "দ্বীন গ্রহণের ব্যাপারে কোন জোর-জবরদস্তি নেই [১]" [quran:2:256]'
    report = check_answer(answer, EVIDENCE, store)
    assert "malformed_marker" not in kinds(report) and "misquote" not in kinds(report), report.to_dict()


def test_repetition_fails(store):
    line = "There is no compulsion in religion [quran:2:256]. "
    assert check_answer(line * 3, EVIDENCE, store).status == "fail"
    assert check_answer(line * 2, EVIDENCE, store).status == "warn"


def test_bangla_inside_english_answer_fails(store):
    answer = "There is no compulsion in religion এর অনুগ্রহের উত্তর [quran:2:256]."
    report = check_answer(answer, EVIDENCE, store, question_lang="en")
    assert report.status == "fail" and "mixed_language" in kinds(report)
    assert "mixed_language" not in kinds(check_answer(answer, EVIDENCE, store))


def test_not_sure_followed_by_a_guess_is_not_a_decline(store):
    answer = "I'm not sure. The Quran requires every Muslim to pay zakat on all jewellery every single year."
    report = check_answer(answer, EVIDENCE, store)
    assert report.status == "fail" and "unsupported_answer" in kinds(report)


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
