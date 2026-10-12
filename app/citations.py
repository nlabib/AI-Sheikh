"""Checks on a generated answer's citations.

What is checked:
  * every [marker] is well-formed and names a passage that exists in the corpus
  * every marker names a passage that was actually given to the model as evidence
  * every quoted span ("..." or “...”) appears in a passage cited in the same sentence
  * each substantive sentence carries at least one citation
  * each cited sentence shares some vocabulary with what it cites (a weak signal)
  * a sentence that says "the Quran" cites a verse, and "the Prophet said" cites a hadith
  * the answer is in the question's language and doesn't repeat itself

What is NOT checked: whether the claim is a correct reading of the source. That
needs claim-level entailment checks and human review (see docs/EVALUATION.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.refs import MARKER, expand_marker, find_markers
from app.store import Store
from app.text import normalize, strip_footnotes, tokenize

_QUOTE = re.compile(r"[\"“«]([^\"”»]{12,}?)[\"”»]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।؟])\s+|\n+")
# Sentences like these do not need a source.
_EXEMPT = re.compile(
    r"(i'?m not sure|i am not sure|আমি নিশ্চিত নই|لست متأكد|allah knows best|(?:ask|consult)\w* (?:with )?(?:a |an |your )?(?:qualified |local |trusted )?"
    r"(?:scholar|imam|mufti)|not (?:found|covered)|do(?: not|n't) (?:have|address|cover|answer|include)|"
    r"won't guess|will not guess|"
    r"cannot (?:find|answer)|আল্লাহই ভালো জানেন|আলেমকে|আলেমের|উত্তর নেই|পারছি না|"
    r"والله أعلم|لا تتناول|لا تجيب|لا تتضمن|لن أخم|لا أستطيع|سؤال عالم|عالم مختص)",
    re.I,
)


@dataclass
class Issue:
    kind: str
    detail: str
    sentence: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "sentence": self.sentence}


@dataclass
class CitationReport:
    status: str  # "pass" | "warn" | "fail"
    cited: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"status": self.status, "cited": self.cited, "issues": [i.to_dict() for i in self.issues]}


FAIL_KINDS = {"nonexistent_reference", "uncited_evidence", "misquote", "unsupported_answer", "no_citations",
              "misattributed", "mixed_language", "repetition", "off_topic_answer"}


_NOT_SURE = re.compile(r"(i'?m not sure|i am not sure|আমি নিশ্চিত নই|لست متأكد|"
                       r"couldn't find anything about this|কিছু পাইনি|لم أجد شيئًا)", re.I)
# A refusal opens by saying so (system prompt rule 4), either "I'm not sure" or "the sources don't cover this".
_REFUSAL_START = re.compile(r"^\W*(?:i'?m not sure|i am not sure|আমি নিশ্চিত নই|لست متأكد)", re.I)
_TALKS_ABOUT_SOURCES = re.compile(r"(sources?|provided|উৎস|المصادر|مصادر)", re.I)
_NEGATION = re.compile(r"(\bnot?\b|n't|\bno\b|নেই|না|নয়|নাই|\bلا\b|\bليس|\bلم\b|\bلن\b)", re.I)
# Attribution words, matched on normalize()d text with quotations masked.
_SAYS_QURAN = re.compile(r"(\bqur'?an\b|\bkoran\b|القران|قران|কুরআন|কোরআন)")
_SAYS_PROPHET = re.compile(r"(\bprophet\b|messenger of allah|\ufdfa|ﷺ|رسول الله|النبي|নবী|রাসূল|রাসুল)")
_SAID = re.compile(r"(\bsaid\b|\bsays\b|\bsay\b|বলেছেন|বলেন|قال|يقول)")
_LETTERS = {"bn": re.compile(r"[\u0980-\u09FF]"), "ar": re.compile(r"[\u0600-\u06FF]"),
            "en": re.compile(r"[A-Za-z]")}


def says_not_sure(answer: str) -> bool:
    return bool(_NOT_SURE.search(answer))


def _sentences(answer: str) -> list[str]:
    """Split into sentences without breaking inside quoted source text."""
    quotes: list[str] = []

    def stash(m: re.Match) -> str:
        quotes.append(m.group(0))
        return f"\x00{len(quotes) - 1}\x00"

    masked = _QUOTE.sub(stash, answer)
    out = []
    for part in _SENTENCE_SPLIT.split(masked):
        if part and part.strip():
            out.append(re.sub(r"\x00(\d+)\x00", lambda m: quotes[int(m.group(1))], part.strip()))
    return out


def _quote_found(quote: str, texts: list[str]) -> bool:
    q = " ".join(tokenize(quote))
    if not q:
        return True
    for t in texts:
        norm = " ".join(tokenize(t))
        if q in norm:
            return True
        # Tolerate small differences (punctuation, a dropped word) on longer quotes.
        m = SequenceMatcher(None, q, norm, autojunk=False).find_longest_match(0, len(q), 0, len(norm))
        if len(q) >= 40 and m.size / len(q) >= 0.9:
            return True
    return False


def _support_text(p) -> str:
    """Passage text plus its grading metadata, which answers may also cite."""
    text = strip_footnotes(p.text)
    grades = " ".join(f"{g.get('grade', '')} {g.get('name', '')}" for g in p.meta.get("grades", []))
    return f"{text} {grades}" if grades else text


def _unquoted(sentence: str) -> str:
    return _QUOTE.sub(" ", MARKER.sub(" ", sentence))


def _misattributed(sentence: str, refs: list[str]) -> str | None:
    """"The Quran says ..." citing only hadith, or "the Prophet said ..." citing only verses."""
    # The attribution comes before the quotation; quoted text may itself mention the Quran or the Prophet.
    plain = normalize(MARKER.sub(" ", re.split(r"[\"“«]", sentence, maxsplit=1)[0]))
    quran = [r for r in refs if r.startswith("quran:")]
    says_quran, says_prophet = bool(_SAYS_QURAN.search(plain)), bool(_SAYS_PROPHET.search(plain))
    if says_quran and not says_prophet and not quran:
        return "attributed to the Quran but cites only hadith"
    if says_prophet and _SAID.search(plain) and not says_quran and quran and len(quran) == len(refs):
        return "attributed to the Prophet but cites only Quran verses"
    return None


def _foreign_letters(answer: str, lang: str) -> int:
    """Letters in a script the question wasn't written in, outside quotations. Arabic is allowed anywhere
    (verse text, names), and so is English in a Bangla or Arabic answer (book names like Sahih Muslim)."""
    plain = _unquoted(answer)
    if lang in ("en", "ar"):
        return len(_LETTERS["bn"].findall(plain))
    return 0


def _overlap(sentence: str, texts: list[str]) -> float:
    s = set(tokenize(MARKER.sub("", sentence), drop_stopwords=True))
    if not s:
        return 1.0
    pool: set[str] = set()
    for t in texts:
        pool |= set(tokenize(t, drop_stopwords=True))
    return len(s & pool) / len(s)


def check_answer(answer: str, evidence_refs: list[str], store: Store, min_overlap: float = 0.15,
                 question_lang: str | None = None, relevance: dict[str, float] | None = None,
                 min_relevance: float = 0.0) -> CitationReport:
    """relevance: how close each evidence ref is in meaning to the question (Retriever.relevance). Citing
    passages below min_relevance is flagged; an answer that cites nothing else fails."""
    answer = strip_footnotes(answer)
    evidence = set(evidence_refs)
    issues: list[Issue] = []
    cited: list[str] = []
    first = next(iter(_sentences(answer)), "")
    refusal = bool(_REFUSAL_START.search(answer)) or bool(
        _TALKS_ABOUT_SOURCES.search(first) and _NEGATION.search(first) and _EXEMPT.search(first))

    for marker in find_markers(answer):
        for ref in expand_marker(marker):
            cited.append(ref)
            if not store.ref_exists(ref):
                issues.append(Issue("nonexistent_reference", f"[{marker}] is not in the corpus"))
            elif ref not in evidence:
                issues.append(Issue("uncited_evidence", f"[{marker}] was not among the retrieved sources"))

    for bad in re.findall(r"\[([^\]\n]{1,40})\]", answer):
        if not MARKER.fullmatch(f"[{bad}]") and re.search(r"\d", bad):
            issues.append(Issue("malformed_marker", f"[{bad}] is not a valid citation marker"))

    seen_sentences: dict[str, int] = {}
    for sentence in _sentences(answer):
        refs = [r for m in find_markers(sentence) for r in expand_marker(m)]
        texts = [_support_text(p) for r in refs for p in store.by_ref(r)]
        plain = MARKER.sub("", sentence).strip()
        key = " ".join(tokenize(plain))
        if len(key.split()) >= 4:
            seen_sentences[key] = seen_sentences.get(key, 0) + 1
        if not refs:
            explains_refusal = refusal and _TALKS_ABOUT_SOURCES.search(plain) and _NEGATION.search(plain)
            if (len(tokenize(plain, drop_stopwords=True)) >= 6 and not _EXEMPT.search(plain)
                    and not explains_refusal):
                issues.append(Issue("uncited_claim", "substantive sentence without a citation", sentence))
            continue
        problem = _misattributed(sentence, refs)
        if problem:
            issues.append(Issue("misattributed", problem, sentence))
        verified_quote = False
        for quote in _QUOTE.findall(sentence):
            if _quote_found(quote, texts):
                verified_quote = True
            else:
                issues.append(Issue("misquote", f"quoted text not found in {', '.join(refs)}", sentence))
        # A sentence carrying a verbatim quote is supported; otherwise look for shared vocabulary.
        if texts and not verified_quote and _overlap(sentence, texts) < min_overlap:
            issues.append(Issue("weak_support", "little shared vocabulary with the cited source", sentence))

    if relevance and min_relevance > 0 and cited:
        off = [r for r in dict.fromkeys(cited) if r in relevance and relevance[r] < min_relevance]
        for r in off:
            issues.append(Issue("off_topic_citation", f"[{r}] does not seem to be about the question "
                                f"(similarity {relevance[r]:.2f} < {min_relevance:.2f})"))
        if off and len(off) == len(set(cited)):
            issues.append(Issue("off_topic_answer", "none of the cited passages is about the question"))

    repeated = max(seen_sentences.values(), default=0)
    if repeated >= 2:
        issues.append(Issue("repetition" if repeated >= 3 else "repeated_sentence",
                            f"the same sentence appears {repeated} times"))
    if question_lang and _foreign_letters(answer, question_lang) > 3:
        issues.append(Issue("mixed_language", f"text in another script inside a {question_lang} answer"))

    uncited_claims = any(i.kind == "uncited_claim" for i in issues)
    # A pure refusal ("I'm not sure...") is correct. A refusal followed by claims is a guess, and so is an
    # uncited answer that doesn't open by declining.
    declined = not cited and refusal and not uncited_claims
    if not cited and not declined:
        # Substantive claims with no citation at all are an answer from memory.
        kind = "unsupported_answer" if uncited_claims else "no_citations"
        issues.append(Issue(kind, "the answer makes claims but cites nothing"
                            if uncited_claims else "the answer cites nothing and doesn't say it is unsure"))
    if any(i.kind in FAIL_KINDS for i in issues):
        status = "fail"
    elif declined:
        status = "pass"  # declining to answer without sources is the correct behaviour
    elif issues or not cited:
        status = "warn"
    else:
        status = "pass"
    return CitationReport(status, list(dict.fromkeys(cited)), issues)
