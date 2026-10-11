"""Checks on a generated answer's citations.

What is checked:
  * every [marker] is well-formed and names a passage that exists in the corpus
  * every marker names a passage that was actually given to the model as evidence
  * every quoted span ("..." or “...”) appears in a passage cited in the same sentence
  * each substantive sentence carries at least one citation
  * each cited sentence shares some vocabulary with what it cites (a weak signal)

What is NOT checked: whether the claim is a correct reading of the source. That
needs claim-level entailment checks and human review (see docs/EVALUATION.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.refs import MARKER, expand_marker, find_markers
from app.store import Store
from app.text import tokenize

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


FAIL_KINDS = {"nonexistent_reference", "uncited_evidence", "misquote", "unsupported_answer"}


_NOT_SURE = re.compile(r"(i'?m not sure|i am not sure|আমি নিশ্চিত নই|لست متأكد)", re.I)


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
    grades = " ".join(f"{g.get('grade', '')} {g.get('name', '')}" for g in p.meta.get("grades", []))
    return f"{p.text} {grades}" if grades else p.text


def _overlap(sentence: str, texts: list[str]) -> float:
    s = set(tokenize(MARKER.sub("", sentence), drop_stopwords=True))
    if not s:
        return 1.0
    pool: set[str] = set()
    for t in texts:
        pool |= set(tokenize(t, drop_stopwords=True))
    return len(s & pool) / len(s)


def check_answer(answer: str, evidence_refs: list[str], store: Store, min_overlap: float = 0.15) -> CitationReport:
    evidence = set(evidence_refs)
    issues: list[Issue] = []
    cited: list[str] = []

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

    for sentence in _sentences(answer):
        refs = [r for m in find_markers(sentence) for r in expand_marker(m)]
        texts = [_support_text(p) for r in refs for p in store.by_ref(r)]
        plain = MARKER.sub("", sentence).strip()
        if not refs:
            if len(tokenize(plain, drop_stopwords=True)) >= 6 and not _EXEMPT.search(plain):
                issues.append(Issue("uncited_claim", "substantive sentence without a citation", sentence))
            continue
        verified_quote = False
        for quote in _QUOTE.findall(sentence):
            if _quote_found(quote, texts):
                verified_quote = True
            else:
                issues.append(Issue("misquote", f"quoted text not found in {', '.join(refs)}", sentence))
        # A sentence carrying a verbatim quote is supported; otherwise look for shared vocabulary.
        if texts and not verified_quote and _overlap(sentence, texts) < min_overlap:
            issues.append(Issue("weak_support", "little shared vocabulary with the cited source", sentence))

    uncited_claims = any(i.kind == "uncited_claim" for i in issues)
    # A pure refusal ("I'm not sure...") is correct. A refusal followed by claims is a guess.
    declined = not cited and bool(_EXEMPT.search(answer)) and not uncited_claims
    if not cited and not declined:
        # Substantive claims with no citation at all are an answer from memory.
        kind = "unsupported_answer" if uncited_claims else "no_citations"
        issues.append(Issue(kind, "the answer makes claims but cites nothing"
                            if uncited_claims else "the answer cites nothing"))
    if any(i.kind in FAIL_KINDS for i in issues):
        status = "fail"
    elif declined:
        status = "pass"  # declining to answer without sources is the correct behaviour
    elif issues or not cited:
        status = "warn"
    else:
        status = "pass"
    return CitationReport(status, list(dict.fromkeys(cited)), issues)

