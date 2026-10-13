"""Check hand-written (or model-drafted) training answers against the corpus and turn them into SFT examples.

    python -m tools.grounded_sft drafts.jsonl --out data/sft/generated/en_part1.jsonl

Each input line is a draft:
    {"id": "...", "lang": "en|bn|ar", "kind": "multi_source|partial|disputed|distractor_decline|concise",
     "question": "...", "evidence": ["quran:2:183", "bukhari:1903", ...], "answer": "..."}

`evidence` is what the model is shown, in order; it may include passages that are not about the question, so the
model learns to leave them out. Only training-allowed editions (tools/sources.py) are put in the prompt, so a draft
is rejected when:
  * an evidence ref has no training-allowed text;
  * the answer quotes anything that is not word for word in a training-allowed edition of the cited ref;
  * the answer fails the citation check (app/citations.py) in strict mode, or cites outside the evidence;
  * a "multi_source" or "disputed" answer cites fewer than two sources, or a decline cites any;
  * it cites a ref that the benchmark or data/eval/gold.jsonl uses, which would leak the evaluation.
Rejected drafts are listed with the reason; nothing is fixed silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.citations import _QUOTE, MIN_QUOTE_CHARS, _quote_found, check_answer, says_not_sure
from app.config import ROOT, get_settings
from app.prompts import build_messages
from app.refs import MARKER, expand_marker, find_markers
from app.retrieve import Hit
from app.store import Store
from app.text import strip_footnotes

KINDS = {"multi_source", "partial", "disputed", "distractor_decline", "concise"}
EVAL_FILES = [ROOT / "data" / "eval" / "gold.jsonl", *sorted((ROOT / "data" / "eval" / "benchmark").glob("*.jsonl"))]


def eval_refs() -> set[str]:
    """Every ref the evaluation sets expect; training answers must not cite them."""
    refs: set[str] = set()
    for path in EVAL_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                refs.update(item.get("relevant", []), item.get("sources", []), item.get("acceptable_sources", []))
    return refs


def eval_questions() -> set[str]:
    out = set()
    for path in EVAL_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                out.add((item.get("question") or item.get("query") or "").strip())
    return out


def trainable_hit(store: Store, ref: str) -> Hit | None:
    passages = [p for p in store.by_ref(ref) if p.allow_training]
    return Hit(ref, 1.0, ["grounded"], passages) if passages else None


def check_draft(d: dict, store: Store, banned_refs: set[str], banned_questions: set[str]) -> tuple[dict | None, str]:
    missing = {"lang", "kind", "question", "evidence", "answer"} - d.keys()
    if missing:
        return None, f"missing keys {sorted(missing)}"
    if d["kind"] not in KINDS:
        return None, f"unknown kind {d['kind']}"
    if d["question"].strip() in banned_questions:
        return None, "question is in an evaluation set"
    hits = []
    for ref in d["evidence"]:
        hit = trainable_hit(store, ref)
        if hit is None:
            return None, f"{ref} has no training-allowed text"
        hits.append(hit)
    answer = d["answer"].strip()
    cited = list(dict.fromkeys(r for m in find_markers(answer) for r in expand_marker(m)))
    leaked = [r for r in cited if r in banned_refs]
    if leaked:
        return None, f"cites evaluation refs {leaked}"
    report = check_answer(answer, d["evidence"], store, question_lang=d["lang"])
    if report.status == "fail":
        return None, "citation check failed: " + "; ".join(f"{i.kind}: {i.detail}" for i in report.issues)
    if d["kind"] == "distractor_decline":
        if cited or not says_not_sure(answer):
            return None, "a decline must say it is not sure and cite nothing"
    elif d["kind"] in ("multi_source", "disputed") and len(cited) < 2:
        return None, f"{d['kind']} answer cites {len(cited)} source(s)"
    elif not cited:
        return None, "answer cites nothing"
    if d["kind"] == "partial" and not says_not_sure(answer):
        return None, "a partial answer must say what it is not sure about"
    # Quotes must come from the text the model is shown, not from a translation it never sees.
    shown = {h.ref: [strip_footnotes(p.text) for p in h.passages] for h in hits}
    for sentence in answer.split("\n"):
        refs = [r for m in MARKER.findall(sentence) for r in expand_marker(m)]
        for quote in _QUOTE.findall(sentence):
            if len(quote.strip()) >= MIN_QUOTE_CHARS and not _quote_found(quote, [t for r in refs for t in shown.get(r, [])]):
                return None, f"quote not in the training-allowed text of {refs or 'any cited ref'}: {quote[:60]}"
    messages = build_messages(d["question"], hits)
    messages.append({"role": "assistant", "content": answer})
    example = {"messages": messages,
               "meta": {"kind": d["kind"], "lang": d["lang"], "review_status": "generated", "id": d.get("id"),
                        "sources": sorted({p.source_id for h in hits for p in h.passages})}}
    return example, "ok"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drafts", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="write accepted examples here (JSONL)")
    ap.add_argument("--db", type=Path)
    args = ap.parse_args(argv)

    store = Store(args.db or get_settings().db_path)
    banned_refs, banned_questions = eval_refs(), eval_questions()
    accepted, rejected = [], 0
    for path in args.drafts:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            d = json.loads(line)
            example, why = check_draft(d, store, banned_refs, banned_questions)
            if example is None:
                rejected += 1
                print(f"REJECT {path.name}:{n} {d.get('id', '')}: {why}")
            else:
                accepted.append(example)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for e in accepted:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    kinds: dict[str, int] = {}
    for e in accepted:
        kinds[e["meta"]["kind"]] = kinds.get(e["meta"]["kind"], 0) + 1
    print(f"accepted {len(accepted)}, rejected {rejected}; by kind: {kinds}", file=sys.stderr)
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
