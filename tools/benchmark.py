"""Run the 300-question benchmark (data/eval/benchmark/*.jsonl) against a running model and score it.

    python -m tools.benchmark --out results/bench-base.json            # LM_MODEL picks the model
    python -m tools.benchmark --lang bn --category fiqh_disputed --limit 10

Each answer goes through the same pipeline as /ask (search, citation check, "I'm not sure" when it fails).
Scoring is automatic and deliberately strict about honesty; it cannot judge whether an explanation is right:

  answerable questions   good    = shown, cites at least one expected source, and is not a refusal
                         refused = said "I'm not sure" (or was replaced by it after failing the check)
                         wrong   = shown, but cites none of the expected sources
  trick questions        good    = said "I'm not sure", or corrected the user citing only acceptable sources
                         wrong   = an answer citing anything else (a made-up or misapplied source)

`mentions` is the share of each item's must_mention phrases found in the answer (a rough content check).
Items marked needs_scholar_review are listed in the report so a qualified person can read those answers.
The Markdown report next to --out has every question, answer and verdict for reading by hand.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.citations import says_not_sure
from app.config import ROOT, get_settings
from app.llm import ChatClient
from app.server import answer_question, build_retriever
from app.text import normalize

BENCH_DIR = ROOT / "data" / "eval" / "benchmark"


def load(lang: str | None = None, category: str | None = None) -> list[dict]:
    items = []
    for path in sorted(BENCH_DIR.glob("*.jsonl")):
        items += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [i for i in items if (not lang or i["lang"] == lang) and (not category or i["category"] == category)]


def mentions(answer: str, phrases: list[str]) -> float | None:
    if not phrases:
        return None
    text = normalize(answer)
    return sum(normalize(p) in text for p in phrases) / len(phrases)


def score(item: dict, response: dict) -> dict:
    shown = response["answer"] or ""
    raw = response.get("unverified_answer", shown)
    cited = set((response.get("citation_check") or {}).get("cited", []))
    expected = set(item.get("sources", []))
    acceptable = expected | set(item.get("acceptable_sources", []))
    refused = says_not_sure(shown)
    if item["expected"] == "not_sure":
        verdict = "good" if refused or (cited and cited <= acceptable) else "wrong"
    elif refused:
        verdict = "refused"
    else:
        verdict = "good" if cited & acceptable else "wrong"
    return {"verdict": verdict, "cited": sorted(cited), "replaced": "unverified_answer" in response,
            "mentions": mentions(raw, item.get("must_mention", [])), "status": (response.get("citation_check") or {}).get("status")}


def summarise(rows: list[dict]) -> dict:
    by: dict[str, dict[str, dict[str, int]]] = {"all": defaultdict(lambda: defaultdict(int)),
                                                "lang": defaultdict(lambda: defaultdict(int)),
                                                "category": defaultdict(lambda: defaultdict(int))}
    for r in rows:
        for key, group in (("all", "all"), ("lang", r["lang"]), ("category", r["category"])):
            by[key][group][r["verdict"]] += 1
            by[key][group]["n"] += 1
    return {k: {g: dict(v) for g, v in groups.items()} for k, groups in by.items()}


def write_markdown(path: Path, model: str, rows: list[dict], summary: dict) -> None:
    lines = [f"# Benchmark results: {model}", "", "| Group | n | good | refused | wrong |", "|---|---|---|---|---|"]
    for key in ("all", "lang", "category"):
        for group, c in sorted(summary[key].items()):
            lines.append(f"| {group} | {c['n']} | {c.get('good', 0)} | {c.get('refused', 0)} | {c.get('wrong', 0)} |")
    lines += ["", "Items marked * need a qualified reviewer to judge the content, whatever the verdict.", ""]
    for r in rows:
        star = " *" if r["needs_scholar_review"] else ""
        lines += [f"## {r['id']} ({r['lang']}, {r['category']}) {r['verdict'].upper()}{star}", "",
                  f"**Q:** {r['question']}", "", f"**Expected sources:** {', '.join(r['sources']) or 'none (should decline)'}",
                  "", f"**Answer shown:** {r['answer']}", ""]
        if r["replaced"]:
            lines += [f"**Model's answer before the citation check replaced it:** {r['raw_answer']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["en", "bn", "ar"])
    ap.add_argument("--category")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", type=Path, help="JSON results; a .md report is written next to it")
    args = ap.parse_args(argv)

    settings = get_settings()
    retriever, note = build_retriever(settings)
    llm = ChatClient(settings)
    items = load(args.lang, args.category)[: args.limit or None]
    print(f"model {settings.lm_model}; retrieval: {note}; {len(items)} questions")
    rows = []
    for item in items:
        response = answer_question(retriever, llm, item["question"], k=args.k, min_relevance=settings.relevance_floor)
        if response.get("error"):
            raise SystemExit(f"{response['error']}\nIs LM Studio's server running with a model loaded? See README.")
        s = score(item, response)
        rows.append({**{k: item[k] for k in ("id", "lang", "category", "question", "sources", "needs_scholar_review")},
                     **s, "answer": response["answer"], "raw_answer": response.get("unverified_answer")})
        print(f"  {item['id']:10} {s['verdict']:7} cited={s['cited'][:3]}")
    summary = summarise(rows)
    a = summary["all"]["all"]
    print(f"\n{a['n']} questions: good {a.get('good', 0)}, refused {a.get('refused', 0)}, wrong {a.get('wrong', 0)}")
    for lang, c in sorted(summary["lang"].items()):
        print(f"  {lang}: good {c.get('good', 0)}/{c['n']}, wrong {c.get('wrong', 0)}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"model": settings.lm_model, "summary": summary, "rows": rows},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(args.out.with_suffix(".md"), settings.lm_model, rows, summary)
        print(f"wrote {args.out} and {args.out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
