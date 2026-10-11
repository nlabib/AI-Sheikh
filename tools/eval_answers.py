"""End-to-end answer check against a running model (LM Studio or any OpenAI-compatible server).

    python -m tools.eval_answers --limit 20
    LM_MODEL=islamic-qwen-lora python -m tools.eval_answers --out results/answers-finetuned.json

Run it once with the base model and once with your fine-tuned one, then compare.
For each gold question it records the citation-check status and whether the
answer cites one of the expected references. It also asks questions the corpus
cannot answer (data/eval/unanswerable.jsonl: misattributed sayings, verses and
hadith numbers that don't exist) and counts how often the model says "I'm not
sure" instead of making something up. It cannot judge whether the
explanation is theologically right; read a sample of answers yourself.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.citations import check_answer, says_not_sure
from app.config import get_settings
from app.llm import ChatClient, LLMError
from app.prompts import build_messages
from app.server import build_retriever
from tools.evaluate import DEFAULT_GOLD, load_gold

DEFAULT_UNANSWERABLE = DEFAULT_GOLD.parent / "unanswerable.jsonl"


def ask(retriever, llm, query: str, k: int):
    mode = "hybrid" if retriever.index is not None else "fts"
    hits = retriever.search_balanced(query, k=k, mode=mode)
    try:
        answer = llm.chat(build_messages(query, hits))
    except LLMError as e:
        raise SystemExit(f"{e}\nIs LM Studio's server running with a model loaded? See README.")
    return hits, answer, check_answer(answer, [h.ref for h in hits], retriever.store)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--unanswerable", type=Path, default=DEFAULT_UNANSWERABLE)
    ap.add_argument("--limit", type=int, default=0, help="only the first N answerable questions")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    settings = get_settings()
    retriever, mode_note = build_retriever(settings)
    llm = ChatClient(settings)
    gold = load_gold(args.gold)[: args.limit or None]
    trick = load_gold(args.unanswerable) if args.unanswerable.exists() else []
    print(f"model {settings.lm_model} at {settings.lm_base_url}; retrieval: {mode_note}; "
          f"{len(gold)} answerable + {len(trick)} unanswerable questions")

    rows, status = [], Counter()
    for item in gold:
        hits, answer, report = ask(retriever, llm, item["query"], args.k)
        cites_expected = bool(set(report.cited) & set(item["relevant"]))
        retrieved_expected = bool({h.ref for h in hits} & set(item["relevant"]))
        not_sure = says_not_sure(answer)
        status[report.status] += 1
        status["cites_expected"] += cites_expected
        status["retrieved_expected"] += retrieved_expected
        status["not_sure_answerable"] += not_sure
        rows.append({"id": item["id"], "query": item["query"], "answer": answer, "citation_check": report.to_dict(),
                     "cites_expected": cites_expected, "retrieved_expected": retrieved_expected,
                     "said_not_sure": not_sure})
        print(f"  {item['id']:8} {report.status:5} cites_expected={cites_expected} not_sure={not_sure}")

    for item in trick:
        _, answer, report = ask(retriever, llm, item["query"], args.k)
        not_sure = says_not_sure(answer)
        status["not_sure_unanswerable"] += not_sure
        status["fabricated"] += (not not_sure) and bool(report.cited)
        rows.append({"id": item["id"], "query": item["query"], "answer": answer, "citation_check": report.to_dict(),
                     "said_not_sure": not_sure})
        print(f"  {item['id']:8} not_sure={not_sure} cited={report.cited[:3]}")

    n, u = len(gold), len(trick)
    print(f"\nanswerable: pass {status['pass']}/{n}, warn {status['warn']}/{n}, fail {status['fail']}/{n}; "
          f"expected source retrieved {status['retrieved_expected']}/{n}, cited {status['cites_expected']}/{n}; "
          f"said not sure {status['not_sure_answerable']}/{n}")
    if u:
        print(f"unanswerable: said not sure {status['not_sure_unanswerable']}/{u}; "
              f"answered with citations anyway {status['fabricated']}/{u}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"model": settings.lm_model, "summary": dict(status), "rows": rows},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
