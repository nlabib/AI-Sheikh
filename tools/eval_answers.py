"""End-to-end answer check against a running model (LM Studio or any OpenAI-compatible server).

    python -m tools.eval_answers --limit 20
    LM_MODEL=islamic-qwen-lora python -m tools.eval_answers --out results/answers-finetuned.json

Run it once with the base model and once with your fine-tuned one, then compare.
For each gold question it records the citation-check status and whether the
answer cites one of the expected references. It cannot judge whether the
explanation is theologically right; read a sample of answers yourself.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.citations import check_answer
from app.config import get_settings
from app.llm import ChatClient, LLMError
from app.prompts import build_messages
from app.server import build_retriever
from tools.evaluate import DEFAULT_GOLD, load_gold


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    settings = get_settings()
    retriever, mode_note = build_retriever(settings)
    llm = ChatClient(settings)
    gold = load_gold(args.gold)[: args.limit or None]
    print(f"model {settings.lm_model} at {settings.lm_base_url}; retrieval: {mode_note}; {len(gold)} questions")

    rows, status = [], Counter()
    for item in gold:
        mode = "hybrid" if retriever.index is not None else "fts"
        hits = retriever.search_balanced(item["query"], k=args.k, mode=mode)
        try:
            answer = llm.chat(build_messages(item["query"], hits))
        except LLMError as e:
            raise SystemExit(f"{e}\nIs LM Studio's server running with a model loaded? See README.")
        report = check_answer(answer, [h.ref for h in hits], retriever.store)
        cites_expected = bool(set(report.cited) & set(item["relevant"]))
        retrieved_expected = bool({h.ref for h in hits} & set(item["relevant"]))
        status[report.status] += 1
        status["cites_expected"] += cites_expected
        status["retrieved_expected"] += retrieved_expected
        rows.append({"id": item["id"], "query": item["query"], "answer": answer, "citation_check": report.to_dict(),
                     "cites_expected": cites_expected, "retrieved_expected": retrieved_expected})
        print(f"  {item['id']:8} {report.status:5} cites_expected={cites_expected}")

    n = len(gold)
    print(f"\npass {status['pass']}/{n}, warn {status['warn']}/{n}, fail {status['fail']}/{n}; "
          f"expected source retrieved {status['retrieved_expected']}/{n}, cited {status['cites_expected']}/{n}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"model": settings.lm_model, "summary": dict(status), "rows": rows},
                                       ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
