"""Pick MIN_RELEVANCE: the E5 similarity below which a passage is treated as not about the question.

    python -m tools.calibrate_relevance

It compares two groups of (question, passage) pairs from data/eval/gold.jsonl:
  * on topic: each question with the passages marked relevant to it;
  * off topic: each question with another question's relevant passages, plus a few questions the corpus
    has nothing on (weather, programming) with whatever retrieval returns for them.
It prints both distributions and suggests the floor that keeps 95% of the on-topic pairs, with the share of
off-topic pairs that floor removes. Set it with MIN_RELEVANCE=<value> when starting the server.
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from app.config import get_settings
from app.server import build_retriever
from tools.evaluate import DEFAULT_GOLD, load_gold

OFF_TOPIC_QUESTIONS = [
    "What's the weather in Dhaka today?",
    "Write me a Python function to sort a list.",
    "Who won the football world cup in 2022?",
    "What is the capital of Bangladesh?",
    "বাংলাদেশের রাজধানী কী?",
    "আজ ঢাকার আবহাওয়া কেমন?",
    "ما هي عاصمة فرنسا؟",
    "كيف أطبخ الأرز؟",
]


def percentiles(values: list[float]) -> str:
    p = np.percentile(values, [5, 25, 50, 75, 95])
    return " ".join(f"p{q}={v:.3f}" for q, v in zip((5, 25, 50, 75, 95), p))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--keep", type=float, default=0.95, help="share of on-topic pairs the floor must keep")
    args = ap.parse_args(argv)

    retriever, note = build_retriever(get_settings())
    if retriever.index is None:
        raise SystemExit(f"needs the E5 vector index ({note}); run python -m tools.index first")
    gold = load_gold(args.gold)
    rng = random.Random(0)

    on, off = [], []
    for item in gold:
        on += retriever.relevance(item["query"], item["relevant"]).values()
        other = rng.choice([g for g in gold if not set(g["relevant"]) & set(item["relevant"])])
        off += retriever.relevance(item["query"], other["relevant"]).values()
    for q in OFF_TOPIC_QUESTIONS:
        hits = retriever.search_balanced(q, k=6)
        off += retriever.relevance(q, [h.ref for h in hits]).values()

    floor = float(np.percentile(on, 100 * (1 - args.keep)))
    print(f"on topic  ({len(on)} pairs): {percentiles(on)}")
    print(f"off topic ({len(off)} pairs): {percentiles(off)}")
    print(f"\nsuggested MIN_RELEVANCE={floor:.3f}: keeps {np.mean(np.array(on) >= floor):.0%} of on-topic pairs, "
          f"removes {np.mean(np.array(off) < floor):.0%} of off-topic pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
