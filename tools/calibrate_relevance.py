"""Pick the relevance floor: the score below which a passage is treated as not about the question.

    python -m tools.calibrate_relevance                                   # E5 similarity -> MIN_RELEVANCE
    RERANKER=BAAI/bge-reranker-v2-m3 python -m tools.calibrate_relevance  # reranker score -> RERANK_MIN

It compares two groups of (question, passage) pairs from data/eval/gold.jsonl:
  * on topic: each question with the passages marked relevant to it;
  * off topic: each question with another question's relevant passages, plus a few questions the corpus
    has nothing on (weather, programming) with whatever retrieval returns for them.
It prints both distributions and suggests the floor that keeps 95% of the on-topic pairs, with the share of
off-topic pairs that floor removes. Set it with MIN_RELEVANCE=<value> (or RERANK_MIN=<value> with a reranker)
when starting the server.

It also scores HARD_CASES: passages search really returned for a question and that a reader judged not to
be about it, next to the passages that do answer it (docs/INFERENCE_REVIEW.md). They are what a floor most
needs to get right, and far harder than another question's passages.
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

# (question, passages search returned that are NOT about it, passages that DO answer it), judged by hand.
HARD_CASES = [
    ("what dose quran say abt riba intrest", ["quran:6:19", "quran:2:143", "quran:27:40"], ["quran:2:275"]),
    ("What do the Quran and hadith say about honouring parents?", ["quran:4:11", "quran:89:17"],
     ["quran:17:23", "quran:31:14"]),
    ("Should I raise my hands before and after ruku?", ["muslim:1549"], ["ibnmajah:858", "ibnmajah:859"]),
    ("ماذا يقول القرآن عن الصبر؟", ["quran:17:41", "quran:46:29"], ["quran:2:153"]),
    ("ما الذي ورد في القرآن والسنة عن بر الوالدين؟", ["quran:27:76"], ["muslim:253"]),
    ("Is Surah al-Fatiha mandatory in every prayer?", ["quran:3:30", "quran:6:108"], ["bukhari:756", "muslim:884"]),
    ("প্রতি নামাজে কি সূরা ফাতিহা পড়া ফরজ?", ["tirmidhi:1027", "quran:89:10"], ["tirmidhi:247"]),
    ("রমজানে রোজা রাখার ফজিলত সম্পর্কে হাদিসে কী আছে?", ["tirmidhi:741", "quran:12:79"], []),
    ("الوضوء؟", ["quran:93:1"], []),
]


def percentiles(values: list[float]) -> str:
    p = np.percentile(values, [5, 25, 50, 75, 95])
    return " ".join(f"p{q}={v:.3f}" for q, v in zip((5, 25, 50, 75, 95), p))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--keep", type=float, default=0.95, help="share of on-topic pairs the floor must keep")
    args = ap.parse_args(argv)

    settings = get_settings()
    retriever, note = build_retriever(settings)
    if retriever.reranker is not None:
        score, setting = retriever.rerank_scores, "RERANK_MIN"
    elif retriever.index is not None:
        score, setting = retriever.relevance, "MIN_RELEVANCE"
    else:
        raise SystemExit(f"needs the E5 vector index or a reranker ({note}); run python -m tools.index first")
    print(f"scoring with {settings.reranker or 'E5 similarity'}")
    gold = load_gold(args.gold)
    rng = random.Random(0)

    on: dict[str, list[float]] = {"quran": [], "hadith": []}
    off: list[float] = []
    for item in gold:
        on[item["type"]] += score(item["query"], item["relevant"]).values()
        other = rng.choice([g for g in gold if not set(g["relevant"]) & set(item["relevant"])])
        off += score(item["query"], other["relevant"]).values()
    for q in OFF_TOPIC_QUESTIONS:
        hits = retriever.search_balanced(q, k=6)
        off += score(q, [h.ref for h in hits]).values()
    all_on = on["quran"] + on["hadith"]

    floor = float(np.percentile(all_on, 100 * (1 - args.keep)))
    print(f"on topic, Quran  ({len(on['quran'])} pairs): {percentiles(on['quran'])}")
    print(f"on topic, hadith ({len(on['hadith'])} pairs): {percentiles(on['hadith'])}")
    print(f"off topic        ({len(off)} pairs): {percentiles(off)}")
    print(f"\nsuggested {setting}={floor:.3f}: keeps {np.mean(np.array(all_on) >= floor):.0%} of on-topic pairs, "
          f"removes {np.mean(np.array(off) < floor):.0%} of off-topic pairs")

    print("\nhard cases (- should fall below the floor, + should stay above it; ! = on the wrong side):")
    right_side = total = 0
    for q, wrong, right in HARD_CASES:
        scores = score(q, wrong + right)
        cells = []
        for ref in wrong + right:
            value = scores.get(ref, float("nan"))
            ok = (value < floor) == (ref in wrong)
            right_side += ok
            total += 1
            cells.append(f"{'-' if ref in wrong else '+'}{ref}={value:.3f}{'' if ok else '!'}")
        print(f"  {q[:42]:42}  " + "  ".join(cells))
    print(f"{right_side}/{total} hard cases on the right side of {floor:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
