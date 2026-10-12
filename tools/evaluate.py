"""Measure retrieval on the gold question set.

    python -m tools.evaluate                       # every mode available (fts, plus vector/hybrid if indexed)
    python -m tools.evaluate --modes fts --k 1 5 10
    python -m tools.evaluate --gold data/eval/gold.jsonl --out results/retrieval.json

For each question, a hit counts if ANY of its relevant refs appears in the top k.
Each mode runs twice: over the whole corpus, and "+scoped" to the question's
side (Quran questions search only the Quran, hadith questions only hadith),
which is how /ask gathers evidence (see Retriever.search_balanced).
Relevance lists were built by hand and are not exhaustive (a hadith often has
more parallel narrations than we listed), so the numbers are a lower bound.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from app.config import ROOT, get_settings
from app.embed import VectorIndex, make_embedder
from app.refs import HADITH_COLLECTIONS
from app.retrieve import Retriever
from app.store import Store

DEFAULT_GOLD = ROOT / "data" / "eval" / "gold.jsonl"


def load_gold(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


HADITH = list(HADITH_COLLECTIONS)


def evaluate(retriever: Retriever, gold: list[dict], mode: str, ks: list[int], scoped: bool = False) -> dict:
    max_k = max(max(ks), 10)
    groups: dict[str, list[dict]] = defaultdict(list)
    misses = []
    t0 = time.perf_counter()
    for item in gold:
        scope = (["quran"] if item["type"] == "quran" else HADITH) if scoped else None
        refs = [h.ref for h in retriever.search(item["query"], k=max_k, mode=mode, collections=scope)]
        relevant = set(item["relevant"])
        rank = next((i + 1 for i, r in enumerate(refs) if r in relevant), None)
        row = {"rank": rank}
        for key in ("all", f"{item['type']}", f"{item['type']}/{item['lang']}"):
            groups[key].append(row)
        if rank is None or rank > 5:
            misses.append({"id": item["id"], "query": item["query"], "expected": item["relevant"][:3],
                           "got": refs[:5], "rank": rank})
    elapsed = time.perf_counter() - t0

    def summarise(rows: list[dict]) -> dict:
        n = len(rows)
        out = {"n": n}
        for k in ks:
            out[f"recall@{k}"] = round(sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n, 3)
        out["mrr@10"] = round(sum(1 / r["rank"] for r in rows if r["rank"] and r["rank"] <= 10) / n, 3)
        return out

    return {
        "mode": mode + ("+scoped" if scoped else ""),
        "ms_per_query": round(1000 * elapsed / len(gold), 1),
        "groups": {g: summarise(rows) for g, rows in sorted(groups.items())},
        "misses": misses,
    }


def to_markdown(results: list[dict], ks: list[int]) -> str:
    cols = [f"recall@{k}" for k in ks] + ["mrr@10"]
    lines = ["| mode | group | n | " + " | ".join(cols) + " |", "|" + "---|" * (3 + len(cols))]
    for res in results:
        for g, s in res["groups"].items():
            lines.append(f"| {res['mode']} | {g} | {s['n']} | " + " | ".join(f"{s[c]:.3f}" for c in cols) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--modes", nargs="*", default=None, help="fts, vector, hybrid (default: all available)")
    ap.add_argument("--k", nargs="*", type=int, default=[1, 5, 10])
    ap.add_argument("--out", type=Path, help="write full JSON results here")
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args(argv)

    settings = get_settings()
    store = Store(settings.db_path)
    if store.count() == 0:
        raise SystemExit("corpus is empty; run `python -m tools.fetch && python -m tools.ingest` first")
    index = VectorIndex.load(settings.index_dir)
    embedder = None
    if index is not None:
        kind = "hashing" if index.embedder_name.startswith("hashing") else "e5"
        embedder = make_embedder(kind, index.embedder_name)
    retriever = Retriever(store, embedder, index)
    modes = args.modes or (["fts", "vector", "hybrid"] if index is not None else ["fts"])

    gold = load_gold(args.gold)
    print(f"{len(gold)} questions, corpus {store.count()} passages, vector index: "
          f"{index.embedder_name if index is not None else 'none'}\n")
    results = [evaluate(retriever, gold, m, args.k, scoped) for scoped in (False, True) for m in modes]
    print(to_markdown(results, args.k))
    for res in results:
        print(f"\n{res['mode']}: {res['ms_per_query']} ms/query, {len(res['misses'])} questions not in top 5")
        if args.show_misses:
            for m in res["misses"]:
                print(f"  {m['id']}: rank={m['rank']} expected {m['expected']} got {m['got']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"embedder": index.embedder_name if index is not None else None,
                                        "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
