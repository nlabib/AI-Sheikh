"""Build a starter fine-tuning set from the corpus.

    python -m tools.build_sft                     # writes data/sft/train.jsonl and data/sft/valid.jsonl
    python -m tools.build_sft --per-kind 400 --seed 1

What these examples teach: the answer FORMAT the server expects (quote the
source verbatim, put the exact [ref] marker after it, answer in the question's
language) and honesty: begin with "I'm not sure" when the sources do not cover
the question, including when it asks for a verse the sources don't contain. They are made by
filling templates with real passages, so every quote is correct by
construction. They do NOT teach fiqh or interpretation; that needs examples
written or reviewed by someone qualified (data/sft/reviewed/*.jsonl, merged in
automatically when present).

Varied answers that combine several sources, answer part of a question, set out differing views or turn down
a question the sources don't answer are in data/sft/generated/*.jsonl. They were drafted against the corpus and
each one passed tools.grounded_sft (exact quotes from training-allowed text, the citation check, and no
citation of an evaluation ref). They are merged in automatically.

Licensing rule enforced here: only passages from sources with
allow_training=True (tools/sources.py) are placed in prompts or answers.
Questions from the evaluation set are never used, so the eval stays honest.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from app.config import ROOT, get_settings
from app.prompts import NOT_SURE, build_messages
from app.refs import HADITH_COLLECTIONS, expand_marker, find_markers, parse_ref
from app.retrieve import Hit, Retriever
from app.store import Passage, Store
from app.surahs import SURAHS
from tools.grounded_sft import eval_refs

SFT_DIR = ROOT / "data" / "sft"
GOLD = ROOT / "data" / "eval" / "gold.jsonl"

_NO_ANSWER = NOT_SURE

# Questions on topics the corpus cannot answer (modern events, other religions' texts, personal data).
_OFF_CORPUS = {
    "en": ["What is the current price of gold in Dhaka?", "Who won the 2022 football World Cup?",
           "What is the population of Indonesia today?", "Which smartphone should I buy?",
           "What is the capital of Canada?", "How do I reset my router password?"],
    "bn": ["আজ ঢাকায় সোনার দাম কত?", "২০২২ সালের ফুটবল বিশ্বকাপ কে জিতেছে?", "কোন স্মার্টফোন কেনা উচিত?",
           "কানাডার রাজধানী কোথায়?"],
    "ar": ["ما سعر الذهب اليوم في دكا؟", "من فاز بكأس العالم لكرة القدم 2022؟", "ما عاصمة كندا؟"],
}


def _trainable_hit(store: Store, ref: str) -> Hit | None:
    passages = [p for p in store.by_ref(ref) if p.allow_training]
    return Hit(ref, 1.0, ["template"], passages) if passages else None


def _quote(p: Passage) -> str:
    return f"\"{p.text}\" [{p.ref}]"


def _quran_lookup(store: Store, rng: random.Random, lang: str, numbers: dict[str, list[str]]) -> dict | None:
    ref = f"quran:{rng.choice(numbers['quran'])}"
    s, a = map(int, parse_ref(ref).number.split(":"))
    hit = _trainable_hit(store, ref)
    if hit is None:
        return None
    name = SURAHS[s][0]
    ar_p = next((p for p in hit.passages if p.lang == "ar"), None)
    if ar_p is None:
        return None
    if lang == "en":
        q = rng.choice([f"What does Quran {s}:{a} say?", f"Quote verse {a} of Surah {name}.",
                        f"What is ayah {s}:{a}?"])
        en_p = next(p for p in hit.passages if p.lang == "en")
        answer = (f"Surah {name} ({s}), ayah {a}, reads in Arabic: {_quote(ar_p)}\n\n"
                  f"In Pickthall's English translation: {_quote(en_p)}")
    elif lang == "bn":
        q = rng.choice([f"কুরআনের {s}:{a} আয়াতে কী বলা হয়েছে?", f"সূরা {name} এর {a} নম্বর আয়াতটি কী?"])
        answer = f"সূরা {name} ({s}) এর {a} নম্বর আয়াতের আরবি পাঠ: {_quote(ar_p)}"
    else:
        q = rng.choice([f"ما نص الآية {a} من سورة رقم {s}؟", f"اذكر الآية {s}:{a}."])
        answer = f"الآية {a} من السورة رقم {s} ({name}): {_quote(ar_p)}"
    return {"question": q, "lang": lang, "hits": [hit], "answer": answer, "kind": "quran_lookup"}


def _hadith_lookup(store: Store, rng: random.Random, lang: str, numbers: dict[str, list[str]]) -> dict | None:
    available = [c for c in HADITH_COLLECTIONS if numbers.get(c)]
    if not available:
        return None
    coll = rng.choice(available)
    ref = f"{coll}:{rng.choice(numbers[coll])}"
    hit = _trainable_hit(store, ref)
    if hit is None:
        return None
    ar_p = next((p for p in hit.passages if p.lang == "ar"), None)
    if ar_p is None or len(ar_p.text) > 1500:
        return None
    title, num = HADITH_COLLECTIONS[coll], parse_ref(ref).number
    grades = ar_p.meta.get("grades") or []
    grade_en = "; ".join(f"{g['grade']} according to {g['name']}" for g in grades[:2])
    if lang == "en":
        q = f"What is the Arabic text of {title} {num}?"
        answer = f"{title} {num} reads: {_quote(ar_p)}"
        answer += f"\n\nGrading: {grade_en} [{ref}]." if grade_en else ""
    elif lang == "bn":
        q = f"{title} এর {num} নম্বর হাদিসের আরবি পাঠ কী?"
        answer = f"{title} {num} এর আরবি পাঠ: {_quote(ar_p)}"
        answer += f"\n\nমান: {grade_en} [{ref}]।" if grade_en else ""
    else:
        q = f"ما نص الحديث رقم {num} في {title}؟"
        answer = f"نص الحديث رقم {num} في {title}: {_quote(ar_p)}"
        answer += f"\n\nالحكم: {grade_en} [{ref}]." if grade_en else ""
    return {"question": q, "lang": lang, "hits": [hit], "answer": answer, "kind": "hadith_lookup"}


def _no_answer(store: Store, rng: random.Random, lang: str, numbers: dict[str, list[str]]) -> dict | None:
    """Unrelated sources + an off-corpus question -> the model should decline, not improvise."""
    hits = []
    for _ in range(3):
        hit = _trainable_hit(store, f"quran:{rng.choice(numbers['quran'])}")
        if hit:
            hits.append(hit)
    if not hits:
        return None
    return {"question": rng.choice(_OFF_CORPUS[lang]), "lang": lang, "hits": hits,
            "answer": _NO_ANSWER[lang], "kind": "no_answer"}


_MISSING = {
    "en": "I'm not sure. The sources I was given do not include {label}, so I cannot quote or explain it. "
          "Please check a printed copy or ask a qualified scholar.",
    "bn": "আমি নিশ্চিত নই। আমাকে দেওয়া উৎসগুলোতে {label} নেই, তাই আমি এটি উদ্ধৃত বা ব্যাখ্যা করতে পারছি না। "
          "অনুগ্রহ করে মুদ্রিত কপি দেখুন বা একজন যোগ্য আলেমকে জিজ্ঞাসা করুন।",
    "ar": "لست متأكدًا. المصادر المتاحة لي لا تتضمن {label}، فلا أستطيع نقله أو شرحه. يرجى الرجوع إلى نسخة مطبوعة أو سؤال عالم مختص.",
}


def _missing_ref(store: Store, rng: random.Random, lang: str, numbers: dict[str, list[str]]) -> dict | None:
    """An Islamic question about a specific verse whose text is NOT among the sources.

    This is the case that matters most for honesty: the model may well "know" the
    verse from pre-training, but it must not answer from memory when the
    evidence isn't in front of it.
    """
    asked = f"quran:{rng.choice(numbers['quran'])}"
    s, a = parse_ref(asked).number.split(":")
    hits = [h for h in (_trainable_hit(store, f"quran:{rng.choice(numbers['quran'])}") for _ in range(3))
            if h and h.ref != asked]
    if not hits:
        return None
    if lang == "en":
        q, label = f"What does Quran {s}:{a} say?", f"Quran {s}:{a}"
    elif lang == "bn":
        q, label = f"কুরআনের {s}:{a} আয়াতে কী বলা হয়েছে?", f"কুরআনের {s}:{a} আয়াত"
    else:
        q, label = f"ما نص الآية {s}:{a}؟", f"الآية {s}:{a}"
    return {"question": q, "lang": lang, "hits": hits, "answer": _MISSING[lang].format(label=label),
            "kind": "missing_ref"}


def _to_example(item: dict) -> dict:
    messages = build_messages(item["question"], item["hits"])
    messages.append({"role": "assistant", "content": item["answer"]})
    return {"messages": messages,
            "meta": {"kind": item["kind"], "lang": item["lang"], "review_status": "template",
                     "sources": sorted({p.source_id for h in item["hits"] for p in h.passages})}}


def build(store: Store, per_kind: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    numbers: dict[str, list[str]] = {}
    for (ref,) in store.conn.execute("SELECT DISTINCT ref FROM passages WHERE allow_training = 1 ORDER BY ref"):
        c, n = ref.split(":", 1)
        numbers.setdefault(c, []).append(n)
    if not numbers.get("quran"):
        raise SystemExit("no trainable Quran passages in the corpus; run tools.fetch and tools.ingest first")
    gold_queries = {json.loads(line)["query"] for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()}

    makers = [lambda l: _quran_lookup(store, rng, l, numbers), lambda l: _hadith_lookup(store, rng, l, numbers),
              lambda l: _no_answer(store, rng, l, numbers), lambda l: _missing_ref(store, rng, l, numbers)]
    langs = ["en", "bn", "ar"]
    out, seen = [], set()
    for make in makers:
        made, tries = 0, 0
        while made < per_kind and tries < per_kind * 10:
            tries += 1
            item = make(langs[tries % 3])
            if item is None or item["question"] in gold_queries:
                continue
            key = (item["question"], item["kind"], tuple(h.ref for h in item["hits"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(_to_example(item))
            made += 1
    rng.shuffle(out)
    return out


def cites_any(example: dict, refs: set[str]) -> bool:
    """True when the example's answer (the last message) cites a ref in `refs`."""
    answer = example["messages"][-1]["content"]
    return any(r in refs for m in find_markers(answer) for r in expand_marker(m))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 150 per template kind: the templates are easy to memorise (the first run's loss hit ~0 by step 30), so they
    # are kept to a minority next to the generated answers.
    ap.add_argument("--per-kind", type=int, default=150, help="template examples per kind (4 kinds)")
    ap.add_argument("--valid-share", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=SFT_DIR)
    args = ap.parse_args(argv)

    store = Store(get_settings().db_path)
    examples = build(store, args.per_kind, args.seed)
    reviewed = []
    for sub in ("reviewed", "generated"):
        folder = args.out_dir / sub
        for path in sorted(folder.glob("*.jsonl")) if folder.exists() else []:
            reviewed += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = reviewed + examples
    # Training answers must not cite what the evaluation sets expect (the benchmark grows over time, so older
    # answers get checked again here, not only when they are written).
    held_out = eval_refs()
    kept = [e for e in examples if not cites_any(e, held_out)]
    print(f"dropped {len(examples) - len(kept)} examples that cite a held-out evaluation source")
    examples = kept
    random.Random(args.seed).shuffle(examples)

    n_valid = max(1, int(len(examples) * args.valid_share))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("valid", examples[:n_valid]), ("train", examples[n_valid:])):
        with (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds: dict[str, int] = {}
    for e in examples:
        kinds[e["meta"]["kind"]] = kinds.get(e["meta"]["kind"], 0) + 1
    print(f"wrote {len(examples) - n_valid} train / {n_valid} valid examples to {args.out_dir} "
          f"({len(reviewed)} reviewed or generated); by kind: {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
