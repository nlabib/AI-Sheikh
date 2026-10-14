"""Topic examples for tools.build_sft: questions answered from several sources, and honest refusals when the
sources are about something else.

The first fine-tune learned from one-source templates only (look up a numbered verse or hadith, or refuse),
and it showed: answers quoted one passage, often the wrong one of six, and quotes drifted from the source
(docs/INFERENCE_REVIEW.md). These examples come from the real retrieval pipeline instead. A topic question
("What does the Quran say about patience?") is run through hybrid search and the reranker; the passages the
reranker is confident about are quoted, word for word as a contiguous excerpt, each with its own marker, and
the rest stay in the prompt as distractors the model must learn to leave alone.

Kinds:
  topic_quran / topic_hadith / topic_mixed   2-3 relevant sources quoted, nothing else cited
  topic_unrelated                            sources about another topic -> "I'm not sure", in the question's
                                             language, saying the sources are about other matters
  surah_lookup                               a short surah quoted ayah by ayah, one marker each

Guards: passages listed as relevant in data/eval/gold.jsonl are never quoted, gold questions are never
asked, and only passages whose licence allows training appear in prompts or answers.
"""

from __future__ import annotations

import random
import re

from app.refs import HADITH_COLLECTIONS, parse_ref
from app.retrieve import Hit, Retriever
from app.store import Passage, Store
from app.surahs import SURAHS

# (English, Bangla, Arabic)
TOPICS = [
    ("patience", "ধৈর্য", "الصبر"),
    ("honesty", "সত্যবাদিতা", "الصدق"),
    ("charity", "দান-সদকা", "الصدقة"),
    ("kindness to parents", "মা-বাবার প্রতি সদাচরণ", "بر الوالدين"),
    ("prayer", "নামাজ", "الصلاة"),
    ("fasting", "রোজা", "الصيام"),
    ("backbiting", "গীবত", "الغيبة"),
    ("anger", "রাগ", "الغضب"),
    ("seeking knowledge", "জ্ঞান অর্জন", "طلب العلم"),
    ("repentance", "তওবা", "التوبة"),
    ("gratitude", "কৃতজ্ঞতা", "الشكر"),
    ("orphans", "এতিম", "اليتيم"),
    ("neighbours", "প্রতিবেশী", "الجار"),
    ("justice", "ন্যায়বিচার", "العدل"),
    ("lying", "মিথ্যা বলা", "الكذب"),
    ("envy", "হিংসা", "الحسد"),
    ("arrogance", "অহংকার", "الكبر"),
    ("humility", "বিনয়", "التواضع"),
    ("supplication", "দোয়া", "الدعاء"),
    ("remembering Allah", "আল্লাহর জিকির", "ذكر الله"),
    ("the Day of Judgement", "কিয়ামতের দিন", "يوم القيامة"),
    ("Paradise", "জান্নাত", "الجنة"),
    ("forgiving others", "অন্যকে ক্ষমা করা", "العفو"),
    ("good character", "উত্তম চরিত্র", "حسن الخلق"),
    ("trust in Allah", "আল্লাহর উপর ভরসা", "التوكل على الله"),
    ("keeping promises", "প্রতিশ্রুতি রক্ষা", "الوفاء بالعهد"),
    ("hypocrisy", "মুনাফেকি", "النفاق"),
    ("the Hajj", "হজ", "الحج"),
    ("trade", "ব্যবসা-বাণিজ্য", "التجارة"),
    ("debts", "ঋণ", "الدين"),
    ("the Friday prayer", "জুমার নামাজ", "صلاة الجمعة"),
    ("mercy", "দয়া", "الرحمة"),
    ("visiting the sick", "রোগী দেখতে যাওয়া", "عيادة المريض"),
    ("greeting with salam", "সালাম দেওয়া", "إفشاء السلام"),
    ("modesty", "লজ্জাশীলতা", "الحياء"),
    ("wasting food and money", "অপচয়", "الإسراف"),
]

QUESTIONS = {
    "quran": {
        "en": ["What does the Quran say about {t}?", "Which verses talk about {t}?"],
        "bn": ["কুরআনে {t} সম্পর্কে কী বলা হয়েছে?", "{t} নিয়ে কুরআনের আয়াত কী?"],
        "ar": ["ماذا يقول القرآن عن {t}؟", "ما الآيات التي تتحدث عن {t}؟"],
    },
    "hadith": {
        "en": ["What did the Prophet say about {t}?", "Is there a hadith about {t}?"],
        "bn": ["{t} সম্পর্কে রাসূল (সা.) কী বলেছেন?", "{t} সম্পর্কে হাদিসে কী আছে?"],
        "ar": ["ماذا قال النبي صلى الله عليه وسلم عن {t}؟", "ما الحديث الوارد في {t}؟"],
    },
    "mixed": {
        "en": ["What do the Quran and hadith say about {t}?", "What do the sources say about {t}?"],
        "bn": ["{t} সম্পর্কে কুরআন ও হাদিসে কী আছে?"],
        "ar": ["ما الذي ورد في القرآن والسنة عن {t}؟"],
    },
}

INTROS = {
    "quran": {"en": ["The Quran speaks about {t} in these verses:", "These verses among the sources are about {t}:"],
              "bn": ["{t} সম্পর্কে কুরআনের এই আয়াতগুলো উৎসে আছে:", "কুরআনে {t} সম্পর্কে বলা হয়েছে:"],
              "ar": ["ورد في القرآن عن {t}:", "من الآيات التي تتحدث عن {t}:"]},
    "hadith": {"en": ["These hadith among the sources are about {t}:", "The Prophet's words on {t} in the sources:"],
               "bn": ["{t} সম্পর্কে উৎসে এই হাদিসগুলো আছে:", "{t} সম্পর্কে হাদিসে এসেছে:"],
               "ar": ["ورد في الحديث عن {t}:", "من الأحاديث في {t}:"]},
    "mixed": {"en": ["The sources say this about {t}:", "Here is what the verses and hadith I was given say about {t}:"],
              "bn": ["{t} সম্পর্কে উৎসে যা আছে:"],
              "ar": ["هذا ما ورد في المصادر عن {t}:"]},
}

UNRELATED = {
    "en": ["I'm not sure. The sources I was given are about other matters and don't address {t}, so I won't guess. "
           "Please ask a qualified scholar.",
           "I'm not sure. None of the passages I was given is about {t}, so I can't answer from them. "
           "Please ask a qualified scholar."],
    "bn": ["আমি নিশ্চিত নই। আমাকে দেওয়া উৎসগুলো অন্য বিষয়ে, {t} সম্পর্কে নয়, তাই আমি অনুমান করব না। "
           "অনুগ্রহ করে একজন যোগ্য আলেমকে জিজ্ঞাসা করুন।"],
    "ar": ["لست متأكدًا. المصادر المتاحة لي تتناول أمورًا أخرى ولا تتحدث عن {t}، فلن أخمّن. يرجى سؤال عالم مختص."],
}

# How sure the reranker must be before a passage is quoted as answering the question. Quran verses score lower
# against modern wording (Pickthall, 1930), so their bar is lower; see docs/RESULTS.md.
QUOTE_MIN = {"quran": 0.3, "hadith": 0.8}
UNRELATED_MAX = 0.05
# Share of questions that also get an unrelated-sources refusal. Every one qualifies, and too many refusals
# teach the model to refuse questions it could answer.
UNRELATED_SHARE = 0.25
EXCERPT_CHARS = 280
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
# The Prophet's words in the Arabic hadith texts sit between quote marks, often wrapped in RLM marks.
_MATN = re.compile(r"\"‏?\s*([^\"]{15,}?)\s*‏?\"")


def excerpt(p: Passage) -> str | None:
    """A contiguous piece of the passage, copied exactly: the whole text if short, else whole sentences (or
    clauses) from the start. Hadith: the Prophet's quoted words, not the chain of narrators."""
    text = p.text.strip()
    if p.collection != "quran":
        spans = _MATN.findall(text)
        if not spans:
            return None
        text = max(spans, key=len).strip().strip("‏").strip()
    if '"' in text:
        return None
    if len(text) <= EXCERPT_CHARS:
        return text
    pieces = _SENTENCE_END.split(text) if p.lang == "en" else re.split(r"(?<=[ۚۖۗ،.؛])\s+", text)
    out = ""
    for piece in pieces:
        if out and len(out) + 1 + len(piece) > EXCERPT_CHARS:
            break
        out = f"{out} {piece}".strip() if out else piece
    if not out or len(out) > EXCERPT_CHARS * 1.5 or out not in p.text:
        return None
    return out


def _side(ref: str) -> str:
    return "quran" if ref.startswith("quran:") else "hadith"


def _trainable(store: Store, ref: str) -> Hit | None:
    passages = [p for p in store.by_ref(ref) if p.allow_training]
    return Hit(ref, 1.0, ["template"], passages) if passages else None


def _cite(hit: Hit, lang: str) -> str | None:
    """One quoted line for an answer in `lang`: the English translation for English answers to Quran
    questions, otherwise the Arabic (the only trainable text for hadith and for Bangla answers)."""
    want = "en" if (lang == "en" and hit.ref.startswith("quran:")) else "ar"
    p = next((p for p in hit.passages if p.lang == want), None)
    text = excerpt(p) if p else None
    if not text:
        return None
    label = parse_ref(hit.ref).label
    if hit.ref.startswith("quran:"):
        s = int(parse_ref(hit.ref).number.split(":")[0])
        label = f"Surah {SURAHS[s][0]} ({parse_ref(hit.ref).number})" if lang == "en" else parse_ref(hit.ref).number
    return f"{label}: \"{text}\" [{hit.ref}]"


def topic_examples(store: Store, retriever: Retriever, rng: random.Random, gold_refs: set[str],
                   gold_queries: set[str], log=print) -> list[dict]:
    if retriever.reranker is None:
        raise SystemExit("topic examples need the reranker (RERANKER=BAAI/bge-reranker-v2-m3) and the E5 index")
    out: list[dict] = []
    for i, names in enumerate(TOPICS):
        for style in ("quran", "hadith", "mixed"):
            collections = (["quran"] if style == "quran" else
                           list(HADITH_COLLECTIONS) if style == "hadith" else None)
            for li, lang in enumerate(("en", "bn", "ar")):
                t = names[li]
                for template in QUESTIONS[style][lang]:  # every wording, not one at random: more examples
                    q = template.format(t=t)
                    if q in gold_queries:
                        continue
                    item = _topic_answer(store, retriever, rng, q, t, style, lang, collections, gold_refs)
                    if item:
                        out.append(item)
                    # The same question shown passages retrieved for a different topic: nothing to quote.
                    if rng.random() > UNRELATED_SHARE:
                        continue
                    other = TOPICS[(i + rng.randrange(1, len(TOPICS))) % len(TOPICS)]
                    q_other = rng.choice(QUESTIONS[style][lang]).format(t=other[li])
                    other_hits = retriever.search_balanced(q_other, k=6, collections=collections, min_rerank=0.01)
                    scores = retriever.rerank_scores(q, [h.ref for h in other_hits])
                    if other_hits and max(scores.values(), default=1.0) < UNRELATED_MAX:
                        unrelated = [h2 for h2 in (_trainable(store, h.ref) for h in other_hits) if h2]
                        out.append({"question": q, "lang": lang, "hits": unrelated,
                                    "answer": rng.choice(UNRELATED[lang]).format(t=t), "kind": "topic_unrelated"})
        log(f"  topic {i + 1}/{len(TOPICS)} {names[0]}: {len(out)} examples so far")
    return out


def _topic_answer(store: Store, retriever: Retriever, rng: random.Random, q: str, t: str, style: str, lang: str,
                  collections: list[str] | None, gold_refs: set[str]) -> dict | None:
    """2-3 passages the reranker is confident about, quoted word for word; the rest stay as distractors."""
    hits = retriever.search_balanced(q, k=6, collections=collections, min_rerank=0.01)
    chosen = [h for h in hits if h.ref not in gold_refs and (h.rerank or 0) >= QUOTE_MIN[_side(h.ref)]]
    if style == "mixed":
        chosen = ([h for h in chosen if _side(h.ref) == "quran"][:1] +
                  [h for h in chosen if _side(h.ref) == "hadith"][:2])
        if {_side(h.ref) for h in chosen} != {"quran", "hadith"}:
            return None
    prompt_hits = [h2 for h2 in (_trainable(store, h.ref) for h in hits) if h2]
    by_ref = {h.ref: h for h in prompt_hits}
    lines = [line for line in (_cite(by_ref[c.ref], lang) for c in chosen[:3] if c.ref in by_ref) if line]
    if len(lines) < 2:
        return None
    end = "।" if lang == "bn" else ""
    answer = rng.choice(INTROS[style][lang]).format(t=t) + "\n\n" + "\n\n".join(line + end for line in lines)
    return {"question": q, "lang": lang, "hits": prompt_hits, "answer": answer, "kind": f"topic_{style}"}


SURAH_QUESTIONS = {"en": ["Show me Surah {n}.", "Recite Surah {n}.", "What does Surah {n} say?"],
                   "bn": ["সূরা {s} দেখাও।", "সূরা {s} পড়ে শোনাও।"],
                   "ar": ["اعرض سورة رقم {s}.", "ما نص سورة رقم {s}؟"]}


def surah_examples(store: Store, rng: random.Random, gold_refs: set[str]) -> list[dict]:
    """Short surahs quoted ayah by ayah, each with its own [quran:s:a] marker (never [1] or [s:a-b])."""
    out = []
    for s, (name, _meaning, n_ayat) in SURAHS.items():
        if n_ayat > 12:
            continue
        refs = [f"quran:{s}:{a}" for a in range(1, n_ayat + 1)]
        if gold_refs & set(refs):
            continue
        hits = [h for h in (_trainable(store, r) for r in refs) if h]
        if len(hits) != n_ayat:
            continue
        for lang in ("en", "bn", "ar"):
            q = rng.choice(SURAH_QUESTIONS[lang]).format(n=name, s=s)
            ar = [f"\"{next(p.text for p in h.passages if p.lang == 'ar')}\" [{h.ref}]" for h in hits]
            if lang == "en":
                en = [f"\"{next(p.text for p in h.passages if p.lang == 'en')}\" [{h.ref}]" for h in hits]
                answer = (f"Surah {name} ({s}) in Arabic:\n" + "\n".join(ar) +
                          "\n\nIn Pickthall's English translation:\n" + "\n".join(en))
            elif lang == "bn":
                answer = f"সূরা {name} ({s}) এর আরবি পাঠ:\n" + "\n".join(ar)
            else:
                answer = f"سورة رقم {s} ({name}):\n" + "\n".join(ar)
            out.append({"question": q, "lang": lang, "hits": hits, "answer": answer, "kind": "surah_lookup"})
    return out
