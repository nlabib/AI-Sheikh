"""Prompt construction. The training data builder uses the same functions, so a
fine-tuned adapter learns exactly the format the server sends at inference time."""

from __future__ import annotations

import unicodedata

from app.text import detect_lang, strip_footnotes

SYSTEM_PROMPT = """You answer questions about Islam using ONLY the numbered sources provided.

Rules:
1. Every factual sentence ends with the marker of the source that supports it, copied exactly, e.g. [quran:2:255] or [bukhari:1].
2. Quote source text only word for word, inside double quotes, followed by its marker.
3. Never cite a marker that is not in the sources. Never invent verse or hadith numbers.
4. If the sources do not clearly answer the question, or you are unsure, begin your answer with "I'm not sure" (in Bangla "আমি নিশ্চিত নই", in Arabic "لست متأكدًا"), say what the sources do and do not cover, and suggest asking a qualified scholar. Never fill gaps from memory.
5. Do not issue fatwas. For rulings that depend on circumstances or differ between schools, present what the sources say and recommend a qualified scholar.
6. Answer in the language of the question ({lang_name})."""

LANG_NAMES = {"en": "English", "bn": "Bangla", "ar": "Arabic"}

# What the model says (and what the server substitutes) when an answer cannot be backed by the sources.
NOT_SURE = {
    "en": "I'm not sure. The sources I was given do not answer this, so I won't guess. "
          "Please ask a qualified scholar.",
    "bn": "আমি নিশ্চিত নই। আমাকে দেওয়া উৎসগুলোতে এর উত্তর নেই, তাই আমি অনুমান করব না। "
          "অনুগ্রহ করে একজন যোগ্য আলেমকে জিজ্ঞাসা করুন।",
    "ar": "لست متأكدًا. المصادر المتاحة لي لا تجيب عن هذا، فلن أخمّن. يرجى سؤال عالم مختص.",
}

# When no passage is about the question at all (off-topic, or nothing in the corpus): no scholar line, since
# "ask a scholar about the weather" helps nobody.
NO_SOURCES = {
    "en": "I couldn't find anything about this in the Quran or the six hadith collections I search, "
          "and I only answer from those sources, so I won't guess.",
    "bn": "কুরআন বা যে ছয়টি হাদিস গ্রন্থে আমি খুঁজি, সেখানে এ বিষয়ে কিছু পাইনি। আমি শুধু এই উৎসগুলো থেকে উত্তর দিই, "
          "তাই অনুমান করব না।",
    "ar": "لم أجد شيئًا عن هذا في القرآن أو في كتب الحديث الستة التي أبحث فيها، وأنا لا أجيب إلا منها، فلن أخمّن.",
}

INTRO = {
    "en": "I answer questions from the Quran and the six major hadith collections (Bukhari, Muslim, Abu Dawud, "
          "Tirmidhi, Nasa'i and Ibn Majah). I quote the passages I find and cite each one, and when they don't "
          "answer your question I say I'm not sure. I don't give fatwas: for rulings, please ask a qualified "
          "scholar.",
    "bn": "আমি কুরআন এবং ছয়টি প্রধান হাদিস গ্রন্থ (বুখারী, মুসলিম, আবু দাউদ, তিরমিযী, নাসাঈ ও ইবনে মাজাহ) থেকে "
          "প্রশ্নের উত্তর দিই। যে অংশগুলো পাই সেগুলো উদ্ধৃত করি এবং প্রতিটির সূত্র দিই; উৎসে উত্তর না থাকলে বলি "
          "আমি নিশ্চিত নই। আমি ফতোয়া দিই না: বিধানের জন্য একজন যোগ্য আলেমকে জিজ্ঞাসা করুন।",
    "ar": "أجيب عن الأسئلة من القرآن ومن كتب الحديث الستة (البخاري ومسلم وأبي داود والترمذي والنسائي وابن ماجه). "
          "أقتبس النصوص التي أجدها وأذكر مصدر كل منها، وإذا لم تُجب عن سؤالك قلت إنني لست متأكدًا. "
          "لا أصدر فتاوى: للأحكام يُرجى سؤال عالم مختص.",
}
SALAM_REPLY = {"en": "Wa alaikum assalam. ", "bn": "ওয়া আলাইকুমুস সালাম। ", "ar": "وعليكم السلام. "}


def format_evidence(hits: list, max_chars: int = 1200) -> str:
    """hits: retrieve.Hit objects or dicts from Hit.to_dict()."""
    blocks = []
    for h in hits:
        d = h.to_dict() if hasattr(h, "to_dict") else h
        lines = [f"[{d['ref']}] {d['label']}"]
        meta = d.get("meta") or {}
        if meta.get("grades"):
            lines.append("Grading: " + "; ".join(f"{g['grade']} ({g['name']})" for g in meta["grades"][:3]))
        for t in d["texts"]:
            text = strip_footnotes(t["text"])
            text = text if len(text) <= max_chars else text[:max_chars] + " …"
            lines.append(f"({t['lang']}) {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list) -> list[dict]:
    lang = detect_lang(question)
    system = SYSTEM_PROMPT.format(lang_name=LANG_NAMES.get(lang, "English"))
    user = f"Sources:\n\n{format_evidence(hits) or '(no sources found)'}\n\nQuestion: {question}"
    # Qwen's own tokenizer NFC-normalizes text before splitting it into tokens, so the model was trained on
    # normalized text. llama.cpp (LM Studio) does not, and the Bangla letters য় and ড় (U+09DF, U+09DC) are
    # stored precomposed in the corpus: unnormalized, they reach the model as byte sequences it has hardly
    # seen, and it garbles words containing them when quoting ("শয়তান" came back as "শতান").
    # Normalizing here gives every backend the same input.
    user = unicodedata.normalize("NFC", user)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# A question about a reference that cannot exist (verse 8 of al-Fatiha, surah 115, a hadith number past the end
# of the collection) is answered with these, without asking the model. Hadith numbering differs between
# editions, so those messages say which numbering they mean.
NO_SUCH_REF = {
    "en": {"start": "I'm not sure what you are referring to: ",
           "ayah": "Surah {name} ({s}) has {n} ayat, so there is no ayah {a}.",
           "surah": "the Quran has 114 surahs, so there is no surah {s}.",
           "hadith": "{title} in the edition I search has no hadith numbered {num} (its numbers run to {max}).",
           "end": " Please check the reference."},
    "bn": {"start": "আমি নিশ্চিত নই আপনি কোনটির কথা বলছেন: ",
           "ayah": "সূরা {name} ({s})-এ {n}টি আয়াত আছে, তাই {a} নম্বর আয়াত নেই।",
           "surah": "কুরআনে ১১৪টি সূরা আছে, তাই {s} নম্বর সূরা নেই।",
           "hadith": "আমি যে সংস্করণে খুঁজি তাতে {title}-এর {num} নম্বর হাদিস নেই (সেখানে {max} পর্যন্ত আছে)।",
           "end": " অনুগ্রহ করে রেফারেন্সটি যাচাই করুন।"},
    "ar": {"start": "لست متأكدًا مما تقصده: ",
           "ayah": "سورة {name} ({s}) عدد آياتها {n}، فلا توجد آية رقم {a}.",
           "surah": "في القرآن 114 سورة، فلا توجد سورة رقم {s}.",
           "hadith": "لا يوجد في الطبعة التي أبحث فيها حديث برقم {num} في {title} (تصل أرقامه إلى {max}).",
           "end": " يرجى التحقق من المرجع."},
}
