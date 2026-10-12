"""Text normalisation and tokenisation for English, Bangla and Arabic.

The same functions are used at index time and query time so that FTS matching,
the hashing embedder and the citation checker all agree on what a "word" is.
"""

from __future__ import annotations

import re
import unicodedata

# Arabic harakat, Quranic annotation marks, tatweel and superscript alef.
_AR_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_AR_FOLD = str.maketrans(
    {
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه",
    }
)
# Bangla and Arabic-Indic digits -> ASCII, so "২:২৫৫" and "٢:٢٥٥" parse as 2:255.
_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 3)

# Footnote marks in the Bangla Quran translation, e.g. "[১]" or "[১২]". Over half of its verses have them.
_BN_FOOTNOTE = re.compile(r"\s*\[\s*\u200c?[০-৯]{1,3}\s*।?\s*\]")

# Python's \w splits Bangla words on vowel signs, so include the whole block.
_TOKEN = re.compile(r"[\wঀ-৿‌‍]+")

# FTS5's unicode61 tokenizer has the same problem; these are passed as tokenchars.
BANGLA_TOKENCHARS = "".join(
    ch for ch in map(chr, range(0x0980, 0x0A00)) if unicodedata.category(ch) in ("Mn", "Mc")
) + "‌‍"

_STOPWORDS = {
    "en": set(
        """a an and are as at be by did do does for from has have how i in is it its of on or
        say says said tell than that the their them there these they this to was what when where
        which who whom why will with about quran hadith verse verses ayah surah islam islamic
        according me my we our you your can should allah's""".split()
    ),
    "bn": set("কি কী কে কোন কোথায় কেন কিভাবে কীভাবে এবং ও এর একটি সম্পর্কে বলা হয়েছে বলে কুরআনে কুরআন হাদিসে হাদিস আছে".split()),
    "ar": set("ما ماذا من في عن على الى إلى هل كيف لماذا او أو و ثم ان إن".split()),
}


def detect_lang(text: str) -> str:
    counts = {"ar": 0, "bn": 0, "en": 0}
    for ch in text:
        o = ord(ch)
        if 0x0600 <= o <= 0x06FF:
            counts["ar"] += 1
        elif 0x0980 <= o <= 0x09FF:
            counts["bn"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["en"] += 1
    return max(counts, key=counts.get) if any(counts.values()) else "en"


def normalize(text: str) -> str:
    """Lower-case, fold Arabic letter variants, strip harakat, unify digits."""
    text = unicodedata.normalize("NFC", text)
    text = _AR_DIACRITICS.sub("", text).translate(_AR_FOLD).translate(_DIGITS)
    text = text.replace("\u00ad", "")  # soft hyphens appear in some hadith texts
    return text.lower()


def strip_footnotes(text: str) -> str:
    """Drop the translator's footnote marks so they are neither shown to the model nor mistaken for
    citation markers when it quotes the verse."""
    return _BN_FOOTNOTE.sub("", text)


def tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    """drop_stopwords also drops bare numbers: references like "2:255" are matched
    separately (app.refs), and the Bangla Quran is full of footnote markers like [১]."""
    tokens = _TOKEN.findall(normalize(text))
    if drop_stopwords:
        stop = _STOPWORDS["en"] | _STOPWORDS["bn"] | _STOPWORDS["ar"]
        tokens = [t for t in tokens if t not in stop and not t.isdigit()]
    return tokens


def normalized_for_index(text: str) -> str:
    return " ".join(t for t in tokenize(text) if not t.isdigit())
