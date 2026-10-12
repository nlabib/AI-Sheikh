"""Canonical source references.

Every passage has a canonical ref string that is also the citation marker the
model must emit, wrapped in square brackets:

    quran:2:255     Surah 2, ayah 255
    bukhari:1       Sahih al-Bukhari, hadith 1 (numbering of the imported edition)

Hadith numbers are the `hadithnumber` of the fawazahmed0/hadith-api editions.
For Bukhari this is the familiar number (bukhari:1 is the hadith on intentions).
Sahih Muslim is numbered sequentially there; the familiar Abd al-Baqi number
(e.g. "Muslim 1907a") is kept in the passage metadata as `alt_number` and shown
in labels. Spot-check other collections against a printed edition or
sunnah.com before publishing references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches

from app.surahs import SURAHS
from app.text import normalize, tokenize

HADITH_COLLECTIONS = {
    "bukhari": "Sahih al-Bukhari",
    "muslim": "Sahih Muslim",
    "abudawud": "Sunan Abi Dawud",
    "tirmidhi": "Jami at-Tirmidhi",
    "nasai": "Sunan an-Nasa'i",
    "ibnmajah": "Sunan Ibn Majah",
}

# Spellings people actually type, in English, Bangla and Arabic.
_ALIASES = {
    "quran": ["quran", "qur'an", "koran", "q", "কুরআন", "কোরআন", "القران", "قران"],
    "bukhari": ["bukhari", "al-bukhari", "sahih bukhari", "বুখারী", "বুখারি", "البخاري"],
    "muslim": ["muslim", "sahih muslim", "মুসলিম", "مسلم"],
    "abudawud": ["abu dawud", "abi dawud", "abu dawood", "abudawud", "abu daud", "আবু দাউদ", "ابو داود", "ابي داود"],
    "tirmidhi": ["tirmidhi", "at-tirmidhi", "তিরমিযী", "তিরমিজি", "الترمذي"],
    "nasai": ["nasai", "an-nasai", "nasa'i", "an-nasa'i", "নাসাঈ", "নাসায়ী", "النسائي"],
    "ibnmajah": ["ibn majah", "ibnmajah", "ইবনে মাজাহ", "ইবনু মাজাহ", "ابن ماجه"],
}

# Words people put between a collection name and the number: "Bukhari hadith no. 1", "مسلم حديث رقم ٣٩٤".
_FILLER = r"(?:\s*(?:hadith|hadees|hadis|number|num|no\.?|nr\.?|#|নং|নম্বর|হাদিস|হাদীস|حديث|رقم)\b\.?)*"
_SURAH_WORD = r"(?:surah|sura|surat|soorah|সূরা|সুরা|سوره|سورة)"
# Surahs this short are given whole when named; longer ones are left to ordinary search.
WHOLE_SURAH_MAX_AYAT = 12
# A named surah is handed over whole only when the question asks for its text ("show me", "recite", "what does
# it say", its meaning), or is little more than the name. "Is Surah al-Fatiha obligatory in every prayer?" is
# about the surah, and its seven ayat don't answer it; ordinary search should find the hadith that do.
_ASKS_FOR_TEXT = re.compile(
    r"\b(?:show|recite|read|quote|text|says?|said|translat\w*|mean\w*|tafsir|explain|words|full|whole)\b"
    r"|দেখাও|পড়|পাঠ|তেলাওয়াত|অর্থ|অনুবাদ|কী বলা|কি বলা|ব্যাখ্যা"
    r"|نص|اقر|اعرض|معن|تفسير|ماذا تقول|ماذا يقول")
_CONNECTORS = {"and", "&", "or", "و", "এবং", "ও"}


def _asks_for_surah_text(q: str) -> bool:
    if _ASKS_FOR_TEXT.search(q):
        return True
    words = [w for w in tokenize(q) if not re.fullmatch(_SURAH_WORD, w) and w not in _CONNECTORS]
    return len(words) <= 3  # "Surah Ikhlas", "surah ikhlas and surah nas", "سورة الإخلاص"


def _fold_name(name: str) -> str:
    """"Al-Ikhlaas", "al ikhlas" and "Ikhlas" all fold to "ikhlas"."""
    n = normalize(name).replace("'", "").replace("’", "")
    n = re.sub(r"^(?:a[lnstdrz]|ash|adh)[-\s]+", "", n)
    n = re.sub(r"[-\s]+", "", n)
    n = n.replace("ee", "i").replace("oo", "u").replace("th", "t").replace("dh", "d").replace("q", "k")
    n = re.sub(r"([aiueo])\1+", r"\1", n)
    return re.sub(r"h$", "", n)


# Arabic and Bangla names of the surahs short enough to hand over whole (WHOLE_SURAH_MAX_AYAT).
_LOCAL_SURAH_NAMES = {
    1: ["الفاتحة", "ফাতিহা", "ফাতিহাহ"], 62: ["الجمعة", "জুমুআ", "জুমুআহ", "জুমা"],
    63: ["المنافقون", "মুনাফিকুন"], 65: ["الطلاق", "তালাক"], 66: ["التحريم", "তাহরীম"],
    93: ["الضحى", "দুহা", "দোহা", "দুহা"], 94: ["الشرح", "الانشراح", "ইনশিরাহ", "শারহ"], 95: ["التين", "তীন", "ত্বীন"],
    97: ["القدر", "কদর", "ক্বদর", "কদর"], 98: ["البينة", "বাইয়্যিনাহ", "বাইয়িনাহ", "বায়্যিনাহ"],
    99: ["الزلزلة", "যিলযাল", "যালযালাহ", "যিলযাল"], 100: ["العاديات", "আদিয়াত"], 101: ["القارعة", "কারিআহ", "ক্বারিআহ"],
    102: ["التكاثر", "তাকাসুর"], 103: ["العصر", "আসর"], 104: ["الهمزة", "হুমাযাহ"], 105: ["الفيل", "ফীল"],
    106: ["قريش", "কুরাইশ"], 107: ["الماعون", "মাউন"], 108: ["الكوثر", "কাওসার", "কাউসার"],
    109: ["الكافرون", "কাফিরুন", "কাফিরূন"], 110: ["النصر", "নাসর"], 111: ["المسد", "اللهب", "লাহাব", "মাসাদ"],
    112: ["الإخلاص", "ইখলাস"], 113: ["الفلق", "ফালাক", "ফালাক্ব"], 114: ["الناس", "নাস"],
}


def _fold_local(name: str) -> str:
    n = re.sub(r"^আল[-\s]?", "", normalize(name).replace("ী", "ি").replace("ূ", "ু"))
    return n[2:] if n.startswith("ال") and len(n) > 4 else n


_SURAH_BY_LOCAL_NAME = {_fold_local(n): num for num, names in _LOCAL_SURAH_NAMES.items() for n in names}
_SURAH_BY_NAME = {_fold_name(name): num for num, (name, _meaning, _n) in SURAHS.items()}

MARKER = re.compile(r"\[((?:quran:\d{1,3}:\d{1,3}(?:-\d{1,3})?)|(?:[a-z]+:\d+(?:\.\d+)?))\]")


@dataclass(frozen=True)
class Ref:
    collection: str  # "quran" or a HADITH_COLLECTIONS key
    number: str  # "2:255" for Quran, "1" for hadith

    def __str__(self) -> str:
        return f"{self.collection}:{self.number}"

    @property
    def label(self) -> str:
        if self.collection == "quran":
            return f"Quran {self.number}"
        return f"{HADITH_COLLECTIONS.get(self.collection, self.collection)} {self.number}"


def quran_ref(surah: int, ayah: int) -> str:
    return f"quran:{surah}:{ayah}"


def hadith_ref(collection: str, number: int | float | str) -> str:
    """Hadith numbers follow the imported edition; a few are fractional (e.g. bukhari:402.2)."""
    if isinstance(number, float) and number.is_integer():
        number = int(number)
    return f"{collection}:{number}"


def parse_ref(ref: str) -> Ref:
    collection, _, number = ref.partition(":")
    if not number:
        raise ValueError(f"not a canonical ref: {ref!r}")
    return Ref(collection, number)


def expand_marker(marker: str) -> list[str]:
    """`quran:2:255-257` -> three refs; anything else -> itself."""
    m = re.fullmatch(r"quran:(\d+):(\d+)-(\d+)", marker)
    if not m:
        return [marker]
    s, a, b = map(int, m.groups())
    if b < a or b - a > 50:
        return [marker]
    return [quran_ref(s, i) for i in range(a, b + 1)]


def find_markers(text: str) -> list[str]:
    return MARKER.findall(text)


def surah_refs(surah: int) -> list[str]:
    """Every ayah of a short surah, or nothing for a long one."""
    info = SURAHS.get(surah)
    if not info or info[2] > WHOLE_SURAH_MAX_AYAT:
        return []
    return [quran_ref(surah, a) for a in range(1, info[2] + 1)]


def find_explicit_refs(query: str) -> list[str]:
    """Pull references the user typed directly, e.g. "2:255", "Bukhari 6018", "Sahih Bukhari hadith number 1",
    "surah 112" or "Surah Al-Ikhlas"."""
    q = normalize(query)
    found: list[str] = []
    for s, a in re.findall(r"\b(\d{1,3})\s*[:/]\s*(\d{1,3})\b", q):
        if 1 <= int(s) <= 114:
            found.append(quran_ref(int(s), int(a)))
    if _asks_for_surah_text(q):
        for num in re.findall(_SURAH_WORD + r"\s*(?:no\.?|number|#|নং)?\s*(\d{1,3})\b(?!\s*[:/]\s*\d)", q):
            found += surah_refs(int(num))
        for m in re.finditer(_SURAH_WORD + r"\s+(?=((?:[a-z']+[-\s]?){1,3}))", q):
            words = re.split(r"\s+", m.group(1).strip())
            for n in (2, 1, 3):  # "al ikhlas" before "al", then "aal-i-imraan"-style names
                name = _fold_name(" ".join(words[:n]))
                close = get_close_matches(name, _SURAH_BY_NAME, n=1, cutoff=0.85) if len(name) >= 5 else []
                num = _SURAH_BY_NAME.get(name) or (_SURAH_BY_NAME[close[0]] if close else None)
                if num:
                    found += surah_refs(num)
                    break
        for name in re.findall(_SURAH_WORD + r"\s*((?:আল-?\s?)?[\u0600-\u06FF\u0980-\u09FF]+)", q):
            num = _SURAH_BY_LOCAL_NAME.get(_fold_local(name))
            if num:
                found += surah_refs(num)
    for key, names in _ALIASES.items():
        if key == "quran":
            continue
        for name in names:
            # Muslim is often cited by Abd al-Baqi number with a letter ("Muslim 1907a"); the retriever looks
            # those up through alt_number.
            for num, letter in re.findall(re.escape(normalize(name)) + _FILLER + r"\s*(\d{1,5})([a-z]?)\b", q):
                found.append(f"{key}:{int(num)}{letter}" if letter and key == "muslim" else hadith_ref(key, int(num)))
    return list(dict.fromkeys(found))
