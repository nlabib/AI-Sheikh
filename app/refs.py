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
    "malik": "Muwatta Malik",
    "nawawi": "Forty Hadith of an-Nawawi",
    "qudsi": "Forty Hadith Qudsi",
}

# Tafsir, keyed by verse like the Quran. Each language edition is its own collection because long commentary
# is split into chunks ("ibnkathiren:2:255.2" is the second chunk on 2:255), and the chunks of different
# translations don't line up.
TAFSIR_COLLECTIONS = {
    "ibnkathir": "Tafsir Ibn Kathir (Arabic)",
    "ibnkathiren": "Tafsir Ibn Kathir (English, abridged)",
    "ibnkathirbn": "Tafsir Ibn Kathir (Bangla)",
    "jalalayn": "Tafsir al-Jalalayn (Arabic)",
}
TAFSIR_LANGS = {"ibnkathir": "ar", "ibnkathiren": "en", "ibnkathirbn": "bn", "jalalayn": "ar"}

# Spellings people actually type, in English, Bangla and Arabic.
_ALIASES = {
    "quran": ["quran", "qur'an", "koran", "q", "কুরআন", "কোরআন", "القران", "قران"],
    "bukhari": ["bukhari", "al-bukhari", "sahih bukhari", "বুখারী", "বুখারি", "البخاري"],
    "muslim": ["muslim", "sahih muslim", "মুসলিম", "مسلم"],
    "abudawud": ["abu dawud", "abi dawud", "abu dawood", "abudawud", "abu daud", "আবু দাউদ", "ابو داود", "ابي داود"],
    "tirmidhi": ["tirmidhi", "at-tirmidhi", "তিরমিযী", "তিরমিজি", "الترمذي"],
    "nasai": ["nasai", "an-nasai", "nasa'i", "an-nasa'i", "নাসাঈ", "নাসায়ী", "النسائي"],
    "ibnmajah": ["ibn majah", "ibnmajah", "ইবনে মাজাহ", "ইবনু মাজাহ", "ابن ماجه"],
    "malik": ["muwatta malik", "muwatta", "muwatta imam malik", "মুয়াত্তা মালিক", "মুওয়াত্তা", "موطأ مالك", "الموطأ"],
    "nawawi": ["nawawi", "an-nawawi", "40 hadith nawawi", "forty hadith nawawi", "নববী", "الأربعين النووية", "النووية"],
    "qudsi": ["hadith qudsi", "40 hadith qudsi", "qudsi", "হাদীসে কুদসী", "হাদিসে কুদসি", "الأحاديث القدسية", "القدسية"],
}

# Words people put between a collection name and the number: "Bukhari hadith no. 1", "مسلم حديث رقم ٣٩٤".
# The word boundary only goes on the Latin words: Bangla words end in a vowel sign, which \b doesn't count as a
# word character, so "নং\b" never matches before a space.
_FILLER = r"(?:\s*(?:(?:hadith|hadees|hadis|number|num|no|nr)\b|#|নং|নম্বর|হাদিস|হাদীস|حديث|رقم)\.?)*"
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
# "Ibn Kathir on 2:255" asks for that tafsir's first part on the verse as well as the verse; "tafsir of 2:255"
# asks for every tafsir's.
_TAFSIR_WORDS = {
    "ibnkathir": ["ibn kathir", "ibn kaseer", "ইবনে কাসীর", "ইবনে কাসির", "ابن كثير"],
    "ibnkathiren": ["ibn kathir", "ibn kaseer"],
    "ibnkathirbn": ["ইবনে কাসীর", "ইবনে কাসির", "ibn kathir"],
    "jalalayn": ["jalalayn", "jalalain", "জালালাইন", "الجلالين"],
}
_TAFSIR_ANY = ["tafsir", "tafseer", "commentary", "তাফসীর", "তাফসির", "تفسير"]
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

MARKER = re.compile(r"\[((?:quran:\d{1,3}:\d{1,3}(?:-\d{1,3})?)|(?:[a-z]+:\d{1,3}:\d{1,3}(?:\.\d{1,2})?)"
                    r"|(?:[a-z]+:\d+(?:\.\d+)?))\]")


@dataclass(frozen=True)
class Ref:
    collection: str  # "quran", a HADITH_COLLECTIONS key or a TAFSIR_COLLECTIONS key
    number: str  # "2:255" for Quran, "1" for hadith

    def __str__(self) -> str:
        return f"{self.collection}:{self.number}"

    @property
    def label(self) -> str:
        if self.collection == "quran":
            return f"Quran {self.number}"
        if self.collection in TAFSIR_COLLECTIONS:
            verse, _, part = self.number.partition(".")
            return f"{TAFSIR_COLLECTIONS[self.collection]} on {verse}" + (f" (part {part})" if part else "")
        return f"{HADITH_COLLECTIONS.get(self.collection, self.collection)} {self.number}"


def quran_ref(surah: int, ayah: int) -> str:
    return f"quran:{surah}:{ayah}"


def tafsir_ref(collection: str, surah: int, ayah: int, part: int = 1) -> str:
    return f"{collection}:{surah}:{ayah}" + (f".{part}" if part > 1 else "")


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
    tafsirs = [c for c, words in _TAFSIR_WORDS.items() if any(w in q for w in words)]
    if not tafsirs and any(w in q for w in _TAFSIR_ANY):
        tafsirs = list(TAFSIR_COLLECTIONS)
    for s, a in re.findall(r"\b(\d{1,3})\s*[:/]\s*(\d{1,3})\b", q):
        if 1 <= int(s) <= 114:
            found.append(quran_ref(int(s), int(a)))
            found += [tafsir_ref(c, int(s), int(a)) for c in tafsirs]
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


_QURAN_WORDS = re.compile(r"quran|qur'an|koran|surah|sura\b|verse|ayah|ayat|aya\b|কুরআন|কোরআন|সূরা|সুরা|আয়াত|القرآن|قرآن|سورة|الآية|آية")
_CLOCK_SUFFIX = re.compile(r"\s*(?:am|pm|a\.m|p\.m|o'?clock)\b")


def find_quran_mentions(query: str) -> list[tuple[int, int]]:
    """Every "S:A" the user typed as a Quran reference, valid or not ("115:1", "2:999"). Unlike
    find_explicit_refs it keeps out-of-range numbers, so a made-up verse can be told apart from a real one.
    A pair only counts when the question talks about the Quran, or the second number can't be a minute
    (60 or more), and never when it is followed by am/pm ("pray at 1:30 pm")."""
    q = normalize(query)
    context = bool(_QURAN_WORDS.search(q))
    out = []
    for m in re.finditer(r"\b(\d{1,3})\s*[:/]\s*(\d{1,3})\b", q):
        if _CLOCK_SUFFIX.match(q, m.end()):
            continue
        if context or int(m.group(2)) >= 60:
            out.append((int(m.group(1)), int(m.group(2))))
    return list(dict.fromkeys(out))


# Verses named in words: "verse 8 of al-Fatiha", "সূরা ফাতিহার ৮ নম্বর আয়াত", "الآية السابعة من سورة الناس".
# Asked for verse 7 of an-Nas (it has 6), the 8B fine-tune quoted 7:158 as if it were that verse.
_VERSE_WORD = r"(?:verses?|ayahs?|ayat|aya|ayath|আয়াত|আয়াত|ايه|اية)"
# Arabic ordinals for "the seventh verse" (after normalize: ة -> ه, ى -> ي, أ -> ا).
_AR_ORDINALS = {normalize(w): n for n, w in enumerate(
    ["الأولى", "الثانية", "الثالثة", "الرابعة", "الخامسة", "السادسة", "السابعة", "الثامنة", "التاسعة", "العاشرة"], 1)}


def _surahs_named(q: str) -> list[int]:
    """Surah numbers a normalized question names: "surah 112", "Surah Al-Ikhlas", "সূরা ফাতিহার", "سورة الناس"."""
    found = [int(n) for n in re.findall(_SURAH_WORD + r"\s*(?:no\.?|number|#|নং|رقم)?\s*(\d{1,3})\b(?!\s*[:/]\s*\d)", q)]
    for m in re.finditer(_SURAH_WORD + r"\s+(?=((?:[a-z']+[-\s]?){1,3}))", q):
        words = re.split(r"\s+", m.group(1).strip())
        for n in (2, 1, 3):
            name = _fold_name(" ".join(words[:n]))
            close = get_close_matches(name, _SURAH_BY_NAME, n=1, cutoff=0.85) if len(name) >= 5 else []
            num = _SURAH_BY_NAME.get(name) or (_SURAH_BY_NAME[close[0]] if close else None)
            if num:
                found.append(num)
                break
    # Arabic letters only (U+0621-U+064A), so "الناس؟" doesn't drag its question mark along.
    for name in re.findall(_SURAH_WORD + r"\s*((?:আল-?\s?)?[ء-يঀ-৿]+)", q):
        folded = _fold_local(name)
        for candidate in (folded, re.sub(r"(?:ের|এর|র)$", "", folded)):  # Bangla possessive: "ফাতিহার"
            if candidate in _SURAH_BY_LOCAL_NAME:
                found.append(_SURAH_BY_LOCAL_NAME[candidate])
                break
    return list(dict.fromkeys(found))


def _verse_numbers(q: str) -> list[int]:
    nums = [int(n) for n in re.findall(_VERSE_WORD + r"\s*(?:no\.?|number|#|নং|নম্বর|رقم)?\s*(\d{1,3})\b", q)]
    nums += [int(n) for n in re.findall(r"\b(\d{1,3})\s*(?:st|nd|rd|th)?\s*(?:no\.?|নং|নম্বর)?\s*" + _VERSE_WORD, q)]
    nums += [_AR_ORDINALS[w] for w in re.findall(r"(?:ايه|اية)\s+(ال\S+)", q) if w in _AR_ORDINALS]
    return list(dict.fromkeys(nums))


def find_impossible_named_refs(query: str) -> list[str]:
    """Quran references written in words that cannot exist: "surah 115" -> "quran:115", "verse 8 of
    al-Fatiha" -> "quran:1:8". Verse numbers only count when the question names exactly one surah."""
    q = normalize(query)
    surahs = _surahs_named(q)
    out = [f"quran:{s}" for s in surahs if not 1 <= s <= 114]
    valid = [s for s in surahs if 1 <= s <= 114]
    if len(valid) == 1:
        s = valid[0]
        out += [quran_ref(s, a) for a in _verse_numbers(q) if not 1 <= a <= SURAHS[s][2]]
    return list(dict.fromkeys(out))
