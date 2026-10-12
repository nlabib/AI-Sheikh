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
from app.text import normalize

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


def _fold_name(name: str) -> str:
    """"Al-Ikhlaas", "al ikhlas" and "Ikhlas" all fold to "ikhlas"."""
    n = normalize(name).replace("'", "").replace("’", "")
    n = re.sub(r"^(?:a[lnstdrz]|ash|adh)[-\s]+", "", n)
    n = re.sub(r"[-\s]+", "", n)
    n = n.replace("ee", "i").replace("oo", "u").replace("th", "t").replace("dh", "d").replace("q", "k")
    n = re.sub(r"([aiueo])\1+", r"\1", n)
    return re.sub(r"h$", "", n)


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
    for key, names in _ALIASES.items():
        if key == "quran":
            continue
        for name in names:
            for num in re.findall(re.escape(normalize(name)) + _FILLER + r"\s*(\d{1,5})\b", q):
                found.append(hadith_ref(key, int(num)))
    return list(dict.fromkeys(found))
