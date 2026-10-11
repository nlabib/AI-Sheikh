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
    "abudawud": ["abu dawud", "abudawud", "abu daud", "আবু দাউদ", "ابو داود"],
    "tirmidhi": ["tirmidhi", "at-tirmidhi", "তিরমিযী", "তিরমিজি", "الترمذي"],
    "nasai": ["nasai", "an-nasai", "nasa'i", "নাসাঈ", "নাসায়ী", "النسائي"],
    "ibnmajah": ["ibn majah", "ibnmajah", "ইবনে মাজাহ", "ইবনু মাজাহ", "ابن ماجه"],
}

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


def find_explicit_refs(query: str) -> list[str]:
    """Pull references the user typed directly, e.g. "2:255" or "Bukhari 6018"."""
    q = normalize(query)
    found: list[str] = []
    for s, a in re.findall(r"\b(\d{1,3})\s*[:/]\s*(\d{1,3})\b", q):
        if 1 <= int(s) <= 114:
            found.append(quran_ref(int(s), int(a)))
    for key, names in _ALIASES.items():
        if key == "quran":
            continue
        for name in names:
            for num in re.findall(re.escape(normalize(name)) + r"\s*(?:no\.?|#|নং)?\s*(\d{1,5})", q):
                found.append(hadith_ref(key, int(num)))
    return list(dict.fromkeys(found))
