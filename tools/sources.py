"""Registry of corpus sources, with licence terms and whether each may be used for training.

Everything is pinned to an exact upstream commit and SHA-256, so two people who
run `python -m tools.fetch` get byte-identical files. Nothing here is committed
to this repository; it is downloaded to data/raw/ (git-ignored).

`allow_training` is the rule that tools.build_sft enforces. Retrieval-only
sources can be shown to the user with attribution but are never copied into
fine-tuning data.
"""

from __future__ import annotations

from dataclasses import dataclass

QURAN_API_COMMIT = "47ca096b0976443ba2eab2e45cdf0fb4096a2610"
HADITH_API_COMMIT = "df57907be35291c91ad6a6691180e22ca9920784"
# spa5k/tafsir_api serves one JSON file per surah; this tag is immutable on GitHub.
TAFSIR_API_TAG = "v1.2.1"

_MIRRORS = {
    "quran": [
        "https://raw.githubusercontent.com/fawazahmed0/quran-api/{commit}/editions/{edition}.json",
        "https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@{commit}/editions/{edition}.json",
    ],
    "hadith": [
        "https://raw.githubusercontent.com/fawazahmed0/hadith-api/{commit}/editions/{edition}.json",
        "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@{commit}/editions/{edition}.json",
    ],
    "tafsir": [
        "https://raw.githubusercontent.com/spa5k/tafsir_api/{commit}/tafsir/{edition}/{surah}.json",
        "https://cdn.jsdelivr.net/gh/spa5k/tafsir_api@{commit}/tafsir/{edition}/{surah}.json",
    ],
}

SHA256 = {
    "ara-quransimple": "e430798cdda553f703b54f5601835bbbd8e9a81dce3db42c7e6f09314a4ec3cb",
    "eng-mohammedmarmadu": "aaff752d4349005282d977eebb86398279330c8df8736d4528a7afe80dd9ce66",
    "ben-abubakrzakaria": "d8e12d62cc24a4c268f8eca7411e89c6ede34532dc210578e66a2dc509f2f6c5",
    "ara-bukhari": "e34a3402889ca378871da3c5984b7875c680920e93f1c8738ac7afe502179562",
    "ara-muslim": "194073b24090c5368e4d14a3f55c9e0ef144c7437bd7e5aa83aa4d2bc54a8ea0",
    "ara-abudawud": "216139c5f40a8147e700d62ffa3b5b2fe45d0e034f86941a193c598c10056189",
    "ara-tirmidhi": "408028cd56329ed78edbe3ad443beb96b58e16c3e5bd49f9bc7b0eac77c8820f",
    "ara-nasai": "c705983c19bb089c87cd6531aff609bd94d862183ed0859115b0ff991826a956",
    "ara-ibnmajah": "b406e6be81588ab0ec1ddbed32f5fd95bbcdda242c460d88331e82eb5c2bc0aa",
    "eng-bukhari": "b3a10b8d50949e86cec82b0d323ce0eae3e2f713fa2f3daa5361638feb795988",
    "eng-muslim": "2ef51a2a1980fb91e395fbf1580cc9c9413297f54bb8cad271e1499417299488",
    "eng-abudawud": "f6bcf73ad96d2f28296232e201e12963e7b3dd5325572fb88365edbbfb0ecd40",
    "eng-tirmidhi": "f64f1a5e6f2614e164e3b10138412af322d3959011bf84d67f428f79b3858afb",
    "eng-nasai": "4f2aaca4229b96210fdaa887915c69ba99dda8726816856cc8660e996b6b3a1f",
    "eng-ibnmajah": "5e9479efd403e935ddcf746a709b2727d535a41fe1f496deadc8a43f1d103cd7",
    "ben-bukhari": "8f5e574422e2c2eecaa42bcd80ce90d9a1f29c82c2143bc789965dc364d91156",
    "ben-muslim": "876591fcf5954af41b14a67b28f732df74713010fef3c9e0117661ed677273dd",
    "ben-abudawud": "4050510ed8277d376130af107f4b32d7a4a5dfd4010867f41651f2cf55a83d4d",
    "ben-tirmidhi": "17080a67cff1fe216e8ba8eda3a628952f745027d7d682cf3101ddd56459e72c",
    "ben-nasai": "11a4ad2a41ec2d259f7fd9b5959496bf5cf8424cdf0944bea197927b68d59d72",
    "ben-ibnmajah": "8263d8d1c65aa5710d0f45198a07e3c62d6e1fe1ca54e4ac54d3685e5b6bc8c9",
    "ara-malik": "088b1f354e110211da006d47e37dd8f8837368f318ed683271f0fdb0f09a3ab5",
    "eng-malik": "fb02a0ae16ebba1a658b107da3b93e72ecb59c2acd871819e2b1f3bd2e90d27b",
    "ben-malik": "c79d3a18a0a34fa3afc11104f7c4933cd45e27b73c2814e8e02e222e6a3741a2",
    "ara-nawawi": "91f559b90422005b768bb449426d87d2e6fabbb43ac3a31e5c24c50ca77465e6",
    "eng-nawawi": "3c8124e3f633a576d1fb6e6a9cb352e162436781b15bc78188c28203ff08743d",
    "ben-nawawi": "00cea983ca9023f7cd05a8e16fd032ff647ebc81446fe3fb26d4daf8b92dc69d",
    "ara-qudsi": "8645482e46b9ab145b3a8b740ffc7dc6902cc1cc0a1845da35b2057ec23bb55a",
    "eng-qudsi": "4d9a35148af7b04ec41f6cc3dbaf0bdb82d2f2c93d03052ee229e86dd0f6c53c",
    "eng-ummmuhammad": "5ac5aab3e152dd76b615d8748e2198c428f88cd3bd19d240b3e8feb89de6f28c",
    "eng-mustafakhattaba": "0d7458a1d8ebe1402d3e142947f482d5f89f67e276fe0dc25e0ff567f8f9ffe6",
    "ben-muhiuddinkhan": "96ca3d09a0c821aae2a0e4265739aa0314b9aefee8ad63bfd7ca8d3872581b00",
    # Tafsir: SHA-256 of the 114 surah files concatenated in order.
    "ar-tafsir-ibn-kathir": "841f59ca557d929971160fb252b7a39406cf72e2fc96a3ff9ee922219eb64a74",
    "en-tafisr-ibn-kathir": "fddbcb18e4c13f09232a6e11ae02e8d37b01b65aa7eea65e6f2ac7c5d5ca0557",
    "bn-tafseer-ibn-e-kaseer": "8dc286c9b2054cbec3dbf7766cada7c0299be53d211ef9c751931fd17d6097f3",
    "ar-tafsir-al-jalalayn": "3e50c50860cd1624d74eb9a9c6d3478a27618f5d61426811582106540deec182",
}

LICENCES = {
    "tanzil": (
        "Tanzil Quran Text (Simple), tanzil.net. Creative Commons Attribution 3.0: may be copied and "
        "redistributed verbatim with attribution to tanzil.net; the text must not be changed."
    ),
    "public-domain-pickthall": (
        "Marmaduke Pickthall, 'The Meaning of the Glorious Koran' (1930). The author died in 1936, so the "
        "translation is in the public domain under life+70 rules, and its US term (95 years) ended in 2026."
    ),
    "kfgqpc-free": (
        "Bangla translation by Dr. Abu Bakr Muhammad Zakaria, published by the King Fahd Glorious Quran "
        "Printing Complex for free distribution. Not openly licensed: kept for local retrieval and display "
        "with attribution only, not for fine-tuning or redistribution."
    ),
    "classical-arabic": (
        "Classical Arabic hadith text (compiled in the 9th century, public domain). Digitised edition from "
        "fawazahmed0/hadith-api, released under the Unlicense."
    ),
    "classical-tafsir": (
        "Classical Arabic tafsir (Ibn Kathir, d. 1373; al-Mahalli and al-Suyuti, 15th century), public domain. "
        "Digitised edition from spa5k/tafsir_api (MIT licence for the repository)."
    ),
    "modern-quran-translation": (
        "Modern Quran translation whose copyright belongs to its translator/publisher. Mirrored by "
        "fawazahmed0/quran-api; kept for local search only (not shown as evidence, not used for fine-tuning)."
    ),
    "modern-tafsir-translation": (
        "Modern translation of a classical tafsir; copyright of its translator/publisher (English: Darussalam "
        "abridgement; Bangla: Tawheed Publication). Mirrored by spa5k/tafsir_api; kept for local retrieval and "
        "display with attribution only, not for fine-tuning or redistribution."
    ),
    "modern-translation": (
        "Modern translation whose copyright belongs to its translator/publisher. Mirrored by "
        "fawazahmed0/hadith-api; kept for local retrieval and display with attribution only, not for "
        "fine-tuning or redistribution."
    ),
}


@dataclass(frozen=True)
class Source:
    source_id: str
    kind: str  # "quran" | "hadith" | "tafsir"
    collection: str  # "quran", a hadith collection key or a tafsir collection key
    edition: str  # upstream edition file name
    lang: str
    title: str
    licence: str  # key into LICENCES
    allow_training: bool
    default: bool = True
    # Shown to the model as evidence. A second translation in a language we already show only helps search.
    display: bool = True

    @property
    def commit(self) -> str:
        return {"quran": QURAN_API_COMMIT, "hadith": HADITH_API_COMMIT, "tafsir": TAFSIR_API_TAG}[self.kind]

    @property
    def urls(self) -> list[str]:
        """Download URLs; a tafsir has one file per surah, so its URLs still contain "{surah}"."""
        return [u.format(commit=self.commit, edition=self.edition, surah="{surah}") for u in _MIRRORS[self.kind]]

    @property
    def sha256(self) -> str:
        return SHA256[self.edition]

    @property
    def filename(self) -> str:
        """A file in data/raw/, or for a tafsir a directory of 1.json … 114.json."""
        return f"tafsir/{self.edition}" if self.kind == "tafsir" else f"{self.edition}.json"

    def info(self) -> dict:
        return {
            "title": self.title, "lang": self.lang, "collection": self.collection, "edition": self.edition,
            "licence": LICENCES[self.licence], "allow_training": self.allow_training, "display": self.display,
            "upstream": self.urls[0], "sha256": self.sha256,
        }


_HADITH_TITLES = {
    "bukhari": "Sahih al-Bukhari", "muslim": "Sahih Muslim", "abudawud": "Sunan Abi Dawud",
    "tirmidhi": "Jami at-Tirmidhi", "nasai": "Sunan an-Nasa'i", "ibnmajah": "Sunan Ibn Majah",
    "malik": "Muwatta Malik", "nawawi": "Forty Hadith of an-Nawawi", "qudsi": "Forty Hadith Qudsi",
}
# Upstream has no Bangla edition of these.
_NO_BANGLA = {"qudsi"}

SOURCES: list[Source] = [
    Source("quran-ar-tanzil", "quran", "quran", "ara-quransimple", "ar",
           "Quran, Arabic (Tanzil Simple)", "tanzil", allow_training=True),
    Source("quran-en-pickthall", "quran", "quran", "eng-mohammedmarmadu", "en",
           "Quran, English (Pickthall, 1930)", "public-domain-pickthall", allow_training=True),
    Source("quran-bn-zakaria", "quran", "quran", "ben-abubakrzakaria", "bn",
           "Quran, Bangla (Abu Bakr Zakaria)", "kfgqpc-free", allow_training=False),
]
SOURCES += [
    Source("quran-en-sahih", "quran", "quran", "eng-ummmuhammad", "en",
           "Quran, English (Saheeh International)", "modern-quran-translation", allow_training=False, display=False),
    Source("quran-en-khattab", "quran", "quran", "eng-mustafakhattaba", "en",
           "Quran, English (The Clear Quran, Mustafa Khattab)", "modern-quran-translation", allow_training=False,
           display=False),
    Source("quran-bn-muhiuddin", "quran", "quran", "ben-muhiuddinkhan", "bn",
           "Quran, Bangla (Muhiuddin Khan)", "modern-quran-translation", allow_training=False, display=False),
]
for _key, _title in _HADITH_TITLES.items():
    SOURCES += [
        Source(f"{_key}-ar", "hadith", _key, f"ara-{_key}", "ar", f"{_title}, Arabic",
               "classical-arabic", allow_training=True),
        Source(f"{_key}-en", "hadith", _key, f"eng-{_key}", "en", f"{_title}, English",
               "modern-translation", allow_training=False),
    ]
    if _key not in _NO_BANGLA:
        SOURCES.append(Source(f"{_key}-bn", "hadith", _key, f"ben-{_key}", "bn", f"{_title}, Bangla",
                              "modern-translation", allow_training=False))
SOURCES += [
    Source("tafsir-ibnkathir-ar", "tafsir", "ibnkathir", "ar-tafsir-ibn-kathir", "ar",
           "Tafsir Ibn Kathir, Arabic", "classical-tafsir", allow_training=True),
    Source("tafsir-ibnkathir-en", "tafsir", "ibnkathiren", "en-tafisr-ibn-kathir", "en",
           "Tafsir Ibn Kathir, English (abridged)", "modern-tafsir-translation", allow_training=False),
    Source("tafsir-ibnkathir-bn", "tafsir", "ibnkathirbn", "bn-tafseer-ibn-e-kaseer", "bn",
           "Tafsir Ibn Kathir, Bangla", "modern-tafsir-translation", allow_training=False),
    Source("tafsir-jalalayn-ar", "tafsir", "jalalayn", "ar-tafsir-al-jalalayn", "ar",
           "Tafsir al-Jalalayn, Arabic", "classical-tafsir", allow_training=True),
]

BY_ID = {s.source_id: s for s in SOURCES}


def select(ids: list[str] | None = None, quran_only: bool = False) -> list[Source]:
    if ids:
        unknown = [i for i in ids if i not in BY_ID]
        if unknown:
            raise SystemExit(f"unknown source ids: {unknown}. Known: {sorted(BY_ID)}")
        return [BY_ID[i] for i in ids]
    return [s for s in SOURCES if s.default and (not quran_only or s.kind == "quran")]
