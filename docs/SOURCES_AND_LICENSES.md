# Sources and licences

No source text is stored in this repository except a small test fixture of public-domain and
CC BY material (`data/sample/`). `python -m tools.fetch` downloads everything below, pinned to
an exact upstream commit and verified by SHA-256 (`tools/sources.py`).

Each source has a `allow_training` flag. `tools/build_sft.py` only puts training-allowed text in
fine-tuning data; everything else is used only for local search and display with attribution.

| Source id | Text | Licence / status | Search | Training |
|---|---|---|---|---|
| `quran-ar-tanzil` | Quran, Arabic, Tanzil "Simple" text | Tanzil Quran Text, CC BY 3.0: verbatim copies with attribution to tanzil.net, no changes to the text | yes | yes |
| `quran-en-pickthall` | Quran, English, M. Pickthall (1930) | Public domain: author died 1936; US term (95 years) ended 2026 | yes | yes |
| `quran-bn-zakaria` | Quran, Bangla, Abu Bakr Muhammad Zakaria | Published free by King Fahd Glorious Quran Printing Complex; not openly licensed | yes | **no** |
| `<book>-ar` (6 books) | Hadith, Arabic | Classical text, public domain; digitised by fawazahmed0/hadith-api (Unlicense) | yes | yes |
| `<book>-en` (6 books) | Hadith, English translations | Modern translations, copyright of their translators/publishers | yes | **no** |
| `<book>-bn` (6 books) | Hadith, Bangla translations | Modern translations, copyright of their translators/publishers | yes | **no** |

Books: Sahih al-Bukhari, Sahih Muslim, Sunan Abi Dawud, Jami at-Tirmidhi, Sunan an-Nasa'i,
Sunan Ibn Majah. Hadith gradings (e.g. Al-Albani's) come with the Abu Dawud, Tirmidhi, Nasa'i
and Ibn Majah editions and are shown alongside each hadith.

Files come from [fawazahmed0/quran-api](https://github.com/fawazahmed0/quran-api) and
[fawazahmed0/hadith-api](https://github.com/fawazahmed0/hadith-api) (both Unlicense for the
repositories themselves). The licence of each underlying text is as listed above; the
repositories' Unlicense does not override a translator's copyright.

This table is a good-faith reading, not legal advice. Before publishing a model or app, check the
terms yourself, especially for any commercial use.

## Why not the Quran Foundation or Sunnah.com APIs?

Both are good sources, but they need API keys, and their terms have to be checked before storing
their data in bulk or using it for training. The files above can be stored locally and searched
offline today.
Adding either API as an optional source is listed in `docs/ROADMAP.md`.

## Hadith numbering

Refs use each edition's `hadithnumber` (`bukhari:1` is the hadith on intentions). Sahih Muslim
there is numbered sequentially; its familiar Abd al-Baqi number is kept as `alt_number` and shown
in labels (e.g. `muslim:4927` is "Abd al-Baqi no. 1907a"). Spot-check numbers in other
collections against a printed edition or sunnah.com before publishing references.
