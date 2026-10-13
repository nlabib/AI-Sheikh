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
| `quran-en-sahih` | Quran, English, Saheeh International | Modern translation, copyright of its publisher | search only, not shown | **no** |
| `quran-en-khattab` | Quran, English, The Clear Quran (Mustafa Khattab) | Modern translation, copyright of its author | search only, not shown | **no** |
| `quran-bn-muhiuddin` | Quran, Bangla, Muhiuddin Khan | Modern translation, copyright of its translator/publisher | search only, not shown | **no** |
| `<book>-ar` (9 books) | Hadith, Arabic | Classical text, public domain; digitised by fawazahmed0/hadith-api (Unlicense) | yes | yes |
| `<book>-en` (9 books) | Hadith, English translations | Modern translations, copyright of their translators/publishers | yes | **no** |
| `<book>-bn` (8 books) | Hadith, Bangla translations | Modern translations, copyright of their translators/publishers | yes | **no** |
| `tafsir-ibnkathir-ar` | Tafsir Ibn Kathir, Arabic | Classical text (Ibn Kathir d. 1373), public domain; digitised by spa5k/tafsir_api (MIT) | yes | yes |
| `tafsir-jalalayn-ar` | Tafsir al-Jalalayn, Arabic | Classical text (15th century), public domain; digitised by spa5k/tafsir_api (MIT) | yes | yes |
| `tafsir-ibnkathir-en` | Tafsir Ibn Kathir, English (abridged) | Modern abridged translation (Darussalam), copyright of its publisher | yes | **no** |
| `tafsir-ibnkathir-bn` | Tafsir Ibn Kathir, Bangla | Modern translation (Tawheed Publication), copyright of its publisher | yes | **no** |

Books: Sahih al-Bukhari, Sahih Muslim, Sunan Abi Dawud, Jami at-Tirmidhi, Sunan an-Nasa'i,
Sunan Ibn Majah, Muwatta Malik (`malik`), the Forty Hadith of an-Nawawi (`nawawi`) and Forty Hadith
Qudsi (`qudsi`, no Bangla edition upstream). Hadith gradings (e.g. Al-Albani's) come with the Abu
Dawud, Tirmidhi, Nasa'i and Ibn Majah editions and are shown alongside each hadith.

"Search only, not shown": the extra Quran translations help search match modern wording ("interest",
"patience"), but the model is shown one translation per language (Pickthall, Abu Bakr Zakaria), so its
quotes stay checkable and it only sees text it was trained to quote.

Tafsir is split into parts of about 1,500 characters, because Ibn Kathir's commentary on one verse runs
to tens of thousands. A part is cited like `[ibnkathiren:2:255.2]` (the second part on 2:255), and a
commentary that covers several verses is stored once under its first verse. At most `TAFSIR_SLOTS`
(default 1) of the sources shown for a question are tafsir, in the question's language. Tafsir files are
pinned to the `v1.2.1` tag of [spa5k/tafsir_api](https://github.com/spa5k/tafsir_api) and checked with one
SHA-256 over the 114 surah files.

Files come from [fawazahmed0/quran-api](https://github.com/fawazahmed0/quran-api) and
[fawazahmed0/hadith-api](https://github.com/fawazahmed0/hadith-api) (both Unlicense for the
repositories themselves) and [spa5k/tafsir_api](https://github.com/spa5k/tafsir_api) (MIT). The licence of each underlying text is as listed above; the
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

## Committed training answers

`data/sft/generated/*.jsonl` is committed. Those examples quote only training-allowed text (the Tanzil
Arabic Quran verbatim, Pickthall's English, classical Arabic hadith and tafsir); everything else in them
is newly written paraphrase. `tools/grounded_sft.py` rejects any example that quotes another edition.
