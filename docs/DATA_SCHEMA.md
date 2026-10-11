# Data schema

## Passages (`data/corpus.sqlite`, table `passages`)

One row per (ref, edition). Quran 2:255 has three rows (ar, en, bn) sharing `ref = "quran:2:255"`.

| column | example | meaning |
|---|---|---|
| `ref` | `quran:2:255`, `bukhari:1`, `bukhari:402.2` | canonical reference, also the citation marker `[ref]` |
| `collection` | `quran`, `bukhari`, `muslim`, `abudawud`, `tirmidhi`, `nasai`, `ibnmajah` | |
| `lang` | `ar`, `en`, `bn` | |
| `edition` | `ara-quransimple` | upstream edition file |
| `text` | | exactly as published; never modified |
| `source_id` | `quran-en-pickthall` | key into `tools/sources.py` and the `sources` table |
| `allow_training` | 0/1 | whether this text may appear in fine-tuning data |
| `meta` | JSON | Quran: `surah`, `ayah`, `surah_name`, `surah_meaning`. Hadith: `number`, `book`, `in_book`, `chapter_title`, `grades` (list of `{name, grade}`), Muslim also `alt_number` |

`passages_fts` is an FTS5 index over a normalised copy of `text` (lower-cased, Arabic harakat
removed, alef/ya/ta-marbuta folded, Bangla/Arabic digits to ASCII; see `app/text.py`).
The `sources` table holds each source's title, licence, upstream URL and checksum.

## Importing your own passages

`python -m tools.ingest my.jsonl` accepts one JSON object per line:

```json
{"ref": "quran:1:1", "collection": "quran", "lang": "en", "edition": "my-edition",
 "source_id": "my-source", "allow_training": false, "text": "...", "meta": {}}
```

`ref`, `collection`, `lang` and `text` are required. Re-importing the same (ref, edition)
replaces the old row.

## Vector index (`data/index/`)

`vectors.npy` (float32, one unit-length row per passage), `ids.npy` (passage id per row) and
`meta.json` (embedder name, count, dimension). Rebuild with `python -m tools.index` after any
ingest.

## Gold questions (`data/eval/gold.jsonl`)

```json
{"id": "q-en-02", "lang": "en", "type": "quran", "query": "Can someone be forced to accept Islam?",
 "relevant": ["quran:2:256"]}
```

`relevant` lists every ref that answers the question (any one counts as a hit). Hadith refs were
found by searching the English texts for the hadith's wording, so parallel narrations with
different wording may be missing; treat scores as a lower bound.

## Fine-tuning examples (`data/sft/*.jsonl`)

See `training/data.py` and `docs/TRAINING_GUIDE.md`, step 5.
