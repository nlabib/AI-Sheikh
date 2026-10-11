# Retrieval results

Measured 2026-10-11 on the full corpus (121,064 passages: 6,236 ayat in each of 3 languages,
plus 34,046 to 34,157 hadith per language across the six books), with
`python -m tools.evaluate` over `data/eval/gold.jsonl` (98 questions: 65 Quran and 33 hadith;
63 English, 20 Bangla, 15 Arabic).

**Semantic (E5) search has not been measured yet.** The machine these were produced on could
not download models from Hugging Face. Run `python -m tools.index && python -m tools.evaluate`
on your computer and add the numbers here. The keyword baseline below is what E5 needs to beat.

## Keyword search (SQLite FTS5, BM25)

"scoped" means Quran questions search only the Quran and hadith questions only hadith, which is
how `/ask` gathers evidence (`Retriever.search_balanced`). Recall@k: share of questions with at
least one correct source in the top k.

| group | n | recall@1 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|---|
| all, scoped | 98 | 0.429 | 0.653 | 0.745 | 0.534 |
| Quran, English | 40 | 0.275 | 0.425 | 0.600 | 0.358 |
| Quran, Bangla | 15 | 0.600 | 0.733 | 0.733 | 0.667 |
| Quran, Arabic | 10 | 0.600 | 0.800 | 0.900 | 0.669 |
| Hadith, English | 23 | 0.435 | 0.826 | 0.870 | 0.599 |
| Hadith, Bangla | 5 | 0.400 | 0.800 | 0.800 | 0.600 |
| Hadith, Arabic | 5 | 0.800 | 1.000 | 1.000 | 0.900 |
| all, unscoped | 98 | 0.306 | 0.551 | 0.592 | 0.410 |

About 10-15 ms per query on a 4-core cloud VM.

## What the numbers say

- **English questions about the Quran are the weak spot (42.5% in the top 5).** Pickthall's 1930
  English ("Squander not your wealth...", "strong drink and games of chance") shares few words
  with how people ask today ("interest", "alcohol and gambling"). This is exactly the vocabulary
  gap that multilingual embeddings close, so expect the biggest gain here from E5.
- **Unscoped search buries the Quran.** Hadith outnumber ayat about three to one, so a
  whole-corpus search often fills the top 5 with hadith. Scoping lifts Quran recall@5 from 0.385
  to 0.554, which is why `/ask` searches each side separately.
- **Arabic questions do well** because the Arabic text is matched after stripping harakat and
  folding letter variants (`app/text.py`).
- **The numbers are a lower bound.** Relevance lists were built by hand and miss some parallel
  narrations; several "misses" are actually correct (e.g. Tirmidhi 2627 for "a Muslim is one from
  whose tongue and hand others are safe").

## Hashing fallback (not recommended)

`--embedder hashing` hashes character n-grams; it needs no download but captures no meaning.
Hybrid with it scored *lower* than keyword search alone (scoped recall@5 0.592 vs 0.653), so the
server ignores a hashing index unless `EMBEDDER=hashing` is set. It exists for tests.
