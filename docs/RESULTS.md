# Results

Measured 2026-10-11 on the full corpus (121,064 passages: 6,236 ayat in each of 3 languages,
plus 34,046 to 34,157 hadith per language across the six books), with
`python -m tools.evaluate` over `data/eval/gold.jsonl` (98 questions: 65 Quran and 33 hadith;
63 English, 20 Bangla, 15 Arabic).

## Semantic (E5) and hybrid search

Measured 2026-10-11 on an RTX 3090 with `intfloat/multilingual-e5-small`
(`python -m tools.evaluate --out results/retrieval-e5.json`). Scoped, all 98 questions:

| mode | recall@1 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|
| keyword (below) | 0.429 | 0.653 | 0.745 | 0.534 |
| E5 only | 0.255 | 0.469 | 0.531 | 0.339 |
| **hybrid (keyword + E5)** | **0.459** | **0.684** | **0.806** | **0.567** |
| hybrid, Quran English (n=40) | 0.425 | 0.575 | 0.700 | 0.504 |

Hybrid search beats keyword search. The gain is largest exactly where predicted below: English
questions about the Quran (recall@5 0.425 → 0.575). E5 on its own is weaker than keyword search,
and poor on Arabic (Quran/ar recall@5 0.100, hadith/ar 0.200), so it only pays off combined with
keyword search, which is what `/ask` does.

## Answers: base Qwen3-4B vs fine-tuned

Measured 2026-10-12 with `python -m tools.eval_answers`: 98 gold questions plus 14 unanswerable
trick questions. Models were served from LM Studio as f16 GGUF, 16k context, non-thinking mode,
hybrid retrieval with k=6. The expected source was among the retrieved ones for 62 of 98 questions.
The fine-tune is QLoRA on the 1,140 template examples (`qwen3-4b-24gb.yaml` with batch 2 x
accumulation 8), either for 20 steps or the full 2 epochs (144 steps). Run details are in
`docs/TRAINING_LOG.md`.

| model | pass | pass, actually answering | fail | cites an expected source | "not sure" to answerable | trick: "not sure" | trick: cited anyway |
|---|---|---|---|---|---|---|---|
| base Qwen3-4B | 18 | 11 | 28 | **60** | 5 | 11/14 | 3/14 |
| fine-tuned, 20 steps | 42 | 12 | 32 | 41 | 29 | **14/14** | **0/14** |
| **fine-tuned, 144 steps** | **57** | **32** | **22** | 39 | 6 | 13/14 | 1/14 |

"pass" counts a clean "I'm not sure" as correct, so the "actually answering" column shows answers
that passed without declining.

What it shows:

- **Fine-tuning made answers more checkable and more honest.** The full fine-tune almost tripled
  the passing real answers (11 → 32), cut failures (28 → 22), and declined 13 of 14 trick questions
  with only one citation-backed answer (base: 3). Its one miss quoted the real, related hadith
  "purity is half of faith" (Muslim 223) for "cleanliness is next to godliness".
- **A short run over-refuses.** After 20 steps the model declines every trick question, but also
  29 of 98 questions it should answer. The full run brings that down to 6.
- **It learned the templates' shape too well.** The templates always quote one source, so the
  fine-tune cites 0.9 sources per answer (base: 2.1) and writes shorter answers (233 vs 602
  characters). When the expected source was retrieved, it cited it 27 times out of 39 (base: 37).
  It sometimes picks the wrong one of the six sources, and 16 times it wrote `[1]` (the source's
  position in the prompt) instead of `[quran:2:255]`. That is the next thing to fix with reviewed,
  multi-source examples (`docs/TRAINING_GUIDE.md`, step 5).
- **Serve it in non-thinking mode.** Shown Qwen3's thinking-mode prompt (LM Studio's default), the
  144-step model emitted junk before 36 of its answers. `app/llm.py` now requests non-thinking mode
  (`reasoning_effort: none`, `chat_template_kwargs`), which fixed this completely.
- Base Qwen3-4B in thinking mode (LM Studio's default) did no better than non-thinking: 5 pass,
  38 fail, 10/14 trick questions declined.

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
