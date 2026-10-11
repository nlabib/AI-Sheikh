# Roadmap

## Done

- [x] FastAPI app: `/ask`, `/search`, `/ref/{ref}`, `/sources`, `/health`
- [x] Corpus: full Quran (Arabic, English, Bangla) and the six major hadith books (Arabic,
      English, Bangla), pinned and checksummed, with licence flags per source
- [x] Keyword search (FTS5) that handles Arabic diacritics and Bangla script correctly
- [x] Multilingual semantic search (E5) and hybrid fusion, balanced between Quran and hadith
- [x] Citation checker: invented refs, refs not in evidence, misquotes, uncited claims
- [x] 98-question gold set in three languages; retrieval benchmark (keyword baseline measured)
- [x] Licence-aware SFT data builder; QLoRA (CUDA) and MLX training; adapter merge
- [x] Step-by-step training guide

## Next

1. **Measure E5 retrieval** on your machine and record it in `docs/RESULTS.md`
   (`python -m tools.index && python -m tools.evaluate`).
2. **First training run** with the 12 GB config; record time, peak VRAM and loss.
3. **Write 200-500 reviewed examples** of real questions with good cited answers
   (`data/sft/reviewed/`). This matters more than any setting.
4. **Grow the gold set** to 300+ questions, including questions whose honest answer is
   "the sources don't cover this", and have someone knowledgeable check the relevance lists.
5. **Reranker**: a cross-encoder (e.g. `BAAI/bge-reranker-v2-m3`) over the top 30 candidates.
6. **Claim-level verification** in the citation checker (see docs/EVALUATION.md).
7. **Tafsir** (e.g. Ibn Kathir, Arabic, public domain) as a third evidence type, kept separate
   from Quran and hadith in prompts and citations.
8. **Optional API sources**: Quran Foundation and Sunnah.com, behind API keys, for more
   translations, with their terms recorded in `tools/sources.py`.
9. **A simple web UI** over `/ask` that shows sources side by side with the answer.
