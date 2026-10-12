# Evaluation

Three layers, cheapest first.

## 1. Retrieval: does the right source reach the model?

`python -m tools.evaluate` runs the 98 gold questions and reports recall@1/5/10 and MRR@10 by
type (Quran/hadith) and language, for keyword, vector and hybrid search, each over the whole
corpus and scoped. Current numbers: `docs/RESULTS.md`. Answer quality can't exceed this.

## 2. Citation checks on every answer

`app/citations.py` runs on every `/ask` response and in `tools/eval_answers.py`:

| check | status if it fails |
|---|---|
| marker names a passage that exists | fail (`nonexistent_reference`) |
| marker names a passage that was given to the model | fail (`uncited_evidence`) |
| text in quotation marks appears in the cited passage (normalised; ≥90% match for long quotes) | fail (`misquote`) |
| each substantive sentence carries a marker | warn (`uncited_claim`) |
| cited sentence shares vocabulary with its source, unless it carries a verified quote | warn (`weak_support`) |
| answer makes substantive claims and cites nothing | fail (`unsupported_answer`) |
| answer cites nothing and doesn't open by declining | fail (`no_citations`) |
| a sentence says "the Quran" but cites only hadith, or "the Prophet said" but cites only verses | fail (`misattributed`) |
| Bangla text outside quotations in an English or Arabic answer | fail (`mixed_language`) |
| one sentence repeated three or more times (twice: warn, `repeated_sentence`) | fail (`repetition`) |
| a cited passage scores below the relevance floor (`RERANK_MIN` with a reranker, else `MIN_RELEVANCE`) | warn (`off_topic_citation`) |
| every cited passage is below the floor | fail (`off_topic_answer`) |

A refusal passes when it opens by declining ("I'm not sure…", or "The sources don't cover this…") and
makes no uncited claims; sentences explaining what the sources lack are allowed. "I'm not sure" followed
by claims is not a refusal: it fails as `unsupported_answer`. A claim with a refusal phrase tucked inside
("wudu is clean, I won't guess") is not a refusal either. Bangla footnote marks like `[১]` are ignored. With
`STRICT_CITATIONS` on (the default), any failing answer is replaced with "I'm not sure".

The relevance checks are off until a floor is set. With `RERANKER` set (a cross-encoder such as
`BAAI/bge-reranker-v2-m3`), they use the reranker's 0-1 score and `RERANK_MIN`; otherwise E5 similarity and
`MIN_RELEVANCE`. E5 similarity can't reliably tell an off-topic passage from a relevant one (wrong verses
often score higher than right ones; `docs/INFERENCE_REVIEW.md`), so prefer the reranker. Pick a value with
`python -m tools.calibrate_relevance`, which also scores hand-checked hard cases. With a floor set, `/ask` also drops passages below the floor before the
model sees them, and answers questions nothing in the corpus is about (weather, programming) with a
fixed "I couldn't find anything about this" message. Greetings and "who are you?" get a fixed
introduction without calling the model.

**What this does not establish:** that the answer's reading of the source is correct. A sentence
can cite the right verse and still misstate what it means. Closing that gap needs:

- claim-level entailment checking (does the cited passage actually support this sentence?),
  e.g. a small NLI model or an LLM judge run over sentence/source pairs, and
- human review of a sample by someone qualified, with a written rubric.

## 3. End-to-end answers against a live model

`python -m tools.eval_answers` asks every gold question through the full pipeline against LM
Studio and records citation status and whether an expected source was cited. Run it for the base
model and the fine-tuned one and compare. Then read the answers.
