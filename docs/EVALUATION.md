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
| answer cites nothing and doesn't decline | warn (`no_citations`) |

Declining ("the sources don't cover this; ask a qualified scholar") passes.

**What this does not establish:** that the answer's reading of the source is correct. A sentence
can cite the right verse and still misstate what it means. Closing that gap needs:

- claim-level entailment checking (does the cited passage actually support this sentence?),
  e.g. a small NLI model or an LLM judge run over sentence/source pairs, and
- human review of a sample by someone qualified, with a written rubric.

## 3. End-to-end answers against a live model

`python -m tools.eval_answers` asks every gold question through the full pipeline against LM
Studio and records citation status and whether an expected source was cited. Run it for the base
model and the fine-tuned one and compare. Then read the answers.
