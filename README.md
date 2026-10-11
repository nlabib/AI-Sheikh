# Islamic Qwen AI

Ask questions about Islam in English, Bangla or Arabic and get answers grounded in the Quran and
the six major hadith collections, with every claim tied to a checkable reference like
`[quran:2:256]` or `[bukhari:1]`. Everything runs locally: retrieval in SQLite, the model in
LM Studio (Qwen3), and optional fine-tuning on your own GPU or Mac.

```
question (en / bn / ar)
  -> keyword search (SQLite FTS5) + meaning search (multilingual E5), Quran and hadith side by side
  -> top sources, with Arabic text, translations and hadith gradings
  -> Qwen3 via LM Studio, told to answer only from those sources and cite them
  -> citation check: invented refs, refs it wasn't given, misquotes, uncited claims
  -> answer + sources + check status ("I'm not sure" replaces any answer that fails the check)
```

**Status:** working retrieval and API over the full corpus (121,064 passages), 50 passing tests,
keyword-search baseline measured (`docs/RESULTS.md`). Not yet done: measuring semantic search,
and any real training run. Read the limits section before relying on an answer.

## Quick start

Python 3.10-3.12.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -r requirements-embed.txt
python -m pytest -q

python -m tools.fetch        # download Quran + hadith (~150 MB, checksummed)
python -m tools.ingest       # build data/corpus.sqlite
python -m tools.index        # semantic index (downloads multilingual-e5-small once)
python -m tools.evaluate     # retrieval scores on 98 gold questions
```

Start Qwen3 in LM Studio (Developer tab, load a Qwen3 model, Start Server), find its id with
`curl http://127.0.0.1:1234/v1/models`, then:

```bash
LM_MODEL=qwen3-4b uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs, or:

```bash
curl -s http://127.0.0.1:8000/ask -H 'content-type: application/json' \
  -d '{"question": "দ্বীনের ব্যাপারে কি জোর জবরদস্তি আছে?"}' | python -m json.tool
```

Without LM Studio running, `/ask` still returns the sources and an error message, and `/search`
works on its own.

## Fine-tuning

Follow **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**. It explains every step for a
beginner: picking a config for your GPU's memory, building licence-safe training data, a dry run,
a 20-step test, the full QLoRA run, comparing base vs fine-tuned, and loading the result in
LM Studio. Configs for 8, 12 and 24 GB cards are in `training/configs/` (default: Qwen3-4B on
12 GB). Apple Silicon users can use `training/mlx_optional.sh`.

## Layout

| Path | What it is |
|---|---|
| `app/server.py` | FastAPI: `/ask`, `/search`, `/ref/{ref}`, `/sources`, `/health` |
| `app/retrieve.py` | hybrid retrieval, Quran/hadith balancing, explicit reference lookup |
| `app/store.py` | SQLite corpus and FTS5 index |
| `app/embed.py` | E5 embedder, offline hashing fallback, vector index |
| `app/text.py` | Arabic/Bangla normalisation and tokenisation |
| `app/citations.py` | answer citation checks |
| `app/prompts.py` | system prompt and evidence formatting (shared with training) |
| `tools/sources.py` | every source: pinned URL, checksum, licence, training permission |
| `tools/fetch.py`, `tools/ingest.py`, `tools/index.py` | build the corpus and index |
| `tools/evaluate.py` | retrieval benchmark |
| `tools/eval_answers.py` | end-to-end answer benchmark against a live model |
| `tools/build_sft.py` | build fine-tuning data from training-allowed sources |
| `training/train_cuda.py`, `training/configs/` | NVIDIA QLoRA |
| `training/mlx_optional.sh` | Apple Silicon LoRA |
| `training/merge_lora.py` | merge adapter into the base model for GGUF export |
| `data/eval/gold.jsonl` | 98 hand-checked questions (en/bn/ar) with correct references |
| `docs/` | training guide, sources & licences, data schema, evaluation, results, roadmap |

## Configuration

Environment variables (see `app/config.py`): `LM_BASE_URL` (default
`http://127.0.0.1:1234/v1`), `LM_MODEL`, `LM_API_KEY`, `EMBED_MODEL`
(default `intfloat/multilingual-e5-small`; `-base` or `-large` are better and slower),
`EMBEDDER` (`e5` or `hashing`), `DB_PATH`, `INDEX_DIR`, `TOP_K`, `STRICT_CITATIONS`
(default `1`: answers that fail the citation check are replaced with "I'm not sure").

## Sources and licences

Arabic Quran (Tanzil, CC BY 3.0), Pickthall's English (public domain), Abu Bakr Zakaria's Bangla
(free to read, not openly licensed), and Bukhari, Muslim, Abu Dawud, Tirmidhi, Nasa'i and Ibn
Majah in Arabic (classical, public domain) with English and Bangla translations (copyrighted).
Only openly licensed and public-domain texts are ever used for training. Details:
[docs/SOURCES_AND_LICENSES.md](docs/SOURCES_AND_LICENSES.md). Nothing is bundled in this repo.

## Limits

- This is a study aid, not a mufti. It does not issue fatwas, and answers on rulings should be
  checked with a qualified scholar.
- The citation checker catches invented or misquoted references, not wrong interpretations.
- Retrieval misses happen (see `docs/RESULTS.md`); when the right verse isn't retrieved, the
  model is told to say so rather than answer from memory, but it may not always comply.
- Hadith gradings are shown as given by the source edition (Al-Albani and others) and only exist
  for four of the six books.
