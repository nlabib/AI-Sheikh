# Training guide: fine-tune Qwen3 on your own computer

This guide walks you through the whole process once, step by step, and explains what each
step does and why. The training scripts were written and unit-tested on a machine without a
GPU and have not yet been run end to end on one, so treat every time and memory figure here as
an estimate and write down your own.

You can do this on:

- **An NVIDIA PC** (recommended): Ubuntu, or Windows with WSL2. Uses `training/train_cuda.py`.
- **An Apple Silicon Mac** (M1 to M4, 16 GB+ unified memory): uses MLX via `training/mlx_optional.sh`.

The two paths share steps 0 to 5 and 7. Step 6 is different for each.

---

## The big picture first

A language model like Qwen3-4B is a file of about 4 billion numbers ("weights"). It already
knows a lot of language and some Islamic content, but it also makes things up, including
verse and hadith numbers. This project attacks that in two ways:

1. **Retrieval (RAG).** Before the model answers, we search the real Quran and hadith texts
   and paste the best matches into the prompt. The model is told to answer only from those.
   This is where accuracy comes from. You already have this working without any training.
2. **Fine-tuning.** We then teach the model the *habits* we want: quote word for word, put
   the exact `[quran:2:255]` marker after each claim, answer in the user's language, and say
   "the sources don't cover this" instead of guessing. Fine-tuning is not how the model learns
   the Quran; it is how it learns to behave.

**LoRA** is the trick that makes fine-tuning affordable. Instead of changing all 4 billion
weights, we freeze them and add small extra matrices ("adapters", a few tens of millions of
numbers) to some layers. Only the adapters learn. The result is a folder of about 100 MB
that sits on top of the original model.

**QLoRA** goes further: the frozen base model is loaded with its weights compressed to
4 bits, so Qwen3-4B fits in about 3 GB of GPU memory instead of about 8 GB. The adapters still
train in 16-bit precision, so quality stays close to normal LoRA.

---

## Step 0: Check your hardware

On the NVIDIA PC, run:

```bash
nvidia-smi
```

Look at the GPU name and the memory column (for example `12288MiB` = 12 GB). Pick the
config that matches:

| GPU memory | Config | Notes |
|---|---|---|
| 8 GB | `training/configs/qwen3-4b-8gb.yaml` | shorter examples (1024 tokens), smaller adapter |
| 12 GB | `training/configs/qwen3-4b-12gb.yaml` | the default |
| 24 GB | `training/configs/qwen3-4b-24gb.yaml` | longer context, faster |
| 24 GB | `training/configs/qwen3-8b-24gb.yaml` | the bigger model; better answers, slower |

On a Mac: Apple menu, About This Mac, and note the chip and memory.

## Step 1: Set up Python

Use Python 3.10 to 3.12. In the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows WSL and Mac: same command
pip install -r requirements-dev.txt
```

A *virtual environment* (`.venv`) keeps this project's packages separate from everything else
on your computer, so versions can't clash. Run `source .venv/bin/activate` again in every new
terminal.

**NVIDIA only**: install PyTorch built for CUDA, then the training packages:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-train-cuda.txt -r requirements-embed.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and your GPU's name. If it prints `False`, the CUDA build of
torch is not installed or the NVIDIA driver is too old; fix that before going on.

**Mac only**:

```bash
pip install -r requirements-train-mlx.txt -r requirements-embed.txt
```

Check everything works:

```bash
python -m pytest -q
```

## Step 2: Download and import the texts

```bash
python -m tools.fetch       # ~150 MB into data/raw/, each file checked against a SHA-256
python -m tools.ingest      # builds data/corpus.sqlite (~290 MB, ~121,000 passages)
```

`fetch` downloads exact, pinned versions of the Quran (Arabic, English, Bangla) and the six
major hadith collections (Arabic, English, Bangla). The checksum check means you have
byte-for-byte the same files this project was tested with. `ingest` loads them into SQLite and
builds a full-text index, so keyword search works straight away.

See `docs/SOURCES_AND_LICENSES.md` for who made each text and what you may do with it.

## Step 3: Build the semantic search index

```bash
python -m tools.index
```

This runs every passage through `intfloat/multilingual-e5-small`, a model that turns text into
a list of 384 numbers such that texts with similar meaning get similar numbers, *across
languages*. That is what lets a Bangla question find an English or Arabic passage. Keyword
search can't do that.

It downloads the model (~470 MB) the first time, then works through 121,000 passages; expect
somewhere around 10 to 30 minutes on a GPU and much longer on CPU (an estimate).

## Step 4: Measure retrieval before you train anything

```bash
python -m tools.evaluate --show-misses
```

This asks 98 real questions (in English, Bangla and Arabic) whose correct verses and hadith
were looked up by hand, and reports how often the right source shows up in the top 1, 5 and
10 results. Write the numbers down. Read `docs/RESULTS.md` for the keyword-only baseline.

**Why measure first:** if the right verse never reaches the model, no amount of fine-tuning
will produce a correct, cited answer. Retrieval quality is the ceiling on answer quality.

## Step 5: Build the training data

```bash
python -m tools.build_sft
```

This writes `data/sft/train.jsonl` (1,140 examples) and `data/sft/valid.jsonl` (60). Open
`train.jsonl` and read a few lines. Each one is a conversation:

- **system**: the rules (cite everything, quote word for word, don't issue fatwas)
- **user**: the retrieved sources, then the question, exactly as the server builds them
- **assistant**: the answer we want the model to learn to write

There are four kinds, in English, Bangla and Arabic: look up a verse, look up a hadith (with
its grading), and two kinds of honest refusal. Those start with "I'm not sure" ("আমি নিশ্চিত নই",
"لست متأكدًا"): one for questions the sources don't cover at all, and one for a question about a
specific verse whose text is *not* among the sources. The second matters most. Qwen has
probably seen that verse during pre-training and could "remember" it, sometimes wrongly; we
want it to say it isn't sure rather than answer from memory. The examples are generated from
templates, so every quote is exact, but they teach *format and honesty*, not knowledge. Only sources whose licence
allows it (Arabic Quran, Pickthall's public-domain English, classical Arabic hadith) are used.

**To teach real answering skill**, write your own examples in the same format, ideally checked
by someone with Islamic knowledge, and save them as `data/sft/reviewed/anything.jsonl`. Re-run
`build_sft`; it merges them in. A few hundred good reviewed examples will do more than
thousands of templates. Set `"review_status": "reviewed"` in their `meta`.

The `valid.jsonl` split is never trained on. The trainer measures loss on it to tell you
whether the model is learning patterns (valid loss falls) or memorising (train loss keeps
falling while valid loss rises).

## Step 6a: Train on NVIDIA

Dry run first. It loads only the tokenizer, encodes the data and reports lengths, then stops:

```bash
python -m training.train_cuda --config training/configs/qwen3-4b-12gb.yaml --dry-run
```

You'll see how many examples fit in `max_len`, how many tokens each has, and how many
optimiser steps the run will take.

Then a short real run to check memory, 20 steps:

```bash
python -m training.train_cuda --config training/configs/qwen3-4b-12gb.yaml --max-steps 20
```

The first run downloads Qwen3-4B (~8 GB). Watch `nvidia-smi` in a second terminal. If you see
`CUDA out of memory`, use the next smaller config or lower `max_len`.

Then the full run:

```bash
python -m training.train_cuda --config training/configs/qwen3-4b-12gb.yaml
```

What the settings mean (all in the YAML file):

| Setting | What it does | Why this value |
|---|---|---|
| `max_len` | longest example in tokens | longer costs memory; our examples are mostly short |
| `batch_size` | examples processed at once | 1 to fit in 12 GB |
| `grad_accum` | batches added up before one update | 1 x 16 behaves like a batch of 16, which is more stable |
| `epochs` | passes over the data | 2; more tends to memorise templates |
| `lr` | learning rate, the size of each update | 2e-4 is the usual LoRA starting point |
| `lora_r` | adapter size (rank) | 16 is plenty to learn a format |
| `lora_alpha` | adapter scaling | conventionally 2 x r |
| `gradient_checkpointing` | recompute instead of storing activations | saves a lot of memory for ~30% more time |

What you'll see while it trains: a `loss` line every 10 steps. Loss is how surprised the model
is by the correct answer; lower is better. It should drop quickly in the first steps (the model
picks up the format) and then flatten. Only the answer part of each example counts towards the
loss: the system prompt, sources and question are masked out (see `training/data.py`), so the
model spends its effort on learning to *answer*, not to repeat prompts.

When it finishes you get `outputs/islamic-qwen-lora/adapter/` and `training_summary.json`
(time taken, loss history, peak GPU memory). Keep the summary; it's your record of the run.

### Watching training

The trainer writes charts for TensorBoard while it runs (`report_to: tensorboard`, the default).
In a second terminal, in the project folder:

```bash
tensorboard --logdir outputs --host 127.0.0.1 --port 6006
```

Then open http://localhost:6006 in your browser. Each training run appears as its own coloured line.
Charts refresh every 30 seconds, so you can leave the page open during the run. To watch the GPU
itself, run `nvidia-smi -l 2` in a third terminal (it prints every 2 seconds), or open Task Manager,
Performance, GPU.

Have a run from before charts existed? `python -m training.summary_to_tensorboard outputs/<run>`
draws its charts from `training_summary.json`.

What each chart means:

| Chart | What it is | Healthy | Problem |
|---|---|---|---|
| `train/loss` | how wrong the model is on the answers it's learning, every 10 steps | drops fast in the first steps (it picks up the format), then flattens | flat from the start (learning rate too low, or data didn't load); jagged and rising (learning rate too high) |
| `eval/loss` | the same, on `valid.jsonl`, which it never trains on; every 20 steps | falls with train loss, then levels off | **turns upward while train loss keeps falling: overfitting.** It has started memorising instead of learning; stop at the lowest point (fewer epochs) |
| `train/learning_rate` | size of each update | ramps up over the first few steps (warm-up), then a smooth cosine curve down to ~0 | — (it's a schedule, not a measurement) |
| `train/grad_norm` | how big each update wants to be | settles to small values | spikes far above its usual level; often comes with a loss spike |
| `gpu/memory_*_gb` | GPU memory the trainer holds (`reserved`) and actually uses (`allocated`), and the peak | flat once training is going, below your card's size | near the card's size. On Windows the driver then quietly spills into system RAM ("shared GPU memory" in Task Manager), and training gets much slower instead of crashing. Lower `--batch-size` and raise `--grad-accum` to keep the same effective batch |
| `gpu/utilization_pct` | how busy the GPU is | near 100% during steps | low: the GPU is waiting (CPU, disk, or memory spilling) |

**A warning these charts can't give you.** On the first real run (RTX 3090, see
`docs/TRAINING_LOG.md`), train loss fell to about 0.0001 and eval loss to 0.002, which looks
perfect. But `valid.jsonl` is made by the same templates as `train.jsonl`, so a model that
memorised the templates also scores near zero on it. The charts said "great", while the real
answer test (step 7) found broken answers. A loss near zero on template data means
"memorised", not "learned". The only real check is step 7, plus reading answers yourself.

## Step 6b: Train on a Mac

```bash
ITERS=50 bash training/mlx_optional.sh      # quick check that everything runs
bash training/mlx_optional.sh               # the full run
```

The script explains its settings at the top. `--mask-prompt` does the same job as the masking
in the CUDA script. The adapter lands in `outputs/islamic-qwen-mlx-adapter/`, and the script
prints the commands to try it and to fuse it into a standalone model.

## Step 7: Compare base and fine-tuned

Get both models into LM Studio (step 8 shows how to package yours), start LM Studio's local
server, then:

```bash
LM_MODEL=<base model id> python -m tools.eval_answers --out results/base.json
LM_MODEL=<your model id> python -m tools.eval_answers --out results/finetuned.json
```

Find model ids with `curl http://127.0.0.1:1234/v1/models`. For each question this records
whether the answer's citations pass the checker and whether it cites one of the expected
sources. It also asks 14 trick questions the sources can't answer (a "verse" that is really a
proverb, a hadith number that doesn't exist, surah 115), and counts how often the model says
"I'm not sure" instead of inventing an answer. Fine-tuning has helped if pass rate, "cited
expected" and the not-sure rate on trick questions go up, without it saying "not sure" to
questions it should answer.

**A safety net that doesn't depend on training:** the server runs the citation checker on every
answer. If an answer cites a verse or hadith that doesn't exist, cites something it wasn't
given, or misquotes, the user gets "I'm not sure" instead, and the original answer is kept in
`unverified_answer` for you to inspect. Set `STRICT_CITATIONS=0` to turn this off. No model can
be guaranteed never to be wrong, but this means an invented reference never reaches the user as
an answer. Then read twenty
answers yourself. The checker can catch a wrong verse number or a misquote, but it cannot tell
whether an explanation is right.

## Step 8: Use your model in LM Studio (NVIDIA path)

LM Studio runs GGUF files, so merge the adapter into the base model and convert:

```bash
python -m training.merge_lora --adapter outputs/islamic-qwen-lora/adapter --out outputs/islamic-qwen-merged

git clone https://github.com/ggml-org/llama.cpp
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py outputs/islamic-qwen-merged --outfile outputs/islamic-qwen-f16.gguf --outtype f16
```

The f16 GGUF is about 8 GB. To shrink it to about 2.5 GB (Q4_K_M, a good quality/size balance),
build llama.cpp (see its README) and run
`llama.cpp/build/bin/llama-quantize outputs/islamic-qwen-f16.gguf outputs/islamic-qwen-Q4_K_M.gguf Q4_K_M`.
Then in LM Studio, import the `.gguf` file, load it, start the server, and point this project at
it with `LM_MODEL=<its id>`.

On a Mac, LM Studio can load the fused MLX folder from step 6b directly.

## Step 9: Run the app with your model

```bash
LM_MODEL=<your model id> uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs and try `/ask`. Every response includes the sources and a
`citation_check` with `pass`, `warn` or `fail`.

---

## Round 2: more sources, varied answers, the 300-question benchmark

The first run taught the answer format, but its 1,200 template answers each quoted one source, and the model
memorised them (training loss near 0 by step 30). Round 2 changes the data, not the method:

| What | Why |
|---|---|
| New sources: Muwatta Malik, Nawawi 40, Hadith Qudsi, Tafsir Ibn Kathir (ar/en/bn), Jalalayn, 3 more Quran translations for search | More to find and cite; tafsir explains verses in the user's language |
| 582 checked answers in `data/sft/generated/` | Several sources per answer, partial answers, differing views, declines, in all three languages |
| Templates cut from 300 to 150 per kind | Less to memorise |
| Learning rate 1e-4 instead of 2e-4 (`qwen3-4b-24gb-v2.yaml`) | Gentler, so the model learns the pattern rather than the wording |
| `tools.benchmark`: 300 questions with reference answers | A fair before/after score; none of its sources is cited in training data |

Run these in the project folder with the virtual environment active. Times are rough guesses for an RTX 3090.

```bash
git pull                                   # on branch claude/benchmark-corpus-v2 until PR #6 is merged
python -m tools.fetch                      # about 260 MB of new files; already-downloaded ones are skipped
python -m tools.ingest                     # corpus grows from ~121,000 to ~177,500 passages
python -m tools.index                      # rebuild the E5 search index for the new passages
# Baseline: score the round-1 model on the new benchmark (LM Studio serving ai-sheikh-ft)
RERANKER=BAAI/bge-reranker-v2-m3 RERANK_MIN=0.01 LM_MODEL=ai-sheikh-ft python -m tools.benchmark --out results/bench-ft-v1.json
python -m tools.build_sft                  # ~1,180 examples: 582 generated + 600 templates
python -m training.train_cuda --config training/configs/qwen3-4b-24gb-v2.yaml
```

Watch it in TensorBoard (`tensorboard --logdir outputs`). This time the training loss should fall more slowly
and level off well above zero; if it hits ~0 within the first epoch, the model is memorising again. Then merge,
convert and load it exactly as in Steps 7 and 8, with `outputs/ai-sheikh-v2` and the name `ai-sheikh-v2`, and
score it:

```bash
RERANKER=BAAI/bge-reranker-v2-m3 RERANK_MIN=0.01 LM_MODEL=ai-sheikh-v2 python -m tools.benchmark --out results/bench-ft-v2.json
```

Compare `good` and `wrong` per language and category in the two reports, and read the starred (scholar-review)
items in `results/bench-ft-v2.md` yourself.

## When things go wrong

| Symptom | Likely cause and fix |
|---|---|
| `torch.cuda.is_available()` is False | CPU-only torch installed; reinstall from the CUDA index URL in step 1 |
| `CUDA out of memory` | use a smaller config, lower `max_len`, or close other GPU programs |
| Loss stays flat from the start | learning rate too low, or the data didn't load; check the dry-run output |
| Loss drops to near zero and answers look copy-pasted | over-training on templates; fewer epochs, add reviewed examples |
| valid loss rises while train loss falls | memorising; stop earlier (fewer epochs) |
| Model writes `<think>` blocks | Qwen3's reasoning mode; the server strips them, and the training data uses non-thinking mode |
| `bitsandbytes` errors on Windows | run inside WSL2 Ubuntu rather than native Windows |

## Glossary

- **Token**: a chunk of text, about ¾ of an English word; Arabic and Bangla use more tokens per word.
- **Epoch**: one full pass over the training data.
- **Loss**: how wrong the model's predictions are on the target text; training pushes it down.
- **Adapter**: the small set of new weights LoRA trains; the base model stays unchanged.
- **Quantisation**: storing weights with fewer bits (4 instead of 16) to save memory.
- **GGUF**: the file format llama.cpp and LM Studio load.
