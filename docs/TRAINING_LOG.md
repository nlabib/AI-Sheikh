# Training log: RTX 3090, native Windows 11

A plain-language record of following `docs/TRAINING_GUIDE.md` on a real NVIDIA PC, 2026-10-11.

**Machine:** RTX 3090 (24 GB VRAM), 64 GB RAM, Windows 11 Pro, no WSL. Config: `training/configs/qwen3-4b-24gb.yaml`.

## Step 0–1: Setup

| What I ran | What it does | Time | Notes |
|---|---|---|---|
| `winget install Python.Python.3.12`, `winget install Git.Git` | Python and git weren't installed (only the Microsoft Store stub for `python`) | ~2 min | ~90 MB |
| `python -m venv .venv` + `pip install -r requirements-dev.txt` | an isolated Python environment with the project's packages | | |
| `pip install torch --index-url .../cu124` | PyTorch built for CUDA (2.5 GB download) | | torch 2.6.0+cu124 |
| `pip install -r requirements-train-cuda.txt -r requirements-embed.txt` | transformers 5.19, peft 0.21, bitsandbytes 0.50 | 4.4 min for all installs | `torch.cuda.is_available()` printed `True NVIDIA GeForce RTX 3090` |
| `python -m pytest -q` | the unit tests | 4 s | **3 failed on Windows** at first, see fix 1 below. 52/52 passed after the fix |

bitsandbytes now ships Windows wheels, so QLoRA loads natively on Windows. WSL2 wasn't needed.

## Step 2: Texts

| Command | Time | Result |
|---|---|---|
| `python -m tools.fetch` | 16 s | 21 files, 127 MB, all checksums OK |
| `python -m tools.ingest` | 26 s | 121,064 passages, `data/corpus.sqlite` 278 MB |

## Step 3: Semantic index

`python -m tools.index` downloaded `intfloat/multilingual-e5-small` and embedded all 121,064 passages
on the GPU in **177 s (3.3 min)**, much faster than the guide's 10–30 min estimate.

## Step 4: Retrieval measured (first E5 numbers ever)

`python -m tools.evaluate --out results/retrieval-e5.json --show-misses` (39 s). Scoped search, 98 questions:

| mode | recall@1 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|
| keyword (baseline) | 0.429 | 0.653 | 0.745 | 0.534 |
| E5 vectors only | 0.255 | 0.469 | 0.531 | 0.339 |
| **hybrid (keyword + E5)** | **0.459** | **0.684** | **0.806** | **0.567** |

Hybrid search wins. The biggest gain is English questions about the Quran (recall@5 0.425 → 0.575).
On its own, E5-small is *worse* than keyword search, and very weak on Arabic (Quran/ar recall@5 0.100),
so it only helps when combined with keyword search, which is what `/ask` does.

## Step 5: Training data

`python -m tools.build_sft` (1 s): 1,140 train / 60 valid, 300 each of quran_lookup, hadith_lookup,
no_answer and missing_ref. All are templates (0 reviewed examples).

Dry run (`--dry-run`, 13 s, downloads only the tokenizer): 0 examples dropped at max_len 4096;
median 773 tokens, max 2,399; 219,760 answer tokens carry the loss; 72 steps per epoch, **144 steps in total**.

## Step 6a: Training

- First smoke attempt (`--max-steps 20`) downloaded Qwen3-4B (~8 GB, ~1 min) and loaded it in 4-bit
  (66.1 M trainable LoRA parameters, 1.6% of 4.09 B). Then it crashed: transformers 5 removed
  `warmup_ratio` (fix 2).
- Second smoke attempt ran. Step-0 validation pass took 40 s, then **~20–25 s per step**
  (so the full 144-step run would take about an hour).
- **Memory problem found:** the GPU showed 23.7 GB dedicated **plus 19.4 GB "shared" (system RAM)
  in use**. With the 24gb config (batch 4, max_len 4096), Windows' WDDM driver silently spills
  overflow into system RAM instead of raising "CUDA out of memory". Training doesn't crash, but it
  runs far slower than it should. Next step: rerun the smoke with `--batch-size 2 --grad-accum 8`
  (same effective batch of 16) or `--max-len 2560` (the longest example is 2,399 tokens), and compare
  s/step and shared memory.

- That second smoke run reached step 20 (checkpoint saved) before my session was cut off, so its final
  eval and adapter save never ran. Numbers: valid loss before training **0.595**; train loss **0.257**
  at step 10 and **0.021** at step 20. A loss near zero after only 20 steps means the templated answers
  are very easy to learn, so the full 2-epoch run is at risk of memorising them. Watch valid loss.
- Third smoke run, `--batch-size 2 --grad-accum 8` (same effective batch of 16): shared memory dropped
  from 19.4 GB to **0.36 GB** and steps took **~18 s** instead of ~22 s. I used these settings for the full run.

### Qwen3's "thinking mode" and LM Studio

Qwen3 can "think out loud" in a `<think>...</think>` block before answering. The training data is
rendered with thinking **off**, so every prompt the model learned from ends with an empty
`<think></think>` and the answer follows straight away. LM Studio uses the chat template stored inside
the GGUF file, and Qwen3's stock template has thinking **on** unless the caller passes
`enable_thinking=false`, which an OpenAI-style request (what `eval_answers` sends) can't do.

So I added `training/no_think_template.py`, which flips the template's default to thinking off before
conversion. I converted the base model twice: once with the stock template (what you get out of the box)
and once with thinking off, the same way as the fine-tuned model, for a fair comparison.

| GGUF | Time | Size | LM Studio id |
|---|---|---|---|
| base, stock template | 65 s | 8.05 GB (f16) | `ai-sheikh-base` |
| base, thinking off | 33 s | 8.05 GB (f16) | `ai-sheikh-base-nothink` |

`lms import -L` hard-links the file into LM Studio's models folder, so it costs no extra disk space.
(Correction found later: LM Studio ignores the template's default. See "The junk-answer hunt" below.)

## Step 6a, continued: the full run

```
python -m training.train_cuda --config training/configs/qwen3-4b-24gb.yaml --batch-size 2 --grad-accum 8
```

| | |
|---|---|
| Time | **57.7 min** (144 steps, ~18 s each, 2 epochs) |
| Peak VRAM | **14.4 GB** of 24 |
| Trainable parameters | 66.1 M of 4.09 B (1.6%) |
| valid loss | **0.753** before, **0.0019** after |
| train loss | 0.32 (step 10), 0.024 (step 20), 0.002 (step 30), then ~0.0001 to the end |
| grad norm | ~0.2 at the start, ~0.001 at the end |

What it means: the format was learned in about 30 steps; after that there was nothing left to learn
from templates, and the second epoch just memorised them. The valid loss can't show this, because
`valid.jsonl` comes from the same templates. See the charts below.

## Step 8: Package for LM Studio

| Command | Time |
|---|---|
| `python -m training.merge_lora --adapter outputs/islamic-qwen-lora/adapter --out outputs/islamic-qwen-merged` | 32 s (CPU) |
| `python -m training.no_think_template outputs/islamic-qwen-merged` | instant |
| `python llama.cpp/convert_hf_to_gguf.py outputs/islamic-qwen-merged --outfile outputs/gguf/ai-sheikh-ft-f16.gguf --outtype f16` | 18 s, 8.05 GB |
| `lms import -L --user-repo local/ai-sheikh-ft outputs/gguf/ai-sheikh-ft-f16.gguf` | instant |

The first merge failed to convert: transformers 5 saves the tokenizer as `tokenizer.json` only, and
llama.cpp's converter needs `vocab.json`/`merges.txt`. LoRA never changes the tokenizer, so
`merge_lora.py` now copies the base model's original tokenizer files (fix 4).

llama.cpp lives at `C:\Users\labib\src\llama.cpp` with its own `.venv`, because its requirements pin a
CPU-only torch that would overwrite the CUDA one in the project venv.

I also packaged the 20-step smoke adapter the same way (`ai-sheikh-ft20`), to see whether a little
training already changes behaviour.

## Step 7: Base vs fine-tuned

Commands: `lms load <id> --gpu max -c 16384`, then `LM_MODEL=<id> python -m tools.eval_answers --out results/answers-<id>.json`.
Each run took 4–7 min (98 questions + 14 trick questions).

First problem: one Bangla question's prompt (six long Bangla hadith) is **8,367 tokens**, which
overflowed LM Studio's 8,192 context with a "400 Bad Request". The fix was a 16,384 context.

### The junk-answer hunt

The first evaluation of the full fine-tune looked broken. 36 answers began with a stray Chinese
character (瓢), 17 were empty, English questions got the Bangla "I'm not sure" sentence, and one
answer repeated "।" hundreds of times. The 20-step model and the base model were fine. I narrowed
it down one step at a time:

| Test | Result |
|---|---|
| 4-bit base + adapter in PyTorch (exactly as trained) | clean, correct answers |
| bf16 base + adapter, unmerged | clean |
| merged folder, loaded in float16 and in bfloat16 | clean |
| GGUF in LM Studio, exact training prompt sent as raw text (`/v1/completions`) | clean |
| GGUF in LM Studio, normal chat request | **junk first, then the answer** |

So the weights were fine at every stage; the prompt was different. Capturing LM Studio's rendered
prompt (`lms log stream -s model --filter input`) showed it ends at `<|im_start|>assistant\n`,
**without** the empty `<think>\n\n</think>\n\n` that every training prompt had. LM Studio turns Qwen3's
thinking mode on by itself, even when the template in the GGUF defaults to off. A model trained for
144 steps on one exact format is brittle when shown another; the 20-step model was barely changed,
so it coped.

Of the request options I tried, only `"reasoning_effort": "none"` made LM Studio add the empty think
block (`chat_template_kwargs`, `enable_thinking` and `reasoning` were ignored). `app/llm.py` now sends
it, plus `chat_template_kwargs: {"enable_thinking": false}` for llama.cpp's own server (fix 6).
`THINKING=1` turns thinking back on. Re-run, the junk disappeared completely: 0 stray characters,
0 empty answers, 0 wrong-language refusals.

**Lesson:** serve a fine-tuned model with exactly the prompt format it was trained on. The longer you
train on one format, the less it tolerates any other.

### Results (non-thinking mode, all from the same retrieval)

| model | pass | pass, actually answering | fail | cites an expected source | "not sure" to answerable | trick: "not sure" | trick: cited anyway |
|---|---|---|---|---|---|---|---|
| base Qwen3-4B | 18 | 18 | 28 | **60** | 5 | 11/14 | 3/14 |
| fine-tuned, 20 steps | 42 | 23 | 32 | 41 | 29 | **14/14** | **0/14** |
| **fine-tuned, 144 steps** | **57** | **51** | **22** | 39 | 6 | 13/14 | 1/14 |

(The expected source was retrieved for 62/98 questions, which caps "cites an expected source".
In thinking mode, LM Studio's default, the base model scored 5 pass / 38 fail / 10 of 14 trick
questions declined.)

How to read it:

- **Did it learn to say "I'm not sure" instead of guessing?** Yes. On trick questions (a proverb
  passed off as a verse, a hadith number that doesn't exist, surah 115) the full fine-tune declined 13
  of 14 (base: 11) and backed only one answer with citations (base: 3). That one quoted the real,
  related hadith "purity is half of faith" (Muslim 223) for "cleanliness is next to godliness".
- **Does it cite its sources?** More reliably. Passing real answers rose from 18 to 51, and failures
  (made-up references, misquotes, uncited claims) fell from 28 to 22.
- **The 20-step model over-refuses:** it said "not sure" to 29 answerable questions. More training
  fixed that (6), so the full run was worth it, even though the loss had flattened.
- **What templates can't teach:** the templates always quote exactly one source, so the fine-tune
  writes short answers (233 vs 602 characters), cites 0.9 sources per answer (base: 2.1), and
  sometimes picks the wrong one of the six, e.g. 2:245 ("a goodly loan") for a question about
  interest. When the right source was retrieved, the base model cited it 37 of 39 times, the fine-tune
  27. It also wrote `[1]` instead of `[quran:2:245]` 16 times. Reviewed, multi-source examples (guide
  step 5) are the way to fix this.
- **Safety net:** the server replaces any answer that fails the citation check with "I'm not sure"
  (`STRICT_CITATIONS=1`, the default), so the 22 failing answers would reach a user as "I'm not sure".

## Watching it: TensorBoard

The full run had already finished when the charts were added, so I drew them from its saved history:
`python -m training.summary_to_tensorboard outputs/islamic-qwen-lora`, then
`tensorboard --logdir outputs` and http://localhost:6006. New runs log live (fix 7). How to read the
charts: `docs/TRAINING_GUIDE.md`, "Watching training".

## Where things are on this PC

| What | Where |
|---|---|
| repo, venv | `C:\Users\labib\src\AI-Sheikh` |
| adapter (~260 MB) + `training_summary.json` | `outputs\islamic-qwen-lora\` |
| merged model | `outputs\islamic-qwen-merged\` |
| GGUFs (8 GB each, hard-linked into LM Studio) | `outputs\gguf\`; LM Studio ids `ai-sheikh-ft`, `ai-sheikh-ft20`, `ai-sheikh-base`, `ai-sheikh-base-nothink` |
| answer evaluations | `results\answers-*.json` (non-thinking), `results\thinking-on\` (first runs) |

## Code fixes made along the way

1. **UTF-8 on Windows.** Several `read_text()`/`write_text()` calls had no `encoding=`, so Windows
   used cp1252 and crashed on Arabic/Bangla (3 failing tests). Added `encoding="utf-8"` in
   `tools/build_sft.py`, `training/data.py`, `training/merge_lora.py`, `app/embed.py`,
   `tools/evaluate.py`, `tools/eval_answers.py`.
2. **transformers 5 compatibility** in `training/train_cuda.py`: pass `warmup_steps=<ratio>` when
   `warmup_ratio` no longer exists.
3. **Validation loss is now actually measured.** `save_steps` (200) was larger than the whole run
   (144 steps), so eval never ran. Added `eval_on_start` and a final `trainer.evaluate()`.
4. **Merge output converts to GGUF** (`training/merge_lora.py`): copy the base model's original
   tokenizer files instead of re-saving them with transformers 5.
5. **`training/no_think_template.py`**: make the GGUF's template default to non-thinking mode (for
   llama.cpp; LM Studio overrides it, see fix 6).
6. **Non-thinking requests** (`app/llm.py`, `THINKING` in `app/config.py`): send
   `reasoning_effort: none` and `chat_template_kwargs: {enable_thinking: false}` so the served model sees
   the prompt format it was trained on.
7. **Training charts** (`training/train_cuda.py`, `training/summary_to_tensorboard.py`): TensorBoard
   logging by default, eval loss every 20 steps, GPU memory/utilisation charts.

## Second fine-tune (v2), 2026-10-14

Goal: fix the weaknesses found in `docs/INFERENCE_REVIEW.md` (one-source answers, quoting drift, `[1]`
markers) with better training data, and train shorter. **Result: v2 is worse than the first fine-tune and
is not used.** Along the way we found and fixed a bug that had been garbling Bangla quotes in LM Studio all
along. The bug fix is the real gain.

### Where it ran

- **Windows blocked it first.** Windows Smart App Control started refusing two unsigned library files
  (`regex`, `tokenizers`) that transformers needs, so nothing that tokenizes could run on Windows. Changing
  that security setting was the owner's call; they chose WSL2.
- **WSL2 (Ubuntu 26.04):** `uv` for Python 3.12, torch 2.6 + cu124, the same library versions as on
  Windows. The corpus, index and models were copied from the Windows side (no re-download). 77 tests pass.
  Two runs there died when the WSL VM was shut down (it uses a lot of RAM), so training moved to the cloud.
- **RunPod, RTX A6000 (48 GB), Secure Cloud, $0.59/h**, official PyTorch 2.8 template, SSH with a key made
  for this. Code and data went up as a 0.7 MB tarball, the adapter (245 MB) came back by `scp`, and the pod
  was deleted right after. About 35 minutes, **about $0.35**. Script: `training/runpod_train.sh`.

Things that went wrong on the way, so you don't repeat them:
- Batch 4 on the A6000 ran at 92 s per step (every batch padded to its longest example, up to 4,096
  tokens). Batch 1 × accumulation 16 ran at 22 s per step.
- `tar` needs `--no-same-owner` on the pod's `/workspace` disk.
- `pkill -f <pattern>` over SSH can match the remote shell's own command line and kill the connection.
  Kill by PID, or launch through a script file.
- Detach long runs with `setsid … < /dev/null &`, so they survive a dropped connection.

### The data (`python -m tools.build_sft --per-kind 120 --topics`, `tools/sft_topics.py`)

> This dataset was built with `--topics` and `tools/sft_topics.py`, which exist only on the unmerged branch
> `claude/retrain-v2` (the model it produced scored worse, so they are not on `main`). The command will not
> run from `main`; it is recorded here for the history.

| kind | n | what it teaches |
|---|---|---|
| quran_lookup, hadith_lookup, no_answer, missing_ref | 120 each | as before |
| topic_hadith / topic_quran / topic_mixed | 134 / 8 / 3 | a topic question answered by quoting 2–3 passages the reranker is confident about (hadith ≥ 0.8, Quran ≥ 0.3), other passages left as distractors |
| topic_unrelated | 110 | sources retrieved for a different topic → "I'm not sure, these sources are about other matters" |
| surah_lookup | 60 | a short surah quoted ayah by ayah, one marker each |

The Quran bar is high on purpose. At 0.1–0.2, the reranker accepted tangential verses (5:45 on
retaliation for "charity"), and those would teach the wrong habit. The cost is that Quran topic examples
are few.

### Training

`training/configs/qwen3-4b-24gb-v2.yaml`: 1 epoch, 46 steps, batch 1 × 16. **16.6 minutes**, peak VRAM 12.9 GB.

| step | train loss | valid loss |
|---|---|---|
| 0 | | 1.564 |
| 10 | 0.320 | 0.187 |
| 20 | 0.056 | 0.039 |
| 30 | 0.012 | 0.021 |
| 46 | | **0.015** |

Healthier than the first run (train loss ~0.0001, i.e. memorised): the curve flattens instead of collapsing.

### The Bangla bug: য় and ড়

v2's first test failed more often, and many failures were garbled Bangla quotes ("শয়তান" → "শতান", "খায়" →
"খাড়াবে"), the same corruption the first model showed. With the same prompt at temperature 0, LM Studio
counted **4,664** tokens and Hugging Face **4,736**. Comparing word by word, every mismatch contained
**য় (U+09DF)** or **ড় (U+09DC)**. The corpus stores these letters as single code points. Qwen's tokenizer
applies Unicode **NFC** normalization first, which turns each into two (য + nukta), and Qwen was trained on
that form. llama.cpp (inside LM Studio) does not normalize. So those words reached the model as byte
sequences it had hardly seen, and it garbled exactly them when quoting. This affects any Qwen GGUF in LM
Studio.

**Fix:** `app/prompts.build_messages` NFC-normalizes the prompt (`tests/test_prompts.py`). Afterwards LM
Studio counts 4,736 tokens, identical, and the Bangla comes out clean.

### Results (with the fix, reranker on, `RERANK_MIN=0.01`, `tools.eval_answers`)

| model | pass | fail | real answers passing | cites an expected source | sources per answer | trick: "not sure" / cited anyway |
|---|---|---|---|---|---|---|
| first fine-tune, before the fix | 66 | 14 | 62 | 62 | 1.2 | 12 / 1 |
| **first fine-tune, with the fix** | **71** | **13** | **66** | **64** | 1.2 | 12 / 1 |
| v2, with the fix | 60 | 24 | 56 | 59 | 1.5 | 11 / 0 |

v2 did learn to use more sources (1.5 per answer), but its quoting got worse, and that's what fails. Of the
18 questions v2 fails and the first model passes, the pattern is:
- **dropped words mid-quote** without "…" (5:90 loses "and idols and divining arrows");
- **transliteration inside quote marks** ("Shahrur Ramadan al-ladhi…");
- **uncited verdicts up front** ("Alcohol and gambling are prohibited in Islam").

**Lesson:** the topic examples quoted *excerpts* (the opening sentences of a passage). A 4B model
generalised that to "cutting words is fine". To teach multi-source answers without this, quote whole
passages, or mark every cut with "…" and teach the checker to accept it. And keep the first model:
`ai-sheikh-ft` with the NFC fix is the best combination so far.

## Round 2 on Qwen3-8B, 2026-10-14

The owner asked for "bigger and better" and chose the 8B model going forward. This run uses round 2 as it is
on `main` (new sources, 582 checked multi-source answers, learning rate 1e-4) on **Qwen3-8B**, trained in
the cloud.

### Preparation (in WSL)

| Step | Result |
|---|---|
| `tools.fetch` + `tools.ingest` | corpus 121,064 → **177,529** passages (Muwatta, Nawawi 40, Hadith Qudsi, Ibn Kathir and Jalalayn tafsir, more translations) |
| `tools.index` | E5 index rebuilt for all of them |
| `tools.build_sft` | 1,123 train / 59 valid: multi_source 223, partial 82, concise 83, disputed 56, distractor_decline 115, plus 564 templates |
| baseline | `tools.benchmark` on the round-1 4B (`ai-sheikh-ft`): **good 112, refused 91, wrong 97** of 300 (24 min) |

### Training on RunPod

`training/configs/qwen3-8b-48gb-round2.yaml` (2 epochs, lr 1e-4, LoRA r32, batch 1 × 16) on an RTX A6000,
Secure Cloud, $0.59/h, with the owner's limits: 100 GB of storage, a 4-hour self-delete timer, at most $3.

| | |
|---|---|
| Training | **63 min**, 142 steps, ~25 s/step, peak VRAM 19.1 GB |
| Loss | train 0.25, valid 0.28 at the end: it learned the patterns without memorising (round 1 reached ~0.0001) |
| Packaging | merged and converted **on the pod** to Q8_0 (8.7 GB), so the 16 GB base model never came home (`training/runpod_package.sh`) |
| Download | GGUF and adapter (325 MB), checksums verified |
| Cost | pod 10:26–11:44 UTC, **about $0.79**; pod and its storage deleted, no network volumes left |

In LM Studio: `ai-sheikh-8b-v2`, 11.4 GB of GPU memory with a 16k context.

### Benchmark: round-1 4B vs round-2 8B (300 questions, reranker on)

| | 4B round 1 | **8B round 2** |
|---|---|---|
| good | 112 | **129–130** |
| refused | 91 | 51–58 |
| "wrong" | 97 | 113–119 |
| trick questions handled | 48/50 | 42–45/50 |

(Two 8B runs; answers vary a little between runs because of sampling.)

| category | n | 4B good | 8B good |
|---|---|---|---|
| quran_locate | 35 | 21 | 23 |
| quran_content | 28 | 10 | 13 |
| hadith_content | 27 | 5 | 10 |
| hadith_locate | 26 | 3 | 6 |
| fiqh_ruling | 31 | 4 | 7 |
| tafsir | 15 | 10 | 10 |
| Bangla, all | 80 | 23 | 27–28 |

**"Wrong" is not the same as incorrect.** The benchmark counts an answer as wrong when it doesn't cite one
of the sources it lists. Reading 12 of the 8B's "wrong" answers (questions the 4B had declined): **7 were
actually good**, citing other valid sources (silk and gold for men from Muslim 5426 and Ibn Majah 3590; both
views on reciting al-Fatiha behind the imam, in Arabic and Bangla), 3 were weak, and 2 were wrong.

**Where the 8B is worse: questions about things that don't exist.** It is more eager to answer, and 4 of its
trick-question misses are misleading. Asked for "verse 7 of an-Nas" (an-Nas has 6), it quoted 7:158 as that
verse. Asked for "verse 8 of al-Fatiha" (7), it answered with unrelated tafsir. For a fabricated hadith
("patriotism is part of faith"), it quoted Bukhari 24 ("modesty is part of faith") without saying the
asked one doesn't exist. Asked for the number of "ikhtilaf ummati rahma" in Muslim, it gave another
hadith's number. The 4B had 2 such misses.

### A checker gap the 8B found

One 8B answer showed a made-up Arabic "verse" under the broken marker `[al-Baqarah 2:268]`. Quotes were only
verified against the passage a *valid* marker points to, so this one was never checked. Now a quote with no
valid marker must appear in one of the sources given, or the answer fails (`unsourced_quote`,
`tests/test_citations.py`).

### Next

- **A guard for references that don't exist:** the server knows how many ayat each surah has, so "verse 8 of
  al-Fatiha" or "surah 115" can be answered "that verse doesn't exist" without asking the model. Three of
  the 8B's four misleading trick answers are this kind.
- **Quantise** (Q4_K_M, about 5 GB) for speed, then re-benchmark to see what it costs.
- **Score "wrong" fairly:** the benchmark's source lists are incomplete. An LLM judge or a reviewer pass
  over the "wrong" answers would show the true picture.
