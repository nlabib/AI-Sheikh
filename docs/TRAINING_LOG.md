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

**Not done yet:** smoke-run final numbers, full run, GGUF conversion, base-vs-fine-tuned evaluation.
llama.cpp is cloned at `C:\Users\labib\src\llama.cpp` with its own `.venv`, because its requirements
pin a CPU torch that would overwrite the CUDA one. LM Studio has no Qwen3-4B, so the plan is to convert
both base and fine-tuned to GGUF the same way (`lms import`) for a fair comparison. Watch out for
Qwen3's thinking mode in the base model: `eval_answers` uses `max_tokens=1024`.

## Code fixes made along the way

1. **UTF-8 on Windows.** Several `read_text()`/`write_text()` calls had no `encoding=`, so Windows
   used cp1252 and crashed on Arabic/Bangla (3 failing tests). Added `encoding="utf-8"` in
   `tools/build_sft.py`, `training/data.py`, `training/merge_lora.py`, `app/embed.py`,
   `tools/evaluate.py`, `tools/eval_answers.py`.
2. **transformers 5 compatibility** in `training/train_cuda.py`: pass `warmup_steps=<ratio>` when
   `warmup_ratio` no longer exists.
3. **Validation loss is now actually measured.** `save_steps` (200) was larger than the whole run
   (144 steps), so eval never ran. Added `eval_on_start` and a final `trainer.evaluate()`.
