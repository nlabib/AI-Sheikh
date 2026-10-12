"""QLoRA fine-tuning of Qwen3 on an NVIDIA GPU.

    python -m training.train_cuda --config training/configs/qwen3-4b-12gb.yaml --dry-run
    python -m training.train_cuda --config training/configs/qwen3-4b-12gb.yaml

Read docs/TRAINING_GUIDE.md first: it explains every step and setting.

QLoRA in one paragraph: the base model is loaded with its weights compressed to
4 bits (so a 4B model fits in ~3 GB of VRAM) and frozen. Small trainable
"adapter" matrices (LoRA) are attached to the attention and MLP layers. Only the
adapters learn, so training needs a fraction of the memory of full fine-tuning
and the result is a ~100 MB adapter folder rather than a new 8 GB model.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import time
from pathlib import Path

from training.data import encode_example, load_config, load_jsonl, pad_batch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="YAML file from training/configs/")
    ap.add_argument("--model")
    ap.add_argument("--data")
    ap.add_argument("--eval-data", dest="eval_data")
    ap.add_argument("--out")
    ap.add_argument("--max-len", dest="max_len", type=int)
    ap.add_argument("--batch-size", dest="batch_size", type=int)
    ap.add_argument("--grad-accum", dest="grad_accum", type=int)
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--max-steps", dest="max_steps", type=int, help="stop after N optimiser steps (try 20 first)")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--eval-steps", dest="eval_steps", type=int, help="measure valid loss every N steps")
    ap.add_argument("--report-to", dest="report_to", help='"tensorboard" (default) or "none"')
    ap.add_argument("--dry-run", action="store_true",
                    help="tokenise the data and report lengths, then stop; no GPU needed")
    return ap.parse_args(argv)


def encode_all(tokenizer, rows: list[dict], max_len: int) -> tuple[list, int]:
    encoded, dropped = [], 0
    for row in rows:
        enc = encode_example(tokenizer, row["messages"], max_len)
        if enc is None:
            dropped += 1
        else:
            encoded.append(enc)
    return encoded, dropped


def report(name: str, encoded: list, dropped: int) -> None:
    if not encoded:
        print(f"{name}: no usable examples ({dropped} dropped)")
        return
    lengths = sorted(len(e.input_ids) for e in encoded)
    trained = sum(e.trained_tokens for e in encoded)
    print(f"{name}: {len(encoded)} examples, {dropped} dropped as too long; "
          f"tokens per example median {lengths[len(lengths) // 2]}, max {lengths[-1]}; "
          f"{trained} answer tokens carry the loss")


def gpu_stats_callback(log_dir: Path):
    """A Trainer callback that writes GPU memory and utilisation to TensorBoard (Trainer doesn't log these)."""
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from transformers import TrainerCallback

    class GpuStats(TrainerCallback):
        def __init__(self):
            self.writer = SummaryWriter(str(log_dir))

        def on_log(self, args, state, control, logs=None, **kwargs):
            step = state.global_step
            self.writer.add_scalar("gpu/memory_allocated_gb", torch.cuda.memory_allocated() / 2**30, step)
            self.writer.add_scalar("gpu/memory_reserved_gb", torch.cuda.memory_reserved() / 2**30, step)
            self.writer.add_scalar("gpu/peak_memory_gb", torch.cuda.max_memory_allocated() / 2**30, step)
            try:  # needs nvidia-ml-py
                self.writer.add_scalar("gpu/utilization_pct", torch.cuda.utilization(), step)
            except Exception:
                pass
            self.writer.flush()

    return GpuStats()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, {k: v for k, v in vars(args).items() if k not in ("config", "dry_run")})
    print(json.dumps(cfg, indent=2))

    from transformers import AutoTokenizer  # imported here so --help works without transformers

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_jsonl(Path(cfg["data"]))
    eval_rows = load_jsonl(Path(cfg["eval_data"])) if cfg["eval_data"] and Path(cfg["eval_data"]).exists() else []
    template_share = sum(r.get("meta", {}).get("review_status") == "template" for r in train_rows) / len(train_rows)
    if template_share > 0.5:
        print(f"note: {template_share:.0%} of training examples are auto-generated templates. They teach the "
              "citation format, not Islamic knowledge; add reviewed examples (docs/TRAINING_GUIDE.md, step 5).")

    train_enc, train_drop = encode_all(tokenizer, train_rows, cfg["max_len"])
    eval_enc, eval_drop = encode_all(tokenizer, eval_rows, cfg["max_len"])
    report("train", train_enc, train_drop)
    report("valid", eval_enc, eval_drop)
    steps_per_epoch = math.ceil(len(train_enc) / (cfg["batch_size"] * cfg["grad_accum"]))
    total = cfg["max_steps"] if cfg["max_steps"] > 0 else math.ceil(steps_per_epoch * cfg["epochs"])
    print(f"{steps_per_epoch} optimiser steps per epoch, {total} in total")
    if args.dry_run:
        print("dry run: stopping before loading the model")
        return 0

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. This script needs an NVIDIA GPU; on a Mac use training/mlx_optional.sh.")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    bf16 = torch.cuda.is_bf16_supported()  # RTX 30xx and newer
    dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB, {'bf16' if bf16 else 'fp16'}")

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
    ) if cfg["load_in_4bit"] else None
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], quantization_config=quant, torch_dtype=dtype,
                                                 device_map={"": 0})
    model.config.use_cache = False  # the KV cache is for generation and conflicts with checkpointing
    if cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg["gradient_checkpointing"])
    elif cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    def collate(batch):
        padded = pad_batch(batch, tokenizer.pad_token_id)
        return {k: torch.tensor(v) for k, v in padded.items()}

    out = Path(cfg["out"])
    # transformers 5 dropped warmup_ratio; there a float warmup_steps below 1 means a ratio of total steps.
    warmup = ({"warmup_ratio": cfg["warmup_ratio"]}
              if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters
              else {"warmup_steps": cfg["warmup_ratio"]})
    training_args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        max_steps=cfg["max_steps"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        **warmup,
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=3,
        eval_strategy="steps" if eval_enc else "no",
        eval_steps=cfg["eval_steps"] if eval_enc else None,
        eval_on_start=bool(eval_enc),  # valid loss of the untouched model, to compare against the end
        bf16=bf16,
        fp16=not bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=cfg["gradient_checkpointing"],
        remove_unused_columns=False,
        report_to=cfg["report_to"],  # TensorBoard writes to <out>/runs/; view with: tensorboard --logdir outputs
        seed=cfg["seed"],
    )
    callbacks = [gpu_stats_callback(out / "runs" / "gpu")] if "tensorboard" in str(cfg["report_to"]) else []
    trainer = Trainer(model=model, args=training_args, train_dataset=train_enc,
                      eval_dataset=eval_enc or None, data_collator=collate, callbacks=callbacks)
    t0 = time.time()
    trainer.train()
    if eval_enc:  # short runs never reach eval_steps, so always measure valid loss once at the end
        trainer.evaluate()
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    summary = {"config": cfg, "minutes": round((time.time() - t0) / 60, 1),
               "train_examples": len(train_enc), "log": trainer.state.log_history,
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"adapter saved to {out / 'adapter'}; peak VRAM {summary['peak_vram_gb']} GB")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
