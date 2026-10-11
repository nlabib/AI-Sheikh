"""Training-data helpers that need no GPU and no torch, so they can be unit tested.

An SFT example is one JSON object per line:

    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}],
     "meta": {"kind": "...", "review_status": "template" | "reviewed", "sources": [...]}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

IGNORE_INDEX = -100  # label value the loss function skips


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            problems = validate_example(row)
            if problems:
                raise ValueError(f"{path}:{n}: {'; '.join(problems)}")
            rows.append(row)
    return rows


def validate_example(row: dict) -> list[str]:
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return ["needs a 'messages' list with at least a user and an assistant turn"]
    problems = []
    if msgs[-1].get("role") != "assistant":
        problems.append("last message must be from the assistant (that is what the model learns to write)")
    for m in msgs:
        if m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str):
            problems.append(f"bad message {str(m)[:60]}")
    return problems


def apply_template(tokenizer: Any, messages: list[dict], add_generation_prompt: bool) -> str:
    """Render with the model's own chat template. Qwen3 takes enable_thinking; others ignore it."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


@dataclass
class Encoded:
    input_ids: list[int]
    labels: list[int]

    @property
    def trained_tokens(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)


def encode_example(tokenizer: Any, messages: list[dict], max_len: int) -> Encoded | None:
    """Tokenise one conversation and mask everything except the final assistant reply.

    The model only gets a loss on the answer it should write, not on the system
    prompt, the sources or the question; otherwise it would spend capacity
    learning to reproduce the prompt. Returns None if the answer does not fit.
    """
    prompt_text = apply_template(tokenizer, messages[:-1], add_generation_prompt=True)
    full_text = apply_template(tokenizer, messages, add_generation_prompt=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template output for the prompt is not a prefix of the full conversation")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) >= max_len:
        return None
    full_ids = full_ids[:max_len]
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    enc = Encoded(full_ids, labels)
    return enc if enc.trained_tokens > 0 else None


def pad_batch(batch: list[Encoded], pad_id: int) -> dict[str, list[list[int]]]:
    width = max(len(e.input_ids) for e in batch)
    return {
        "input_ids": [e.input_ids + [pad_id] * (width - len(e.input_ids)) for e in batch],
        "labels": [e.labels + [IGNORE_INDEX] * (width - len(e.labels)) for e in batch],
        "attention_mask": [[1] * len(e.input_ids) + [0] * (width - len(e.input_ids)) for e in batch],
    }


DEFAULTS: dict[str, Any] = {
    "model": "Qwen/Qwen3-4B",
    "data": "data/sft/train.jsonl",
    "eval_data": "data/sft/valid.jsonl",
    "out": "outputs/islamic-qwen-lora",
    "max_len": 2048,
    "batch_size": 1,
    "grad_accum": 16,
    "epochs": 2,
    "max_steps": -1,
    "lr": 2.0e-4,
    "warmup_ratio": 0.03,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gradient_checkpointing": True,
    "load_in_4bit": True,
    "logging_steps": 10,
    "save_steps": 200,
    "seed": 42,
}


def load_config(path: Path | None, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if path:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        cfg.update(loaded)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
