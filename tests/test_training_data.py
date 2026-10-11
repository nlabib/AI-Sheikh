import json

import pytest

from training.data import IGNORE_INDEX, encode_example, load_config, pad_batch, validate_example


class CharTokenizer:
    """Stands in for a Hugging Face tokenizer: one token per character."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


MSGS = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "Answer [quran:1:1]"}]


def test_only_the_answer_carries_loss():
    enc = encode_example(CharTokenizer(), MSGS, max_len=500)
    trained = "".join(chr(t) for t, l in zip(enc.input_ids, enc.labels) if l != IGNORE_INDEX)
    assert trained == "Answer [quran:1:1]</assistant>"


def test_examples_whose_prompt_is_too_long_are_dropped():
    assert encode_example(CharTokenizer(), MSGS, max_len=10) is None


def test_padding():
    tok = CharTokenizer()
    a, b = encode_example(tok, MSGS, 500), encode_example(tok, MSGS[:2] + [{"role": "assistant", "content": "A"}], 500)
    batch = pad_batch([a, b], pad_id=0)
    assert len(set(map(len, batch["input_ids"]))) == 1
    assert batch["labels"][1][-1] == IGNORE_INDEX and batch["attention_mask"][1][-1] == 0


def test_validation():
    assert validate_example({"messages": MSGS}) == []
    assert validate_example({"messages": MSGS[:2]})


def test_configs_load_and_reject_typos(tmp_path):
    from pathlib import Path

    for path in Path("training/configs").glob("*.yaml"):
        cfg = load_config(path, {})
        assert cfg["model"].startswith("Qwen/") and cfg["batch_size"] >= 1
    bad = tmp_path / "bad.yaml"
    bad.write_text("learning_rate: 1\n")
    with pytest.raises(ValueError):
        load_config(bad, {})
    assert load_config(None, {"lr": 1e-4, "epochs": None})["lr"] == 1e-4
