"""Make a Qwen3 model folder default to non-thinking mode before converting it to GGUF.

    python -m training.no_think_template outputs/islamic-qwen-merged

The training data is rendered with enable_thinking=False, so every prompt the model
learned from ends in an empty "<think></think>" block. LM Studio and llama.cpp use the
chat template stored in the GGUF, and Qwen3's template only adds that empty block when
the caller passes enable_thinking=false, which an OpenAI-style request can't do. This
flips the default so the served model sees prompts in the same shape it was trained on.
A caller can still turn thinking on with enable_thinking=true.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OLD = "enable_thinking is defined and enable_thinking is false"
NEW = "enable_thinking is not defined or enable_thinking is false"


def patch(folder: Path) -> bool:
    """Patch every copy of the template in the folder; True if the folder ends up non-thinking by default."""
    found = False
    jinja = folder / "chat_template.jinja"  # transformers 5 saves the template here
    if jinja.exists():
        text = jinja.read_text(encoding="utf-8")
        jinja.write_text(text.replace(OLD, NEW), encoding="utf-8")
        found = found or OLD in text or NEW in text
    path = folder / "tokenizer_config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    template = cfg.get("chat_template")
    if isinstance(template, str):
        cfg["chat_template"] = template.replace(OLD, NEW)
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        found = found or OLD in template or NEW in template
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="model folder containing tokenizer_config.json")
    args = ap.parse_args(argv)
    if not patch(args.folder):
        raise SystemExit(f"no Qwen3 thinking switch found in the chat template in {args.folder}")
    print(f"{args.folder}: chat template now defaults to non-thinking mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
