"""Merge a trained LoRA adapter into the base model so LM Studio / llama.cpp can load it.

    python -m training.merge_lora --adapter outputs/islamic-qwen-lora/adapter --out outputs/islamic-qwen-merged

The merge runs on CPU in 16-bit precision (needs ~2x the model size in RAM, ~16 GB
for Qwen3-4B). Next, convert the merged folder to GGUF with llama.cpp; the exact
commands are in docs/TRAINING_GUIDE.md, step 8.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", help="base model id (default: read from the adapter config)")
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = args.base or json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))["base_model_name_or_path"]
    print(f"loading {base} in float16 on CPU")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    # LoRA leaves the tokenizer unchanged. Copy the base model's original files rather than re-saving:
    # transformers 5 writes only tokenizer.json, and llama.cpp's GGUF converter needs vocab.json/merges.txt.
    if Path(base).is_dir():
        src = Path(base)
    else:
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(base, allow_patterns=TOKENIZER_FILES))
    (args.out / "chat_template.jinja").unlink(missing_ok=True)  # the copied tokenizer_config.json carries it
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copyfile(src / name, args.out / name)
    print(f"merged model written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
