"""Merge a trained LoRA adapter into the base model so LM Studio / llama.cpp can load it.

    python -m training.merge_lora --adapter outputs/islamic-qwen-lora/adapter --out outputs/islamic-qwen-merged

The merge runs on CPU in 16-bit precision (needs ~2x the model size in RAM, ~16 GB
for Qwen3-4B). Next, convert the merged folder to GGUF with llama.cpp; the exact
commands are in docs/TRAINING_GUIDE.md, step 8.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", help="base model id (default: read from the adapter config)")
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = args.base or json.loads((args.adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
    print(f"loading {base} in float16 on CPU")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"merged model written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
