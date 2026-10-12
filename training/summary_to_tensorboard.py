"""Turn a finished run's training_summary.json into TensorBoard charts.

    python -m training.summary_to_tensorboard outputs/islamic-qwen-lora
    tensorboard --logdir outputs

New runs log to TensorBoard while they train (report_to: tensorboard). This is for runs
made before that, or with report_to: none: their loss history is in training_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Trainer's log keys -> the chart names its own TensorBoard logger uses, so old and new runs line up.
KEYS = {"loss": "train/loss", "grad_norm": "train/grad_norm", "learning_rate": "train/learning_rate",
        "eval_loss": "eval/loss"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="folder with training_summary.json (the trainer's --out)")
    args = ap.parse_args(argv)

    from torch.utils.tensorboard import SummaryWriter

    summary = json.loads((args.run_dir / "training_summary.json").read_text(encoding="utf-8"))
    writer = SummaryWriter(str(args.run_dir / "runs" / "from-summary"))
    written = 0
    for entry in summary["log"]:
        step = int(entry.get("step", 0))
        for key, tag in KEYS.items():
            if key in entry:
                writer.add_scalar(tag, float(entry[key]), step)
                written += 1
    if "peak_vram_gb" in summary:
        writer.add_scalar("gpu/peak_memory_gb", summary["peak_vram_gb"], int(summary["log"][-1].get("step", 0)))
    writer.close()
    print(f"wrote {written} points to {args.run_dir / 'runs' / 'from-summary'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
