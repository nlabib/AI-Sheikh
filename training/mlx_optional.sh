#!/usr/bin/env bash
# LoRA fine-tuning on Apple Silicon (M1-M4) with MLX. Same data as the CUDA script.
#
#   pip install -r requirements-train-mlx.txt
#   python -m tools.build_sft
#   bash training/mlx_optional.sh
#
# Override anything with environment variables, e.g. ITERS=50 bash training/mlx_optional.sh
# for a quick smoke run. MLX cannot use an NVIDIA GPU, and the CUDA script cannot use a Mac GPU.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-data/sft}"
ADAPTER="${ADAPTER:-outputs/islamic-qwen-mlx-adapter}"
ITERS="${ITERS:-2300}"   # micro-batches, not optimiser steps: ~2 epochs of the default 1,140 examples
BATCH="${BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"  # effective batch = BATCH x GRAD_ACCUM, as in the CUDA config
LAYERS="${LAYERS:-16}"      # how many transformer layers get LoRA adapters
MAX_LEN="${MAX_LEN:-2048}"

# mlx_lm expects train.jsonl and valid.jsonl with a "messages" field; extra fields are ignored.
test -f "$DATA_DIR/train.jsonl" || { echo "run: python -m tools.build_sft"; exit 1; }

python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --iters "$ITERS" \
  --batch-size "$BATCH" \
  --grad-accumulation-steps "$GRAD_ACCUM" \
  --num-layers "$LAYERS" \
  --max-seq-length "$MAX_LEN" \
  --mask-prompt \
  --grad-checkpoint \
  --adapter-path "$ADAPTER"

echo
echo "Adapter saved to $ADAPTER. Try it:"
echo "  python -m mlx_lm generate --model $MODEL --adapter-path $ADAPTER --prompt 'What does Quran 2:255 say?'"
echo "Fuse it into a standalone model for LM Studio (MLX format):"
echo "  python -m mlx_lm fuse --model $MODEL --adapter-path $ADAPTER --save-path outputs/islamic-qwen-mlx-fused"
