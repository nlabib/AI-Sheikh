#!/usr/bin/env bash
# Train on a rented GPU pod (RunPod's official PyTorch template), so the home PC stays free.
#
#   On your PC:  git archive -o code.tar HEAD && tar --append -f code.tar data/sft/train.jsonl data/sft/valid.jsonl
#                gzip code.tar   then copy code.tar.gz and this script to the pod's /workspace (scp over SSH)
#   On the pod:  setsid bash runpod_train.sh training/configs/qwen3-8b-48gb-round2.yaml ai-sheikh-8b-v2 \
#                    > /workspace/train.log 2>&1 < /dev/null &
#   Then fetch /workspace/<out>.tar.gz (adapter + training_summary.json + TensorBoard logs) and delete the pod.
#
# Lessons from the first cloud run (docs/TRAINING_LOG.md): use batch 1 with gradient accumulation (bigger
# batches pad to the longest example), extract with --no-same-owner on /workspace, and launch detached.
set -e
CONFIG=${1:?config file}
OUT=${2:?output folder name under outputs/}
cd /workspace
mkdir -p ai-sheikh && tar --no-same-owner -xzf code.tar.gz -C ai-sheikh
cd ai-sheikh
export HF_HOME=/workspace/hf-cache PYTHONUNBUFFERED=1
pip install --break-system-packages -q "transformers>=4.51" "peft>=0.11" "bitsandbytes>=0.43" "accelerate>=0.30" \
    pyyaml tensorboard nvidia-ml-py
python -c "import torch, transformers, peft; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'transformers', transformers.__version__)"
s=$(date +%s)
python -m training.train_cuda --config "$CONFIG"
echo "TRAIN_MINUTES $(( ($(date +%s) - s) / 60 ))"
tar -czf "/workspace/$OUT.tar.gz" -C outputs "$OUT/adapter" "$OUT/training_summary.json" "$OUT/runs"
ls -la "/workspace/$OUT.tar.gz"
echo TRAIN_DONE
