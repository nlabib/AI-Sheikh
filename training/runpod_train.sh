#!/usr/bin/env bash
# Runs ON the RunPod pod (official PyTorch template): unpack code + data, install the training libraries
# into the template's Python, train the v2 adapter, pack it for download. Log: /workspace/train.log
set -e
cd /workspace
mkdir -p ai-sheikh && tar --no-same-owner -xzf ai-sheikh-code.tar.gz -C ai-sheikh
cd ai-sheikh
export HF_HOME=/workspace/hf-cache PYTHONUNBUFFERED=1
pip install --break-system-packages -q "transformers>=4.51" "peft>=0.11" "bitsandbytes>=0.43" "accelerate>=0.30" \
    pyyaml tensorboard nvidia-ml-py
python -c "import torch, transformers, peft; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'transformers', transformers.__version__, 'peft', peft.__version__)"
s=$(date +%s)
python -m training.train_cuda --config training/configs/qwen3-4b-24gb-v2.yaml
echo "TRAIN_MINUTES $(( ($(date +%s) - s) / 60 ))"
tar -czf /workspace/adapter-v2.tar.gz -C outputs islamic-qwen-lora-v2/adapter islamic-qwen-lora-v2/training_summary.json islamic-qwen-lora-v2/runs
ls -la /workspace/adapter-v2.tar.gz
echo TRAIN_DONE
