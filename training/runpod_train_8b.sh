#!/usr/bin/env bash
# Round 2 with Qwen3-8B on a RunPod pod (official PyTorch template, RTX A6000 48 GB). Run it ON the pod, in /workspace:
#   git clone https://github.com/nlabib/AI-Sheikh && cd AI-Sheikh && setsid bash training/runpod_train_8b.sh > /workspace/train.log 2>&1 < /dev/null &
# Not yet run end to end: watch the first minutes of the log. Markers to look for: STEP_DONE_<name> and TRAIN_DONE.
# It trains, merges the adapter, converts to GGUF and quantizes, so only the ~5 GB Q4_K_M file has to come back.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/workspace/hf-cache PYTHONUNBUFFERED=1
step() { echo "STEP_DONE_$1 $(date -u +%H:%M:%S)"; }
free_gb=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt 70 ]; then echo "NEED_MORE_DISK: only ${free_gb} GB free on /workspace, need 70+ (100 recommended)"; exit 1; fi

pip install --break-system-packages -q -r requirements.txt "transformers>=4.51" "peft>=0.11" "bitsandbytes>=0.43" \
    "accelerate>=0.30" sentence-transformers tensorboard nvidia-ml-py
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
step install

python -m tools.fetch && python -m tools.ingest
step corpus
python -m tools.build_sft
step build_sft

python -m training.train_cuda --config training/configs/qwen3-8b-48gb-v2.yaml
step train

python -m training.merge_lora --adapter outputs/ai-sheikh-8b-v2/adapter --out outputs/ai-sheikh-8b-v2-merged
step merge

git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
pip install --break-system-packages -q -r /workspace/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python /workspace/llama.cpp/convert_hf_to_gguf.py outputs/ai-sheikh-8b-v2-merged --outfile outputs/ai-sheikh-8b-v2-f16.gguf --outtype f16
rm -rf outputs/ai-sheikh-8b-v2-merged
step convert
cmake -S /workspace/llama.cpp -B /workspace/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build /workspace/llama.cpp/build --target llama-quantize -j"$(nproc)" >/dev/null
/workspace/llama.cpp/build/bin/llama-quantize outputs/ai-sheikh-8b-v2-f16.gguf outputs/ai-sheikh-8b-v2-Q4_K_M.gguf Q4_K_M
rm -f outputs/ai-sheikh-8b-v2-f16.gguf
step quantize

tar -czf /workspace/adapter-8b-v2.tar.gz -C outputs ai-sheikh-8b-v2/adapter ai-sheikh-8b-v2/training_summary.json ai-sheikh-8b-v2/runs
ls -la /workspace/adapter-8b-v2.tar.gz outputs/ai-sheikh-8b-v2-Q4_K_M.gguf
echo TRAIN_DONE
