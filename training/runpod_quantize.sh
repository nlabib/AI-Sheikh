#!/usr/bin/env bash
# On a fresh pod: merge a trained adapter into its base model, make the chat template non-thinking by default,
# convert to f16 GGUF and quantize to 4-bit (Q4_K_M) for LM Studio. No training; the adapter is uploaded.
#
#   upload ai-sheikh-code.tar.gz and <name>.tar.gz (the adapter tar) to /workspace, then
#   setsid bash runpod_quantize.sh ai-sheikh-8b-v2 > quant.log 2>&1 < /dev/null &
#   and fetch /workspace/<name>-Q4_K_M.gguf (check its sha256)
set -e
NAME=${1:?adapter folder name, e.g. ai-sheikh-8b-v2}
cd /workspace
mkdir -p ai-sheikh/outputs && tar --no-same-owner -xzf ai-sheikh-code.tar.gz -C ai-sheikh
tar --no-same-owner -xzf "$NAME.tar.gz" -C ai-sheikh/outputs
export HF_HOME=/workspace/hf-cache PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=1
s=$(date +%s)
[ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp llama.cpp
# llama-quantize is CPU-only; build it while the model downloads and merges.
(apt-get install -y -qq cmake > /dev/null 2>&1 || true
 cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /workspace/build.log 2>&1 &&
 cmake --build llama.cpp/build --target llama-quantize -j "$(nproc)" >> /workspace/build.log 2>&1) &
build=$!
pip install --break-system-packages -q "transformers>=4.51" "peft>=0.11" accelerate sentencepiece protobuf gguf hf_transfer
cd ai-sheikh
python -m training.merge_lora --adapter "outputs/$NAME/adapter" --out "outputs/$NAME-merged"
python -m training.no_think_template "outputs/$NAME-merged"
python /workspace/llama.cpp/convert_hf_to_gguf.py "outputs/$NAME-merged" --outfile "/workspace/$NAME-f16.gguf" --outtype f16
wait $build
/workspace/llama.cpp/build/bin/llama-quantize "/workspace/$NAME-f16.gguf" "/workspace/$NAME-Q4_K_M.gguf" Q4_K_M
sha256sum "/workspace/$NAME-Q4_K_M.gguf"
ls -la "/workspace/$NAME-Q4_K_M.gguf"
echo "QUANT_MINUTES $(( ($(date +%s) - s) / 60 ))"
echo QUANT_DONE
